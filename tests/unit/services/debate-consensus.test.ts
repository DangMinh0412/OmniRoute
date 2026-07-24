/**
 * Consensus measurement + consensus-stop for the debate strategy.
 *
 * The consensus functions (tokenizeForConsensus / jaccard / measureConsensus) are
 * pure and deterministic — no LLM call — so early-stop behavior is fully unit-
 * testable. These tests pin the math and the end-to-end early-stop wiring in
 * handleDebateChat (a panel that converges stops before running all rounds).
 */
import test from "node:test";
import assert from "node:assert/strict";

const {
  tokenizeForConsensus,
  jaccard,
  measureConsensus,
  cosineSimilarity,
  measureConsensusFromVectors,
  selectDevilsAdvocates,
  buildDebateRoundPrompt,
  buildVerifyPrompt,
  buildRefinePrompt,
  handleDebateChat,
} = await import("../../../open-sse/services/debate.ts");

const noop = () => {};
const log = { info: noop, warn: noop, debug: noop, error: noop };

type Body = Record<string, unknown>;

function okResponse(content: string): Response {
  return new Response(JSON.stringify({ choices: [{ message: { role: "assistant", content } }] }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

// --- Pure math ---------------------------------------------------------------

test("tokenizeForConsensus: lowercases, strips punctuation, drops short tokens", () => {
  const set = tokenizeForConsensus("The Quick, brown FOX! a to");
  assert.ok(set.has("the"));
  assert.ok(set.has("quick"));
  assert.ok(set.has("brown"));
  assert.ok(set.has("fox"));
  // "a"/"to" are < 3 chars → dropped
  assert.ok(!set.has("a"));
  assert.ok(!set.has("to"));
});

test("jaccard: identical sets = 1, disjoint = 0, two empty = 1", () => {
  assert.equal(jaccard(new Set(["a", "b"]), new Set(["a", "b"])), 1);
  assert.equal(jaccard(new Set(["a"]), new Set(["b"])), 0);
  assert.equal(jaccard(new Set(), new Set()), 1);
  assert.equal(jaccard(new Set(["a"]), new Set()), 0);
});

test("jaccard: partial overlap", () => {
  // {a,b,c} vs {b,c,d}: ∩={b,c}=2, ∪={a,b,c,d}=4 → 0.5
  assert.equal(jaccard(new Set(["a", "b", "c"]), new Set(["b", "c", "d"])), 0.5);
});

test("measureConsensus: <2 answers → 1", () => {
  assert.equal(measureConsensus([]), 1);
  assert.equal(measureConsensus([{ model: "a", text: "hello world" }]), 1);
});

test("measureConsensus: identical answers → 1, distinct → low", () => {
  const identical = measureConsensus([
    { model: "a", text: "the sky is blue today" },
    { model: "b", text: "the sky is blue today" },
  ]);
  assert.equal(identical, 1);

  const distinct = measureConsensus([
    { model: "a", text: "quantum entanglement physics theory" },
    { model: "b", text: "medieval european castle architecture" },
  ]);
  assert.ok(distinct < 0.2, `distinct answers should score low, got ${distinct}`);
});

// --- Semantic vector math (the world-class upgrade over lexical Jaccard) -----

test("cosineSimilarity: identical vectors = 1, orthogonal = 0, opposite = -1", () => {
  assert.equal(cosineSimilarity([1, 0], [1, 0]), 1);
  assert.equal(cosineSimilarity([1, 0], [0, 1]), 0);
  assert.equal(cosineSimilarity([1, 0], [-1, 0]), -1);
  // parallel but different magnitude → still 1 (direction only)
  assert.equal(cosineSimilarity([2, 0], [5, 0]), 1);
});

test("cosineSimilarity: guards — empty / mismatched length / zero vector → 0", () => {
  assert.equal(cosineSimilarity([], []), 0);
  assert.equal(cosineSimilarity([1, 2], [1, 2, 3]), 0);
  assert.equal(cosineSimilarity([0, 0], [1, 1]), 0);
});

test("measureConsensusFromVectors: <2 vectors → 1 (matches Jaccard contract)", () => {
  assert.equal(measureConsensusFromVectors([]), 1);
  assert.equal(measureConsensusFromVectors([[1, 0, 0]]), 1);
});

test("measureConsensusFromVectors: aligned vectors → high, opposed → floored at 0", () => {
  // Two answers that mean the same thing (aligned embeddings) score high even
  // though lexical Jaccard on different wording would score low.
  const aligned = measureConsensusFromVectors([
    [1, 0, 0],
    [0.99, 0.01, 0],
  ]);
  assert.ok(aligned > 0.99, `aligned vectors should score ~1, got ${aligned}`);

  // Genuinely opposed answers: negative cosine is clamped to 0 so it composes
  // with the same [0,1] consensusThreshold.
  const opposed = measureConsensusFromVectors([
    [1, 0, 0],
    [-1, 0, 0],
  ]);
  assert.equal(opposed, 0);
});

test("measureConsensusFromVectors: mean over pairs", () => {
  // 3 vectors: pair(0,1)=1, pair(0,2)=0, pair(1,2)=0 → mean = 1/3
  const score = measureConsensusFromVectors([
    [1, 0],
    [1, 0],
    [0, 1],
  ]);
  assert.ok(Math.abs(score - 1 / 3) < 1e-9, `expected 1/3, got ${score}`);
});

// --- Devil's-advocate diversity control (arXiv 2311.17371 countermeasure) ----

test("selectDevilsAdvocates: fraction <= 0 or tiny panel selects none", () => {
  assert.equal(selectDevilsAdvocates(["a", "b", "c"], 0).size, 0);
  assert.equal(selectDevilsAdvocates(["a", "b", "c"], -1).size, 0);
  // panel < 3 never gets an advocate (needs ≥1 collaborative voice + ≥1 dissent)
  assert.equal(selectDevilsAdvocates(["a", "b"], 0.5).size, 0);
});

test("selectDevilsAdvocates: caps at n-1 so the panel is never all-contrarian", () => {
  const all = selectDevilsAdvocates(["a", "b", "c", "d"], 1);
  assert.equal(all.size, 3, `fraction 1 on 4 models caps at n-1=3, got ${all.size}`);
});

test("selectDevilsAdvocates: deterministic count + membership for a given input", () => {
  const models = ["m1", "m2", "m3", "m4", "m5", "m6"];
  const a = selectDevilsAdvocates(models, 0.34);
  const b = selectDevilsAdvocates(models, 0.34);
  // ceil(0.34 * 6) = 3
  assert.equal(a.size, 3);
  // fully deterministic — same input → same set
  assert.deepEqual([...a].sort(), [...b].sort());
  // every chosen model is a real panel member
  for (const m of a) assert.ok(models.includes(m));
});

test("buildDebateRoundPrompt: devil's-advocate stance injects adversarial framing", () => {
  const prior = [
    { model: "a", text: "answer one" },
    { model: "b", text: "answer two" },
  ];
  const standard = buildDebateRoundPrompt(prior, 1, 2, "standard");
  const devil = buildDebateRoundPrompt(prior, 1, 2, "devils_advocate");

  assert.ok(standard.includes("AGREEMENTS"), "standard prompt keeps collaborative framing");
  assert.ok(!standard.includes("DEVIL'S ADVOCATE"));

  assert.ok(devil.includes("DEVIL'S ADVOCATE"), "devil prompt sets the adversarial role");
  assert.ok(devil.includes("CHALLENGE"), "devil prompt asks to attack the shared claim");
  // both still show the peer responses + ask for a final answer
  assert.ok(devil.includes("answer one") && devil.includes("answer two"));
});

// --- Evaluator-optimizer verify pass -----------------------------------------

test("buildVerifyPrompt: adversarial critic framing embeds the draft", () => {
  const p = buildVerifyPrompt("the earth is 5000 years old");
  // Frames the critic as a defect-finder, not a rewriter or praiser.
  assert.ok(/fact-checker|editor|reviewing/i.test(p));
  assert.ok(p.includes("the earth is 5000 years old"), "draft is embedded for review");
  // Asks for actionable defects (factual errors / gaps / missing caveats).
  assert.ok(/Factual errors|unsupported claims/i.test(p));
});

test("buildRefinePrompt: judge stays authority, embeds draft + critique", () => {
  const p = buildRefinePrompt("draft text here", "defect: missing a caveat");
  assert.ok(p.includes("draft text here"), "draft embedded");
  assert.ok(p.includes("defect: missing a caveat"), "critique embedded");
  // The judge must weigh the critique, not blindly accept it.
  assert.ok(/authority|merits|disregard/i.test(p));
  // No meta-commentary about the review leaking to the user.
  assert.ok(/Do NOT mention the review|write the final answer directly/i.test(p));
});

// --- End-to-end early stop ---------------------------------------------------

test("handleDebateChat: converged panel stops before running all rounds", async () => {
  // All models return the SAME text every round → consensus = 1 ≥ 0.85 → early stop
  // after round 1. With debateRounds=5 we'd otherwise see R0..R4; early stop means
  // the panel is called for R0 + R1 only (then judge).
  const roundsSeen = new Set<string>();
  let panelCalls = 0;
  const handleSingleModel = async (b: Body, m: string) => {
    const msgs = (b.messages as Array<{ content: string }>) ?? [];
    const last = msgs[msgs.length - 1]?.content ?? "";
    const isJudge = last.includes("JUDGE");
    if (isJudge) return okResponse("FINAL");
    panelCalls++;
    // identical answer regardless of model → forces consensus
    return okResponse("the answer is exactly forty two units precisely");
  };

  const res = await handleDebateChat({
    body: { messages: [{ role: "user", content: "q" }] },
    models: ["p/a", "p/b"],
    handleSingleModel,
    log,
    tuning: { debateRounds: 5, consensusThreshold: 0.85 },
  });

  assert.equal(res.status, 200);
  // R0 (2 calls) + R1 (2 calls) = 4 panel calls, then early stop. Without early stop
  // a 5-round debate on 2 models would make 10 panel calls.
  assert.equal(panelCalls, 4, `expected early stop after R1 (4 panel calls), got ${panelCalls}`);
  void roundsSeen;
});

test("handleDebateChat: consensusThreshold > 1 disables early stop (runs all rounds)", async () => {
  let panelCalls = 0;
  const handleSingleModel = async (b: Body, m: string) => {
    const msgs = (b.messages as Array<{ content: string }>) ?? [];
    const last = msgs[msgs.length - 1]?.content ?? "";
    if (last.includes("JUDGE")) return okResponse("FINAL");
    panelCalls++;
    return okResponse("identical text every round for all models here");
  };

  await handleDebateChat({
    body: { messages: [{ role: "user", content: "q" }] },
    models: ["p/a", "p/b"],
    handleSingleModel,
    log,
    tuning: { debateRounds: 3, consensusThreshold: 1.01 },
  });

  // 3 rounds × 2 models = 6 panel calls (no early stop despite identical answers).
  assert.equal(panelCalls, 6, `expected all 3 rounds (6 calls), got ${panelCalls}`);
});
