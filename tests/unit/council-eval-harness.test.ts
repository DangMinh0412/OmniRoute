/**
 * Council eval harness — pairwise LM-judge win-rate protocol.
 *
 * These tests pin the two pieces that make the harness trustworthy:
 *   1. `resolveVerdict` — the position-bias cancellation: a win counts ONLY when
 *      the judge is consistent across both answer orderings; a flipped/tied judge
 *      is scored a tie. This is the whole reason the harness is credible, so it
 *      gets exhaustive coverage of the 3×3 (order1 × order2) verdict matrix.
 *   2. `summarize` — the AlpacaEval-style aggregation (tie = half a win, decisive
 *      win-rate ignores ties, null when nothing is decided).
 *   3. `runCouncilWinRateEval` — end-to-end with deterministic fake caller/judge:
 *      swapped-order judging, per-case error isolation, tag passthrough.
 *   4. `parseJudgeChoice` / `buildJudgePrompt` — deterministic verdict parsing +
 *      neutral prompt (no system-identity leak that would defeat the blind test).
 *
 * All fakes are synchronous + deterministic — no model, no network, no keys.
 */
import test from "node:test";
import assert from "node:assert/strict";

import {
  resolveVerdict,
  summarize,
  runCouncilWinRateEval,
  type CouncilEvalCase,
  type CouncilEvalCaseResult,
  type JudgeChoice,
  type JudgeFn,
} from "../../src/lib/council/evalHarness.ts";
import { buildJudgePrompt, parseJudgeChoice } from "../../src/lib/council/evalJudge.ts";

// --- resolveVerdict: the full bias-cancellation matrix ------------------------

test("resolveVerdict: council wins ONLY when consistent across both orderings", () => {
  // order1: A=baseline B=council → "B" = council preferred
  // order2: A=council  B=baseline → "A" = council preferred
  assert.equal(resolveVerdict("B", "A"), "council");
});

test("resolveVerdict: baseline wins ONLY when consistent across both orderings", () => {
  // order1 "A" = baseline preferred; order2 "B" = baseline preferred
  assert.equal(resolveVerdict("A", "B"), "baseline");
});

test("resolveVerdict: a judge that just followed position resolves to tie", () => {
  // "picked A both times" = pure position bias (baseline then council) → tie
  assert.equal(resolveVerdict("A", "A"), "tie");
  // "picked B both times" = pure position bias (council then baseline) → tie
  assert.equal(resolveVerdict("B", "B"), "tie");
});

test("resolveVerdict: any tie in either pass → tie (no half-credit for a single win)", () => {
  const choices: JudgeChoice[] = ["A", "B", "tie"];
  for (const o1 of choices) {
    for (const o2 of choices) {
      const v = resolveVerdict(o1, o2);
      // The only non-tie outcomes are the two consistent-preference corners.
      if (o1 === "B" && o2 === "A") assert.equal(v, "council");
      else if (o1 === "A" && o2 === "B") assert.equal(v, "baseline");
      else assert.equal(v, "tie", `expected tie for order1=${o1} order2=${o2}`);
    }
  }
});

// --- summarize: AlpacaEval-style aggregation ----------------------------------

function res(id: string, verdict: CouncilEvalCaseResult["verdict"]): CouncilEvalCaseResult {
  return { id, verdict, order1: null, order2: null };
}

test("summarize: counts each verdict class and computes both win-rates", () => {
  const s = summarize([
    res("1", "council"),
    res("2", "council"),
    res("3", "baseline"),
    res("4", "tie"),
  ]);
  assert.equal(s.total, 4);
  assert.equal(s.councilWins, 2);
  assert.equal(s.baselineWins, 1);
  assert.equal(s.ties, 1);
  assert.equal(s.errors, 0);
  // decided = 2+1+1 = 4; winRate = (2 + 1/2) / 4 = 0.625
  assert.equal(s.winRate, 0.625);
  // decisive = 2+1 = 3; decisiveWinRate = 2/3
  assert.ok(Math.abs((s.decisiveWinRate ?? 0) - 2 / 3) < 1e-9);
});

test("summarize: errors are excluded from win-rate denominators", () => {
  const s = summarize([res("1", "council"), res("2", "error"), res("3", "error")]);
  assert.equal(s.errors, 2);
  // Only one decided case (council) → winRate 1, decisiveWinRate 1
  assert.equal(s.winRate, 1);
  assert.equal(s.decisiveWinRate, 1);
});

test("summarize: all-error run yields null win-rates (never divides by zero)", () => {
  const s = summarize([res("1", "error"), res("2", "error")]);
  assert.equal(s.winRate, null);
  assert.equal(s.decisiveWinRate, null);
});

test("summarize: all-tie run has a 0.5 winRate but null decisiveWinRate", () => {
  const s = summarize([res("1", "tie"), res("2", "tie")]);
  assert.equal(s.winRate, 0.5);
  assert.equal(s.decisiveWinRate, null);
});

// --- runCouncilWinRateEval: end-to-end with deterministic fakes ---------------

const CASES: CouncilEvalCase[] = [
  { id: "c1", prompt: "q1", tags: ["reasoning"] },
  { id: "c2", prompt: "q2" },
];

/**
 * A judge that ALWAYS prefers whichever answer contains the marker "COUNCIL",
 * regardless of position. Because the harness swaps positions, this correctly
 * resolves to a consistent council win in both orderings.
 */
const contentJudge: JudgeFn = async (_q, a, b) => {
  const aCouncil = a.includes("COUNCIL");
  const bCouncil = b.includes("COUNCIL");
  if (aCouncil && !bCouncil) return "A";
  if (bCouncil && !aCouncil) return "B";
  return "tie";
};

test("runCouncilWinRateEval: a content-based judge credits the council consistently", async () => {
  const report = await runCouncilWinRateEval(CASES, {
    baseline: async (p) => `baseline answer to ${p}`,
    council: async (p) => `COUNCIL answer to ${p}`,
    judge: contentJudge,
  });
  assert.equal(report.summary.total, 2);
  assert.equal(report.summary.councilWins, 2);
  assert.equal(report.summary.winRate, 1);
  // Tag passthrough survives to the result row.
  const c1 = report.results.find((r) => r.id === "c1");
  assert.deepEqual(c1?.tags, ["reasoning"]);
});

test("runCouncilWinRateEval: a position-only judge is neutralised to ties", async () => {
  // This judge ALWAYS says "A" — pure position bias. Bias cancellation must
  // turn every case into a tie, not a baseline sweep.
  const positionJudge: JudgeFn = async () => "A";
  const report = await runCouncilWinRateEval(CASES, {
    baseline: async () => "base",
    council: async () => "coun",
    judge: positionJudge,
  });
  assert.equal(report.summary.ties, 2);
  assert.equal(report.summary.councilWins, 0);
  assert.equal(report.summary.baselineWins, 0);
  assert.equal(report.summary.winRate, 0.5);
});

test("runCouncilWinRateEval: a failing completion isolates to one error case", async () => {
  const report = await runCouncilWinRateEval(CASES, {
    baseline: async (p) => {
      if (p === "q1") throw new Error("baseline boom");
      return "ok";
    },
    council: async () => "COUNCIL",
    judge: contentJudge,
  });
  const c1 = report.results.find((r) => r.id === "c1");
  const c2 = report.results.find((r) => r.id === "c2");
  assert.equal(c1?.verdict, "error");
  assert.match(c1?.error ?? "", /baseline boom/);
  // The other case still ran and was judged.
  assert.equal(c2?.verdict, "council");
  assert.equal(report.summary.errors, 1);
});

test("runCouncilWinRateEval: judges each case in BOTH orderings (swap verified)", async () => {
  const seen: Array<{ a: string; b: string }> = [];
  const recordingJudge: JudgeFn = async (_q, a, b) => {
    seen.push({ a, b });
    return "tie";
  };
  await runCouncilWinRateEval([{ id: "only", prompt: "p" }], {
    baseline: async () => "BASE",
    council: async () => "COUN",
    judge: recordingJudge,
  });
  // Two judge calls: (baseline, council) then (council, baseline).
  assert.equal(seen.length, 2);
  assert.deepEqual(seen[0], { a: "BASE", b: "COUN" });
  assert.deepEqual(seen[1], { a: "COUN", b: "BASE" });
});

test("runCouncilWinRateEval: concurrency > 1 preserves per-case result order", async () => {
  const manyCases: CouncilEvalCase[] = Array.from({ length: 6 }, (_, i) => ({
    id: `k${i}`,
    prompt: `p${i}`,
  }));
  const report = await runCouncilWinRateEval(
    manyCases,
    {
      baseline: async () => "base",
      council: async () => "COUNCIL",
      judge: contentJudge,
    },
    { concurrency: 3 }
  );
  assert.deepEqual(
    report.results.map((r) => r.id),
    manyCases.map((c) => c.id)
  );
  assert.equal(report.summary.councilWins, 6);
});

// --- evalJudge: prompt neutrality + verdict parsing ---------------------------

test("buildJudgePrompt: embeds both answers and leaks NO system identity", () => {
  const p = buildJudgePrompt("What is 2+2?", "four", "the answer is 4");
  assert.match(p, /What is 2\+2\?/);
  assert.match(p, /ANSWER A/);
  assert.match(p, /ANSWER B/);
  assert.match(p, /four/);
  // The judge must not learn which side is the council / baseline.
  assert.doesNotMatch(p, /council/i);
  assert.doesNotMatch(p, /baseline/i);
});

test("parseJudgeChoice: extracts A / B / TIE across common reply shapes", () => {
  assert.equal(parseJudgeChoice("A"), "A");
  assert.equal(parseJudgeChoice("B"), "B");
  assert.equal(parseJudgeChoice("TIE"), "tie");
  assert.equal(parseJudgeChoice("**A**\nBecause it is more complete."), "A");
  assert.equal(parseJudgeChoice("B.\nMore accurate."), "B");
  assert.equal(parseJudgeChoice("tie — both equal"), "tie");
  assert.equal(parseJudgeChoice("Equal"), "tie");
});

test("parseJudgeChoice: ambiguous / empty replies fall back to tie (never a fake win)", () => {
  assert.equal(parseJudgeChoice(""), "tie");
  assert.equal(parseJudgeChoice("I think it depends"), "tie");
  assert.equal(parseJudgeChoice(null as unknown as string), "tie");
  // A verdict buried below the first line is NOT read (contract: first line only).
  assert.equal(parseJudgeChoice("Let me think.\nA"), "tie");
});
