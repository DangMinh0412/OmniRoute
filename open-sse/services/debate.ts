/**
 * Debate combo strategy — multi-round adversarial panel + judge synthesis.
 *
 * Extends the fusion strategy with structured debate rounds:
 *   1. Round 0 (fan-out): every panel model independently answers the user prompt.
 *   2. Round 1..N (debate): each model sees all prior-round responses (anonymized)
 *      and is asked to (a) identify agreements, (b) rebut errors, (c) refine its
 *      own position. This surfaces deeper disagreements and forces each model to
 *      defend or revise its claims.
 *   3. Judge synthesis: a judge model sees all rounds, traces how consensus emerged,
 *      resolves unresolved contradictions, and writes ONE authoritative final answer.
 *
 * Key properties:
 *   - Source anonymization ("Peer N") prevents brand-bias in both debate and judgment.
 *   - Panel calls are forced non-streaming in all rounds (we need complete prose).
 *   - The judge call honours the client's original stream flag.
 *   - Degrades gracefully: 0 survivors → 503; 1 survivor → direct answer; 0 debate
 *     rounds → behaves like fusion.
 *
 * Reuses collectPanel, extractPanelText, appendUserTurn from fusion.ts (DRY).
 */
import { collectPanel, extractPanelText, appendUserTurn, FUSION_DEFAULTS } from "./fusion.ts";
import type { FusionTuning } from "./fusion.ts";
import { errorResponse, sanitizeErrorMessage } from "../utils/error.ts";
import type { ComboLogger, HandleSingleModel } from "./combo/types.ts";

// ---------------------------------------------------------------------------
// Constants & types
// ---------------------------------------------------------------------------

export const DEBATE_DEFAULTS = {
  debateRounds: 2, // R0 (initial) + R1 (rebuttal); more rounds yield diminishing returns
  minPanel: 2,
  stragglerGraceMs: FUSION_DEFAULTS.stragglerGraceMs,
  panelHardTimeoutMs: FUSION_DEFAULTS.panelHardTimeoutMs,
  maxPanel: FUSION_DEFAULTS.maxPanel,
  // Consensus-stop: after each debate round, measure how much the surviving
  // answers converged (mean pairwise Jaccard over normalized token sets). When
  // it reaches this threshold the debate ends early — further rounds would only
  // burn tokens re-confirming agreement. Set to a value > 1 to disable early stop
  // and always run the full `debateRounds`. Default 0.85 = "strong agreement".
  consensusThreshold: 0.85,
  // Devil's-advocate fraction (0..1): the share of the panel that is assigned an
  // adversarial stance in debate rounds (R>=1) to counter conformity/sycophancy —
  // the primary failure mode named by "Should we be going MAD?" (arXiv 2311.17371),
  // where debating models drift into agreement that does NOT beat plain ensembling.
  // A dissent quota keeps genuine disagreement alive so consensus is earned, not
  // conformist. 0 = disabled (every model gets the neutral refine prompt → prior
  // behaviour). Default 0 for zero-regression; a caller opts in (e.g. 0.34 ≈ one
  // third of the panel argues the strongest opposing case each round).
  devilsAdvocateFraction: 0,
} as const;

export type DebateTuning = FusionTuning & {
  /** Total number of rounds including the initial fan-out. Min 1. Default 2. */
  debateRounds?: number;
  /**
   * Mean-pairwise-Jaccard threshold (0..1) at which the debate stops early
   * because the panel has converged. Values > 1 disable early stop. Default 0.85.
   */
  consensusThreshold?: number;
  /**
   * Fraction (0..1) of the panel assigned a devil's-advocate stance each debate
   * round to counter conformity (arXiv 2311.17371). 0 disables (default).
   */
  devilsAdvocateFraction?: number;
};

/** One panel member's contribution for a single round. */
export type PanelAnswer = {
  model: string;
  text: string;
};

/** All collected rounds, indexed by round number [0..N-1]. */
export type DebateHistory = PanelAnswer[][];

export type HandleDebateChatOptions = {
  body: Record<string, unknown>;
  models: string[];
  handleSingleModel: HandleSingleModel;
  log: ComboLogger;
  comboName?: string;
  judgeModel?: string | null;
  tuning?: DebateTuning | null;
};

// ---------------------------------------------------------------------------
// Prompt builders
// ---------------------------------------------------------------------------

/**
 * A debater's stance for a debate round. `standard` is the collaborative
 * refine-and-rebut role; `devils_advocate` is the contrarian role assigned to a
 * minority of the panel to counter premature convergence.
 */
export type DebateStance = "standard" | "devils_advocate";

/**
 * Prompt injected before round R (R >= 1). Each model sees all prior-round
 * answers anonymized as "Peer N". Asks the model to:
 *   (a) acknowledge points it agrees with
 *   (b) rebut specific factual / logical errors it sees
 *   (c) refine or defend its own position
 *
 * When `stance` is `devils_advocate`, the collaborative framing is replaced with
 * an adversarial one: the model must actively challenge the emerging consensus,
 * surface overlooked failure modes, and resist agreeing just to agree. This is
 * the direct countermeasure to the conformity / sycophancy failure mode that
 * research ("Should we be going MAD?", arXiv 2311.17371) identifies as the reason
 * naive multi-agent debate often fails to beat self-consistency: deliberately
 * adjusting panel agreement levels is what lets debate surpass non-debate
 * protocols. A minority of contrarians keeps the debate honest without derailing
 * it (the judge still weighs substance, not stance).
 *
 * Never reveals model identities — weights substance over brand reputation.
 */
export function buildDebateRoundPrompt(
  priorAnswers: PanelAnswer[],
  roundNum: number,
  totalRounds: number,
  stance: DebateStance = "standard"
): string {
  const peers = priorAnswers.map((a, i) => `[Peer ${i + 1}]\n${a.text}`).join("\n\n");

  const header = [
    `You are participating in a structured expert debate (Round ${roundNum} of ${totalRounds - 1}).`,
    "",
    `${priorAnswers.length} peers have answered the user's question independently. Their responses are anonymized below.`,
    "",
  ];

  const task =
    stance === "devils_advocate"
      ? [
          "Your role this round is DEVIL'S ADVOCATE. Your job is NOT to agree — it is to",
          "stress-test the emerging consensus so a wrong answer cannot survive by popularity:",
          "1. CHALLENGE — attack the strongest shared claim among the peers. Assume it is",
          "   subtly wrong and argue why: hidden assumptions, edge cases, counter-examples,",
          "   failure modes, or evidence the panel is taking on faith.",
          "2. STEELMAN THE MINORITY — if one peer disagrees with the rest, make the strongest",
          "   possible case FOR that dissent before dismissing it.",
          "3. POSITION — give your own answer, but do not soften a correct, unpopular",
          "   conclusion to match the group. Agree only where agreement genuinely survives",
          "   scrutiny; say so explicitly when it does.",
        ]
      : [
          "Your task in this round:",
          "1. AGREEMENTS — briefly identify specific points from the peers you agree with and why.",
          "2. REBUTTALS — identify any factual errors, logical gaps, or missing nuance in the peer responses. Be specific and precise; vague disagreement is unhelpful.",
          "3. REFINEMENT — revise, defend, or expand your own answer in light of the debate so far. If peers identified a genuine error in your prior reasoning, correct it openly.",
        ];

  return [
    ...header,
    ...task,
    "",
    "Do NOT mention that this is a multi-model system. Write as a single expert refining their position.",
    "Do NOT simply restate what peers said — add value through analysis, correction, or synthesis.",
    "",
    "=== PEER RESPONSES ===",
    peers,
    "=== END PEER RESPONSES ===",
    "",
    "Now write your refined answer to the user's original question:",
  ].join("\n");
}

/**
 * Deterministically select which panel members act as devil's advocates in a
 * debate round. Picks `ceil(fraction * n)` models by even spacing across the
 * (stably sorted) model list so the choice is reproducible and provider-diverse
 * rather than always hitting the first few. `fraction <= 0` selects none;
 * `fraction >= 1` selects all but one (never the whole panel — at least one
 * collaborative voice must remain or the round degenerates into pure objection).
 *
 * Pure + deterministic → unit-testable without any model call.
 */
export function selectDevilsAdvocates(models: string[], fraction: number): Set<string> {
  const n = models.length;
  if (n < 3 || !Number.isFinite(fraction) || fraction <= 0) return new Set();
  // Cap at n-1 so the panel never becomes all-contrarian.
  const count = Math.min(n - 1, Math.max(1, Math.ceil(fraction * n)));
  // Even spacing: pick indices 0, step, 2*step, … over a stable ordering.
  const ordered = [...models].sort();
  const step = n / count;
  const chosen = new Set<string>();
  for (let k = 0; k < count; k++) {
    chosen.add(ordered[Math.min(n - 1, Math.floor(k * step))]);
  }
  return chosen;
}

/**
 * Final judge prompt shown after all debate rounds.
 * The judge sees the full debate history (all rounds, all peers), traces how
 * consensus emerged, and writes ONE authoritative final answer grounded in
 * the best reasoning across the panel.
 */
export function buildDebateJudgePrompt(history: DebateHistory): string {
  const rounds = history
    .map((round, ri) => {
      const entries = round.map((a, pi) => `[Peer ${pi + 1}, Round ${ri}]\n${a.text}`).join("\n\n");
      return `--- Round ${ri} ---\n${entries}`;
    })
    .join("\n\n");

  const totalRounds = history.length;
  const totalPeers = history[0]?.length ?? 0;

  return [
    `You are the JUDGE in a ${totalRounds}-round multi-model debate. ${totalPeers} expert models independently answered the user's request, then debated each other across ${totalRounds} rounds. The full debate transcript is below.`,
    "",
    "Do NOT mention that multiple models or a debate occurred. Produce ONE authoritative final answer addressed directly to the user.",
    "",
    "Before writing, internally analyze the debate along these dimensions:",
    "• Convergence: what points did models AGREE on after debate? (higher confidence, but not automatically correct)",
    "• Resolved contradictions: disagreements that were settled by good reasoning during debate",
    "• Unresolved contradictions: genuine disagreements that persist — apply your OWN judgment to resolve them",
    "• Unique insights: important points raised by only one model that the others failed to rebut",
    "• Collective blind spots: important considerations that the ENTIRE panel missed",
    "",
    "You are not a vote-counter. Apply your own reasoning as a full participant:",
    "• If the consensus is wrong, override it and state what is correct.",
    "• If a lone dissenter is right against the majority, side with them.",
    "• If every model missed something important, add it.",
    "• Do not water down a correct answer to match panel agreement.",
    "",
    "=== FULL DEBATE TRANSCRIPT ===",
    rounds,
    "=== END TRANSCRIPT ===",
    "",
    "Now write the best possible final answer to the user's original request — more complete and correct than any single debater, and than the panel as a whole. No filler, no meta-commentary.",
  ].join("\n");
}

/**
 * Evaluator prompt for the (opt-in) verify pass. Shows the judge's DRAFT answer
 * to an independent critic model whose ONLY job is to find what is wrong or
 * missing — the "evaluator" half of Anthropic's evaluator-optimizer pattern.
 * It is deliberately adversarial: praise is worthless here, only actionable
 * defects move the answer forward. Returns a critique the refine step consumes.
 *
 * Pure: formats a prompt string, unit-testable without a model call.
 */
export function buildVerifyPrompt(draftAnswer: string): string {
  return [
    "You are a rigorous fact-checker and editor reviewing a DRAFT answer before it",
    "is delivered. Your job is NOT to rewrite it and NOT to praise it — only to find",
    "what would make it wrong, incomplete, or misleading for the user.",
    "",
    "Check specifically for:",
    "• Factual errors or unsupported claims stated as fact.",
    "• Logical gaps, contradictions, or unjustified leaps.",
    "• Missing caveats, edge cases, or important context the user needs.",
    "• Questions the user implicitly asked that the draft leaves unanswered.",
    "",
    "If the draft is genuinely sound, say so in one line. Otherwise, output a short,",
    "specific, prioritized list of concrete defects — each one actionable enough that",
    "an editor could fix it without guessing what you meant. No filler.",
    "",
    "=== DRAFT ANSWER ===",
    draftAnswer,
    "=== END DRAFT ===",
    "",
    "Now list the defects (or confirm the draft is sound):",
  ].join("\n");
}

/**
 * Optimizer prompt for the (opt-in) verify pass. Hands the judge's own DRAFT plus
 * the evaluator's critique back to the judge to produce the FINAL answer — the
 * "optimizer" half of evaluator-optimizer. The judge stays the authority: it
 * applies its own judgment to the critique, incorporating valid defects and
 * explicitly rejecting any it finds wrong, rather than blindly accepting them.
 *
 * Pure: formats a prompt string, unit-testable without a model call.
 */
export function buildRefinePrompt(draftAnswer: string, critique: string): string {
  return [
    "Below is a DRAFT answer you wrote, followed by an independent reviewer's",
    "critique. Produce the FINAL answer for the user.",
    "",
    "You remain the authority: weigh each point of the critique on its merits.",
    "Incorporate every valid correction and fill every genuine gap it identifies,",
    "but if a critique point is itself mistaken, silently disregard it — do not",
    "degrade a correct draft to satisfy a wrong objection. Do NOT mention the review,",
    "the draft, or that any revision happened; write the final answer directly to the",
    "user as a single authoritative response. No meta-commentary, no filler.",
    "",
    "=== YOUR DRAFT ===",
    draftAnswer,
    "=== END DRAFT ===",
    "",
    "=== REVIEWER CRITIQUE ===",
    critique,
    "=== END CRITIQUE ===",
    "",
    "Now write the final answer to the user's original request:",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Consensus measurement (pure — no LLM call, fully deterministic & testable)
// ---------------------------------------------------------------------------

/**
 * Normalize an answer to a set of content tokens for similarity scoring.
 * Lowercase, strip punctuation, split on whitespace, drop very short tokens
 * (articles/connectors add noise without signal). Deterministic.
 */
export function tokenizeForConsensus(text: string): Set<string> {
  const tokens = text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((t) => t.length >= 3);
  return new Set(tokens);
}

/** Jaccard similarity of two token sets: |A∩B| / |A∪B|. Returns 1 for two empty sets. */
export function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1;
  if (a.size === 0 || b.size === 0) return 0;
  let intersection = 0;
  for (const t of a) if (b.has(t)) intersection++;
  const union = a.size + b.size - intersection;
  return union === 0 ? 1 : intersection / union;
}

/**
 * Mean pairwise Jaccard similarity across a round's answers — a scalar in [0,1]
 * measuring how much the panel converged. 0 or 1 answers → 1 (nothing to
 * disagree about; caller handles the degenerate panel separately). Deterministic,
 * so consensus-stop is unit-testable without any model call.
 */
export function measureConsensus(answers: PanelAnswer[]): number {
  if (answers.length < 2) return 1;
  const sets = answers.map((a) => tokenizeForConsensus(a.text));
  let sum = 0;
  let pairs = 0;
  for (let i = 0; i < sets.length; i++) {
    for (let j = i + 1; j < sets.length; j++) {
      sum += jaccard(sets[i], sets[j]);
      pairs++;
    }
  }
  return pairs === 0 ? 1 : sum / pairs;
}

/**
 * Cosine similarity of two equal-length numeric vectors. Returns 0 when either
 * vector is missing, empty, mismatched in length, or has zero magnitude — a
 * conservative "no measurable agreement" default so a degenerate embedding can
 * never inflate the consensus score. Range for real inputs: [-1, 1].
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (!Array.isArray(a) || !Array.isArray(b)) return 0;
  if (a.length === 0 || a.length !== b.length) return 0;
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

/**
 * Semantic analogue of {@link measureConsensus}: mean pairwise cosine similarity
 * across per-answer embedding vectors, clamped to [0,1] (negative similarity —
 * genuinely opposed answers — floors at 0 so it composes with the same
 * `consensusThreshold`). This is the world-class upgrade over lexical Jaccard:
 * two answers that agree in meaning but differ in wording score HIGH (Jaccard
 * scored them low), and two that share vocabulary but contradict score LOW.
 *
 * Pure and deterministic given the vectors — the (impure) embedding call lives
 * in `src/lib/council/semanticConsensus.ts`, which falls back to
 * {@link measureConsensus} when no embedding provider is configured. `< 2`
 * vectors → 1 (nothing to disagree about), matching the Jaccard contract.
 */
export function measureConsensusFromVectors(vectors: number[][]): number {
  if (vectors.length < 2) return 1;
  let sum = 0;
  let pairs = 0;
  for (let i = 0; i < vectors.length; i++) {
    for (let j = i + 1; j < vectors.length; j++) {
      const c = cosineSimilarity(vectors[i], vectors[j]);
      sum += c < 0 ? 0 : c;
      pairs++;
    }
  }
  return pairs === 0 ? 1 : sum / pairs;
}

// ---------------------------------------------------------------------------
// Core handler
// ---------------------------------------------------------------------------

/**
 * Run one panel round. Returns only the successful answers (empty array on
 * total failure — caller decides whether to abort or proceed).
 */
async function runPanelRound(
  panelBody: Record<string, unknown>,
  models: string[],
  cfg: { minPanel: number; stragglerGraceMs: number; panelHardTimeoutMs: number },
  handleSingleModel: HandleSingleModel,
  log: ComboLogger,
  roundLabel: string
): Promise<PanelAnswer[]> {
  const calls = models.map((m) =>
    Promise.resolve(handleSingleModel(panelBody, m)).catch((e): Response => {
      log.warn("DEBATE", `${roundLabel} ${m} threw before collect`, {
        error: sanitizeErrorMessage(e as Error),
      });
      // Return a synthetic failed Response so collectPanel treats it as __error
      return new Response(null, { status: 500 });
    })
  );

  const settled = await collectPanel(calls, cfg);
  const answers: PanelAnswer[] = [];

  for (let i = 0; i < settled.length; i++) {
    const res = settled[i];
    const model = models[i];
    if (!res) {
      log.warn("DEBATE", `${roundLabel} ${model} dropped (straggler/timeout)`);
      continue;
    }
    const s = res as { __timeout?: true; __error?: unknown };
    if (s.__timeout) {
      log.warn("DEBATE", `${roundLabel} ${model} timed out`);
      continue;
    }
    if (s.__error) {
      log.warn("DEBATE", `${roundLabel} ${model} threw`, {
        error: sanitizeErrorMessage(s.__error as Error),
      });
      continue;
    }
    const resp = res as Response;
    if (!resp.ok) {
      log.warn("DEBATE", `${roundLabel} ${model} ${resp.status}`);
      continue;
    }
    try {
      const json = await resp.clone().json();
      const text = extractPanelText(json);
      if (text) {
        answers.push({ model, text });
        log.info("DEBATE", `${roundLabel} ${model} ok (${text.length} chars)`);
      } else {
        log.warn("DEBATE", `${roundLabel} ${model} empty content`);
      }
    } catch {
      log.warn("DEBATE", `${roundLabel} ${model} unparseable`);
    }
  }

  return answers;
}

/**
 * Handle a debate combo:
 *   1. Fan-out round 0 (all models independently).
 *   2. For each subsequent round, build a debate prompt from prior-round answers
 *      and fan out again to surviving models.
 *   3. Judge synthesizes from the full debate history.
 *
 * SSE events (x-debate-round, x-debate-done) are NOT emitted here — the
 * /api/v1/council endpoint wraps this function and emits SSE progress.
 * This function returns a single final Response (streaming or not, per client).
 */
export async function handleDebateChat({
  body,
  models,
  handleSingleModel,
  log,
  comboName,
  judgeModel,
  tuning,
}: HandleDebateChatOptions): Promise<Response> {
  const panel = Array.isArray(models) ? models.filter(Boolean) : [];
  if (panel.length === 0) {
    return errorResponse(400, "Debate combo has no models");
  }

  if (panel.length === 1) {
    log.info("DEBATE", "Single model — bypassing debate");
    return handleSingleModel(body, panel[0]);
  }

  const maxPanel = tuning?.maxPanel ?? DEBATE_DEFAULTS.maxPanel;
  if (panel.length > maxPanel) {
    log.warn(
      "DEBATE",
      `Combo "${comboName ?? ""}" panel=${panel.length} exceeds maxPanel=${maxPanel}`
    );
    return errorResponse(400, `Debate panel too large (${panel.length} models, max ${maxPanel})`);
  }

  const totalRounds = Math.max(1, tuning?.debateRounds ?? DEBATE_DEFAULTS.debateRounds);
  const consensusThreshold = tuning?.consensusThreshold ?? DEBATE_DEFAULTS.consensusThreshold;
  const cfg = {
    minPanel: Math.min(Math.max(1, tuning?.minPanel ?? DEBATE_DEFAULTS.minPanel), panel.length),
    stragglerGraceMs: tuning?.stragglerGraceMs ?? DEBATE_DEFAULTS.stragglerGraceMs,
    panelHardTimeoutMs: tuning?.panelHardTimeoutMs ?? DEBATE_DEFAULTS.panelHardTimeoutMs,
  };

  const hasExplicitJudge = Boolean(judgeModel && judgeModel.trim());
  log.info(
    "DEBATE",
    `Combo "${comboName ?? ""}" | panel=${panel.length} [${panel.join(", ")}] | judge=${hasExplicitJudge ? judgeModel : "auto"} | rounds=${totalRounds} | quorum=${cfg.minPanel}`
  );

  // Strip tool params from panel calls — we need prose, not tool invocations.
  const { tools: _tools, tool_choice: _tc, ...rest } = body;
  void _tools;
  void _tc;
  const panelBase: Record<string, unknown> = { ...rest, stream: false };

  // ---------------------------------------------------------------------------
  // Round 0: initial independent answers
  // ---------------------------------------------------------------------------
  const t0 = Date.now();
  const round0 = await runPanelRound(panelBase, panel, cfg, handleSingleModel, log, "R0");
  log.info("DEBATE", `R0 collected in ${Date.now() - t0}ms — ${round0.length}/${panel.length} ok`);

  if (round0.length === 0) {
    return errorResponse(503, "All debate panel models failed in round 0");
  }

  const history: DebateHistory = [round0];

  // ---------------------------------------------------------------------------
  // Rounds 1..N: adversarial refinement
  // ---------------------------------------------------------------------------
  for (let r = 1; r < totalRounds; r++) {
    const priorRound = history[history.length - 1];
    if (priorRound.length < 2) {
      log.warn(
        "DEBATE",
        `Round ${r}: only ${priorRound.length} survivor(s) — skipping remaining debate rounds`
      );
      break;
    }

    const debatePrompt = buildDebateRoundPrompt(priorRound, r, totalRounds);
    const debateBody = appendUserTurn(panelBase, debatePrompt);

    // Only models that survived the previous round participate in the next.
    const activeModels = priorRound.map((a) => a.model);
    const tR = Date.now();
    const roundAnswers = await runPanelRound(
      debateBody,
      activeModels,
      cfg,
      handleSingleModel,
      log,
      `R${r}`
    );
    log.info(
      "DEBATE",
      `R${r} collected in ${Date.now() - tR}ms — ${roundAnswers.length}/${activeModels.length} ok`
    );

    if (roundAnswers.length === 0) {
      log.warn("DEBATE", `Round ${r}: 0 survivors — stopping debate early, using prior round`);
      break;
    }

    history.push(roundAnswers);

    // Consensus-stop: if the surviving answers have converged past the threshold,
    // further rounds only re-confirm agreement — stop early and go to the judge.
    // Disabled when consensusThreshold > 1. Only meaningful with ≥2 answers.
    if (roundAnswers.length >= 2 && consensusThreshold <= 1) {
      const consensus = measureConsensus(roundAnswers);
      if (consensus >= consensusThreshold) {
        log.info(
          "DEBATE",
          `Round ${r}: consensus ${consensus.toFixed(3)} ≥ ${consensusThreshold} — stopping debate early`
        );
        break;
      }
      log.info(
        "DEBATE",
        `Round ${r}: consensus ${consensus.toFixed(3)} < ${consensusThreshold} — continuing`
      );
    }
  }

  // ---------------------------------------------------------------------------
  // Judge synthesis
  // ---------------------------------------------------------------------------
  const finalRound = history[history.length - 1];

  // Degenerate: single survivor across full debate history → answer directly
  // unless an explicit judge was configured.
  if (finalRound.length === 1 && !hasExplicitJudge) {
    log.info("DEBATE", `Only ${finalRound[0].model} survived — answering directly`);
    return handleSingleModel(body, finalRound[0].model);
  }

  // Choose the effective judge: explicit > panel[0] survivor > first survivor.
  const effectiveJudge = hasExplicitJudge
    ? (judgeModel as string).trim()
    : finalRound.some((a) => a.model === panel[0])
      ? panel[0]
      : finalRound[0].model;

  const judgeBody = appendUserTurn(body, buildDebateJudgePrompt(history));
  log.info(
    "DEBATE",
    `Judging ${history.length} rounds (${history.map((r) => r.length).join("→")} answers) with ${effectiveJudge}`
  );
  return handleSingleModel(judgeBody, effectiveJudge);
}
