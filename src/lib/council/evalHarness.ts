/**
 * Council eval harness — measures whether the multi-model council actually
 * produces BETTER answers than a single model, instead of merely asserting it.
 *
 * This is the "measurement" piece that turns "we believe debate helps" into
 * "debate won N% of head-to-head comparisons on this suite". It implements a
 * pairwise LM-as-judge win-rate protocol, the same family used by MT-Bench /
 * AlpacaEval:
 *
 *   For each eval case:
 *     1. Get answer A from the baseline (single model).
 *     2. Get answer B from the council (debate + judge synthesis).
 *     3. Ask a judge model which answer better serves the user.
 *
 * POSITION BIAS: LM judges have a well-documented tendency to favour whichever
 * answer is shown first (or second). Naively asking once would bake that bias
 * into the score. This harness mitigates it the standard way: it judges EACH
 * pair TWICE with the two answers in swapped positions, and only credits a win
 * when the judge is CONSISTENT across both orderings. An inconsistent verdict
 * (the judge just followed position) is scored a tie. This makes the win-rate a
 * conservative lower bound on the council's true advantage, which is the honest
 * direction to err.
 *
 * The core is PURE + fully injectable: callers pass a `baseline` completion fn,
 * a `council` completion fn, and a `judge` fn. That keeps it unit-testable with
 * deterministic fakes (no network, no model keys) while the same code path runs
 * against real models when wired to the live router. Nothing here calls a model
 * directly — the harness only orchestrates and scores.
 *
 * @module lib/council/evalHarness
 */

/** A single evaluation prompt. `id` is stable so results can be joined/diffed. */
export interface CouncilEvalCase {
  id: string;
  /** The user question posed identically to baseline and council. */
  prompt: string;
  /** Optional freeform tags for slicing results (e.g. "reasoning", "coding"). */
  tags?: string[];
}

/**
 * Produces an answer for a prompt. Both the single-model baseline and the
 * council are expressed as this same shape so the harness treats them
 * symmetrically. Implementations may call a live model or return a canned
 * string (tests). Must resolve to the answer text; throw to signal failure.
 */
export type CompletionFn = (prompt: string) => Promise<string>;

/** Which side the judge picked for one comparison. */
export type JudgeChoice = "A" | "B" | "tie";

/**
 * Judges which of two answers better serves the user's question. `a` and `b`
 * are the two candidate answers in the order the judge sees them. The harness
 * calls this twice with swapped arguments to cancel position bias, so a correct
 * implementation must judge purely on the content of `a` vs `b`, not on any
 * external knowledge of which is baseline vs council.
 */
export type JudgeFn = (question: string, a: string, b: string) => Promise<JudgeChoice>;

/** The outcome of one case after the two-ordering, bias-cancelled comparison. */
export type CaseVerdict = "council" | "baseline" | "tie" | "error";

export interface CouncilEvalCaseResult {
  id: string;
  verdict: CaseVerdict;
  /** First pass: judge saw (baseline=A, council=B). */
  order1: JudgeChoice | null;
  /** Second pass: judge saw (council=A, baseline=B). */
  order2: JudgeChoice | null;
  /** Present when verdict === "error": what went wrong. */
  error?: string;
  tags?: string[];
}

export interface CouncilEvalSummary {
  total: number;
  councilWins: number;
  baselineWins: number;
  ties: number;
  errors: number;
  /**
   * Council win-rate over DECIDED (non-error) comparisons, counting a tie as
   * half a win for each side — the standard AlpacaEval convention. Range [0,1].
   * `null` when there are no decided comparisons (every case errored).
   */
  winRate: number | null;
  /**
   * Council win-rate ignoring ties entirely: wins / (wins + losses). A stricter
   * "when someone won, how often was it the council" view. `null` when no case
   * had a decisive (non-tie) winner.
   */
  decisiveWinRate: number | null;
}

export interface CouncilEvalReport {
  summary: CouncilEvalSummary;
  results: CouncilEvalCaseResult[];
}

/**
 * Resolve two swapped-order judge calls into a single bias-cancelled verdict.
 *
 * Pass 1 shows the judge (A=baseline, B=council). Pass 2 shows (A=council,
 * B=baseline). We translate each raw A/B/tie choice into "who did the judge
 * prefer" in council-vs-baseline terms, then require agreement:
 *   - both passes prefer council            → "council"
 *   - both passes prefer baseline           → "baseline"
 *   - anything else (incl. any tie, or the
 *     two passes disagreeing = pure position
 *     bias)                                  → "tie"
 *
 * Pure + exported so the bias logic is unit-testable in isolation.
 */
export function resolveVerdict(order1: JudgeChoice, order2: JudgeChoice): CaseVerdict {
  // order1: A=baseline, B=council  → "B" means council preferred.
  const pass1: "council" | "baseline" | "tie" =
    order1 === "B" ? "council" : order1 === "A" ? "baseline" : "tie";
  // order2: A=council, B=baseline  → "A" means council preferred.
  const pass2: "council" | "baseline" | "tie" =
    order2 === "A" ? "council" : order2 === "B" ? "baseline" : "tie";

  if (pass1 === "council" && pass2 === "council") return "council";
  if (pass1 === "baseline" && pass2 === "baseline") return "baseline";
  return "tie";
}

/**
 * Run the full pairwise win-rate evaluation over a set of cases.
 *
 * For each case: fetch the baseline answer and the council answer, then judge
 * them in both orderings and resolve a bias-cancelled verdict. A failure in
 * either completion or in the judge marks that single case as an "error" and
 * the run continues — one bad case never aborts the suite.
 *
 * Cases run sequentially by default (LLM backends are usually the bottleneck
 * and rate-limited); pass `concurrency` > 1 to parallelize in fixed-size
 * batches when the backend can take it.
 */
export async function runCouncilWinRateEval(
  cases: CouncilEvalCase[],
  deps: { baseline: CompletionFn; council: CompletionFn; judge: JudgeFn },
  options: { concurrency?: number } = {}
): Promise<CouncilEvalReport> {
  const concurrency = Math.max(1, Math.floor(options.concurrency ?? 1));
  const results: CouncilEvalCaseResult[] = new Array(cases.length);

  async function evaluateOne(evalCase: CouncilEvalCase, index: number): Promise<void> {
    try {
      // Fetch both answers. We deliberately await them independently rather than
      // Promise.all so a baseline failure gives a precise error message.
      const baselineAnswer = await deps.baseline(evalCase.prompt);
      const councilAnswer = await deps.council(evalCase.prompt);

      const order1 = await deps.judge(evalCase.prompt, baselineAnswer, councilAnswer);
      const order2 = await deps.judge(evalCase.prompt, councilAnswer, baselineAnswer);
      const verdict = resolveVerdict(order1, order2);

      results[index] = {
        id: evalCase.id,
        verdict,
        order1,
        order2,
        ...(evalCase.tags ? { tags: evalCase.tags } : {}),
      };
    } catch (err) {
      results[index] = {
        id: evalCase.id,
        verdict: "error",
        order1: null,
        order2: null,
        error: err instanceof Error ? err.message : String(err),
        ...(evalCase.tags ? { tags: evalCase.tags } : {}),
      };
    }
  }

  for (let start = 0; start < cases.length; start += concurrency) {
    const batch = cases.slice(start, start + concurrency);
    await Promise.all(batch.map((c, i) => evaluateOne(c, start + i)));
  }

  return { summary: summarize(results), results };
}

/** Fold per-case results into the aggregate win-rate summary. Pure + exported. */
export function summarize(results: CouncilEvalCaseResult[]): CouncilEvalSummary {
  let councilWins = 0;
  let baselineWins = 0;
  let ties = 0;
  let errors = 0;

  for (const r of results) {
    switch (r.verdict) {
      case "council":
        councilWins++;
        break;
      case "baseline":
        baselineWins++;
        break;
      case "tie":
        ties++;
        break;
      case "error":
        errors++;
        break;
    }
  }

  const decided = councilWins + baselineWins + ties;
  const decisive = councilWins + baselineWins;

  return {
    total: results.length,
    councilWins,
    baselineWins,
    ties,
    errors,
    // Tie = half a win each (AlpacaEval convention).
    winRate: decided > 0 ? (councilWins + ties / 2) / decided : null,
    decisiveWinRate: decisive > 0 ? councilWins / decisive : null,
  };
}
