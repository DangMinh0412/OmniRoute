/**
 * Judge prompt + parser for the council eval harness.
 *
 * Separated from evalHarness.ts because the harness is transport-agnostic (it
 * only knows the `JudgeFn` contract), while THIS module knows how to phrase the
 * pairwise-comparison prompt to a real LLM and how to parse its verdict back
 * into the `JudgeChoice` the harness expects. Both pieces are pure and
 * unit-testable without a model call.
 *
 * @module lib/council/evalJudge
 */

import type { JudgeChoice } from "./evalHarness.ts";

/**
 * Build the pairwise-comparison prompt shown to the judge model. The judge is
 * asked to pick which of two answers better serves the user's question, or to
 * call it a tie. It is told NOTHING about which answer came from which system —
 * the harness relies on that neutrality (and on judging both orderings) to
 * cancel position/brand bias.
 *
 * The output contract is deliberately rigid ("respond with exactly one of: A,
 * B, TIE") so `parseJudgeChoice` can extract the verdict deterministically.
 */
export function buildJudgePrompt(question: string, answerA: string, answerB: string): string {
  return [
    "You are an impartial judge comparing two answers to the same user question.",
    "Decide which answer would be more helpful, correct, and complete for the user.",
    "",
    "Judge ONLY on the substance of the two answers. Do not favour an answer for",
    "being longer, more confident, or appearing in a particular position. If both",
    "are equally good (or equally bad), call it a tie.",
    "",
    "=== USER QUESTION ===",
    question,
    "",
    "=== ANSWER A ===",
    answerA,
    "",
    "=== ANSWER B ===",
    answerB,
    "=== END ===",
    "",
    "Respond with EXACTLY one token on the first line — one of: A, B, TIE.",
    "You may add a one-line reason on the next line, but the first line must be the verdict.",
  ].join("\n");
}

/**
 * Parse a judge model's raw reply into a `JudgeChoice`. Robust to the common
 * shapes a model returns: a bare token, a verdict line, markdown emphasis, or a
 * short "Answer: A" style preamble. Falls back to "tie" when the reply is
 * ambiguous or empty — an unparseable judgment must never be silently scored as
 * a win for either side.
 *
 * The parse reads the FIRST line (per the prompt contract) and looks for a
 * standalone A / B / TIE token, tolerating surrounding punctuation/markdown.
 */
export function parseJudgeChoice(raw: string): JudgeChoice {
  if (typeof raw !== "string") return "tie";
  const firstLine = raw.trim().split(/\r?\n/, 1)[0] ?? "";
  // Strip markdown emphasis + common punctuation, collapse to uppercase tokens.
  const cleaned = firstLine
    .replace(/[*_`>#]/g, " ")
    .replace(/[.,:;!?]/g, " ")
    .toUpperCase();
  const tokens = cleaned.split(/\s+/).filter(Boolean);

  // "TIE" wins if present anywhere on the verdict line.
  if (tokens.includes("TIE") || tokens.includes("EQUAL") || tokens.includes("DRAW")) {
    return "tie";
  }
  // Otherwise the first standalone A or B token is the verdict.
  for (const tok of tokens) {
    if (tok === "A") return "A";
    if (tok === "B") return "B";
  }
  return "tie";
}
