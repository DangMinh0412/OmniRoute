/**
 * Live wiring for the council eval harness — connects the transport-agnostic
 * `runCouncilWinRateEval` to the real in-process routes.
 *
 * `evalHarness.ts` deliberately knows nothing about HTTP: it takes a `baseline`
 * fn, a `council` fn, and a `judge` fn. This module builds those three fns
 * against the running OmniRoute instance:
 *
 *   - baseline → POST /api/v1/chat/completions (one model, one shot)
 *   - council  → POST /api/v1/council (debate + judge synthesis, SSE)
 *   - judge    → POST /api/v1/chat/completions (a neutral judge model)
 *
 * It is kept OUT of evalHarness.ts (and untested by the harness's pure unit
 * tests) precisely because it does I/O. The route handlers are imported lazily
 * so importing this module never drags the Next runtime into a context that
 * only wants the pure harness (e.g. the harness's own unit tests).
 *
 * @module lib/council/evalLive
 */

import type { CompletionFn, JudgeFn } from "./evalHarness.ts";
import { buildJudgePrompt, parseJudgeChoice } from "./evalJudge.ts";

/** Options shared by the live baseline/council/judge builders. */
export interface LiveEvalWiringOptions {
  /** Model id for the single-model baseline. */
  baselineModel: string;
  /** Explicit council panel; omit to let the council auto-pick every model. */
  councilModels?: string[];
  /** Judge model for the head-to-head comparison (distinct from panel is ideal). */
  judgeModel: string;
  /** Optional API key forwarded as a Bearer token to the in-process routes. */
  apiKey?: string | null;
  /** Turn on the council's opt-in evaluator-optimizer verify pass. */
  verifyPass?: boolean;
  /** Max tokens for baseline + judge single calls. Default 512. */
  maxTokens?: number;
  /** Absolute origin used for the synthetic Request objects. Default localhost. */
  origin?: string;
}

/** Extract assistant text from a non-stream OpenAI-shaped completion. "" if none. */
function extractAssistantText(payload: unknown): string {
  if (!payload || typeof payload !== "object") return "";
  const choices = (payload as { choices?: unknown }).choices;
  if (!Array.isArray(choices) || choices.length === 0) return "";
  const message = (choices[0] as { message?: unknown }).message;
  const content = (message as { content?: unknown })?.content;
  return typeof content === "string" ? content.trim() : "";
}

function buildHeaders(apiKey?: string | null): Headers {
  const headers = new Headers({ "Content-Type": "application/json" });
  if (apiKey) headers.set("Authorization", `Bearer ${apiKey}`);
  return headers;
}

/**
 * Build a `CompletionFn` that answers with ONE model via the chat completions
 * route. This is the single-model baseline the council is measured against.
 */
export function makeBaselineCompletion(options: LiveEvalWiringOptions): CompletionFn {
  const origin = options.origin ?? "http://localhost";
  const maxTokens = options.maxTokens ?? 512;
  return async (prompt: string): Promise<string> => {
    const { POST } = await import("@/app/api/v1/chat/completions/route");
    const request = new Request(`${origin}/api/v1/chat/completions`, {
      method: "POST",
      headers: buildHeaders(options.apiKey),
      body: JSON.stringify({
        model: options.baselineModel,
        messages: [{ role: "user", content: prompt }],
        stream: false,
        max_tokens: maxTokens,
      }),
    });
    const response = await POST(request as never);
    const payload = (await response.json().catch(() => null)) as Record<string, unknown> | null;
    if (!response.ok) {
      const message =
        (payload?.error as { message?: string } | undefined)?.message ?? `HTTP ${response.status}`;
      throw new Error(`baseline completion failed: ${message}`);
    }
    const text = extractAssistantText(payload);
    if (!text) throw new Error("baseline completion returned no content");
    return text;
  };
}

/**
 * Read a council SSE response to completion and return the final synthesized
 * answer. The council emits metadata events (round_start / panel_answer / …)
 * then, for non-streaming callers, the synthesis as an OpenAI completion JSON in
 * a `synthesis` event (or streamed `token` events). We accumulate whichever form
 * arrives so this works regardless of the client stream flag.
 */
async function readCouncilSynthesis(response: Response): Promise<string> {
  if (!response.ok || !response.body) {
    const payload = (await response.json().catch(() => null)) as Record<string, unknown> | null;
    const message =
      (payload?.error as { message?: string } | undefined)?.message ?? `HTTP ${response.status}`;
    throw new Error(`council request failed: ${message}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let streamed = "";
  let block = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const data = trimmed.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      let ev: Record<string, unknown>;
      try {
        ev = JSON.parse(data) as Record<string, unknown>;
      } catch {
        continue;
      }
      if (ev.type === "token" && typeof ev.text === "string") {
        streamed += ev.text;
      } else if (ev.type === "synthesis") {
        if (typeof ev.text === "string") block += ev.text;
        else block += extractAssistantText(ev.completion);
      } else if (ev.type === "error") {
        throw new Error(`council error: ${String(ev.message ?? "unknown")}`);
      }
    }
  }

  const answer = (block || streamed).trim();
  if (!answer) throw new Error("council returned no synthesis");
  return answer;
}

/**
 * Build a `CompletionFn` that answers via the full council (debate + judge
 * synthesis) over the /api/v1/council route. This is the "system under test"
 * the win-rate compares against the baseline.
 */
export function makeCouncilCompletion(options: LiveEvalWiringOptions): CompletionFn {
  const origin = options.origin ?? "http://localhost";
  return async (prompt: string): Promise<string> => {
    const { POST } = await import("@/app/api/v1/council/route");
    const body: Record<string, unknown> = {
      messages: [{ role: "user", content: prompt }],
      stream: false,
    };
    if (options.councilModels && options.councilModels.length > 0) {
      body.models = options.councilModels;
    }
    if (options.judgeModel) body.judgeModel = options.judgeModel;
    if (options.verifyPass) body.verifyPass = true;

    const request = new Request(`${origin}/api/v1/council`, {
      method: "POST",
      headers: buildHeaders(options.apiKey),
      body: JSON.stringify(body),
    });
    const response = await POST(request as never);
    return readCouncilSynthesis(response);
  };
}

/**
 * Build a `JudgeFn` that asks the judge model which of two answers is better.
 * The harness calls it twice per case with swapped positions to cancel bias, so
 * this only needs to render one neutral comparison and parse the verdict.
 */
export function makeLiveJudge(options: LiveEvalWiringOptions): JudgeFn {
  const origin = options.origin ?? "http://localhost";
  const maxTokens = Math.min(options.maxTokens ?? 512, 256);
  return async (question: string, a: string, b: string) => {
    const { POST } = await import("@/app/api/v1/chat/completions/route");
    const request = new Request(`${origin}/api/v1/chat/completions`, {
      method: "POST",
      headers: buildHeaders(options.apiKey),
      body: JSON.stringify({
        model: options.judgeModel,
        messages: [{ role: "user", content: buildJudgePrompt(question, a, b) }],
        stream: false,
        max_tokens: maxTokens,
        temperature: 0,
      }),
    });
    const response = await POST(request as never);
    const payload = (await response.json().catch(() => null)) as Record<string, unknown> | null;
    if (!response.ok) {
      const message =
        (payload?.error as { message?: string } | undefined)?.message ?? `HTTP ${response.status}`;
      throw new Error(`judge call failed: ${message}`);
    }
    return parseJudgeChoice(extractAssistantText(payload));
  };
}

/**
 * Convenience: build all three live dependencies at once, ready to pass to
 * `runCouncilWinRateEval(cases, buildLiveEvalDeps(opts))`.
 */
export function buildLiveEvalDeps(options: LiveEvalWiringOptions): {
  baseline: CompletionFn;
  council: CompletionFn;
  judge: JudgeFn;
} {
  return {
    baseline: makeBaselineCompletion(options),
    council: makeCouncilCompletion(options),
    judge: makeLiveJudge(options),
  };
}
