/**
 * Agentic ReAct loop — turns a single OmniRoute model into a tool-using agent.
 *
 * The loop is the missing piece between "the council answers a question" and
 * "the council DOES work": it lets a model emit OpenAI-style `tool_calls`, runs
 * each one through the permission gate (`./permissions.ts`) into the sandboxed
 * builtin handlers (`./toolRegistry.ts`), threads the results back as
 * `role:"tool"` messages, and re-prompts — until the model stops calling tools
 * (final answer) or the iteration budget is exhausted.
 *
 * Design constraints (kept deliberately narrow so this is testable + reusable):
 *   - NO transport concerns here. The loop takes a `handleSingleModel` adapter
 *     (the same per-turn primitive the council route uses:
 *     `handleSingleModelChat` bound to a request) and an `emit` callback for
 *     progress events. The SSE endpoint owns the TransformStream; the loop only
 *     calls `emit(...)`.
 *   - Panel calls are forced non-streaming — we need the complete `tool_calls`
 *     array before we can dispatch, and streaming partial tool-call deltas would
 *     add reassembly complexity for no benefit inside the loop.
 *   - Tool execution reuses the sandbox (workspace jail, read-only network-off
 *     containers, SSRF guard) already inside each builtin handler; this module
 *     never touches the filesystem or network directly.
 */
import { getAgentTool, type AgentToolSchema } from "./toolRegistry";
import { evaluateToolCall, DEFAULT_PERMISSION_POLICY, type PermissionPolicy } from "./permissions";

type Body = Record<string, unknown>;

/** A per-turn model primitive: given a chat body + model id, resolve a Response. */
export type HandleSingleModel = (body: Body, model: string) => Promise<Response>;

/** Progress events emitted by the loop. The endpoint forwards these over SSE. */
export type AgentEvent =
  | { type: "iteration_start"; iteration: number }
  | { type: "assistant_message"; iteration: number; text: string }
  | {
      type: "tool_call";
      iteration: number;
      id: string;
      name: string;
      arguments: Record<string, unknown>;
    }
  | {
      type: "tool_result";
      iteration: number;
      id: string;
      name: string;
      ok: boolean;
      result: Record<string, unknown>;
    }
  | {
      type: "tool_denied";
      iteration: number;
      id: string;
      name: string;
      reason: string;
    }
  | { type: "reflection"; iteration: number; failures: number; text: string }
  | { type: "final"; iteration: number; text: string }
  | { type: "error"; message: string };

export type AgentEmit = (event: AgentEvent) => void;

export interface AgentLoopOptions {
  /** The model that drives the loop (emits tool calls, writes the final answer). */
  model: string;
  /** Per-turn model primitive (already bound to the incoming request). */
  handleSingleModel: HandleSingleModel;
  /** Progress sink. */
  emit: AgentEmit;
  /**
   * The full OpenAI-style chat body from the caller: `messages`, plus any
   * sampling params. `tools`/`tool_choice`/`stream` are managed by the loop and
   * overwritten — callers do not pre-set them.
   */
  body: Body;
  /** Permission policy for tool dispatch. Defaults to `gated`. */
  policy?: PermissionPolicy;
  /** Restrict the exposed tool set by name; omit to expose every registered tool. */
  allowedTools?: readonly string[];
  /** Max model↔tool round-trips before forcing a final answer. Default 8. */
  maxIterations?: number;
  /** Identity for the sandboxed handlers (workspace isolation key). */
  context: { apiKeyId: string; sessionId: string };
  /** Optional pre-built tool schema list (else derived from the registry). */
  toolSchemas: AgentToolSchema[];
  /**
   * Reflexion (arXiv 2303.11366): after an iteration whose tool calls FAILED,
   * inject a self-reflection turn asking the model to diagnose what went wrong
   * before it retries, converting raw error output into a corrective plan. This
   * is the research-backed self-correction path that lifts agentic success on
   * decision/reasoning/code tasks without any fine-tuning. Default `false`
   * (byte-identical to the prior loop); a caller opts in.
   */
  reflection?: boolean;
}

/** One assistant tool call parsed from an OpenAI completion. */
interface ParsedToolCall {
  id: string;
  name: string;
  /** Raw arguments string as emitted by the model (may be invalid JSON). */
  rawArguments: string;
}

/**
 * Pull the assistant message + any tool calls out of a non-stream completion.
 * Only the OpenAI chat-completion shape carries `tool_calls` in a stable place;
 * council panel responses are already translated to the client format by
 * chatCore, and the agentic endpoint always requests OpenAI format, so we parse
 * that shape here. Returns text + parsed calls (empty array when none).
 */
export function parseAssistantTurn(json: unknown): {
  message: Record<string, unknown> | null;
  text: string;
  toolCalls: ParsedToolCall[];
} {
  if (!json || typeof json !== "object") {
    return { message: null, text: "", toolCalls: [] };
  }
  const j = json as Record<string, unknown>;
  const choices = j.choices as Array<Record<string, unknown>> | undefined;
  const choice = choices?.[0];
  const message = (choice?.message ?? null) as Record<string, unknown> | null;
  if (!message) return { message: null, text: "", toolCalls: [] };

  const content = message.content;
  const text = typeof content === "string" ? content : "";

  const rawCalls = message.tool_calls as Array<Record<string, unknown>> | undefined;
  const toolCalls: ParsedToolCall[] = Array.isArray(rawCalls)
    ? rawCalls
        .map((c, i) => {
          const fn = c.function as Record<string, unknown> | undefined;
          const name = typeof fn?.name === "string" ? fn.name : "";
          const id = typeof c.id === "string" && c.id ? c.id : `call_${i}`;
          const rawArguments = typeof fn?.arguments === "string" ? fn.arguments : "{}";
          return name ? { id, name, rawArguments } : null;
        })
        .filter((c): c is ParsedToolCall => c !== null)
    : [];

  return { message, text, toolCalls };
}

/** Parse a tool-call arguments string into an object; never throws. */
export function parseToolArguments(raw: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(raw);
    // Tool arguments must be a plain object. A top-level array or primitive is
    // never a valid argument bag (and `typeof [] === "object"`), so fall back
    // to `{}` rather than letting an array through as arguments.
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

/**
 * Build the Reflexion self-reflection turn (arXiv 2303.11366). Given the tool
 * calls that FAILED this iteration, ask the model to diagnose the root cause and
 * plan a concrete correction before it retries — turning raw error output into
 * "verbal reinforcement" the next iteration can act on. Pure: it only formats a
 * prompt string, so it is unit-testable without a model call. Returns "" when
 * there were no failures (caller then skips reflection entirely).
 */
export function buildReflectionPrompt(failures: Array<{ name: string; error: string }>): string {
  if (failures.length === 0) return "";
  const lines = failures.map((f) => `- ${f.name}: ${f.error}`).join("\n");
  return [
    "One or more of your tool calls just failed:",
    "",
    lines,
    "",
    "Before trying again, briefly reflect: what was the root cause of each failure",
    "(wrong arguments, a tool that cannot do this, a bad assumption)? State the",
    "specific correction you will make. Then either retry with the corrected call,",
    "or, if a tool genuinely cannot accomplish the goal, proceed without it and",
    "explain the limitation in your final answer. Do not repeat the same failing",
    "call unchanged.",
  ].join("\n");
}

/**
 * Run one tool call through the permission gate into its sandboxed handler.
 * Returns the structured result object that will be threaded back to the model
 * as the `role:"tool"` message content. Errors are captured as
 * `{ error: string }` rather than thrown so a single bad call never aborts the
 * loop — the model sees the failure and can adapt.
 */
async function runToolCall(
  call: ParsedToolCall,
  args: Record<string, unknown>,
  policy: PermissionPolicy,
  context: { apiKeyId: string; sessionId: string },
  model: string
): Promise<{ ok: boolean; result: Record<string, unknown>; denied?: string }> {
  const decision = evaluateToolCall(call.name, policy);
  if (decision.verdict === "deny") {
    return { ok: false, result: { error: decision.reason }, denied: decision.reason };
  }
  if (decision.verdict === "needs-approval") {
    return { ok: false, result: { error: decision.reason }, denied: decision.reason };
  }

  const entry = getAgentTool(call.name);
  if (!entry) {
    return { ok: false, result: { error: `Unknown tool "${call.name}"` } };
  }

  try {
    const output = await entry.handler(args, {
      apiKeyId: context.apiKeyId,
      sessionId: context.sessionId,
      provider: undefined,
      model,
    });
    return { ok: true, result: output };
  } catch (err) {
    // Surface a sanitized message to the model; the full error is not leaked.
    const message = err instanceof Error ? err.message : "Tool execution failed";
    return { ok: false, result: { error: message } };
  }
}

/**
 * Execute the ReAct loop to completion. Returns the final assistant text (best
 * effort — the last non-empty assistant message if the budget is hit before the
 * model produces a tool-free turn).
 */
export async function runAgentLoop(options: AgentLoopOptions): Promise<{
  finalText: string;
  iterations: number;
  toolCallCount: number;
}> {
  const {
    model,
    handleSingleModel,
    emit,
    body,
    policy = DEFAULT_PERMISSION_POLICY,
    maxIterations = 8,
    context,
    toolSchemas,
    reflection = false,
  } = options;

  // Working copy of the conversation. We only ever append (assistant turn +
  // tool results), never mutate the caller's array.
  const messages: unknown[] = Array.isArray(body.messages) ? [...(body.messages as unknown[])] : [];

  let lastAssistantText = "";
  let toolCallCount = 0;
  let iteration = 0;

  for (; iteration < maxIterations; iteration++) {
    emit({ type: "iteration_start", iteration });

    // On the final permitted iteration, drop the tools so the model is forced to
    // produce a prose answer instead of another tool call it can't act on.
    const isLastIteration = iteration === maxIterations - 1;
    const turnBody: Body = {
      ...body,
      messages,
      stream: false,
      ...(isLastIteration ? { tool_choice: "none" } : { tools: toolSchemas, tool_choice: "auto" }),
    };

    let response: Response;
    try {
      response = await handleSingleModel(turnBody, model);
    } catch (err) {
      emit({
        type: "error",
        message: err instanceof Error ? err.message : "Model dispatch failed",
      });
      break;
    }

    if (!response.ok) {
      emit({ type: "error", message: `Model returned HTTP ${response.status}` });
      break;
    }

    let json: unknown;
    try {
      json = await response.clone().json();
    } catch {
      emit({ type: "error", message: "Model returned an unparseable completion" });
      break;
    }

    const { message, text, toolCalls } = parseAssistantTurn(json);
    if (text.trim()) {
      lastAssistantText = text;
      emit({ type: "assistant_message", iteration, text });
    }

    // No tool calls → the model produced its final answer. Done.
    if (toolCalls.length === 0) {
      emit({ type: "final", iteration, text: lastAssistantText });
      return { finalText: lastAssistantText, iterations: iteration + 1, toolCallCount };
    }

    // Thread the assistant tool-call turn verbatim so provider bookkeeping (the
    // tool_call ids the tool results reference) stays consistent.
    messages.push(message ?? { role: "assistant", content: text || null, tool_calls: toolCalls });

    // Execute every requested tool call, appending a role:"tool" result per call.
    // Collect this iteration's failures so the (opt-in) reflection step can turn
    // them into a self-critique before the next attempt (Reflexion, Shinn et al.).
    const failures: { name: string; error: string }[] = [];
    for (const call of toolCalls) {
      const args = parseToolArguments(call.rawArguments);
      emit({ type: "tool_call", iteration, id: call.id, name: call.name, arguments: args });
      toolCallCount++;

      const outcome = await runToolCall(call, args, policy, context, model);
      if (outcome.denied) {
        emit({
          type: "tool_denied",
          iteration,
          id: call.id,
          name: call.name,
          reason: outcome.denied,
        });
        failures.push({ name: call.name, error: outcome.denied });
      } else {
        emit({
          type: "tool_result",
          iteration,
          id: call.id,
          name: call.name,
          ok: outcome.ok,
          result: outcome.result,
        });
        if (!outcome.ok) {
          const errText =
            typeof outcome.result.error === "string"
              ? outcome.result.error
              : "tool reported failure";
          failures.push({ name: call.name, error: errText });
        }
      }

      messages.push({
        role: "tool",
        tool_call_id: call.id,
        name: call.name,
        content: JSON.stringify(outcome.result),
      });
    }

    // Reflexion (opt-in): when tool calls failed this iteration, inject a verbal
    // self-reflection as a user turn so the model diagnoses WHY before retrying,
    // rather than blindly re-calling. This is the linguistic-feedback loop from
    // Reflexion (Shinn et al.) — no fine-tuning, just reflective memory threaded
    // into the next attempt. Skipped on the last iteration (no retry follows) and
    // when disabled (default), keeping the prior behaviour byte-identical.
    if (reflection && failures.length > 0 && !isLastIteration) {
      const reflectionText = buildReflectionPrompt(failures);
      emit({ type: "reflection", iteration, failures: failures.length, text: reflectionText });
      messages.push({ role: "user", content: reflectionText });
    }
  }

  // Budget exhausted without a tool-free turn — emit the best answer we have.
  emit({ type: "final", iteration, text: lastAssistantText });
  return { finalText: lastAssistantText, iterations: iteration, toolCallCount };
}
