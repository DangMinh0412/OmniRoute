/**
 * Council panel tool-runner — lets a debate panel member USE tools before it
 * speaks. This is the join between the two halves of the AI-empire vision: the
 * council route (`src/app/api/v1/council/route.ts`) debates models to the best
 * answer, and the agentic loop (`./loop.ts`) lets a model DO work. Here a panel
 * member runs the agentic loop (search / fetch / run code in a sandbox) and its
 * tool-free final answer is the prose that enters the debate — so models argue
 * from evidence they gathered, not just from parametric memory.
 *
 * Why a separate module (not inline in the route):
 *   - `debate.ts` lives in `open-sse/` and must NOT import `src/lib/agentic/`
 *     (that would invert the dependency direction). The council route lives in
 *     `src/app/` and may import both, so the join belongs on this side.
 *   - The route handler is not unit-testable in isolation; the two exported
 *     functions here (`buildPanelCompletionResponse`, `runPanelMemberWithTools`)
 *     are, so the "loop result → OpenAI completion the debate loop can read"
 *     contract is covered by tests.
 */
import { runAgentLoop, type AgentEvent, type HandleSingleModel } from "./loop";
import type { AgentToolSchema } from "./toolRegistry";
import type { PermissionPolicy } from "./permissions";

/**
 * Wrap a panel member's final text as a minimal OpenAI chat-completion Response,
 * shaped so the debate loop's `extractPanelText()` reads it back unchanged
 * (`choices[0].message.content`). Non-streaming JSON — the debate loop always
 * consumes panel responses via `resp.clone().json()`.
 */
export function buildPanelCompletionResponse(finalText: string, model: string): Response {
  const completion = {
    id: `council-agent-${Date.now()}`,
    object: "chat.completion",
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: finalText },
        finish_reason: "stop",
      },
    ],
  };
  return new Response(JSON.stringify(completion), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

export interface PanelToolRunOptions {
  /** The panel member model driving this tool loop. */
  model: string;
  /** Round index, forwarded on emitted events for UI grouping. */
  round: number;
  /** Raw per-turn primitive (same one the debate loop uses for plain calls). */
  handleSingleModel: HandleSingleModel;
  /** The round body (system + conversation + any debate prompt already appended). */
  roundBody: Record<string, unknown>;
  /** Pre-built tool schema list exposed to the panel member. */
  toolSchemas: AgentToolSchema[];
  /** Permission policy for tool dispatch (council uses `auto`). */
  policy: PermissionPolicy;
  /** Max model↔tool round-trips per panel member. Bounds cost across a big panel. */
  maxIterations: number;
  /** Sandbox workspace identity. */
  context: { apiKeyId: string; sessionId: string };
  /** Council SSE sink — panel tool activity is surfaced as `panel_tool_*` events. */
  emit: (payload: Record<string, unknown>) => void;
}

/**
 * Run one panel member through the agentic loop and return its tool-free final
 * answer as a completion Response the debate loop can consume. Tool activity is
 * forwarded to the council SSE stream as `panel_tool_call` / `panel_tool_result`
 * / `panel_tool_denied` events (carrying round + model); the council reducer
 * ignores unknown event types, so this is additive and never breaks existing
 * consumers.
 *
 * A failed loop (dispatch error / empty answer) resolves to an empty-content
 * completion; the debate loop already treats empty panel text as a drop, so a
 * broken tool run degrades gracefully to "this member did not contribute this
 * round" rather than aborting the debate.
 */
export async function runPanelMemberWithTools(options: PanelToolRunOptions): Promise<Response> {
  const {
    model,
    round,
    handleSingleModel,
    roundBody,
    toolSchemas,
    policy,
    maxIterations,
    context,
    emit,
  } = options;

  const agentEmit = (event: AgentEvent): void => {
    switch (event.type) {
      case "tool_call":
        emit({
          type: "panel_tool_call",
          round,
          model,
          iteration: event.iteration,
          name: event.name,
          arguments: event.arguments,
        });
        break;
      case "tool_result":
        emit({
          type: "panel_tool_result",
          round,
          model,
          iteration: event.iteration,
          name: event.name,
          ok: event.ok,
        });
        break;
      case "tool_denied":
        emit({
          type: "panel_tool_denied",
          round,
          model,
          iteration: event.iteration,
          name: event.name,
          reason: event.reason,
        });
        break;
      default:
        // iteration_start / assistant_message / final / error are internal to the
        // loop; the debate stream only surfaces tool activity per panel member.
        break;
    }
  };

  const result = await runAgentLoop({
    model,
    handleSingleModel,
    emit: agentEmit,
    body: roundBody,
    policy,
    maxIterations,
    context,
    toolSchemas,
  });

  return buildPanelCompletionResponse(result.finalText, model);
}
