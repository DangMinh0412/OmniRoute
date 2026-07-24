/**
 * POST /api/v1/agent — Agentic ReAct SSE endpoint.
 *
 * Turns a single OmniRoute model into a tool-using agent: the model emits
 * OpenAI-style `tool_calls`, each runs through the permission gate
 * (`src/lib/agentic/permissions.ts`) into the sandboxed builtin handlers
 * (`src/lib/agentic/toolRegistry.ts`), results are threaded back, and the model
 * re-prompts until it produces a tool-free final answer or the iteration budget
 * is hit. This is the "DO work, not just answer" half of the AI-empire vision;
 * the council route (`../council/route.ts`) is the "debate to the best answer"
 * half — the two share the same `handleSingleModelChat` per-turn primitive and
 * the same SSE scaffolding.
 *
 * Request body (OpenAI-compatible + agent fields):
 *   messages:      ChatMessage[]     — required conversation
 *   model:         string            — the driver model (required)
 *   maxIterations: number?           — model↔tool round-trips (default 8, max 20)
 *   permission:    "readonly"|"gated"|"auto"?  — tool policy (default "gated")
 *   approvedTools: string[]?         — mutate tools pre-authorized in "gated" mode
 *   allowedTools:  string[]?         — restrict the exposed tool set by name
 *
 * SSE event stream (one JSON object per `data:` line, terminated by [DONE]):
 *   {"type":"iteration_start","iteration":0}
 *   {"type":"assistant_message","iteration":0,"text":"..."}
 *   {"type":"tool_call","iteration":0,"id":"call_0","name":"web_search","arguments":{...}}
 *   {"type":"tool_result","iteration":0,"id":"call_0","name":"web_search","ok":true,"result":{...}}
 *   {"type":"tool_denied","iteration":0,"id":"call_1","name":"file_write","reason":"..."}
 *   {"type":"final","iteration":2,"text":"..."}
 *   {"type":"done","iterations":3,"toolCalls":2,"durationMs":12345}
 *   [DONE]
 */
import { z } from "zod";
import { buildErrorBody } from "@omniroute/open-sse/utils/error.ts";
import { CORS_HEADERS, handleCorsOptions } from "@/shared/utils/cors";
import { extractApiKey, isValidApiKey } from "@/sse/services/auth";
import { isDashboardSessionAuthenticated } from "@/shared/utils/apiAuth";
import { isRequireApiKeyEnabled } from "@/shared/utils/featureFlags";
import { logger } from "@omniroute/open-sse/utils/logger.ts";
import { handleSingleModelChat } from "@/sse/handlers/chat";
import { initTranslators } from "@omniroute/open-sse/translator/index.ts";
import { runAgentLoop, type AgentEvent, type HandleSingleModel } from "@/lib/agentic/loop";
import { buildToolSchemas } from "@/lib/agentic/toolRegistry";
import { type PermissionPolicy } from "@/lib/agentic/permissions";

// ---------------------------------------------------------------------------
// One-time translator init (mirrors the council route)
// ---------------------------------------------------------------------------
let _initPromise: Promise<void> | null = null;
function ensureInitialized(): Promise<void> {
  if (!_initPromise) {
    _initPromise = Promise.resolve(initTranslators()).then(() => {});
  }
  return _initPromise;
}

// ---------------------------------------------------------------------------
// Request schema
// ---------------------------------------------------------------------------
const agentBodySchema = z
  .object({
    model: z.string().trim().min(1).max(300),
    messages: z.array(z.record(z.string(), z.unknown())).min(1),
    maxIterations: z.coerce.number().int().min(1).max(20).optional(),
    permission: z.enum(["readonly", "gated", "auto"]).optional(),
    approvedTools: z.array(z.string().trim().min(1).max(100)).max(20).optional(),
    allowedTools: z.array(z.string().trim().min(1).max(100)).max(20).optional(),
    reflection: z.boolean().optional(),
  })
  .passthrough();

// ---------------------------------------------------------------------------
// SSE helpers (identical contract to the council route)
// ---------------------------------------------------------------------------
function sseEvent(payload: Record<string, unknown>): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}
const SSE_DONE = "data: [DONE]\n\n";

const SSE_HEADERS = {
  ...CORS_HEADERS,
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache, no-transform",
  Connection: "keep-alive",
  "X-Accel-Buffering": "no",
};

// ---------------------------------------------------------------------------
// CORS preflight
// ---------------------------------------------------------------------------
export async function OPTIONS(): Promise<Response> {
  return handleCorsOptions();
}

// ---------------------------------------------------------------------------
// POST handler
// ---------------------------------------------------------------------------
export async function POST(request: Request): Promise<Response> {
  await ensureInitialized();

  // Content-Type guard
  const contentType = request.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().split(";")[0].trim().startsWith("application/json")) {
    return new Response(
      JSON.stringify(buildErrorBody(415, "Content-Type must be application/json")),
      { status: 415, headers: { ...CORS_HEADERS, "Content-Type": "application/json" } }
    );
  }

  // Auth (mirrors the council route)
  const apiKey = extractApiKey(request);
  const sessionOk = await isDashboardSessionAuthenticated(request);
  const requireApiKey = await isRequireApiKeyEnabled();
  if (requireApiKey && !sessionOk) {
    if (!apiKey) {
      return new Response(JSON.stringify(buildErrorBody(401, "API key required")), {
        status: 401,
        headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
      });
    }
    const validKey = await isValidApiKey(apiKey);
    if (!validKey) {
      return new Response(JSON.stringify(buildErrorBody(401, "Invalid API key")), {
        status: 401,
        headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
      });
    }
  }

  // Parse + validate body
  let rawBody: unknown;
  try {
    rawBody = await request.json();
  } catch {
    return new Response(JSON.stringify(buildErrorBody(400, "Invalid JSON body")), {
      status: 400,
      headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
    });
  }

  const parsed = agentBodySchema.safeParse(rawBody);
  if (!parsed.success) {
    return new Response(
      JSON.stringify(
        buildErrorBody(
          400,
          `Invalid agent request: ${parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ")}`
        )
      ),
      { status: 400, headers: { ...CORS_HEADERS, "Content-Type": "application/json" } }
    );
  }

  const {
    model,
    maxIterations,
    permission,
    approvedTools,
    allowedTools,
    reflection,
    ...chatFields
  } = parsed.data;

  const log = logger("AGENT");

  // ---------------------------------------------------------------------------
  // SSE stream setup
  // ---------------------------------------------------------------------------
  const { readable, writable } = new TransformStream<Uint8Array, Uint8Array>();
  const writer = writable.getWriter();
  const encoder = new TextEncoder();

  function emit(payload: Record<string, unknown>): void {
    writer.write(encoder.encode(sseEvent(payload))).catch(() => {});
  }

  async function runAgent(): Promise<void> {
    const tStart = Date.now();

    // Per-turn primitive routed through the full OmniRoute stack, bound to this
    // request — identical adapter shape to the council route.
    const handleSingleModel: HandleSingleModel = (body, m) =>
      handleSingleModelChat(body, m, request, request, "agent", null, null, {
        sessionId: null,
        forceLiveComboTest: false,
      });

    const policy: PermissionPolicy = {
      mode: permission ?? "gated",
      approvedTools,
    };

    // The sandboxed builtins isolate the workspace by apiKeyId; fall back to a
    // stable anonymous key when the request is unauthenticated (REQUIRE_API_KEY
    // off) so a run still gets a private, deterministic workspace.
    const apiKeyId = apiKey || "anonymous-agent";

    try {
      const toolSchemas = buildToolSchemas(allowedTools);

      const result = await runAgentLoop({
        model,
        handleSingleModel,
        emit: (event: AgentEvent) => emit(event as unknown as Record<string, unknown>),
        body: chatFields as Record<string, unknown>,
        policy,
        allowedTools,
        maxIterations,
        reflection,
        context: { apiKeyId, sessionId: "" },
        toolSchemas,
      });

      emit({
        type: "done",
        iterations: result.iterations,
        toolCalls: result.toolCallCount,
        durationMs: Date.now() - tStart,
      });
    } catch (err) {
      log.error("AGENT", "Unhandled error in agent run", {
        error: err instanceof Error ? err.message : String(err),
      });
      emit({ type: "error", message: "Internal agent error" });
    }

    writer.write(encoder.encode(SSE_DONE)).catch(() => {});
    writer.close().catch(() => {});
  }

  // Fire-and-forget — response is already streaming
  runAgent().catch(() => {});

  return new Response(readable as unknown as BodyInit, { status: 200, headers: SSE_HEADERS });
}
