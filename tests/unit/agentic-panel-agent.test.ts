/**
 * Council panel tool-runner — the join between the debate route and the agentic
 * loop. buildPanelCompletionResponse + runPanelMemberWithTools are exported pure
 * enough to test with a fake handleSingleModel: we drive a scripted model that
 * emits one tool call then a final answer, and assert (a) the returned Response
 * is a minimal OpenAI completion the debate loop's extractPanelText reads back,
 * and (b) tool activity is surfaced as panel_tool_* SSE events carrying round +
 * model. No real sandbox / docker is ever touched — a readonly policy denies the
 * mutate-tier tool so the loop stays fully deterministic.
 */
import test from "node:test";
import assert from "node:assert/strict";

const { buildPanelCompletionResponse, runPanelMemberWithTools } =
  await import("../../src/lib/agentic/panelAgent.ts");
const { buildToolSchemas } = await import("../../src/lib/agentic/toolRegistry.ts");

type Body = Record<string, unknown>;

/** Minimal OpenAI chat-completion Response the debate loop consumes. */
function completion(content: string, toolCalls?: unknown[]): Response {
  const message: Record<string, unknown> = { role: "assistant", content };
  if (toolCalls) message.tool_calls = toolCalls;
  return new Response(
    JSON.stringify({
      id: "x",
      object: "chat.completion",
      choices: [{ index: 0, message, finish_reason: toolCalls ? "tool_calls" : "stop" }],
    }),
    { status: 200, headers: { "Content-Type": "application/json" } }
  );
}

test("buildPanelCompletionResponse: shapes final text as an OpenAI completion", async () => {
  const resp = buildPanelCompletionResponse("the answer", "p/model");
  assert.equal(resp.status, 200);
  const json = (await resp.json()) as {
    object: string;
    model: string;
    choices: Array<{ message: { role: string; content: string } }>;
  };
  assert.equal(json.object, "chat.completion");
  assert.equal(json.model, "p/model");
  assert.equal(json.choices[0].message.role, "assistant");
  assert.equal(json.choices[0].message.content, "the answer");
});

test("runPanelMemberWithTools: no tool call → final answer flows straight through", async () => {
  const events: Array<Record<string, unknown>> = [];
  const handleSingleModel = async (_body: Body, _model: string): Promise<Response> =>
    completion("direct answer, no tools");

  const resp = await runPanelMemberWithTools({
    model: "p/a",
    round: 0,
    handleSingleModel,
    roundBody: { messages: [{ role: "user", content: "hi" }] },
    toolSchemas: buildToolSchemas(),
    policy: { mode: "auto" },
    maxIterations: 4,
    context: { apiKeyId: "test", sessionId: "" },
    emit: (e) => events.push(e),
  });

  const json = (await resp.json()) as { choices: Array<{ message: { content: string } }> };
  assert.equal(json.choices[0].message.content, "direct answer, no tools");
  // No tool activity → no panel_tool_* events surfaced.
  assert.equal(events.filter((e) => String(e.type).startsWith("panel_tool")).length, 0);
});

test("runPanelMemberWithTools: a denied mutate tool surfaces panel_tool_denied, still answers", async () => {
  // Scripted model: turn 1 asks to run code (mutate tier), turn 2 (tools dropped
  // on the last iteration, or after seeing the denial) gives a prose answer.
  let turn = 0;
  const handleSingleModel = async (_body: Body, _model: string): Promise<Response> => {
    turn++;
    if (turn === 1) {
      return completion("", [
        {
          id: "call_0",
          type: "function",
          function: { name: "eval_code", arguments: '{"code":"1+1"}' },
        },
      ]);
    }
    return completion("final answer after the tool was denied");
  };

  const events: Array<Record<string, unknown>> = [];
  const resp = await runPanelMemberWithTools({
    model: "p/b",
    round: 1,
    handleSingleModel,
    roundBody: { messages: [{ role: "user", content: "compute" }] },
    toolSchemas: buildToolSchemas(),
    // readonly denies the mutate-tier eval_code tool → the sandbox is NEVER run.
    policy: { mode: "readonly" },
    maxIterations: 4,
    context: { apiKeyId: "test", sessionId: "" },
    emit: (e) => events.push(e),
  });

  const denied = events.filter((e) => e.type === "panel_tool_denied");
  assert.equal(denied.length, 1, "the mutate tool must be denied under readonly");
  assert.equal(denied[0].round, 1, "denial event carries the debate round");
  assert.equal(denied[0].model, "p/b", "denial event carries the panel model");
  assert.equal(denied[0].name, "eval_code");

  const json = (await resp.json()) as { choices: Array<{ message: { content: string } }> };
  assert.equal(json.choices[0].message.content, "final answer after the tool was denied");
});

test("runPanelMemberWithTools: a failed loop degrades to empty content (debate drops it)", async () => {
  // A model that always errors → the loop yields no final text → empty completion.
  // The debate loop treats empty panel text as a drop, so this member simply does
  // not contribute rather than aborting the whole debate.
  const handleSingleModel = async (_body: Body, _model: string): Promise<Response> =>
    new Response(null, { status: 500 });

  const resp = await runPanelMemberWithTools({
    model: "p/c",
    round: 0,
    handleSingleModel,
    roundBody: { messages: [{ role: "user", content: "hi" }] },
    toolSchemas: buildToolSchemas(),
    policy: { mode: "auto" },
    maxIterations: 4,
    context: { apiKeyId: "test", sessionId: "" },
    emit: () => {},
  });

  const json = (await resp.json()) as { choices: Array<{ message: { content: string } }> };
  assert.equal(json.choices[0].message.content, "");
});
