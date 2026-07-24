/**
 * Agentic ReAct loop — end-to-end control flow with a fake model.
 *
 * The loop takes a `handleSingleModel` adapter, so we drive it with a scripted
 * fake that returns canned OpenAI completions (tool_calls then a tool-free final
 * answer). This exercises the full emit→execute→thread→re-prompt cycle WITHOUT a
 * live provider. The denial path uses permission `readonly` so a `mutate` tool is
 * rejected by the gate BEFORE any real sandbox/fs handler runs — keeping the test
 * hermetic. The pure parsers are asserted directly.
 */
import test from "node:test";
import assert from "node:assert/strict";

const { runAgentLoop, parseAssistantTurn, parseToolArguments, buildReflectionPrompt } =
  await import("../../src/lib/agentic/loop.ts");
const { buildToolSchemas } = await import("../../src/lib/agentic/toolRegistry.ts");

type Body = Record<string, unknown>;

/** Build an OpenAI completion Response carrying an assistant turn. */
function completion(message: Record<string, unknown>): Response {
  return new Response(JSON.stringify({ choices: [{ message }] }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

const TOOLS = buildToolSchemas();

test("parseAssistantTurn: extracts text + tool calls from an OpenAI completion", () => {
  const parsed = parseAssistantTurn({
    choices: [
      {
        message: {
          role: "assistant",
          content: "thinking",
          tool_calls: [
            { id: "c1", function: { name: "file_read", arguments: '{"path":"a.txt"}' } },
          ],
        },
      },
    ],
  });
  assert.equal(parsed.text, "thinking");
  assert.equal(parsed.toolCalls.length, 1);
  assert.equal(parsed.toolCalls[0].name, "file_read");
  assert.equal(parsed.toolCalls[0].id, "c1");
});

test("parseAssistantTurn: no tool_calls → empty array", () => {
  const parsed = parseAssistantTurn({
    choices: [{ message: { role: "assistant", content: "final" } }],
  });
  assert.equal(parsed.text, "final");
  assert.equal(parsed.toolCalls.length, 0);
});

test("parseToolArguments: valid JSON parses; invalid falls back to {}", () => {
  assert.deepEqual(parseToolArguments('{"a":1}'), { a: 1 });
  assert.deepEqual(parseToolArguments("not json"), {});
  assert.deepEqual(parseToolArguments("[1,2]"), {}); // arrays are not arg objects
});

test("loop: model with no tool call returns its answer in one iteration", async () => {
  const events: Array<Record<string, unknown>> = [];
  const handleSingleModel = async () => completion({ role: "assistant", content: "direct answer" });

  const result = await runAgentLoop({
    model: "p/m",
    handleSingleModel,
    emit: (e) => events.push(e as unknown as Record<string, unknown>),
    body: { messages: [{ role: "user", content: "hi" }] },
    context: { apiKeyId: "k", sessionId: "" },
    toolSchemas: TOOLS,
  });

  assert.equal(result.finalText, "direct answer");
  assert.equal(result.iterations, 1);
  assert.equal(result.toolCallCount, 0);
  assert.ok(events.some((e) => e.type === "final" && e.text === "direct answer"));
});

test("loop: readonly denies a mutate tool, then the model finalizes", async () => {
  const events: Array<Record<string, unknown>> = [];
  let turn = 0;
  const seenBodies: Body[] = [];
  const handleSingleModel = async (body: Body) => {
    seenBodies.push(body);
    turn++;
    if (turn === 1) {
      // First turn: model tries to write a file (a mutate-tier tool).
      return completion({
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "w1",
            function: { name: "file_write", arguments: '{"path":"o.txt","content":"x"}' },
          },
        ],
      });
    }
    // Second turn: after seeing the denial, the model gives a prose answer.
    return completion({ role: "assistant", content: "could not write, here is advice" });
  };

  const result = await runAgentLoop({
    model: "p/m",
    handleSingleModel,
    emit: (e) => events.push(e as unknown as Record<string, unknown>),
    body: { messages: [{ role: "user", content: "write a file" }] },
    policy: { mode: "readonly" },
    context: { apiKeyId: "k", sessionId: "" },
    toolSchemas: TOOLS,
  });

  // The mutate tool was denied by the gate — never executed.
  assert.ok(events.some((e) => e.type === "tool_denied" && e.name === "file_write"));
  assert.equal(result.finalText, "could not write, here is advice");
  assert.equal(result.toolCallCount, 1);
  // The denial was threaded back as a role:"tool" message before the 2nd turn.
  const secondTurnMessages = seenBodies[1].messages as Array<Record<string, unknown>>;
  assert.ok(secondTurnMessages.some((m) => m.role === "tool" && m.tool_call_id === "w1"));
});

test("loop: last iteration drops tools to force a prose answer", async () => {
  const events: Array<Record<string, unknown>> = [];
  const seenBodies: Body[] = [];
  // A model that ALWAYS tries to call a tool — the loop must still terminate.
  const handleSingleModel = async (body: Body) => {
    seenBodies.push(body);
    return completion({
      role: "assistant",
      content: "still working",
      tool_calls: [
        { id: "r", function: { name: "file_read", arguments: '{"path":"missing.txt"}' } },
      ],
    });
  };

  const result = await runAgentLoop({
    model: "p/m",
    handleSingleModel,
    emit: (e) => events.push(e as unknown as Record<string, unknown>),
    body: { messages: [{ role: "user", content: "loop forever" }] },
    policy: { mode: "auto" },
    maxIterations: 3,
    context: { apiKeyId: "k", sessionId: "" },
    toolSchemas: TOOLS,
  });

  assert.equal(result.iterations, 3);
  // The final turn must have been sent with tool_choice:"none" and no tools.
  const lastBody = seenBodies[seenBodies.length - 1];
  assert.equal(lastBody.tool_choice, "none");
  assert.equal(lastBody.tools, undefined);
  assert.ok(events.some((e) => e.type === "final"));
});

// --- Reflexion (opt-in self-correction) --------------------------------------

test('buildReflectionPrompt: empty failures → ""; failures list names + errors', () => {
  assert.equal(buildReflectionPrompt([]), "");
  const p = buildReflectionPrompt([
    { name: "file_write", error: "denied: readonly mode" },
    { name: "eval_code", error: "sandbox timeout" },
  ]);
  assert.ok(p.includes("file_write"), "prompt names the failed tool");
  assert.ok(p.includes("denied: readonly mode"), "prompt carries the error text");
  assert.ok(p.includes("eval_code"));
  assert.ok(/root cause/i.test(p), "prompt asks for a root-cause reflection");
});

test("loop: reflection injects a self-critique user turn after a failed tool call", async () => {
  const events: Array<Record<string, unknown>> = [];
  const seenBodies: Body[] = [];
  let turn = 0;
  const handleSingleModel = async (body: Body) => {
    seenBodies.push(body);
    turn++;
    if (turn === 1) {
      // A mutate tool that the readonly gate will DENY → a failure to reflect on.
      return completion({
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "w1",
            function: { name: "file_write", arguments: '{"path":"o.txt","content":"x"}' },
          },
        ],
      });
    }
    return completion({ role: "assistant", content: "reflected, here is the answer" });
  };

  const result = await runAgentLoop({
    model: "p/m",
    handleSingleModel,
    emit: (e) => events.push(e as unknown as Record<string, unknown>),
    body: { messages: [{ role: "user", content: "write a file" }] },
    policy: { mode: "readonly" },
    reflection: true,
    context: { apiKeyId: "k", sessionId: "" },
    toolSchemas: TOOLS,
  });

  // A reflection event was emitted, counting the single failure.
  const refl = events.find((e) => e.type === "reflection");
  assert.ok(refl, "a reflection event was emitted");
  assert.equal(refl?.failures, 1);
  // The 2nd model turn saw a user-role reflection message threaded in.
  const secondTurnMessages = seenBodies[1].messages as Array<Record<string, unknown>>;
  const reflectionTurn = secondTurnMessages.find(
    (m) => m.role === "user" && typeof m.content === "string" && /root cause/i.test(m.content)
  );
  assert.ok(reflectionTurn, "a reflection user-turn was threaded before the retry");
  assert.equal(result.finalText, "reflected, here is the answer");
});

test("loop: reflection OFF (default) injects no self-critique turn", async () => {
  const events: Array<Record<string, unknown>> = [];
  const seenBodies: Body[] = [];
  let turn = 0;
  const handleSingleModel = async (body: Body) => {
    seenBodies.push(body);
    turn++;
    if (turn === 1) {
      return completion({
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "w1",
            function: { name: "file_write", arguments: '{"path":"o.txt","content":"x"}' },
          },
        ],
      });
    }
    return completion({ role: "assistant", content: "done" });
  };

  await runAgentLoop({
    model: "p/m",
    handleSingleModel,
    emit: (e) => events.push(e as unknown as Record<string, unknown>),
    body: { messages: [{ role: "user", content: "write a file" }] },
    policy: { mode: "readonly" },
    // reflection omitted → defaults to false → prior behaviour
    context: { apiKeyId: "k", sessionId: "" },
    toolSchemas: TOOLS,
  });

  assert.ok(!events.some((e) => e.type === "reflection"), "no reflection event when disabled");
  const secondTurnMessages = seenBodies[1].messages as Array<Record<string, unknown>>;
  assert.ok(
    !secondTurnMessages.some(
      (m) => m.role === "user" && typeof m.content === "string" && /root cause/i.test(m.content)
    ),
    "no reflection user-turn when disabled"
  );
});

test("loop: model HTTP error ends the loop with an error event", async () => {
  const events: Array<Record<string, unknown>> = [];
  const handleSingleModel = async () => new Response(null, { status: 503 });

  await runAgentLoop({
    model: "p/m",
    handleSingleModel,
    emit: (e) => events.push(e as unknown as Record<string, unknown>),
    body: { messages: [{ role: "user", content: "hi" }] },
    context: { apiKeyId: "k", sessionId: "" },
    toolSchemas: TOOLS,
  });

  assert.ok(events.some((e) => e.type === "error"));
});
