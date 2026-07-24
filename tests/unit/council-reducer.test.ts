/**
 * useCouncilStream reducer — pure SSE event folding.
 *
 * reduceCouncilEvent / extractSynthesisText are exported so the council console's
 * streaming state machine is unit-testable without a live SSE stream or React.
 * These tests assert every event type folds correctly and unknown events are inert.
 */
import test from "node:test";
import assert from "node:assert/strict";

const { reduceCouncilEvent, extractSynthesisText } =
  await import("../../src/app/(dashboard)/dashboard/council/useCouncilStream.ts");

type State = ReturnType<typeof reduceCouncilEvent>;

const INITIAL: State = {
  running: true,
  rounds: [],
  judge: null,
  synthesis: "",
  done: null,
  error: null,
  warnings: [],
  verifyEvaluator: null,
  verifyCritique: "",
};

function fold(events: Array<Record<string, unknown>>, from: State = INITIAL): State {
  return events.reduce<State>((s, ev) => reduceCouncilEvent(s, ev), from);
}

test("extractSynthesisText: pulls assistant content, else empty string", () => {
  assert.equal(
    extractSynthesisText({ choices: [{ message: { role: "assistant", content: "hello" } }] }),
    "hello"
  );
  assert.equal(extractSynthesisText({}), "");
  assert.equal(extractSynthesisText(null), "");
  assert.equal(extractSynthesisText({ choices: [] }), "");
});

test("round_start: appends a round; duplicate round is ignored", () => {
  const s = fold([
    { type: "round_start", round: 0, models: ["a", "b"] },
    { type: "round_start", round: 0, models: ["a", "b"] }, // dup
  ]);
  assert.equal(s.rounds.length, 1);
  assert.deepEqual(s.rounds[0].models, ["a", "b"]);
});

test("panel_answer: attaches answers to the matching round", () => {
  const s = fold([
    { type: "round_start", round: 0, models: ["a", "b"] },
    { type: "panel_answer", round: 0, model: "a", text: "ans-a" },
    { type: "panel_answer", round: 0, model: "b", text: "ans-b" },
  ]);
  assert.equal(s.rounds[0].answers.length, 2);
  assert.deepEqual(
    s.rounds[0].answers.map((x) => x.model),
    ["a", "b"]
  );
});

test("consensus: records the score on the matching round", () => {
  const s = fold([
    { type: "round_start", round: 1, models: ["a", "b"] },
    { type: "consensus", round: 1, score: 0.91 },
  ]);
  assert.equal(s.rounds[0].consensusScore, 0.91);
});

test("synthesis_start + token: sets judge and accumulates streamed tokens", () => {
  const s = fold([
    { type: "synthesis_start", judge: "j/model" },
    { type: "token", text: "Hello " },
    { type: "token", text: "world" },
  ]);
  assert.equal(s.judge, "j/model");
  assert.equal(s.synthesis, "Hello world");
});

test("synthesis (non-stream): appends text or completion content", () => {
  const fromText = fold([{ type: "synthesis", text: "final" }]);
  assert.equal(fromText.synthesis, "final");

  const fromCompletion = fold([
    { type: "synthesis", completion: { choices: [{ message: { content: "done" } }] } },
  ]);
  assert.equal(fromCompletion.synthesis, "done");
});

test("done: records the run summary", () => {
  const s = fold([{ type: "done", rounds: 2, totalAnswers: 4, durationMs: 1234 }]);
  assert.deepEqual(s.done, { rounds: 2, totalAnswers: 4, durationMs: 1234 });
});

test("error: records the message", () => {
  const s = fold([{ type: "error", message: "boom" }]);
  assert.equal(s.error, "boom");
});

test("panel_tool_call: appends a call entry to the matching round's toolActivity", () => {
  const s = fold([
    { type: "round_start", round: 0, models: ["a", "b"] },
    {
      type: "panel_tool_call",
      round: 0,
      model: "a",
      name: "web_search",
      iteration: 0,
      arguments: { query: "x" },
    },
  ]);
  assert.equal(s.rounds[0].toolActivity?.length, 1);
  const entry = s.rounds[0].toolActivity![0];
  assert.equal(entry.kind, "call");
  assert.equal(entry.model, "a");
  assert.equal(entry.name, "web_search");
});

test("panel_tool_result: records ok flag; failure preserves ok:false", () => {
  const s = fold([
    { type: "round_start", round: 0, models: ["a"] },
    { type: "panel_tool_result", round: 0, model: "a", name: "eval_code", iteration: 1, ok: true },
    { type: "panel_tool_result", round: 0, model: "a", name: "eval_code", iteration: 2, ok: false },
  ]);
  const acts = s.rounds[0].toolActivity!;
  assert.equal(acts.length, 2);
  assert.equal(acts[0].kind, "result");
  assert.equal(acts[0].ok, true);
  assert.equal(acts[1].ok, false);
});

test("panel_tool_denied: records the reason", () => {
  const s = fold([
    { type: "round_start", round: 0, models: ["a"] },
    {
      type: "panel_tool_denied",
      round: 0,
      model: "a",
      name: "file_write",
      iteration: 0,
      reason: "readonly policy",
    },
  ]);
  const entry = s.rounds[0].toolActivity![0];
  assert.equal(entry.kind, "denied");
  assert.equal(entry.reason, "readonly policy");
});

test("panel_tool_* for an unknown round is inert (no matching round)", () => {
  const s = fold([
    { type: "round_start", round: 0, models: ["a"] },
    { type: "panel_tool_call", round: 5, model: "a", name: "web_search", iteration: 0 },
  ]);
  assert.equal(s.rounds[0].toolActivity, undefined);
});

test("warning: accumulates messages at the top level", () => {
  const s = fold([
    { type: "warning", message: "panelTools disabled — non-loopback" },
    { type: "warning", message: "second warning" },
  ]);
  assert.deepEqual(s.warnings, ["panelTools disabled — non-loopback", "second warning"]);
});

test("verify_start + verify_critique: records the evaluator and its critique", () => {
  const s = fold([
    { type: "verify_start", evaluator: "p/critic" },
    { type: "verify_critique", text: "claim 2 is unsupported" },
  ]);
  assert.equal(s.verifyEvaluator, "p/critic");
  assert.equal(s.verifyCritique, "claim 2 is unsupported");
});

test("unknown event type: state is returned unchanged (inert)", () => {
  const s = fold([{ type: "totally_unknown", foo: 1 }]);
  assert.deepEqual(s, INITIAL);
});

test("full realistic stream folds into coherent final state", () => {
  const s = fold([
    { type: "round_start", round: 0, models: ["a", "b"] },
    { type: "panel_answer", round: 0, model: "a", text: "a0" },
    { type: "panel_answer", round: 0, model: "b", text: "b0" },
    { type: "round_end", round: 0, answers: 2 },
    { type: "round_start", round: 1, models: ["a", "b"] },
    { type: "panel_answer", round: 1, model: "a", text: "a1" },
    { type: "panel_answer", round: 1, model: "b", text: "b1" },
    { type: "consensus", round: 1, score: 0.9 },
    { type: "synthesis_start", judge: "a" },
    { type: "token", text: "Final " },
    { type: "token", text: "answer." },
    { type: "done", rounds: 2, totalAnswers: 4, durationMs: 500 },
  ]);
  assert.equal(s.rounds.length, 2);
  assert.equal(s.rounds[1].consensusScore, 0.9);
  assert.equal(s.judge, "a");
  assert.equal(s.synthesis, "Final answer.");
  assert.equal(s.done?.totalAnswers, 4);
  assert.equal(s.error, null);
});
