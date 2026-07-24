/**
 * Agentic permission layer — per-tool-call policy gate.
 *
 * evaluateToolCall is pure; these tests pin the decision matrix across the three
 * policy modes and the read/network/mutate risk tiers, plus the unknown-tool
 * deny path (a hallucinated tool must never silently no-op).
 */
import test from "node:test";
import assert from "node:assert/strict";

const { evaluateToolCall, DEFAULT_PERMISSION_POLICY } =
  await import("../../src/lib/agentic/permissions.ts");

test("read/network tools are allowed regardless of mode", () => {
  for (const mode of ["readonly", "gated", "auto"] as const) {
    assert.equal(evaluateToolCall("file_read", { mode }).verdict, "allow");
    assert.equal(evaluateToolCall("web_search", { mode }).verdict, "allow");
    assert.equal(evaluateToolCall("http_request", { mode }).verdict, "allow");
  }
});

test("readonly mode denies every mutate tool", () => {
  const d = evaluateToolCall("file_write", { mode: "readonly" });
  assert.equal(d.verdict, "deny");
  assert.equal(d.tier, "mutate");
  assert.equal(evaluateToolCall("eval_code", { mode: "readonly" }).verdict, "deny");
  assert.equal(evaluateToolCall("execute_command", { mode: "readonly" }).verdict, "deny");
});

test("auto mode allows mutate tools without approval", () => {
  assert.equal(evaluateToolCall("file_write", { mode: "auto" }).verdict, "allow");
  assert.equal(evaluateToolCall("execute_command", { mode: "auto" }).verdict, "allow");
});

test("gated mode: mutate tool needs approval unless pre-approved", () => {
  const needs = evaluateToolCall("file_write", { mode: "gated" });
  assert.equal(needs.verdict, "needs-approval");
  assert.equal(needs.tier, "mutate");

  const approved = evaluateToolCall("file_write", {
    mode: "gated",
    approvedTools: ["file_write"],
  });
  assert.equal(approved.verdict, "allow");

  // A different approved tool does not unlock this one.
  const other = evaluateToolCall("eval_code", {
    mode: "gated",
    approvedTools: ["file_write"],
  });
  assert.equal(other.verdict, "needs-approval");
});

test("unknown tool is always denied with tier 'unknown'", () => {
  for (const mode of ["readonly", "gated", "auto"] as const) {
    const d = evaluateToolCall("hallucinated_tool", { mode });
    assert.equal(d.verdict, "deny");
    assert.equal(d.tier, "unknown");
  }
});

test("default policy is gated (safe default)", () => {
  assert.equal(DEFAULT_PERMISSION_POLICY.mode, "gated");
  assert.equal(evaluateToolCall("file_write", DEFAULT_PERMISSION_POLICY).verdict, "needs-approval");
});
