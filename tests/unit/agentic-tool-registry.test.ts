/**
 * Agentic tool registry — schema bridge + risk-tier classification.
 *
 * The registry maps the 7 sandboxed builtin skill handlers into OpenAI tool
 * schemas and assigns each a risk tier the permission layer gates on. These
 * tests assert the bridge stays aligned with the real builtins (no drift) and
 * the schema shape the panel models receive is well-formed.
 */
import test from "node:test";
import assert from "node:assert/strict";

const { AGENT_TOOL_NAMES, getAgentTool, buildToolSchemas, getToolRiskTier } =
  await import("../../src/lib/agentic/toolRegistry.ts");
const { builtinSkills } = await import("../../src/lib/skills/builtins.ts");

test("every registered tool name resolves to a real builtin handler", () => {
  for (const name of AGENT_TOOL_NAMES) {
    const entry = getAgentTool(name);
    assert.ok(entry, `tool ${name} must have a registry entry`);
    assert.equal(typeof entry?.handler, "function", `tool ${name} handler must be a function`);
    // The registry handler must be the SAME reference as the builtin — proving
    // we dispatch into the sandbox, not a re-implementation.
    assert.equal(
      entry?.handler,
      (builtinSkills as Record<string, unknown>)[name],
      `tool ${name} must bind the real builtin handler`
    );
  }
});

test("registry does not expose a tool that has no builtin", () => {
  for (const name of AGENT_TOOL_NAMES) {
    assert.ok(
      name in (builtinSkills as Record<string, unknown>),
      `registered tool ${name} has no matching builtin`
    );
  }
});

test("risk tiers: read/network never mutate, mutate covers code + fs writes", () => {
  assert.equal(getToolRiskTier("file_read"), "read");
  assert.equal(getToolRiskTier("web_search"), "network");
  assert.equal(getToolRiskTier("web_fetch"), "network");
  assert.equal(getToolRiskTier("http_request"), "network");
  assert.equal(getToolRiskTier("file_write"), "mutate");
  assert.equal(getToolRiskTier("eval_code"), "mutate");
  assert.equal(getToolRiskTier("execute_command"), "mutate");
});

test("getToolRiskTier returns undefined for an unknown tool", () => {
  assert.equal(getToolRiskTier("does_not_exist"), undefined);
});

test("buildToolSchemas: exposes every tool by default, all well-formed", () => {
  const schemas = buildToolSchemas();
  assert.equal(schemas.length, AGENT_TOOL_NAMES.length);
  for (const s of schemas) {
    assert.equal(s.type, "function");
    assert.equal(typeof s.function.name, "string");
    assert.ok(s.function.description.length > 0, "each tool needs a description");
    assert.equal(s.function.parameters.type, "object");
    assert.equal(s.function.parameters.additionalProperties, false);
  }
});

test("buildToolSchemas: restricts to allowedTools, ignoring unknown names", () => {
  const schemas = buildToolSchemas(["file_read", "not_a_tool"]);
  assert.equal(schemas.length, 1);
  assert.equal(schemas[0].function.name, "file_read");
});

test("buildToolSchemas: empty allowedTools falls back to the full set", () => {
  assert.equal(buildToolSchemas([]).length, AGENT_TOOL_NAMES.length);
});
