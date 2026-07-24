import { test } from "node:test";
import assert from "node:assert/strict";
import {
  isLocalOnlyPath,
  isLocalOnlyBypassableByManageScope,
} from "../../../src/server/authz/routeGuard.ts";
import { SPAWN_CAPABLE_PREFIXES } from "../../../src/shared/constants/spawnCapablePrefixes.ts";

// ─── /api/v1/agent: agentic ReAct loop spawns Docker containers TRANSITIVELY ──
// The endpoint lets a model emit eval_code / execute_command tool calls, which
// run through src/lib/agentic/toolRegistry → src/lib/skills/builtins →
// sandboxRunner.run → childProcess.spawn("docker", …). The spawn is INDIRECT:
// route.ts does not import child_process itself, so the source-scan subcheck of
// scripts/check/check-route-guard-membership.ts CANNOT see it — this test is the
// regression guard that keeps the classification in place. Same RCE-via-tunnel
// CVE class (GHSA-fhh6-4qxv-rpqj) as /api/vnc-session and /api/local/, both of
// which spawn docker and are LOCAL_ONLY (Hard Rules #15 + #17).

test("isLocalOnlyPath: /api/v1/agent is local-only (transitive docker spawn via eval_code/execute_command)", () => {
  assert.equal(isLocalOnlyPath("/api/v1/agent"), true);
});

test("isLocalOnlyBypassableByManageScope: /api/v1/agent is NOT bypassable (defence in depth)", () => {
  assert.ok(
    SPAWN_CAPABLE_PREFIXES.includes("/api/v1/agent"),
    "/api/v1/agent must be in SPAWN_CAPABLE_PREFIXES so a malformed DB bypass row can never open it"
  );
  assert.equal(isLocalOnlyBypassableByManageScope("/api/v1/agent"), false);
});
