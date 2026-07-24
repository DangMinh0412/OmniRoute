/**
 * Agentic permission layer — decides, per tool call, whether the agentic loop
 * may execute it. This is the safety gate that sits between the model's emitted
 * `tool_calls` and the sandboxed builtin handlers (`./toolRegistry.ts`).
 *
 * The sandbox already contains a tool's blast radius (workspace jail, read-only
 * network-disabled containers, SSRF guard). This layer adds POLICY on top: which
 * risk tiers a given agentic session is even allowed to invoke, so a run can be
 * pinned to read-only, or require explicit pre-authorization before any mutation
 * or code execution happens.
 *
 * Pure and synchronous by design so it is trivially unit-testable and cannot
 * itself perform I/O or be a source of side effects.
 */
import { getToolRiskTier, type ToolRiskTier } from "./toolRegistry";

/**
 * Policy mode for an agentic run:
 *   - `readonly`: only `read` + `network` tools run; every `mutate` tool is denied.
 *   - `gated`:    `read` + `network` auto-run; `mutate` tools run ONLY when the
 *                 tool name is in `approvedTools`, otherwise they need approval.
 *   - `auto`:     every registered tool runs without approval (trusted caller).
 *
 * `gated` is the safe default: the loop can read and gather freely, but cannot
 * write files or run code until the caller has explicitly approved those tools.
 */
export type PermissionMode = "readonly" | "gated" | "auto";

export interface PermissionPolicy {
  mode: PermissionMode;
  /**
   * Tool names the caller has pre-authorized. Only consulted in `gated` mode for
   * `mutate`-tier tools; ignored in `readonly` (always denied) and `auto` (always
   * allowed). Unknown names are harmless — they simply never match a real tool.
   */
  approvedTools?: readonly string[];
}

export type PermissionDecision =
  | { verdict: "allow"; tier: ToolRiskTier }
  | { verdict: "deny"; tier: ToolRiskTier | "unknown"; reason: string }
  | { verdict: "needs-approval"; tier: ToolRiskTier; reason: string };

export const DEFAULT_PERMISSION_POLICY: PermissionPolicy = { mode: "gated" };

/**
 * Decide whether a single tool call is permitted under the given policy.
 *
 * An unknown tool is always denied — the model hallucinated a tool that does not
 * exist, and executing "nothing" silently would hide the mistake from the loop.
 */
export function evaluateToolCall(toolName: string, policy: PermissionPolicy): PermissionDecision {
  const tier = getToolRiskTier(toolName);
  if (!tier) {
    return {
      verdict: "deny",
      tier: "unknown",
      reason: `Unknown tool "${toolName}" is not registered`,
    };
  }

  // `read` and `network` tiers never mutate local state; they are always allowed
  // regardless of mode (the sandbox/SSRF guard still contains network reach).
  if (tier === "read" || tier === "network") {
    return { verdict: "allow", tier };
  }

  // tier === "mutate" from here on.
  switch (policy.mode) {
    case "auto":
      return { verdict: "allow", tier };
    case "readonly":
      return {
        verdict: "deny",
        tier,
        reason: `Tool "${toolName}" mutates state but the run is read-only`,
      };
    case "gated": {
      const approved = policy.approvedTools?.includes(toolName) ?? false;
      return approved
        ? { verdict: "allow", tier }
        : {
            verdict: "needs-approval",
            tier,
            reason: `Tool "${toolName}" requires explicit approval before it can run`,
          };
    }
    default: {
      // Exhaustiveness guard — a new mode must be handled explicitly rather than
      // silently falling through to "allow".
      const _never: never = policy.mode;
      return {
        verdict: "deny",
        tier,
        reason: `Unhandled permission mode: ${String(_never)}`,
      };
    }
  }
}
