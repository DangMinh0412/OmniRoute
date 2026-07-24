/**
 * Agentic tool registry — bridges the sandboxed builtin skill handlers
 * (`src/lib/skills/builtins.ts`) into OpenAI-compatible tool schemas the LLM
 * panel can call, and classifies each tool by a risk tier that the permission
 * layer (`./permissions.ts`) gates on.
 *
 * Design decisions:
 *   - We call `builtinSkills[name]` DIRECTLY rather than routing through
 *     `skillExecutor.execute()`. The executor requires a `Skill` DB row
 *     (`skillRegistry.getSkill`) per builtin; the agentic loop instead reuses
 *     only the sandboxed handler + its own audit trail (`agent_tool_calls`).
 *     Security still comes from the sandbox (workspace jail, container isolation,
 *     SSRF guard) that lives inside each handler.
 *   - Tool schemas are hand-authored (not derived) so the descriptions steer the
 *     model toward correct, safe usage — the builtin `SkillHandler`s carry no
 *     machine-readable JSON schema of their own.
 *   - The registry is the single source of truth for BOTH the LLM-facing schema
 *     and the risk tier, so the two can never drift.
 */
import { builtinSkills } from "@/lib/skills/builtins";
import type { SkillHandler } from "@/lib/skills/types";

/**
 * Risk tier drives the permission gate. `read` tools never mutate state and are
 * auto-approved; `network` tools reach outbound (SSRF-guarded) endpoints;
 * `mutate` tools change the workspace or run arbitrary code in a container and
 * require per-action approval unless the caller pre-authorizes them.
 */
export type ToolRiskTier = "read" | "network" | "mutate";

/** OpenAI-compatible function-tool definition sent to the panel models. */
export interface AgentToolSchema {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: "object";
      properties: Record<string, unknown>;
      required?: string[];
      additionalProperties?: boolean;
    };
  };
}

/** Internal registry entry: schema + risk tier + the sandboxed handler. */
export interface AgentToolEntry {
  name: string;
  riskTier: ToolRiskTier;
  schema: AgentToolSchema;
  handler: SkillHandler;
}

function schema(
  name: string,
  description: string,
  properties: Record<string, unknown>,
  required: string[]
): AgentToolSchema {
  return {
    type: "function",
    function: {
      name,
      description,
      parameters: {
        type: "object",
        properties,
        required,
        additionalProperties: false,
      },
    },
  };
}

/**
 * The seven sandboxed builtins, each mapped to a risk tier and an LLM-facing
 * schema. `handler` is resolved from `builtinSkills` at module load so a missing
 * builtin fails fast (throws) rather than silently registering a broken tool.
 */
function resolveHandler(name: string): SkillHandler {
  const handler = builtinSkills[name];
  if (!handler) {
    throw new Error(`Agentic tool registry: builtin skill "${name}" is not registered`);
  }
  return handler;
}

const ENTRIES: AgentToolEntry[] = [
  {
    name: "file_read",
    riskTier: "read",
    handler: resolveHandler("file_read"),
    schema: schema(
      "file_read",
      "Read a UTF-8 or base64 file from the agent's private workspace. Paths are relative to the workspace root; absolute paths and restricted segments (.env, .git, .ssh) are rejected.",
      {
        path: { type: "string", description: "Workspace-relative file path" },
        encoding: {
          type: "string",
          enum: ["utf8", "base64"],
          description: "Read encoding (default utf8)",
        },
      },
      ["path"]
    ),
  },
  {
    name: "file_write",
    riskTier: "mutate",
    handler: resolveHandler("file_write"),
    schema: schema(
      "file_write",
      "Write or append a UTF-8 file inside the agent's private workspace. Creates parent directories as needed. Mutating — requires approval.",
      {
        path: { type: "string", description: "Workspace-relative file path" },
        content: { type: "string", description: "File contents to write" },
        append: { type: "boolean", description: "Append instead of overwrite (default false)" },
        overwrite: {
          type: "boolean",
          description: "Overwrite if the file exists (default true); ignored when append=true",
        },
      },
      ["path", "content"]
    ),
  },
  {
    name: "http_request",
    riskTier: "network",
    handler: resolveHandler("http_request"),
    schema: schema(
      "http_request",
      "Make an outbound HTTP request to a PUBLIC endpoint. Private/internal addresses are blocked by an SSRF guard; redirects are disabled; auth/cookie headers are stripped.",
      {
        url: { type: "string", description: "Absolute http(s) URL of a public endpoint" },
        method: {
          type: "string",
          enum: ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
          description: "HTTP method (default GET)",
        },
        headers: { type: "object", description: "Request headers (unsafe headers are stripped)" },
        body: { type: "string", description: "Request body for non-GET/HEAD methods" },
      },
      ["url"]
    ),
  },
  {
    name: "web_search",
    riskTier: "network",
    handler: resolveHandler("web_search"),
    schema: schema(
      "web_search",
      "Search the web and return ranked results (title, url, snippet) plus an optional synthesized answer. Use to gather current information before answering.",
      {
        query: { type: "string", description: "Search query" },
        max_results: { type: "number", description: "Maximum number of results" },
        search_type: {
          type: "string",
          enum: ["web", "news"],
          description: "Result category (default web)",
        },
      },
      ["query"]
    ),
  },
  {
    name: "web_fetch",
    riskTier: "network",
    handler: resolveHandler("web_fetch"),
    schema: schema(
      "web_fetch",
      "Fetch and extract the readable content of a single web page (markdown, html, or extracted links).",
      {
        url: { type: "string", description: "Absolute URL of the page to fetch" },
        format: {
          type: "string",
          enum: ["markdown", "html", "links"],
          description: "Extraction format (default markdown)",
        },
      },
      ["url"]
    ),
  },
  {
    name: "eval_code",
    riskTier: "mutate",
    handler: resolveHandler("eval_code"),
    schema: schema(
      "eval_code",
      "Execute a short JavaScript or Python snippet in an isolated, read-only, network-disabled container and return stdout/stderr. Use for calculation, data transformation, and verification. Mutating tier — requires approval.",
      {
        code: { type: "string", description: "Source code to execute" },
        language: {
          type: "string",
          enum: ["javascript", "python"],
          description: "Language runtime (default javascript)",
        },
      },
      ["code"]
    ),
  },
  {
    name: "execute_command",
    riskTier: "mutate",
    handler: resolveHandler("execute_command"),
    schema: schema(
      "execute_command",
      "Run a single command with arguments in an isolated, read-only, network-disabled container and return stdout/stderr. Mutating tier — requires approval.",
      {
        command: { type: "string", description: "Executable name (no shell interpolation)" },
        args: {
          type: "array",
          items: { type: "string" },
          description: "Command arguments as an array of strings",
        },
      },
      ["command"]
    ),
  },
];

const BY_NAME = new Map<string, AgentToolEntry>(ENTRIES.map((e) => [e.name, e]));

/** All registered tool names (stable order). */
export const AGENT_TOOL_NAMES: readonly string[] = ENTRIES.map((e) => e.name);

/** Look up a single tool entry by name, or `undefined` if unknown. */
export function getAgentTool(name: string): AgentToolEntry | undefined {
  return BY_NAME.get(name);
}

/**
 * Build the OpenAI `tools` array for a panel request. When `allowedTools` is
 * provided, only those tools are exposed (unknown names are ignored); otherwise
 * every registered tool is exposed.
 */
export function buildToolSchemas(allowedTools?: readonly string[]): AgentToolSchema[] {
  const source =
    allowedTools && allowedTools.length > 0
      ? ENTRIES.filter((e) => allowedTools.includes(e.name))
      : ENTRIES;
  return source.map((e) => e.schema);
}

/** Risk tier for a tool name, or `undefined` when the tool is unknown. */
export function getToolRiskTier(name: string): ToolRiskTier | undefined {
  return BY_NAME.get(name)?.riskTier;
}
