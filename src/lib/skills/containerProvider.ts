/**
 * Container runtime providers for the OmniRoute skill sandbox.
 *
 * The sandbox historically hardcoded the `docker` CLI. This module abstracts
 * the container runtime so OmniRoute can pick the most performant / native
 * runtime available on each host:
 *
 *   - macOS:  Apple Container (`container` CLI)  >  OrbStack (docker shim)  >  Podman  >  Docker
 *   - Windows: WSL Container (`wslc` CLI)       >  Docker Desktop          >  Podman
 *   - Linux:   Podman (rootless, daemonless)    >  Docker
 *
 * The user can override the auto-detected choice with `SKILLS_SANDBOX_RUNTIME`
 * (`auto | docker | apple | wsl | orbstack | podman`). Each provider maps the
 * sandbox's intent (resource caps, network isolation, capability drops,
 * read-only fs, tmpfs workspaces) onto the runtime's native flag set.
 */

import { createRequire } from "module";
import os from "os";

const require = createRequire(import.meta.url);
const childProcess = require("child_process") as typeof import("child_process");

export type SandboxRuntimeId = "docker" | "apple" | "wsl" | "orbstack" | "podman" | "native";

export interface SandboxConfig {
  cpuLimit: number;
  memoryLimit: number;
  timeout: number;
  networkEnabled: boolean;
  readOnly: boolean;
}

export interface ResolvedContainerCommand {
  /** Absolute command to spawn (e.g. `"docker"`, `"container"`, `"wslc"`). */
  command: string;
  /** Arguments for the command. */
  args: string[];
  /** Arguments appended for the `kill` cleanup path. */
  killArgs: (containerName: string) => string[];
}

export interface ContainerProvider {
  readonly id: SandboxRuntimeId;
  readonly displayName: string;
  /** Returns true when this runtime is installed and usable on the host. */
  detect(): boolean;
  /** Build a run command for the given image, command, and config. */
  buildRun(
    image: string,
    command: string[],
    sandboxId: string,
    config: SandboxConfig
  ): ResolvedContainerCommand;
  /** Build a kill/stop command for a running container. */
  killCommand: string;
  buildKillArgs(name: string): string[];
}

// ----------------------------------------------------------------
//  Helpers
// ----------------------------------------------------------------

const SANDBOX_NAME = (sandboxId: string) => `omniroute-${sandboxId}`;

/**
 * Probe whether a CLI binary exists on PATH.
 * Uses `where` on Windows, `which` on *nix — both via spawnSync so existing
 * test mocks on `spawn` (but not `spawnSync`) are not disturbed.
 */
function probeCommand(binary: string): boolean {
  const args = process.platform === "win32" ? ["where", binary] : ["which", binary];
  const r = childProcess.spawnSync(args[0], args.slice(1), {
    encoding: "utf8",
    stdio: "ignore",
  });
  return r.status === 0;
}

/**
 * Probe whether a binary responds to `--version` with exit 0 and
 * a plausible version string.
 */
function probeVersion(binary: string, expects = "v"): boolean {
  const r = childProcess.spawnSync(binary, ["--version"], {
    encoding: "utf8",
    stdio: "pipe",
  });
  return r.status === 0 && !!r.stdout?.trim()?.includes(expects);
}

// ----------------------------------------------------------------
//  DockerProvider
// ----------------------------------------------------------------

class DockerProvider implements ContainerProvider {
  readonly id: SandboxRuntimeId = "docker";
  readonly displayName = "Docker";
  readonly killCommand = "docker";

  detect(): boolean {
    return probeCommand("docker") && probeVersion("docker");
  }

  buildRun(
    image: string,
    command: string[],
    sandboxId: string,
    config: SandboxConfig
  ): ResolvedContainerCommand {
    const args = [
      "run",
      "--rm",
      "--name",
      SANDBOX_NAME(sandboxId),
      "--cpus",
      `${config.cpuLimit / 100}`,
      "--memory",
      `${config.memoryLimit}m`,
      "--network",
      config.networkEnabled ? "bridge" : "none",
      "--cap-drop",
      "ALL",
      "--security-opt",
      "no-new-privileges",
      "--pids-limit",
      "100",
      "--tmpfs",
      "/tmp:rw,noexec,nosuid,size=64m",
      "--tmpfs",
      "/workspace:rw,noexec,nosuid,size=64m",
      "--workdir",
      "/workspace",
    ];
    if (config.readOnly) args.push("--read-only");
    args.push(image, ...command);
    return {
      command: "docker",
      args,
      killArgs: (name) => ["kill", name],
    };
  }

  buildKillArgs(name: string): string[] {
    return ["kill", name];
  }
}

// ----------------------------------------------------------------
//  AppleContainerProvider  (native Apple Container on macOS)
// ----------------------------------------------------------------

class AppleContainerProvider implements ContainerProvider {
  readonly id: SandboxRuntimeId = "apple";
  readonly displayName = "Apple Container";
  readonly killCommand = "container";

  detect(): boolean {
    return probeCommand("container") && probeVersion("container", "c");
  }

  buildRun(
    image: string,
    command: string[],
    sandboxId: string,
    config: SandboxConfig
  ): ResolvedContainerCommand {
    const args = [
      "run",
      "--rm",
      "--name",
      SANDBOX_NAME(sandboxId),
      "--cpus",
      `${config.cpuLimit}`,
      "--memory",
      `${config.memoryLimit}m`,
      "--network",
      config.networkEnabled ? "bridge" : "none",
      "--cap-drop",
      "ALL",
      "--security-opt",
      "no-new-privileges",
      "--tmpfs",
      "/tmp:rw,noexec,nosuid,size=64m",
      "--tmpfs",
      "/workspace:rw,noexec,nosuid,size=64m",
      "--workdir",
      "/workspace",
    ];
    if (config.readOnly) args.push("--read-only");
    args.push(image, ...command);
    return {
      command: "container",
      args,
      killArgs: (name) => ["kill", name],
    };
  }

  buildKillArgs(name: string): string[] {
    return ["kill", name];
  }
}

// ----------------------------------------------------------------
//  WslContainerProvider  (WSL 2 container CLI on Windows)
// ----------------------------------------------------------------

class WslContainerProvider implements ContainerProvider {
  readonly id: SandboxRuntimeId = "wsl";
  readonly displayName = "WSL Container";
  readonly killCommand = "wslc";

  detect(): boolean {
    return probeCommand("wslc") && probeVersion("wslc");
  }

  buildRun(
    image: string,
    command: string[],
    sandboxId: string,
    config: SandboxConfig
  ): ResolvedContainerCommand {
    const args = [
      "run",
      "--rm",
      "--name",
      SANDBOX_NAME(sandboxId),
      "--cpus",
      `${config.cpuLimit}`,
      "--memory",
      `${config.memoryLimit}m`,
      "--network",
      config.networkEnabled ? "bridge" : "none",
      "--tmpfs",
      "/tmp:rw,noexec,nosuid,size=64m",
      "--tmpfs",
      "/workspace:rw,noexec,nosuid,size=64m",
      "--workdir",
      "/workspace",
    ];
    if (config.readOnly) args.push("--read-only");
    args.push(image, ...command);
    return {
      command: "wslc",
      args,
      killArgs: (name) => ["kill", name],
    };
  }

  buildKillArgs(name: string): string[] {
    return ["kill", name];
  }
}

// ----------------------------------------------------------------
//  OrbStackProvider  (high-perf Linux VM on macOS)
// ----------------------------------------------------------------

class OrbStackProvider implements ContainerProvider {
  readonly id: SandboxRuntimeId = "orbstack";
  readonly displayName = "OrbStack";
  readonly killCommand = "orbstack";

  detect(): boolean {
    return probeCommand("orbstack") && probeVersion("orbstack");
  }

  buildRun(
    image: string,
    command: string[],
    sandboxId: string,
    config: SandboxConfig
  ): ResolvedContainerCommand {
    // OrbStack wraps Docker inside a Linux VM.  We invoke the `orbstack`
    // binary which shims `docker` transparently.
    const args = [
      "run",
      "--rm",
      "--name",
      SANDBOX_NAME(sandboxId),
      "--cpus",
      `${config.cpuLimit}`,
      "--memory",
      `${config.memoryLimit}m`,
      "--network",
      config.networkEnabled ? "bridge" : "none",
      "--tmpfs",
      "/tmp:rw,noexec,nosuid,size=64m",
      "--tmpfs",
      "/workspace:rw,noexec,nosuid,size=64m",
      "--workdir",
      "/workspace",
    ];
    if (config.readOnly) args.push("--read-only");
    args.push(image, ...command);
    return {
      command: "orbstack",
      args,
      killArgs: (name) => ["kill", name],
    };
  }

  buildKillArgs(name: string): string[] {
    return ["kill", name];
  }
}

// ----------------------------------------------------------------
//  PodmanProvider  (rootless Linux alternative)
// ----------------------------------------------------------------

class PodmanProvider implements ContainerProvider {
  readonly id: SandboxRuntimeId = "podman";
  readonly displayName = "Podman";
  readonly killCommand = "podman";

  detect(): boolean {
    return probeCommand("podman") && probeVersion("podman");
  }

  buildRun(
    image: string,
    command: string[],
    sandboxId: string,
    config: SandboxConfig
  ): ResolvedContainerCommand {
    const args = [
      "run",
      "--rm",
      "--name",
      SANDBOX_NAME(sandboxId),
      "--cpus",
      `${config.cpuLimit / 100}`,
      "--memory",
      `${config.memoryLimit}m`,
      "--network",
      config.networkEnabled ? "bridge" : "none",
      "--cap-drop",
      "ALL",
      "--security-opt",
      "no-new-privileges",
      "--tmpfs",
      "/tmp:rw,noexec,nosuid,size=64m",
      "--tmpfs",
      "/workspace:rw,noexec,nosuid,size=64m",
      "--workdir",
      "/workspace",
    ];
    if (config.readOnly) args.push("--read-only");
    args.push(image, ...command);
    return {
      command: "podman",
      args,
      killArgs: (name) => ["kill", name],
    };
  }

  buildKillArgs(name: string): string[] {
    return ["kill", name];
  }
}

// ----------------------------------------------------------------
//  NativeProvider  (DIRECT host execution — NO container isolation)
// ----------------------------------------------------------------

/**
 * Runs the requested command DIRECTLY on the host machine, with no container,
 * no resource caps, no filesystem jail, and no capability drops. This is the
 * "run anything on my computer like Claude Code" surface the operator explicitly
 * opted into.
 *
 * ⚠️ SECURITY: this provider removes the sandbox's entire containment layer. It
 * is therefore:
 *   - NEVER auto-detected / auto-selected. It is returned by `resolveProvider()`
 *     ONLY when `SKILLS_ALLOW_NATIVE_EXEC` is explicitly enabled AND
 *     `SKILLS_SANDBOX_RUNTIME=native` is set — two independent opt-ins.
 *   - Guarded upstream by the destructive-command screen (`destructiveGuard.ts`)
 *     applied in `SandboxRunner.run()` before any native spawn, and by the
 *     loopback route gate + permission policy in front of that.
 *
 * `image` is ignored (there is no container to pull); `command[0]` is the binary
 * and `command.slice(1)` its arguments. The runtime `readOnly` / network / cpu /
 * memory config fields are inert here — the host does not enforce them.
 */
class NativeProvider implements ContainerProvider {
  readonly id: SandboxRuntimeId = "native";
  readonly displayName = "Native (host, no isolation)";
  readonly killCommand = process.platform === "win32" ? "taskkill" : "kill";

  detect(): boolean {
    // Only "available" when the operator has explicitly enabled native exec.
    // This keeps it out of any accidental auto-selection path.
    return (
      process.env.SKILLS_ALLOW_NATIVE_EXEC === "1" ||
      process.env.SKILLS_ALLOW_NATIVE_EXEC === "true"
    );
  }

  buildRun(
    image: string,
    command: string[],
    _sandboxId: string,
    _config: SandboxConfig
  ): ResolvedContainerCommand {
    void image;
    void _sandboxId;
    void _config;
    const [binary, ...rest] = command;
    return {
      command: binary ?? "",
      args: rest,
      killArgs: (name) => ["kill", name],
    };
  }

  buildKillArgs(name: string): string[] {
    return ["kill", name];
  }
}

// ----------------------------------------------------------------
//  Registry & auto-detection
// ----------------------------------------------------------------

export const ALL_PROVIDERS: ContainerProvider[] = [
  new DockerProvider(),
  new AppleContainerProvider(),
  new WslContainerProvider(),
  new OrbStackProvider(),
  new PodmanProvider(),
  new NativeProvider(),
];

export const PROVIDER_BY_ID = new Map<SandboxRuntimeId, ContainerProvider>(
  ALL_PROVIDERS.map((p) => [p.id, p])
);

/** Priority order for auto-detection on each platform. */
export function platformPriority(): SandboxRuntimeId[] {
  switch (os.platform()) {
    case "darwin":
      // Apple Container is the native micro-VM runtime on Apple Silicon —
      // fastest startup, lowest overhead.  OrbStack provides a Docker shim
      // inside a tuned Linux VM; better than stock Docker Desktop.
      return ["apple", "orbstack", "podman", "docker"];
    case "win32":
      // WSL Container CLI (wslc.exe) is Windows-native via WSL 2.
      return ["wsl", "docker", "podman"];
    default:
      // Linux — podman is rootless + daemonless and therefore preferred.
      return ["podman", "docker"];
  }
}

// Detect-once memoization
let detectionInFlight: Promise<void> | null = null;
const detectionCache = new Map<SandboxRuntimeId, boolean>();

function clearDetectionCache(): void {
  detectionInFlight = null;
  detectionCache.clear();
}

async function runDetection(): Promise<void> {
  // Run all probes in parallel for speed
  await Promise.all(
    ALL_PROVIDERS.map(async (provider) => {
      const ok = await Promise.resolve(provider.detect());
      detectionCache.set(provider.id, ok);
    })
  );
}

function normaliseRuntimeOverride(raw: string | undefined): SandboxRuntimeId | null {
  if (!raw || raw === "auto") return null;
  const lowered = raw.toLowerCase().trim();
  if (PROVIDER_BY_ID.has(lowered as SandboxRuntimeId)) return lowered as SandboxRuntimeId;
  return null;
}

/**
 * Resolves which runtime the sandbox should use for the current host.
 *
 * Resolution rules (in order):
 *   1. Explicit override via `SKILLS_SANDBOX_RUNTIME`.
 *   2. Auto-detect: walk the platform priority list and pick the first
 *      runtime whose `detect()` succeeds.
 *   3. Fall back to the Docker provider (the historical default) even if
 *      detection fails — the spawn will surface a clear "docker not
 *      found" error if Docker really is missing.
 */
export async function resolveProvider(): Promise<ContainerProvider> {
  if (!detectionInFlight) {
    detectionInFlight = runDetection();
  }
  await detectionInFlight;

  const override = normaliseRuntimeOverride(process.env.SKILLS_SANDBOX_RUNTIME);
  if (override) {
    const provider = PROVIDER_BY_ID.get(override)!;
    // SECURITY: `native` (host exec, no isolation) is the ONE runtime whose
    // override is NOT honoured on faith. It requires the SECOND, independent
    // opt-in (`SKILLS_ALLOW_NATIVE_EXEC`, surfaced as its detect()). Without
    // that, setting `SKILLS_SANDBOX_RUNTIME=native` alone must NOT drop the
    // container — fall through to safe auto-detection instead. Every other
    // runtime keeps the "honour override even if the probe failed" behaviour
    // (a locked-down CI may hide an installed docker/podman from the probe).
    if (provider.id === "native") {
      if (detectionCache.get("native")) return provider;
      // native override requested but not enabled → ignore it, stay contained.
    } else {
      if (detectionCache.get(provider.id)) return provider;
      return provider;
    }
  }

  for (const id of platformPriority()) {
    if (detectionCache.get(id)) return PROVIDER_BY_ID.get(id)!;
  }
  return PROVIDER_BY_ID.get("docker")!;
}

/** Exposed for tests — forces a fresh detection pass. */
export function _resetProviderCacheForTests(): void {
  clearDetectionCache();
}

/**
 * Returns the kill command for the given provider, parameterised with the
 * sandbox's container name. Used by SandboxRunner.kill/killAll.
 */
export function buildKillCommand(
  provider: ContainerProvider,
  sandboxId: string
): { command: string; args: string[] } {
  const name = SANDBOX_NAME(sandboxId);
  return {
    command: provider.killCommand,
    args: provider.buildKillArgs(name),
  };
}
