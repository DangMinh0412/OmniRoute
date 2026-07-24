/**
 * Native host-execution LIVE proof — actually spawns real processes on the host.
 *
 * The sibling suite `skills-native-exec-guard.test.ts` pins the *gating logic*
 * (guard block/allow decisions + two-opt-in provider resolution) with no I/O.
 * This suite closes the honesty gap called out during review: it proves the
 * native surface works END-TO-END by really running a command on the host
 * through `sandboxRunner.run()` and asserting on the actual captured stdout /
 * exit code, and by proving the destructive guard blocks a real dangerous
 * command BEFORE any spawn happens.
 *
 * All tests set BOTH native opt-ins (SKILLS_ALLOW_NATIVE_EXEC=1 +
 * SKILLS_SANDBOX_RUNTIME=native) around a forced provider-cache reset, and
 * restore the prior env afterwards, so the process default (contained) is never
 * changed for other suites.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import path from "node:path";

async function importFresh(modulePath: string) {
  const url = pathToFileURL(path.resolve(modulePath)).href;
  return import(`${url}?live=${Date.now()}-${Math.random().toString(16).slice(2)}`);
}

const NATIVE_ENV = ["SKILLS_ALLOW_NATIVE_EXEC", "SKILLS_SANDBOX_RUNTIME"] as const;
const saved: Record<string, string | undefined> = {};

function enableNative() {
  for (const k of NATIVE_ENV) saved[k] = process.env[k];
  process.env.SKILLS_ALLOW_NATIVE_EXEC = "1";
  process.env.SKILLS_SANDBOX_RUNTIME = "native";
}
function restore() {
  for (const k of NATIVE_ENV) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k];
  }
}

/** A trivially safe, cross-platform command that echoes a known token. */
function echoCommand(token: string): string[] {
  return process.platform === "win32"
    ? ["cmd", "/c", `echo ${token}`]
    : ["sh", "-c", `echo ${token}`];
}

test("LIVE: native provider really runs a host command and captures stdout", async () => {
  enableNative();
  try {
    const providerMod = await importFresh("src/lib/skills/containerProvider.ts");
    providerMod._resetProviderCacheForTests();
    const provider = await providerMod.resolveProvider();
    assert.equal(provider.id, "native", "both opt-ins set → native provider");

    // Fresh sandbox module so it resolves the (native) provider we just enabled.
    const sandboxMod = await importFresh("src/lib/skills/sandbox.ts");
    const runner = sandboxMod.sandboxRunner;

    const token = `omniroute-live-${Math.random().toString(16).slice(2)}`;
    // image is ignored on the native path; command[0] is the real host binary.
    const result = await runner.run("", echoCommand(token), {}, { timeout: 15_000 });

    assert.equal(result.runtime, "native", "ran on the native (host) runtime");
    assert.equal(result.exitCode, 0, `expected clean exit, got stderr: ${result.stderr}`);
    assert.ok(
      result.stdout.includes(token),
      `host stdout should echo the token; got: ${JSON.stringify(result.stdout)}`
    );
    assert.equal(result.killed, false);
  } finally {
    restore();
  }
});

test("LIVE: destructive guard blocks a real `rm -rf /` BEFORE spawning", async () => {
  enableNative();
  try {
    const providerMod = await importFresh("src/lib/skills/containerProvider.ts");
    providerMod._resetProviderCacheForTests();
    await providerMod.resolveProvider();

    const sandboxMod = await importFresh("src/lib/skills/sandbox.ts");
    const runner = sandboxMod.sandboxRunner;

    // This must never reach childProcess.spawn — the guard short-circuits it.
    const result = await runner.run("", ["rm", "-rf", "/"], {}, { timeout: 15_000 });

    assert.equal(result.exitCode, -1, "blocked command returns the sentinel exit code");
    assert.match(
      result.stderr,
      /Blocked by destructive-command guard \[recursive-root-delete\]/,
      `expected guard block message; got: ${result.stderr}`
    );
    assert.equal(result.stdout, "", "no stdout — the command never ran");
    assert.equal(result.killed, false);
  } finally {
    restore();
  }
});

test("LIVE: native run surfaces a nonzero exit code from the host process", async () => {
  enableNative();
  try {
    const providerMod = await importFresh("src/lib/skills/containerProvider.ts");
    providerMod._resetProviderCacheForTests();
    await providerMod.resolveProvider();

    const sandboxMod = await importFresh("src/lib/skills/sandbox.ts");
    const runner = sandboxMod.sandboxRunner;

    // `exit 3` on both shells — proves the real child exit code is threaded back.
    const cmd = process.platform === "win32" ? ["cmd", "/c", "exit 3"] : ["sh", "-c", "exit 3"];
    const result = await runner.run("", cmd, {}, { timeout: 15_000 });

    assert.equal(result.runtime, "native");
    assert.equal(result.exitCode, 3, "host process exit code is threaded through unchanged");
  } finally {
    restore();
  }
});
