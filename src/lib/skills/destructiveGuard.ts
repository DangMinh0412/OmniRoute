/**
 * Destructive-command guard — the safety brake for host-level agent execution.
 *
 * When the operator opts into full-disk access + native (non-container) command
 * execution (see `builtins.ts` / `containerProvider.ts`, both env-gated OFF by
 * default), the sandbox's container isolation no longer contains the blast
 * radius. This module is the replacement guardrail: a pure, deterministic
 * classifier that BLOCKS commands whose obvious intent is to destroy the machine
 * or exfiltrate/wipe data, while allowing ordinary development commands through.
 *
 * Design principles:
 *   - Deny-by-pattern, not allow-by-list: we cannot enumerate every safe command,
 *     but we CAN enumerate the catastrophic ones (recursive root deletes, disk
 *     formatting, raw-device writes, fork bombs, privilege escalation, firmware).
 *   - Pure + synchronous → unit-testable with zero I/O, and cheap to call on
 *     every dispatch.
 *   - Conservative: a match blocks. False positives (a blocked safe command) are
 *     acceptable; a false negative (an allowed `rm -rf /`) is not. The user can
 *     always narrow the command.
 *   - This is defense-in-depth, NOT the only gate. Loopback enforcement (route
 *     guard) and the permission policy still sit in front of it. It is the LAST
 *     line specifically for the "native host exec" surface.
 *
 * IMPORTANT: this guard reduces risk, it does not eliminate it. A determined,
 * cleverly-obfuscated command can still slip a pattern matcher. The honest
 * security posture for native host exec is "loopback-only + permission-gated +
 * destructive-guarded", not "safe against an adversarial prompt". Operators
 * enabling native exec accept that residual risk (documented at the call site).
 */

/** The result of screening one command for destructive intent. */
export interface DestructiveVerdict {
  /** True when the command is blocked (matched a catastrophic pattern). */
  blocked: boolean;
  /** Human-readable reason, present only when blocked. */
  reason?: string;
  /** Short slug of the matched rule, for telemetry/tests. */
  rule?: string;
}

/**
 * Each rule: a slug, a human reason, and a matcher over the normalized,
 * whitespace-collapsed, lowercased full command line (`command + args`).
 * Kept as bounded, non-backtracking regexes / substring checks — no unbounded
 * quantifiers over untrusted input (ReDoS-safe).
 */
interface Rule {
  rule: string;
  reason: string;
  test: (normalized: string) => boolean;
}

/** Collapse a command + args into one normalized, lowercased line for matching. */
export function normalizeCommandLine(command: string, args: readonly string[]): string {
  return [command, ...args].join(" ").replace(/\s+/g, " ").trim().toLowerCase();
}

const RULES: Rule[] = [
  {
    rule: "recursive-root-delete",
    reason: "Recursive delete of a root or system path (rm -rf / , del /s of a drive root)",
    test: (s) => {
      // rm -rf / , rm -fr /, rm -r --force / , rm -rf /* , rm -rf ~ , rm -rf .
      if (/\brm\s+(-[a-z]*\s+)*-?[a-z]*(r|f){2,}[a-z]*\s+(\/|~|\/\*|\.\/?|\*)(\s|$)/.test(s)) {
        return true;
      }
      // rm -rf with r and f split across flags then a root-ish target
      if (
        /\brm\b/.test(s) &&
        /(^|\s)-[a-z]*r[a-z]*(\s|$)/.test(s) &&
        /(^|\s)-[a-z]*f[a-z]*(\s|$)/.test(s) &&
        /(\s)(\/|~|\/\*|c:\\?\*?|d:\\?\*?)(\s|$)/.test(s)
      ) {
        return true;
      }
      // Windows: del /s /q c:\  ,  rd /s /q c:\  ,  rmdir /s
      if (/\b(del|erase|rd|rmdir)\b/.test(s) && /\/s\b/.test(s) && /[a-z]:(\\|\s|$)/.test(s)) {
        return true;
      }
      // PowerShell Remove-Item -Recurse -Force against a drive/system root
      if (
        /remove-item\b/.test(s) &&
        /-recurse\b/.test(s) &&
        /(-force\b)/.test(s) &&
        /([a-z]:\\?(\s|$|\\\*)|\/)/.test(s)
      ) {
        return true;
      }
      return false;
    },
  },
  {
    rule: "disk-format",
    reason: "Formatting or partitioning a disk (format, mkfs, diskpart, fdisk)",
    test: (s) =>
      /\bmkfs(\.[a-z0-9]+)?\b/.test(s) ||
      /\bformat\s+[a-z]:/.test(s) ||
      /\bdiskpart\b/.test(s) ||
      /\b(fdisk|parted|gdisk)\b/.test(s) ||
      /\bnewfs\b/.test(s),
  },
  {
    rule: "raw-device-write",
    reason: "Writing directly to a raw disk device (dd of=/dev/..., > /dev/sd*)",
    test: (s) =>
      /\bdd\b[^\n]*\bof=\/dev\/(sd|nvme|hd|disk|rdisk|mmcblk)/.test(s) ||
      />\s*\/dev\/(sd|nvme|hd|disk|rdisk|mmcblk)/.test(s) ||
      /\bof=\\\\\.\\physicaldrive/.test(s),
  },
  {
    rule: "fork-bomb",
    reason: "Fork bomb / resource-exhaustion pattern",
    test: (s) =>
      s.includes(":(){:|:&};:") ||
      s.replace(/\s+/g, "").includes(":(){:|:&};:") ||
      /\bwhile\s+true\s*;\s*do\b.*\bfork\b/.test(s),
  },
  {
    rule: "privilege-escalation-pipe",
    reason: "Piping a network download straight into a shell (curl|sh) — remote code execution",
    test: (s) =>
      /\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh|python|node|pwsh|powershell)\b/.test(s) ||
      /\biwr\b[^|]*\|\s*iex\b/.test(s) ||
      /\binvoke-webrequest\b[^|]*\|\s*invoke-expression\b/.test(s),
  },
  {
    rule: "firmware-mbr-write",
    reason: "Overwriting boot sector / firmware (dd to MBR, efibootmgr delete, nvram)",
    test: (s) =>
      /\bof=\/dev\/(sda|nvme0n1|disk0)\b/.test(s) ||
      /\bbcdedit\b.*\/delete/.test(s) ||
      /\befibootmgr\b.*(-b|-B)\b/.test(s),
  },
  {
    rule: "mass-permission-change",
    reason: "Recursive chmod/chown/icacls against a filesystem or drive root",
    test: (s) =>
      /\bchmod\s+(-[a-z]*r[a-z]*\s+)*(-r\b|--recursive\b)?\s*[0-7]{3,4}\s+\/(\s|$)/.test(s) ||
      /\bchown\s+.*(-r\b|--recursive\b).*\s+\/(\s|$)/.test(s) ||
      /\bicacls\s+[a-z]:\\?\s+\/grant.*\/t\b/.test(s),
  },
  {
    rule: "shutdown-halt",
    reason: "Powering off / halting the machine",
    test: (s) =>
      /\b(shutdown|halt|poweroff|reboot)\b/.test(s) ||
      /\binit\s+0\b/.test(s) ||
      /\bstop-computer\b/.test(s),
  },
  {
    rule: "wipe-overwrite-tool",
    reason: "Secure-wipe / shred utility against files or devices",
    test: (s) => /\b(shred|wipe|scrub|sdelete|cipher\s+\/w)\b/.test(s),
  },
];

/**
 * Screen a command (and its args) for catastrophic/destructive intent.
 *
 * Returns `{ blocked: true, reason, rule }` when any rule matches, else
 * `{ blocked: false }`. Pure — safe to call on every native dispatch.
 *
 * This is intentionally conservative for the NATIVE host-execution path only;
 * container execution (the default) does not need it because the container
 * already contains the blast radius.
 */
export function screenCommand(command: string, args: readonly string[] = []): DestructiveVerdict {
  if (typeof command !== "string" || !command.trim()) {
    return { blocked: true, reason: "Empty command", rule: "empty" };
  }
  const normalized = normalizeCommandLine(command, args);
  for (const rule of RULES) {
    if (rule.test(normalized)) {
      return { blocked: true, reason: rule.reason, rule: rule.rule };
    }
  }
  return { blocked: false };
}

/** Exposed for tests + telemetry: the list of rule slugs this guard enforces. */
export const DESTRUCTIVE_RULE_SLUGS: readonly string[] = RULES.map((r) => r.rule);
