/**
 * CouncilTranscript — the live deliberation transcript.
 *
 * Renders each debate round (round 0 = initial answers, rounds ≥1 = rebuttals)
 * with its per-model panel answers, and a consensus marker when the debate
 * converged and stopped early. Pure presentational.
 */
"use client";

import { useTranslations } from "next-intl";
import type { CouncilRound } from "../useCouncilStream";

export type CouncilTranscriptProps = {
  rounds: CouncilRound[];
};

export function CouncilTranscript({ rounds }: CouncilTranscriptProps) {
  const t = useTranslations("council");
  if (rounds.length === 0) return null;

  return (
    <div className="space-y-4">
      {rounds.map((round) => (
        <section
          key={round.round}
          className="rounded-lg border border-border bg-surface p-4"
          aria-label={
            round.round === 0
              ? t("initialRoundHeading")
              : t("rebuttalRoundHeading", { round: round.round })
          }
        >
          <header className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-text-main">
              {round.round === 0
                ? t("initialRoundHeading")
                : t("rebuttalRoundHeading", { round: round.round })}
            </h2>
            {typeof round.consensusScore === "number" && (
              <span className="rounded-full bg-green-500/15 px-2.5 py-0.5 text-xs font-medium text-green-400">
                {t("consensusReached", { score: round.consensusScore.toFixed(3) })}
              </span>
            )}
          </header>

          <div className="space-y-3">
            {round.answers.map((answer, i) => (
              <article
                key={`${answer.model}-${i}`}
                className="rounded-md border border-border/60 bg-background p-3"
              >
                <p className="mb-1 text-xs font-medium text-text-muted">
                  {t("panelAnswerFrom", { model: answer.model })}
                </p>
                <p className="whitespace-pre-wrap text-sm text-text-main">{answer.text}</p>
              </article>
            ))}
          </div>

          {round.toolActivity && round.toolActivity.length > 0 && (
            <div className="mt-3 border-t border-border/60 pt-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
                {t("toolActivityHeading")}
              </p>
              <ul className="space-y-1">
                {round.toolActivity.map((entry, i) => (
                  <li
                    key={`${entry.model}-${entry.name}-${entry.iteration}-${entry.kind}-${i}`}
                    className="flex items-center gap-2 text-xs text-text-muted"
                  >
                    <span
                      aria-hidden="true"
                      className={
                        entry.kind === "denied"
                          ? "text-red-400"
                          : entry.kind === "result"
                            ? entry.ok
                              ? "text-green-400"
                              : "text-amber-400"
                            : "text-text-muted"
                      }
                    >
                      {entry.kind === "denied"
                        ? "⛔"
                        : entry.kind === "result"
                          ? entry.ok
                            ? "✓"
                            : "✕"
                          : "→"}
                    </span>
                    <span className="font-mono">
                      {entry.kind === "call"
                        ? t("toolCallLabel", { model: entry.model, tool: entry.name })
                        : entry.kind === "denied"
                          ? t("toolDeniedLabel", {
                              model: entry.model,
                              tool: entry.name,
                              reason: entry.reason ?? "",
                            })
                          : entry.ok
                            ? t("toolResultOk", { model: entry.model, tool: entry.name })
                            : t("toolResultFail", { model: entry.model, tool: entry.name })}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      ))}
    </div>
  );
}
