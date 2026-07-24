/**
 * Semantic consensus for the AI Council — the world-class upgrade over lexical
 * Jaccard (research: "Should we be going MAD?", arXiv 2311.17371, names
 * lexical-overlap consensus a weak proxy that can reward conformity).
 *
 * It embeds each surviving panel answer with OmniRoute's own embedding stack
 * (`src/lib/memory/embedding/embed()` → whichever provider the operator has
 * configured: remote API, static Potion, or local Transformers) and returns the
 * mean pairwise cosine similarity of those vectors — so two answers that AGREE
 * IN MEANING but differ in wording score high, and two that share vocabulary but
 * contradict score low.
 *
 * Zero-regression fallback: if no embedding source is available, any embed call
 * fails, or the panel is degenerate, it transparently falls back to the pure
 * lexical `measureConsensus()` from `debate.ts`. The consensus-stop threshold is
 * unchanged, so an operator with no embedding provider sees exactly the prior
 * behaviour.
 *
 * Why here (not in `open-sse/services/debate.ts`): `embed()` lives under
 * `src/lib/memory/` (`@/lib/...`) and `debate.ts` is in the `open-sse/`
 * workspace, which must NOT import back into `src/` (that inverts the dependency
 * direction). The pure vector math (`measureConsensusFromVectors`,
 * `cosineSimilarity`) stays in `debate.ts` and is imported here; only the impure
 * embedding I/O lives on this side.
 */
import {
  measureConsensus,
  measureConsensusFromVectors,
  type PanelAnswer,
} from "@omniroute/open-sse/services/debate.ts";
import { embed } from "@/lib/memory/embedding";
import { getMemorySettings } from "@/lib/memory/settings";
import type { MemorySettingsExtended } from "@/shared/schemas/memory";

export type ConsensusMethod = "semantic" | "lexical";

export interface ConsensusResult {
  /** Mean pairwise agreement in [0,1] — compared against `consensusThreshold`. */
  score: number;
  /** Which path produced the score, for telemetry + SSE transparency. */
  method: ConsensusMethod;
}

/**
 * Measure how much a round's answers converged, preferring semantic (embedding
 * cosine) similarity and falling back to lexical Jaccard. Never throws — every
 * failure path degrades to the lexical score so consensus-stop keeps working.
 *
 * @param answers  the round's surviving panel answers
 * @param settings optional pre-loaded memory settings (avoids a redundant DB
 *                 read when the caller already has them); loaded on demand
 *                 otherwise. Pass `null` to force a load.
 */
export async function measureSemanticConsensus(
  answers: PanelAnswer[],
  settings?: MemorySettingsExtended | null
): Promise<ConsensusResult> {
  // Degenerate panel — identical contract to the pure helpers (nothing to
  // disagree about). Skip the embedding round-trip entirely.
  if (answers.length < 2) {
    return { score: 1, method: "lexical" };
  }

  const lexical = (): ConsensusResult => ({
    score: measureConsensus(answers),
    method: "lexical",
  });

  let resolved: MemorySettingsExtended;
  try {
    resolved = settings ?? (await getMemorySettings());
  } catch {
    return lexical();
  }

  try {
    const results = await Promise.all(answers.map((a) => embed(a.text, resolved)));
    const vectors: number[][] = [];
    for (const r of results) {
      // embed() returns EmbeddingResult | EmbeddingError; only the success shape
      // carries `vector`. A single failed/empty embedding aborts to lexical
      // rather than scoring a partial panel (which would bias the mean).
      if (!("vector" in r) || !Array.isArray(r.vector) || r.vector.length === 0) {
        return lexical();
      }
      vectors.push(r.vector);
    }
    return { score: measureConsensusFromVectors(vectors), method: "semantic" };
  } catch {
    return lexical();
  }
}
