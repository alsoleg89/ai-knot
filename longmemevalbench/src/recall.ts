import type { LmeQuestion } from "./loader.js";

/**
 * Memory-recall scoring (P7) — turn-level and session-level.
 *
 * LongMemEval reports two recall numbers alongside QA accuracy:
 *   - turn-level    : did a turn flagged ``has_answer: true`` reach the recall pool?
 *   - session-level : did a session in ``answer_session_ids`` reach the recall pool?
 *
 * The official scorer matches retrieved *unit ids* against the evidence ids. Our
 * harness recall surfaces formatted TEXT (the core returns prompt-ready strings,
 * not fact→session links), so we approximate the official id-match with a
 * content-overlap check: an evidence turn/session "made the pool" if a sufficient
 * fraction of its tokens appears in the recalled context. This is a documented
 * approximation — see longmemevalbench/README.md ("Recall scoring") — and is
 * intentionally conservative (it can undercount, never leaks the gold answer).
 *
 * Abstention (``_abs``) questions have no evidence location and are EXCLUDED from
 * recall scoring (matching the official protocol — 30 instances skipped).
 */

export interface RecallScore {
  turnHit: boolean | null; // null => not scored (abstention or no evidence turns)
  sessionHit: boolean | null;
}

const STOP = new Set([
  "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of",
  "in", "on", "at", "for", "with", "i", "you", "it", "that", "this", "my", "me",
  "we", "they", "he", "she", "do", "did", "have", "has", "had", "be", "been",
]);

function contentTokens(s: string): Set<string> {
  const toks = s
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2 && !STOP.has(t));
  return new Set(toks);
}

/** Fraction of evidence tokens present in the context token set. */
function overlap(evidence: string, contextTokens: Set<string>): number {
  const evTokens = contentTokens(evidence);
  if (evTokens.size === 0) return 0;
  let hit = 0;
  for (const t of evTokens) if (contextTokens.has(t)) hit++;
  return hit / evTokens.size;
}

/** Default: an evidence unit "made the pool" at >= 60% content-token overlap. */
const DEFAULT_THRESHOLD = 0.6;

export function scoreRecall(
  q: LmeQuestion,
  recalledContext: string,
  threshold = DEFAULT_THRESHOLD
): RecallScore {
  if (q.isAbstention) {
    return { turnHit: null, sessionHit: null };
  }

  const ctxTokens = contentTokens(recalledContext);

  // Turn-level: any has_answer turn that overlaps the context.
  const answerTurns = q.sessions.flatMap((s) => s.turns.filter((t) => t.hasAnswer));
  const turnHit =
    answerTurns.length === 0
      ? null
      : answerTurns.some((t) => overlap(t.content, ctxTokens) >= threshold);

  // Session-level: any answer_session that overlaps the context.
  const answerSessions = q.sessions.filter((s) => q.answerSessionIds.includes(s.id));
  const sessionHit =
    answerSessions.length === 0
      ? null
      : answerSessions.some((s) => {
          const text = s.turns.map((t) => t.content).join(" ");
          return overlap(text, ctxTokens) >= threshold;
        });

  return { turnHit, sessionHit };
}
