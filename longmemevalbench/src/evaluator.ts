import { generateText } from "ai";
import type { LanguageModelV1 } from "ai";

export type Verdict = "CORRECT" | "WRONG";

// Strip control chars that break JSON serialization.
function sanitize(s: string): string {
  // eslint-disable-next-line no-control-regex
  return s.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
}

// Retry with exponential backoff for rate-limit errors.
async function withRetry<T>(fn: () => Promise<T>, maxAttempts = 6): Promise<T> {
  let delay = 5000;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const msg = String(err);
      const isRateLimit = msg.includes("Rate limit") || msg.includes("429");
      if (!isRateLimit || attempt === maxAttempts) throw err;
      const wait = delay + Math.random() * 1000;
      process.stderr.write(
        `  [rate limit] waiting ${(wait / 1000).toFixed(1)}s (attempt ${attempt}/${maxAttempts})\n`
      );
      await new Promise((r) => setTimeout(r, wait));
      delay = Math.min(delay * 2, 60000);
    }
  }
  throw new Error("unreachable");
}

// ---- The deterministic "I don't know" sentinel ------------------------------

export const IDK = "I don't know.";

/**
 * Phrases that indicate a recall pool carried nothing usable. When the memory
 * surface is empty we short-circuit to a deterministic IDK WITHOUT an LLM call —
 * this is the cheap, generic abstention path (no benchmark-specific text).
 */
function isEmptyContext(context: string): boolean {
  const t = context.trim().toLowerCase();
  return t === "" || t === "no relevant facts found." || t === "no relevant facts found";
}

// ---- Answer generation (with opt-in abstention contract) --------------------

/**
 * Reader system prompt WITHOUT the abstention contract — identical in spirit to
 * the LOCOMO default ("answer concisely"). Used when ``idkContract`` is off so
 * the LongMemEval harness can A/B the contract's effect.
 */
const ANSWER_SYSTEM_PLAIN = `Answer the question based on the memory context below. Answer concisely.`;

/**
 * Reader system prompt WITH the abstention contract (prerequisite C). This is a
 * GENERIC instruction — "if the memory does not support an answer, say you don't
 * know" — not a benchmark-tailored rule. It is the standard contract any
 * grounded-QA reader needs to avoid confabulating on unanswerable / false-premise
 * questions. Enabled by default for LongMemEval (the LOCOMO reader is untouched).
 */
const ANSWER_SYSTEM_IDK = `Answer the question using ONLY the memory context below.
- If the context does not contain enough information to answer, reply exactly: "${IDK}"
- If the question presupposes a fact that the context does not support (a false premise), reply exactly: "${IDK}"
- When the context shows a fact changed over time, use the MOST RECENT value.
- Answer concisely. Do not invent details that are not in the context.`;

export interface Usage {
  promptTokens: number;
  completionTokens: number;
}

export interface AnswerResult {
  text: string;
  usage: Usage;
  /** True if the answer was produced by the deterministic empty-pool short-circuit. */
  shortCircuited: boolean;
}

export interface AnswerOptions {
  /** Enable the generic IDK / abstention reader contract (default true for LME). */
  idkContract?: boolean;
}

export async function answerQuestion(
  model: LanguageModelV1,
  context: string,
  question: string,
  options: AnswerOptions = {}
): Promise<AnswerResult> {
  const idkContract = options.idkContract ?? true;

  // (a) Empty-pool short-circuit: no memory surfaced → deterministic IDK, no LLM.
  if (idkContract && isEmptyContext(context)) {
    return { text: IDK, usage: { promptTokens: 0, completionTokens: 0 }, shortCircuited: true };
  }

  const system = idkContract ? ANSWER_SYSTEM_IDK : ANSWER_SYSTEM_PLAIN;
  const { text, usage } = await withRetry(() =>
    generateText({
      model,
      system,
      messages: [
        {
          role: "user",
          content: `Context:\n${sanitize(context)}\n\nQuestion: ${sanitize(question)}`,
        },
      ],
      maxTokens: 256,
      temperature: 0,
    })
  );
  return {
    text: text.trim(),
    usage: { promptTokens: usage.promptTokens, completionTokens: usage.completionTokens },
    shortCircuited: false,
  };
}

// ---- Abstention detection ---------------------------------------------------

/**
 * Detect whether a reader answer is a refusal / "I don't know". Used to score the
 * ``_abs`` (false-premise) questions: an abstention question is CORRECT iff the
 * reader declined. Deterministic + generic (no gold-answer leakage).
 */
export function isAbstentionAnswer(answer: string): boolean {
  const t = answer.trim().toLowerCase().replace(/[.!]+$/g, "");
  if (t === "") return true;
  const patterns = [
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "no information",
    "not enough information",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not mentioned",
    "not specified",
    "there is no",
    "no mention of",
    "the context does not",
    "context doesn't",
  ];
  return patterns.some((p) => t.includes(p));
}

// ---- LLM judge (LongMemEval gpt-4o judge with per-type rules) ----------------

const JUDGE_SYSTEM = `You are an evaluation judge for a long-term-memory QA benchmark.
Given the question, a candidate answer, and the gold answer, decide whether the
candidate is correct.

Return JSON exactly like: {"verdict": "CORRECT"} or {"verdict": "WRONG"}

Rules:
- CORRECT if the candidate conveys the same essential information as the gold answer.
- Exact wording is not required; semantic equivalence is sufficient.
- For knowledge-update questions, the candidate must reflect the LATEST value; an
  outdated value is WRONG.
- WRONG if the candidate is missing key facts, contradicts the gold, or is irrelevant.
- Output nothing other than the JSON object.`;

export interface JudgeResult {
  verdict: Verdict;
  usage: Usage;
}

export async function judgeAnswer(
  model: LanguageModelV1,
  question: string,
  candidateAnswer: string,
  goldAnswer: string
): Promise<JudgeResult> {
  const { text, usage } = await withRetry(() =>
    generateText({
      model,
      system: JUDGE_SYSTEM,
      messages: [
        {
          role: "user",
          content: `Question: ${sanitize(question)}\nCandidate answer: ${sanitize(candidateAnswer)}\nGold answer: ${sanitize(goldAnswer)}`,
        },
      ],
      maxTokens: 32,
      temperature: 0,
    })
  );
  return {
    verdict: parseVerdict(text),
    usage: { promptTokens: usage.promptTokens, completionTokens: usage.completionTokens },
  };
}

/**
 * Parse a CORRECT/WRONG verdict from raw LLM output.
 * Tries JSON first, then a regex fallback, then defaults to WRONG.
 */
export function parseVerdict(raw: string): Verdict {
  const trimmed = raw.trim();
  try {
    const parsed = JSON.parse(trimmed) as { verdict?: string };
    if (parsed.verdict === "CORRECT") return "CORRECT";
    if (parsed.verdict === "WRONG") return "WRONG";
  } catch {
    // fall through
  }
  if (/\bCORRECT\b/i.test(trimmed)) return "CORRECT";
  if (/\bWRONG\b/i.test(trimmed)) return "WRONG";
  return "WRONG";
}
