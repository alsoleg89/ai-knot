import { generateText } from "ai";
import type { LanguageModelV1 } from "ai";

export type Verdict = "CORRECT" | "WRONG";

// Strip null bytes and other control characters that break JSON serialization
function sanitize(s: string): string {
  // eslint-disable-next-line no-control-regex
  return s.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
}

// Retry with exponential backoff for rate limit errors
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
      process.stderr.write(`  [rate limit] waiting ${(wait / 1000).toFixed(1)}s (attempt ${attempt}/${maxAttempts})\n`);
      await new Promise((r) => setTimeout(r, wait));
      delay = Math.min(delay * 2, 60000);
    }
  }
  throw new Error("unreachable");
}

// ---- Answer generation ------------------------------------------------------

const ANSWER_SYSTEM = `Answer the question based on the memory context below. Answer concisely.`;

export interface Usage {
  promptTokens: number;
  completionTokens: number;
}

export interface AnswerResult {
  text: string;
  usage: Usage;
}

export async function answerQuestion(
  model: LanguageModelV1,
  context: string,
  question: string
): Promise<AnswerResult> {
  const { text, usage } = await withRetry(() => generateText({
    model,
    system: ANSWER_SYSTEM,
    messages: [
      {
        role: "user",
        content: `Context:\n${sanitize(context)}\n\nQuestion: ${sanitize(question)}`,
      },
    ],
    maxTokens: 256,
    temperature: 0,
  }));
  return {
    text: text.trim(),
    usage: { promptTokens: usage.promptTokens, completionTokens: usage.completionTokens },
  };
}

// ---- LLM judge --------------------------------------------------------------

const JUDGE_SYSTEM = `You are an evaluation judge. Given a question, a candidate answer, and the gold answer,
decide whether the candidate answer is correct.

Return JSON exactly like: {"verdict": "CORRECT"} or {"verdict": "WRONG"}

Rules:
- CORRECT if the candidate answer conveys the same essential information as the gold answer.
- Exact wording is not required; semantic equivalence is sufficient.
- WRONG if the candidate answer is missing key facts, contradicts the gold, or is irrelevant.
- Do not output anything other than the JSON object.`;

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
  const { text, usage } = await withRetry(() => generateText({
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
  }));
  return {
    verdict: parseVerdict(text),
    usage: { promptTokens: usage.promptTokens, completionTokens: usage.completionTokens },
  };
}

/**
 * Parse a CORRECT/WRONG verdict from the raw LLM output.
 * Tries JSON first, then regex fallback, then defaults to WRONG.
 */
export function parseVerdict(raw: string): Verdict {
  const trimmed = raw.trim();

  // JSON path
  try {
    const parsed = JSON.parse(trimmed) as { verdict?: string };
    if (parsed.verdict === "CORRECT") return "CORRECT";
    if (parsed.verdict === "WRONG") return "WRONG";
  } catch {
    // fall through
  }

  // Regex fallback
  if (/\bCORRECT\b/i.test(trimmed)) return "CORRECT";
  if (/\bWRONG\b/i.test(trimmed)) return "WRONG";

  // Default — unknown output counts as wrong
  return "WRONG";
}
