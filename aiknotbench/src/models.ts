import { createOpenAI, openai } from "@ai-sdk/openai";
import { anthropic } from "@ai-sdk/anthropic";
import type { LanguageModelV1 } from "ai";

/**
 * Default models — override via env vars DEFAULT_JUDGE_MODEL / DEFAULT_ANSWER_MODEL.
 * Use "ollama:<model>" syntax for local Ollama, e.g. "ollama:qwen2.5:7b".
 */
export const DEFAULT_JUDGE_MODEL =
  process.env["DEFAULT_JUDGE_MODEL"] ?? "ollama:qwen2.5:7b";
export const DEFAULT_ANSWER_MODEL =
  process.env["DEFAULT_ANSWER_MODEL"] ?? "ollama:qwen2.5:7b";

/**
 * Resolve a model alias string to an AI SDK LanguageModelV1.
 *
 * Supported prefixes:
 *   ollama:<name>  → Ollama via OpenAI-compat endpoint (OLLAMA_BASE_URL, default: localhost:11434)
 *   claude-*       → @ai-sdk/anthropic (requires ANTHROPIC_API_KEY)
 *   everything else → @ai-sdk/openai (requires OPENAI_API_KEY)
 */
export function resolveModel(modelId: string): LanguageModelV1 {
  if (modelId.startsWith("ollama:")) {
    const name = modelId.slice("ollama:".length);
    const baseURL =
      process.env["OLLAMA_BASE_URL"] ?? "http://localhost:11434/v1";
    const provider = createOpenAI({ baseURL, apiKey: "ollama" });
    return provider(name) as LanguageModelV1;
  }
  if (modelId.startsWith("claude-")) {
    return anthropic(modelId) as LanguageModelV1;
  }
  return openai(modelId) as LanguageModelV1;
}
