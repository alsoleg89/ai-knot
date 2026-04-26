// Config is loaded from process.env.
// When running with Bun, .env is loaded automatically.
// For Node/tsx, use dotenv or export vars before running.

export interface Config {
  openaiApiKey: string | undefined;
  anthropicApiKey: string | undefined;
  ollamaBaseUrl: string;
  aiKnotCommand: string;
}

export function loadConfig(): Config {
  const openaiApiKey = process.env["OPENAI_API_KEY"];
  const ollamaBaseUrl =
    process.env["OLLAMA_BASE_URL"] ?? "http://localhost:11434/v1";

  // Require at least one LLM backend
  const defaultJudge =
    process.env["DEFAULT_JUDGE_MODEL"] ?? "ollama:qwen2.5:7b";
  const needsOpenAI =
    defaultJudge.startsWith("gpt-") ||
    (process.env["DEFAULT_ANSWER_MODEL"] ?? "").startsWith("gpt-");

  if (needsOpenAI && !openaiApiKey) {
    throw new Error(
      "OPENAI_API_KEY is required when using gpt-* models. " +
        "Set DEFAULT_JUDGE_MODEL and DEFAULT_ANSWER_MODEL to ollama:* to use Ollama instead."
    );
  }

  return {
    openaiApiKey,
    anthropicApiKey: process.env["ANTHROPIC_API_KEY"],
    ollamaBaseUrl,
    aiKnotCommand: process.env["AI_KNOT_COMMAND"] ?? "ai-knot-mcp",
  };
}
