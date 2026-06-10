// Config is loaded from process.env.
// For Node/tsx, export vars before running (or use a .env loader).

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

  // LongMemEval's official judge is gpt-4o; the answerer mirrors LOCOMO (gpt-4.1-nano).
  const defaultJudge = process.env["DEFAULT_JUDGE_MODEL"] ?? "gpt-4o";
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
