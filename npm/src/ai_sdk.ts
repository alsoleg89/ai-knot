import type { RecallOptions } from "./types.js";

export interface AISDKMessageLike {
  role: string;
  content?: unknown;
  parts?: unknown;
  [key: string]: unknown;
}

export interface AiKnotAISDKMemoryOptions {
  topK?: number;
  now?: string;
  header?: string;
}

export interface AiKnotAISDKBuildSystemOptions extends AiKnotAISDKMemoryOptions {
  baseSystem?: string;
}

export interface AiKnotAISDKBuildMessagesOptions
  extends AiKnotAISDKBuildSystemOptions {
  query?: string;
  replaceLeadingSystem?: boolean;
}

type RecallClient = {
  recall(query: string, options?: RecallOptions): Promise<string>;
};

type SystemMessage = { role: "system"; content: string };

const DEFAULT_HEADER = "Relevant long-term memory (ai-knot):";

export class AiKnotAISDKMemory {
  private readonly recallClient: RecallClient;
  private readonly defaults: AiKnotAISDKMemoryOptions;

  constructor(
    recallClient: RecallClient,
    defaults: AiKnotAISDKMemoryOptions = {},
  ) {
    this.recallClient = recallClient;
    this.defaults = defaults;
  }

  /**
   * Build an AI SDK `system` string by combining a base instruction block with
   * recalled ai-knot facts for the current user input.
   */
  async buildSystem(
    input: string,
    options: AiKnotAISDKBuildSystemOptions = {},
  ): Promise<string | undefined> {
    const query = input.trim();
    const baseSystem = options.baseSystem?.trim();

    if (query === "") {
      return baseSystem || undefined;
    }

    const recall = await this.recallClient.recall(query, {
      topK: options.topK ?? this.defaults.topK,
      now: options.now ?? this.defaults.now,
    });

    if (!this.hasUsefulRecall(recall)) {
      return baseSystem || undefined;
    }

    const header = (options.header ?? this.defaults.header ?? DEFAULT_HEADER).trim();
    const parts: string[] = [];
    if (baseSystem) {
      parts.push(baseSystem);
    }
    parts.push(`${header}\n${recall.trim()}`);
    return parts.join("\n\n");
  }

  /**
   * Prepend a normalized AI SDK `system` message to an existing messages array.
   * The recall query defaults to the latest user text in the message list.
   */
  async buildMessages<T extends AISDKMessageLike>(
    messages: readonly T[],
    options: AiKnotAISDKBuildMessagesOptions = {},
  ): Promise<Array<T | SystemMessage>> {
    const replaceLeadingSystem = options.replaceLeadingSystem ?? true;
    let baseSystem = options.baseSystem?.trim();
    let body = [...messages];

    if (replaceLeadingSystem && body.length > 0 && body[0]?.role === "system") {
      const leadingSystem = this.extractText(body[0]);
      if (leadingSystem) {
        baseSystem = baseSystem
          ? `${baseSystem}\n\n${leadingSystem}`
          : leadingSystem;
        body = body.slice(1);
      }
    }

    const query =
      options.query?.trim() ||
      this.extractLatestUserText(body) ||
      this.extractLatestUserText(messages);

    if (!query) {
      return baseSystem
        ? [{ role: "system", content: baseSystem }, ...body]
        : body;
    }

    const system = await this.buildSystem(query, {
      baseSystem,
      topK: options.topK,
      now: options.now,
      header: options.header,
    });

    return system
      ? [{ role: "system", content: system }, ...body]
      : body;
  }

  private hasUsefulRecall(recall: string): boolean {
    const trimmed = recall.trim();
    return trimmed !== "" && !/^No relevant facts found\.?$/i.test(trimmed);
  }

  private extractLatestUserText(messages: readonly AISDKMessageLike[]): string | undefined {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message?.role !== "user") {
        continue;
      }
      const text = this.extractText(message);
      if (text) {
        return text;
      }
    }
    return undefined;
  }

  private extractText(message: AISDKMessageLike): string | undefined {
    return this.extractTextValue(message.content ?? message.parts);
  }

  private extractTextValue(value: unknown): string | undefined {
    if (typeof value === "string") {
      const trimmed = value.trim();
      return trimmed || undefined;
    }

    if (Array.isArray(value)) {
      const pieces = value
        .flatMap((item) => {
          if (typeof item === "string") {
            return [item];
          }
          if (item && typeof item === "object") {
            const record = item as Record<string, unknown>;
            if (typeof record.text === "string") {
              return [record.text];
            }
            if (typeof record.content === "string") {
              return [record.content];
            }
          }
          return [];
        })
        .map((piece) => piece.trim())
        .filter((piece) => piece.length > 0);

      return pieces.length > 0 ? pieces.join("\n") : undefined;
    }

    return undefined;
  }
}
