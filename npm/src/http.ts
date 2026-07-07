import type {
  AddOptions,
  Fact,
  HttpKnowledgeBaseListOptions,
  HttpKnowledgeBaseOptions,
  LearnMessage,
  LearnOptions,
  LearnResult,
  LineageFact,
  LineageOptions,
  RecallOptions,
  ResolvedFact,
  ResolvedResult,
  Stats,
} from "./types.js";

type HttpFetch = NonNullable<HttpKnowledgeBaseOptions["fetch"]>;

type RecallResponse = {
  context: string;
  facts: Fact[];
};

type ListFactsResponse = {
  facts: Fact[];
  returned: number;
  total_matching: number;
  include_inactive: boolean;
  now: string;
};

type HttpHealth = {
  status: string;
  version: string;
};

function normalizeBaseUrl(baseUrl: string): string {
  const trimmed = baseUrl.trim();
  if (trimmed === "") {
    throw new Error("HttpKnowledgeBase requires a non-empty baseUrl.");
  }
  return trimmed.replace(/\/+$/, "");
}

function resolveFetch(fetchImpl: HttpKnowledgeBaseOptions["fetch"]): HttpFetch {
  if (fetchImpl) {
    return fetchImpl;
  }
  if (typeof globalThis.fetch === "function") {
    return (url, init) =>
      globalThis.fetch(url, {
        method: init?.method,
        headers: init?.headers,
        body: init?.body,
      }) as ReturnType<HttpFetch>;
  }
  throw new Error(
    "HttpKnowledgeBase requires fetch. Use Node.js 18+ or pass a custom fetch implementation.",
  );
}

function buildQuery(params: Record<string, string | number | boolean | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined) continue;
    search.set(key, String(value));
  }
  const serialized = search.toString();
  return serialized ? `?${serialized}` : "";
}

export class HttpKnowledgeBase {
  private readonly baseUrl: string;
  private readonly token?: string;
  private readonly extraHeaders: Record<string, string>;
  private readonly fetchImpl: HttpFetch;

  constructor(options: HttpKnowledgeBaseOptions) {
    this.baseUrl = normalizeBaseUrl(options.baseUrl);
    this.token = options.token;
    this.extraHeaders = { ...(options.headers ?? {}) };
    this.fetchImpl = resolveFetch(options.fetch);
  }

  async add(content: string, options: AddOptions = {}): Promise<Fact> {
    return this.requestJson<Fact>("/v1/facts", {
      method: "POST",
      body: {
        content,
        type: options.type,
        importance: options.importance,
        tags: options.tags,
        event_time: options.eventTime,
      },
    });
  }

  async recall(query: string, options: RecallOptions = {}): Promise<string> {
    const response = await this.requestJson<RecallResponse>("/v1/recall", {
      method: "POST",
      body: {
        query,
        top_k: options.topK,
        now: options.now,
      },
    });
    return response.context;
  }

  async search(query: string, options: RecallOptions = {}): Promise<string> {
    const response = await this.requestJson<RecallResponse>("/v1/search", {
      method: "POST",
      body: {
        query,
        top_k: options.topK,
        now: options.now,
      },
    });
    return response.context;
  }

  async learn(messages: LearnMessage[], options: LearnOptions = {}): Promise<LearnResult> {
    return this.requestJson<LearnResult>("/v1/learn", {
      method: "POST",
      body: {
        messages,
        provider: options.provider,
        api_key: options.apiKey,
        model: options.model,
        event_time: options.eventTime,
      },
    });
  }

  async addResolved(facts: ResolvedFact[]): Promise<ResolvedResult[]> {
    const wire = facts.map((fact) => {
      const payload: Record<string, unknown> = { content: fact.content };
      if (fact.entity !== undefined) payload["entity"] = fact.entity;
      if (fact.attribute !== undefined) payload["attribute"] = fact.attribute;
      if (fact.valueText !== undefined) payload["value_text"] = fact.valueText;
      if (fact.slotKey !== undefined) payload["slot_key"] = fact.slotKey;
      if (fact.op !== undefined) payload["op"] = fact.op;
      if (fact.eventTime !== undefined) payload["event_time"] = fact.eventTime;
      return payload;
    });
    return this.requestJson<ResolvedResult[]>("/v1/facts/resolved", {
      method: "POST",
      body: { facts: wire },
    });
  }

  async listFacts(options: HttpKnowledgeBaseListOptions = {}): Promise<Fact[]> {
    const response = await this.requestJson<ListFactsResponse>(
      `/v1/facts${buildQuery({
        include_inactive: options.includeInactive,
        limit: options.limit,
        now: options.now,
      })}`,
      { method: "GET" },
    );
    return response.facts;
  }

  async list(options: HttpKnowledgeBaseListOptions = {}): Promise<Fact[]> {
    return this.listFacts(options);
  }

  async get(factId: string): Promise<Fact> {
    return this.requestJson<Fact>(`/v1/facts/${encodeURIComponent(factId)}`, {
      method: "GET",
    });
  }

  async lineage(factId: string, options: LineageOptions = {}): Promise<LineageFact[]> {
    return this.requestJson<LineageFact[]>(
      `/v1/facts/${encodeURIComponent(factId)}/lineage${buildQuery({ now: options.now })}`,
      { method: "GET" },
    );
  }

  async forget(factId: string): Promise<void> {
    await this.requestVoid(`/v1/facts/${encodeURIComponent(factId)}`, { method: "DELETE" });
  }

  async delete(factId: string): Promise<void> {
    await this.forget(factId);
  }

  async stats(): Promise<Stats> {
    return this.requestJson<Stats>("/v1/stats", { method: "GET" });
  }

  async health(): Promise<HttpHealth> {
    return this.requestJson<HttpHealth>("/health", {
      method: "GET",
      auth: false,
    });
  }

  async close(): Promise<void> {
    return Promise.resolve();
  }

  private async requestJson<T>(
    path: string,
    options: {
      method: "GET" | "POST";
      body?: Record<string, unknown>;
      auth?: boolean;
    },
  ): Promise<T> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method: options.method,
      headers: this.buildHeaders(options.body !== undefined, options.auth !== false),
      body: options.body !== undefined ? JSON.stringify(stripUndefined(options.body)) : undefined,
    });
    if (!response.ok) {
      throw await this.buildHttpError(response);
    }
    return (await response.json()) as T;
  }

  private async requestVoid(
    path: string,
    options: {
      method: "DELETE";
      auth?: boolean;
    },
  ): Promise<void> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method: options.method,
      headers: this.buildHeaders(false, options.auth !== false),
    });
    if (!response.ok) {
      throw await this.buildHttpError(response);
    }
  }

  private buildHeaders(includeJson: boolean, includeAuth: boolean): Record<string, string> {
    const headers: Record<string, string> = { ...this.extraHeaders };
    if (includeJson) {
      headers["Content-Type"] = "application/json";
    }
    if (includeAuth && this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }
    return headers;
  }

  private async buildHttpError(response: {
    status: number;
    statusText: string;
    json(): Promise<unknown>;
    text(): Promise<string>;
  }): Promise<Error> {
    let detail = "";
    try {
      const payload = await response.json();
      if (payload && typeof payload === "object" && "detail" in payload) {
        const candidate = (payload as Record<string, unknown>).detail;
        detail = typeof candidate === "string" ? candidate : JSON.stringify(candidate);
      } else {
        detail = JSON.stringify(payload);
      }
    } catch {
      detail = await response.text();
    }

    const suffix = detail ? `: ${detail}` : "";
    return new Error(`ai-knot HTTP ${response.status} ${response.statusText}${suffix}`);
  }
}

function stripUndefined(value: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(value).filter(([, entry]) => entry !== undefined));
}
