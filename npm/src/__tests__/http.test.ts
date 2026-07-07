import { beforeEach, describe, expect, it, vi } from "vitest";

import { HttpKnowledgeBase } from "../index.js";

type FakeResponse = {
  ok: boolean;
  status: number;
  statusText: string;
  json: () => Promise<unknown>;
  text: () => Promise<string>;
};

type FetchCall = {
  url: string;
  init: {
    method?: string;
    headers?: Record<string, string>;
    body?: string;
  } | undefined;
};

const FACT = {
  id: "abcd1234",
  content: "User prefers TypeScript",
  type: "semantic",
  importance: 0.8,
  retention_score: 1,
  access_count: 0,
  tags: ["prefs"],
  created_at: "2026-01-01T00:00:00.000Z",
  last_accessed: "2026-01-01T00:00:00.000Z",
};

function okJson(value: unknown, status = 200): FakeResponse {
  return {
    ok: true,
    status,
    statusText: status === 201 ? "Created" : "OK",
    json: async () => value,
    text: async () => JSON.stringify(value),
  };
}

function errorJson(status: number, detail: string): FakeResponse {
  return {
    ok: false,
    status,
    statusText: "Bad Request",
    json: async () => ({ detail }),
    text: async () => detail,
  };
}

describe("HttpKnowledgeBase", () => {
  let calls: FetchCall[];

  beforeEach(() => {
    calls = [];
  });

  it("uses the HTTP sidecar for add/get/search/list/delete/stats", async () => {
    const fetch = vi.fn(async (url: string, init) => {
      calls.push({ url, init });
      if (url.endsWith("/v1/facts") && init?.method === "POST") {
        return okJson(FACT, 201);
      }
      if (url.endsWith("/v1/facts/abcd1234") && init?.method === "GET") {
        return okJson({ ...FACT, active: true });
      }
      if (
        url.endsWith("/v1/facts/abcd1234/lineage?now=2026-01-02T00%3A00%3A00%2B00%3A00") &&
        init?.method === "GET"
      ) {
        return okJson([
          {
            ...FACT,
            active: true,
            slot_key: "user::preference",
            entity: "user",
            attribute: "preference",
            value_text: "TypeScript",
            version: 1,
            supersedes_id: "deadbeef",
            published_by: null,
          },
        ]);
      }
      if (url.endsWith("/v1/search") && init?.method === "POST") {
        return okJson({
          context: "[1] User prefers TypeScript",
          facts: [FACT],
        });
      }
      if (url.includes("/v1/facts?") && init?.method === "GET") {
        return okJson({
          facts: [FACT],
          returned: 1,
          total_matching: 1,
          include_inactive: false,
          now: "2026-01-01T00:00:00.000Z",
        });
      }
      if (url.endsWith("/v1/facts/abcd1234") && init?.method === "DELETE") {
        return {
          ok: true,
          status: 204,
          statusText: "No Content",
          json: async () => null,
          text: async () => "",
        };
      }
      if (url.endsWith("/v1/stats") && init?.method === "GET") {
        return okJson({
          total_facts: 1,
          by_type: { semantic: 1 },
          avg_importance: 0.8,
          avg_retention: 1.0,
        });
      }
      throw new Error(`unexpected request: ${url} ${init?.method}`);
    });

    const kb = new HttpKnowledgeBase({
      baseUrl: "http://127.0.0.1:8000/",
      token: "secret",
      fetch,
    });

    const fact = await kb.add("User prefers TypeScript", {
      tags: ["prefs"],
      eventTime: "2026-01-01T00:00:00+00:00",
    });
    expect(fact.id).toBe("abcd1234");

    const fetched = await kb.get("abcd1234");
    expect(fetched.content).toBe("User prefers TypeScript");

    const lineage = await kb.lineage("abcd1234", { now: "2026-01-02T00:00:00+00:00" });
    expect(lineage[0]).toMatchObject({
      id: "abcd1234",
      slot_key: "user::preference",
      value_text: "TypeScript",
      version: 1,
      supersedes_id: "deadbeef",
    });

    const context = await kb.search("what language does the user prefer?");
    expect(context).toContain("TypeScript");

    const listed = await kb.list({ limit: 10 });
    expect(listed).toHaveLength(1);
    expect(listed[0]?.retention_score).toBe(1);

    await kb.delete("abcd1234");

    const stats = await kb.stats();
    expect(stats.total_facts).toBe(1);

    expect(calls[0]?.url).toBe("http://127.0.0.1:8000/v1/facts");
    expect(calls[0]?.init?.headers).toMatchObject({
      Authorization: "Bearer secret",
      "Content-Type": "application/json",
    });
    expect(JSON.parse(calls[0]?.init?.body ?? "{}")).toMatchObject({
      content: "User prefers TypeScript",
      tags: ["prefs"],
      event_time: "2026-01-01T00:00:00+00:00",
    });
    expect(calls[1]?.url).toBe("http://127.0.0.1:8000/v1/facts/abcd1234");
    expect(calls[2]?.url).toBe(
      "http://127.0.0.1:8000/v1/facts/abcd1234/lineage?now=2026-01-02T00%3A00%3A00%2B00%3A00",
    );
    expect(calls[4]?.url).toContain("/v1/facts?");
  });

  it("supports learn() and addResolved() over the HTTP sidecar", async () => {
    const fetch = vi.fn(async (url: string, init) => {
      calls.push({ url, init });
      if (url.endsWith("/v1/learn") && init?.method === "POST") {
        return okJson({
          stored: 1,
          ids: ["abcd1234"],
          note: "provider not configured; stored the last user message verbatim",
        });
      }
      if (url.endsWith("/v1/facts/resolved") && init?.method === "POST") {
        return okJson([{ id: "abcd1234", content: "Alex earns 120k", slot_key: "alex::salary", version: 0 }], 201);
      }
      throw new Error(`unexpected request: ${url} ${init?.method}`);
    });

    const kb = new HttpKnowledgeBase({
      baseUrl: "http://127.0.0.1:8000",
      token: "secret",
      fetch,
    });

    const learned = await kb.learn(
      [{ role: "user", content: "I prefer dark mode." }],
      { provider: "openai", apiKey: "sk-test", model: "gpt-5-mini", eventTime: "2023-05-08T00:00:00+00:00" },
    );
    expect(learned).toEqual({
      stored: 1,
      ids: ["abcd1234"],
      note: "provider not configured; stored the last user message verbatim",
    });

    const resolved = await kb.addResolved([
      {
        content: "Alex earns 120k",
        entity: "Alex",
        attribute: "salary",
        valueText: "120k",
        slotKey: "alex::salary",
        op: "update",
        eventTime: "2023-05-08T00:00:00+00:00",
      },
    ]);
    expect(resolved).toEqual([
      { id: "abcd1234", content: "Alex earns 120k", slot_key: "alex::salary", version: 0 },
    ]);

    expect(calls[0]?.url).toBe("http://127.0.0.1:8000/v1/learn");
    expect(JSON.parse(calls[0]?.init?.body ?? "{}")).toMatchObject({
      messages: [{ role: "user", content: "I prefer dark mode." }],
      provider: "openai",
      api_key: "sk-test",
      model: "gpt-5-mini",
      event_time: "2023-05-08T00:00:00+00:00",
    });
    expect(calls[1]?.url).toBe("http://127.0.0.1:8000/v1/facts/resolved");
    expect(JSON.parse(calls[1]?.init?.body ?? "{}")).toMatchObject({
      facts: [
        {
          content: "Alex earns 120k",
          entity: "Alex",
          attribute: "salary",
          value_text: "120k",
          slot_key: "alex::salary",
          op: "update",
          event_time: "2023-05-08T00:00:00+00:00",
        },
      ],
    });
  });

  it("supports health checks without auth", async () => {
    const fetch = vi.fn(async (url: string, init) => {
      calls.push({ url, init });
      if (url.endsWith("/health")) {
        return okJson({ status: "ok", version: "0.11.0" });
      }
      throw new Error(`unexpected request: ${url}`);
    });

    const kb = new HttpKnowledgeBase({
      baseUrl: "http://127.0.0.1:8000",
      token: "secret",
      fetch,
    });

    await expect(kb.health()).resolves.toEqual({ status: "ok", version: "0.11.0" });
    expect(calls[0]?.init?.headers).not.toHaveProperty("Authorization");
  });

  it("surfaces HTTP errors with server detail", async () => {
    const fetch = vi.fn(async () => errorJson(401, "missing or invalid bearer token"));
    const kb = new HttpKnowledgeBase({
      baseUrl: "http://127.0.0.1:8000",
      fetch,
    });

    await expect(kb.search("anything")).rejects.toThrow(
      /HTTP 401 Bad Request: missing or invalid bearer token/,
    );
  });

  it("fails fast when baseUrl is empty", () => {
    expect(() => new HttpKnowledgeBase({ baseUrl: "   ", fetch: vi.fn() })).toThrow(
      /non-empty baseUrl/,
    );
  });
});
