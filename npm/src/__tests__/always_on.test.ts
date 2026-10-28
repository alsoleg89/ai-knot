import { describe, expect, it, vi } from "vitest";
import { AlwaysOnError, HttpAlwaysOnMemory, HttpKnowledgeBase, KnowledgeBase, McpAlwaysOnMemory } from "../index.js";
import type { HttpKnowledgeBaseOptions } from "../types.js";

describe("always-on adapters", () => {
  it("loads a queryless task snapshot through HTTP and MCP", async () => {
    const calls: unknown[] = [];
    const facts = [{ ref: { namespace: "owner", fact_id: "receipt", revision: "r1" }, content: "Receipt verified", slot_key: "" }];
    const request = { task_key: "restore", max_facts: 3, max_chars: 600 };
    const mcp = new McpAlwaysOnMemory(async (name, args) => { calls.push([name, args]); return JSON.stringify(facts); });
    const fetch = vi.fn<NonNullable<HttpKnowledgeBaseOptions["fetch"]>>(async (url, options) => { calls.push([url, options]); return { ok: true, status: 200, statusText: "OK", json: async () => facts, text: async () => "" }; });
    const http = new HttpKnowledgeBase({ baseUrl: "http://localhost", token: "secret", fetch });
    expect(await mcp.taskSnapshot(request)).toEqual(facts);
    expect(await http.alwaysOn.taskSnapshot(request)).toEqual(facts);
    expect(calls).toEqual([
      ["memory_context_snapshot", request],
      ["http://localhost/v1/always-on/evidence/snapshot", {
        method: "POST", headers: { Authorization: "Bearer secret", "Content-Type": "application/json" },
        body: JSON.stringify(request),
      }],
    ]);
  });
  it("preserves task snapshot overflow errors in both transports", async () => {
    const payload = { error: { code: "SnapshotTooLarge", message: "The complete snapshot exceeds the limit." } };
    const fetch = vi.fn(async () => ({ ok: false, status: 413, statusText: "Content Too Large", json: async () => payload, text: async () => "" }));
    const http = new HttpKnowledgeBase({ baseUrl: "http://localhost", fetch });
    await expect(http.alwaysOn.taskSnapshot({ task_key: "restore" })).rejects.toMatchObject({ code: "SnapshotTooLarge", status: 413 });
    const mcp = new McpAlwaysOnMemory(async () => JSON.stringify(payload));
    await expect(mcp.taskSnapshot({ task_key: "restore" })).rejects.toMatchObject({ code: "SnapshotTooLarge" });
  });
  it("records revision-bound experience through HTTP and MCP", async () => {
    const calls: unknown[] = [];
    const mcp = new McpAlwaysOnMemory(async (name, args) => { calls.push([name, args]); return "{}"; });
    const http = new HttpAlwaysOnMemory(async (path, options) => { calls.push([path, options]); return {} as never; }, async () => {});
    const request = { expected_action_revision: 3, experience_key: "lesson", content: "Restore warning", basis: [{ namespace: "owner", fact_id: "receipt", revision: "r" }], task_key: "rollout42", for_role: "operator" };
    await mcp.recordExperience("action/id", request); await http.recordExperience("action/id", request);
    expect(calls).toEqual([
      ["action_experience_record", { ...request, action_id: "action/id" }],
      ["/v1/always-on/actions/action%2Fid/experience", { method: "POST", body: request }],
    ]);
  });
  it("selects task context through HTTP and MCP without changing its request", async () => {
    const calls: unknown[] = [];
    const mcp = new McpAlwaysOnMemory(async (name, args) => { calls.push([name, args]); return "{}"; });
    const http = new HttpAlwaysOnMemory(async (path, options) => { calls.push([path, options]); return {} as never; }, async () => {});
    const request = { requirements: ["rollback checkpoint"], task_key: "release42", role: "operator", max_facts: 3, max_chars: 800 };
    await mcp.selectContext(request); await http.selectContext(request);
    expect(calls).toEqual([
      ["memory_context_select", request],
      ["/v1/always-on/evidence/select", { method: "POST", body: request }],
    ]);
  });
  it("routes snapshot recovery and checked context with bounded inputs", async () => {
    const calls: unknown[] = [];
    const mcp = new McpAlwaysOnMemory(async (name, args) => { calls.push([name, args]); return "{}"; });
    const http = new HttpAlwaysOnMemory(async (path, options) => { calls.push([path, options]); return {} as never; }, async () => {});
    const refs = [{ namespace: "owner", fact_id: "fact", revision: "r" }];
    await mcp.resync("watch/id", 5); await mcp.reviewContext(refs, 100);
    await http.resync("watch/id", 5); await http.reviewContext(refs, 100);
    expect(calls).toEqual([
      ["memory_resync", { subscription_id: "watch/id", limit: 5 }],
      ["memory_context_review", { refs, max_chars: 100 }],
      ["/v1/always-on/subscriptions/watch%2Fid/resync", { method: "POST", body: { limit: 5 } }],
      ["/v1/always-on/evidence/review", { method: "POST", body: { refs, max_chars: 100 } }],
    ]);
  });
  it("constructs without polling or a subprocess", () => {
    const fetch = vi.fn();
    expect(new HttpKnowledgeBase({ baseUrl: "http://localhost", fetch }).alwaysOn).toBeInstanceOf(HttpAlwaysOnMemory);
    expect(new KnowledgeBase().alwaysOn).toBeInstanceOf(McpAlwaysOnMemory);
    expect(fetch).not.toHaveBeenCalled();
  });
  it("ack preserves JSON body, authorization, custom headers and never reads 204 JSON", async () => {
    const json = vi.fn();
    const fetch = vi.fn(async () => ({ ok: true, status: 204, statusText: "No Content", json, text: async () => "" }));
    const client = new HttpKnowledgeBase({ baseUrl: "http://localhost", token: "secret", headers: { "X-Host": "test" }, fetch });
    await client.alwaysOn.ack("watch/id", "delivery");
    expect(fetch).toHaveBeenLastCalledWith("http://localhost/v1/always-on/subscriptions/watch%2Fid/ack", {
      method: "POST", headers: { "X-Host": "test", Authorization: "Bearer secret", "Content-Type": "application/json" }, body: '{"delivery_id":"delivery"}',
    });
    await client.alwaysOn.unsubscribe("watch/id");
    expect(fetch).toHaveBeenLastCalledWith("http://localhost/v1/always-on/subscriptions/watch%2Fid", {
      method: "DELETE", headers: { "X-Host": "test", Authorization: "Bearer secret" }, body: undefined,
    });
    expect(json).not.toHaveBeenCalled();
  });
  it("preserves structured HTTP and MCP domain codes", async () => {
    const payload = { error: { code: "LeaseLost", message: "The lease is no longer current." } };
    const fetch = vi.fn(async () => ({ ok: false, status: 409, statusText: "Conflict", json: async () => payload, text: async () => "" }));
    const http = new HttpKnowledgeBase({ baseUrl: "http://localhost", fetch });
    await expect(http.alwaysOn.beginAction("a", "token")).rejects.toMatchObject({ code: "LeaseLost", status: 409 });
    const mcp = new McpAlwaysOnMemory(async () => JSON.stringify(payload));
    await expect(mcp.beginAction("a", "token")).rejects.toBeInstanceOf(AlwaysOnError);
    const broken = new McpAlwaysOnMemory(async () => '{"error":"bad"}');
    await expect(broken.resume("x")).rejects.toMatchObject({ code: "InvalidResponse" });
  });
  it("routes every public operation consistently through existing transports", async () => {
    const httpCalls: unknown[] = [], mcpCalls: unknown[] = [];
    const http = new HttpAlwaysOnMemory(async (path, options) => { httpCalls.push([path, options]); return {} as never; },
      async (path, options) => { httpCalls.push([path, options]); });
    const mcp = new McpAlwaysOnMemory(async (name, args) => { mcpCalls.push([name, args]); return "{}"; });
    const ref = { namespace: "owner", fact_id: "f", revision: "r" };
    for (const client of [http, mcp]) {
      await client.createIntent({ client_key: "k", goal: "g" }); await client.getIntent("i");
      await client.listIntents(); await client.claim("i", { worker_id: "host" });
      await client.renew("i", { lease_token: "t" });
      await client.checkpoint("i", { lease_token: "t", expected_revision: 1, state: "ready", progress: {} });
      await client.resume("i"); await client.captureEvidence([["owner", "f"]]);
      await client.subscribe({ subscription_id: "s", selector: { namespace: "owner" } });
      await client.poll("s"); await client.ack("s", "d"); await client.unsubscribe("s");
      await client.prepareAction({ intent_id: "i", lease_token: "t", action_key: "a", request: {} });
      await client.beginAction("a", "t"); await client.finishAction("a", { lease_token: "t", outcome: "succeeded", result: {} });
      await client.reconcileAction("a", { lease_token: "t", outcome: "succeeded", evidence: {} });
      await client.setFreshness({ ref, source_ref: "connector", recheck_after_seconds: 60 });
      await client.dueVerifications(); await client.recordVerification({ ref, result: { outcome: "confirmed", evidence: {}, source_version: "1" } });
      await client.runMaintenance({ run_key: "run", budget: {} });
      await client.taskSnapshot({ task_key: "restore" });
    }
    expect(httpCalls).toHaveLength(21); expect(mcpCalls).toHaveLength(21);
    expect(mcpCalls[10]).toEqual(["memory_ack", { subscription_id: "s", delivery_id: "d" }]);
    expect(httpCalls[2]).toEqual(["/v1/always-on/intents?limit=100", { method: "GET" }]);
  });
});
