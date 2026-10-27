import { describe, expect, it, vi } from "vitest";
import { AlwaysOnError, HttpAlwaysOnMemory, HttpKnowledgeBase, KnowledgeBase, McpAlwaysOnMemory } from "../index.js";

describe("always-on adapters", () => {
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
    }
    expect(httpCalls).toHaveLength(20); expect(mcpCalls).toHaveLength(20);
    expect(mcpCalls[10]).toEqual(["memory_ack", { subscription_id: "s", delivery_id: "d" }]);
    expect(httpCalls[2]).toEqual(["/v1/always-on/intents?limit=100", { method: "GET" }]);
  });
});
