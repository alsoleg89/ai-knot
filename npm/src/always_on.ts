/** Durable memory operations. Constructors never start a worker or poll. */
export type JsonValue = null | boolean | number | string | JsonValue[] | JsonObject;
export type JsonObject = { [key: string]: JsonValue };
export interface EvidenceRef { namespace: string; fact_id: string; revision: string }
export type WaitCondition =
  | { kind: "until"; at: string }
  | { kind: "slot"; namespace: string; slot_key: string; value_text: string }
  | { kind: "change"; ref: EvidenceRef };
export type IntentState = "ready" | "running" | "waiting" | "blocked" | "completed" | "cancelled";
export interface Lease { token: string; generation: number; worker_id: string; expires_at: string }
export interface Intent {
  id: string; owner_agent_id: string; client_key: string; goal: string; state: IntentState;
  progress: JsonObject; next_step: string; waiting_for: WaitCondition | null; revision: number;
  basis: EvidenceRef[]; lease: Lease | null; lease_generation: number; blocked_reasons: string[];
  deadline: string | null; last_transition_at: string;
}
export interface ActionRecord {
  id: string; intent_id: string; action_key: string; request: JsonObject; request_digest: string;
  basis: EvidenceRef[]; state: "prepared" | "started" | "succeeded" | "failed" | "unknown";
  revision: number; idempotency_key: string; result: JsonObject | null; error: JsonObject | null;
  created_at: string; updated_at: string; reconciliation_evidence: JsonObject | null;
}
export interface ActionPermit { action_id: string; idempotency_key: string }
export interface BasisIssue { reason: string; ref: EvidenceRef | null }
export interface ResumePacket {
  intent: Intent; open_actions: ActionRecord[]; stale_basis: BasisIssue[];
  freshness_due: EvidenceRef[]; ready: boolean; reasons: string[];
}
export interface WatchSelector { namespace: string; fact_ids?: string[]; slot_keys?: string[] }
export interface Subscription {
  id: string; owner_agent_id: string; selector: WatchSelector;
  cursor: number; debounce_seconds: number;
}
export interface FactChange {
  namespace: string; seq: number; fact_id: string; kind: string;
  before_revision: string | null; after_revision: string | null;
  slot_key: string;
}
export interface ChangeBatch {
  delivery_id: string | null; events: FactChange[]; resync_required: boolean;
  through_seq: number; head_seq: number; has_more: boolean; is_snapshot: boolean;
}
export interface ContextFact { ref: EvidenceRef; content: string; slot_key: string }
export interface ContextReview {
  accepted: ContextFact[]; issues: BasisIssue[]; omitted: EvidenceRef[];
  checked_at: string; complete: boolean;
}
export interface TaskContext {
  facts: ContextFact[]; covered_requirements: string[]; missing_requirements: string[];
  budget_limited: boolean; checked_at: string; complete: boolean;
}
export interface RecordedExperience {
  lesson: ContextFact; outcome: "succeeded" | "failed"; action_revision: number;
}
export type VerificationOutcome = "confirmed" | "changed" | "unavailable";
export interface VerificationResult {
  outcome: VerificationOutcome; evidence: JsonObject; source_version: string;
}
export interface VerificationRequest { ref: EvidenceRef; source_ref: string }
export interface FreshnessRecord {
  ref: EvidenceRef; subject_kind: "slot" | "fact"; subject_key: string; source_ref: string;
  recheck_after_seconds: number; checked_at: string | null; verified_until: string; retry_at: string;
  last_outcome: VerificationOutcome | null; failure_count: number;
  evidence: JsonObject | null; source_version: string;
}
export interface MaintenanceBudget {
  max_items?: number; max_model_calls?: number; max_input_bytes?: number; max_wall_seconds?: number;
}
export interface MaintenanceReport {
  run_id: string; status: "completed" | "budget_exhausted" | "conflict" | "failed";
  inspected: number; verified: number; consolidated: number; expired: number;
  model_calls_reserved: number; skipped: number; reasons: string[];
}
export interface CheckpointInput {
  lease_token: string; expected_revision: number; state: IntentState; progress: JsonObject;
  next_step?: string; waiting_for?: WaitCondition | null; basis?: EvidenceRef[];
}
export interface AlwaysOnClient {
  createIntent(input: { client_key: string; goal: string; next_step?: string; deadline?: string | null }): Promise<Intent>;
  getIntent(intentId: string): Promise<Intent>;
  listIntents(limit?: number): Promise<Intent[]>;
  claim(intentId: string, input: { worker_id: string; ttl_seconds?: number }): Promise<Lease>;
  renew(intentId: string, input: { lease_token: string; ttl_seconds?: number }): Promise<Lease>;
  checkpoint(intentId: string, input: CheckpointInput): Promise<Intent>;
  resume(intentId: string): Promise<ResumePacket>;
  captureEvidence(refs: [string, string][]): Promise<EvidenceRef[]>;
  subscribe(input: { subscription_id: string; selector: WatchSelector; start?: "now" | "beginning"; debounce_seconds?: number }): Promise<Subscription>;
  poll(subscriptionId: string, limit?: number): Promise<ChangeBatch>;
  resync(subscriptionId: string, limit?: number): Promise<ChangeBatch>;
  reviewContext(refs: EvidenceRef[], maxChars?: number): Promise<ContextReview>;
  selectContext(input: { requirements: string[]; task_key?: string; role?: string; max_facts?: number; max_chars?: number }): Promise<TaskContext>;
  ack(subscriptionId: string, deliveryId: string): Promise<void>;
  unsubscribe(subscriptionId: string): Promise<void>;
  prepareAction(input: { intent_id: string; lease_token: string; action_key: string; request: JsonObject; basis?: EvidenceRef[] }): Promise<ActionRecord>;
  beginAction(actionId: string, leaseToken: string): Promise<ActionPermit>;
  finishAction(actionId: string, input: { lease_token: string; outcome: "succeeded" | "failed" | "unknown"; result: JsonObject }): Promise<ActionRecord>;
  reconcileAction(actionId: string, input: { lease_token: string; outcome: "succeeded" | "failed" | "retryable"; evidence: JsonObject }): Promise<ActionRecord>;
  recordExperience(actionId: string, input: { expected_action_revision: number; experience_key: string; content: string; basis: EvidenceRef[]; namespace?: string | null; visibility_scope?: string; task_key?: string; for_role?: string }): Promise<RecordedExperience>;
  setFreshness(input: { ref: EvidenceRef; source_ref: string; recheck_after_seconds: number }): Promise<FreshnessRecord>;
  dueVerifications(limit?: number): Promise<VerificationRequest[]>;
  recordVerification(input: { ref: EvidenceRef; result: VerificationResult }): Promise<FreshnessRecord>;
  runMaintenance(input: { run_key: string; budget: MaintenanceBudget }): Promise<MaintenanceReport>;
}
export class AlwaysOnError extends Error {
  constructor(readonly code: string, message: string, readonly status?: number) {
    super(message); this.name = "AlwaysOnError";
  }
}
export function parseAlwaysOnError(value: unknown, status?: number): AlwaysOnError | null {
  if (!value || typeof value !== "object" || !("error" in value)) return null;
  const error = (value as { error: unknown }).error;
  if (!error || typeof error !== "object" || !("code" in error) || !("message" in error)
      || typeof error.code !== "string" || typeof error.message !== "string") {
    return new AlwaysOnError("InvalidResponse", "Invalid always-on error response.", status);
  }
  return new AlwaysOnError(error.code, error.message, status);
}

type Operation = { tool: string; path: string; verb?: "GET" | "DELETE"; id?: string };
const operations: Record<string, Operation> = {
  createIntent: { tool: "intent_create", path: "/intents" },
  getIntent: { tool: "intent_get", path: "/intents/{id}", verb: "GET", id: "intent_id" },
  listIntents: { tool: "intent_list", path: "/intents", verb: "GET" },
  claim: { tool: "intent_claim", path: "/intents/{id}/claim", id: "intent_id" },
  renew: { tool: "intent_renew", path: "/intents/{id}/renew", id: "intent_id" },
  checkpoint: { tool: "intent_checkpoint", path: "/intents/{id}/checkpoint", id: "intent_id" },
  resume: { tool: "intent_resume", path: "/intents/{id}/resume", id: "intent_id" },
  captureEvidence: { tool: "memory_evidence_capture", path: "/evidence/capture" },
  subscribe: { tool: "memory_subscribe", path: "/subscriptions" },
  poll: { tool: "memory_poll", path: "/subscriptions/{id}/poll", id: "subscription_id" },
  resync: { tool: "memory_resync", path: "/subscriptions/{id}/resync", id: "subscription_id" },
  reviewContext: { tool: "memory_context_review", path: "/evidence/review" },
  selectContext: { tool: "memory_context_select", path: "/evidence/select" },
  ack: { tool: "memory_ack", path: "/subscriptions/{id}/ack", id: "subscription_id" },
  unsubscribe: { tool: "memory_unsubscribe", path: "/subscriptions/{id}", verb: "DELETE", id: "subscription_id" },
  prepareAction: { tool: "action_prepare", path: "/actions" },
  beginAction: { tool: "action_begin", path: "/actions/{id}/begin", id: "action_id" },
  finishAction: { tool: "action_finish", path: "/actions/{id}/finish", id: "action_id" },
  reconcileAction: { tool: "action_reconcile", path: "/actions/{id}/reconcile", id: "action_id" },
  recordExperience: { tool: "action_experience_record", path: "/actions/{id}/experience", id: "action_id" },
  setFreshness: { tool: "memory_freshness_set", path: "/freshness" },
  dueVerifications: { tool: "memory_freshness_due", path: "/freshness/due", verb: "GET" },
  recordVerification: { tool: "memory_freshness_record", path: "/freshness/record" },
  runMaintenance: { tool: "memory_maintenance", path: "/maintenance/run" },
};
abstract class AlwaysOnMethods implements AlwaysOnClient {
  protected abstract invoke<T>(name: string, body: Record<string, unknown>, id?: string): Promise<T>;
  createIntent(input: Parameters<AlwaysOnClient["createIntent"]>[0]): Promise<Intent> { return this.invoke("createIntent", input); }
  getIntent(id: string): Promise<Intent> { return this.invoke("getIntent", {}, id); }
  listIntents(limit = 100): Promise<Intent[]> { return this.invoke("listIntents", { limit }); }
  claim(id: string, input: Parameters<AlwaysOnClient["claim"]>[1]): Promise<Lease> { return this.invoke("claim", input, id); }
  renew(id: string, input: Parameters<AlwaysOnClient["renew"]>[1]): Promise<Lease> { return this.invoke("renew", input, id); }
  checkpoint(id: string, input: CheckpointInput): Promise<Intent> { return this.invoke("checkpoint", { ...input }, id); }
  resume(id: string): Promise<ResumePacket> { return this.invoke("resume", {}, id); }
  captureEvidence(refs: [string, string][]): Promise<EvidenceRef[]> { return this.invoke("captureEvidence", { refs }); }
  subscribe(input: Parameters<AlwaysOnClient["subscribe"]>[0]): Promise<Subscription> { return this.invoke("subscribe", input); }
  poll(id: string, limit = 100): Promise<ChangeBatch> { return this.invoke("poll", { limit }, id); }
  resync(id: string, limit = 500): Promise<ChangeBatch> { return this.invoke("resync", { limit }, id); }
  reviewContext(refs: EvidenceRef[], maxChars = 12000): Promise<ContextReview> { return this.invoke("reviewContext", { refs, max_chars: maxChars }); }
  selectContext(input: Parameters<AlwaysOnClient["selectContext"]>[0]): Promise<TaskContext> { return this.invoke("selectContext", input); }
  async ack(id: string, deliveryId: string): Promise<void> { await this.invoke("ack", { delivery_id: deliveryId }, id); }
  async unsubscribe(id: string): Promise<void> { await this.invoke("unsubscribe", {}, id); }
  prepareAction(input: Parameters<AlwaysOnClient["prepareAction"]>[0]): Promise<ActionRecord> { return this.invoke("prepareAction", input); }
  beginAction(id: string, leaseToken: string): Promise<ActionPermit> { return this.invoke("beginAction", { lease_token: leaseToken }, id); }
  finishAction(id: string, input: Parameters<AlwaysOnClient["finishAction"]>[1]): Promise<ActionRecord> { return this.invoke("finishAction", input, id); }
  reconcileAction(id: string, input: Parameters<AlwaysOnClient["reconcileAction"]>[1]): Promise<ActionRecord> { return this.invoke("reconcileAction", input, id); }
  recordExperience(id: string, input: Parameters<AlwaysOnClient["recordExperience"]>[1]): Promise<RecordedExperience> { return this.invoke("recordExperience", input, id); }
  setFreshness(input: Parameters<AlwaysOnClient["setFreshness"]>[0]): Promise<FreshnessRecord> { return this.invoke("setFreshness", input); }
  dueVerifications(limit = 100): Promise<VerificationRequest[]> { return this.invoke("dueVerifications", { limit }); }
  recordVerification(input: Parameters<AlwaysOnClient["recordVerification"]>[0]): Promise<FreshnessRecord> { return this.invoke("recordVerification", input); }
  runMaintenance(input: Parameters<AlwaysOnClient["runMaintenance"]>[0]): Promise<MaintenanceReport> { return this.invoke("runMaintenance", input); }
}

type JsonRequest = <T>(path: string, options: { method: "GET" | "POST"; body?: Record<string, unknown> }) => Promise<T>;
type VoidRequest = (path: string, options: { method: "POST" | "DELETE"; body?: Record<string, unknown> }) => Promise<void>;
export class HttpAlwaysOnMemory extends AlwaysOnMethods {
  constructor(private readonly requestJson: JsonRequest, private readonly requestVoid: VoidRequest) { super(); }
  protected async invoke<T>(name: string, body: Record<string, unknown>, id?: string): Promise<T> {
    const operation = operations[name]!;
    const path = "/v1/always-on" + operation.path.replace("{id}", encodeURIComponent(id ?? ""));
    if (name === "ack" || operation.verb === "DELETE") {
      await this.requestVoid(path, operation.verb === "DELETE" ? { method: "DELETE" } : { method: "POST", body });
      return undefined as T;
    }
    if (operation.verb === "GET") {
      const query = new URLSearchParams(Object.entries(body).map(([key, value]) => [key, String(value)])).toString();
      return this.requestJson<T>(path + (query ? "?" + query : ""), { method: "GET" });
    }
    return this.requestJson<T>(path, { method: "POST", body });
  }
}
export class McpAlwaysOnMemory extends AlwaysOnMethods {
  constructor(private readonly call: (name: string, args: Record<string, unknown>) => Promise<string>) { super(); }
  protected async invoke<T>(name: string, body: Record<string, unknown>, id?: string): Promise<T> {
    const operation = operations[name]!;
    const args = operation.id ? { ...body, [operation.id]: id } : body;
    const text = await this.call(operation.tool, args);
    let value: unknown;
    try { value = JSON.parse(text); }
    catch { throw new AlwaysOnError("InvalidResponse", "Invalid always-on JSON response."); }
    const error = parseAlwaysOnError(value);
    if (error) throw error;
    return value as T;
  }
}
