// ---------------------------------------------------------------------------
// Trace types — produced by the recall_with_trace MCP tool
// ---------------------------------------------------------------------------

/** Stage-1 candidate sets from the three parallel retrieval arms */
export interface Stage1Candidates {
  /** Pack-fact IDs returned by the BM25 lexical arm */
  from_bm25: string[]
  /** Pack-fact IDs returned by the rare-token expansion arm */
  from_rare_tokens: string[]
  /** Pack-fact IDs returned by entity-hop traversal */
  from_entity_hop: string[]
  /** Total unique candidates across all arms (optional, may be computed) */
  total?: number
}

/** Full trace object attached to every recall result */
export interface RecallTrace {
  /** Classified query intent: factual | aggregational | exploratory | temporal | comparative */
  intent?: string
  /** Stage-1 candidate breakdown */
  stage1_candidates: Stage1Candidates
  /** Stage-3 RRF score map: pack_fact_id → float */
  stage3_rrf?: Record<string, unknown>
  /** Stage-3b dense guarantee — IDs promoted by dense similarity floor */
  stage3b_dense_guarantee?: string[]
  /** Stage-4a DDSA — IDs surviving date-distance score adjustment */
  stage4a_ddsa?: string[]
  /** Stage-4b MMR — final ranked IDs after maximal-marginal-relevance re-rank */
  stage4b_mmr?: string[]
  /** Allow unknown future trace fields */
  [key: string]: unknown
}

/** Top-level result from recall_with_trace */
export interface TraceResult {
  /** LLM-ready context string (rendered pack facts) */
  context: string
  /** Ordered list of pack-fact IDs used to build context */
  pack_fact_ids: string[]
  /** Full pipeline trace */
  trace: RecallTrace
}

/** A single annotated trace example used in the UX prototype */
export interface TraceExample {
  question: string
  result: TraceResult
}

// ---------------------------------------------------------------------------
// Knot view types — the entity-strand / bead graph
// ---------------------------------------------------------------------------

/** Memory category of a bead */
export type MemoryType = 'semantic' | 'procedural' | 'episodic'

/**
 * A single atomic fact extracted from a conversation turn.
 * A bead belongs to exactly one entity strand.
 */
export interface Bead {
  /** Globally unique ID (UUID-like) */
  id: string
  /** Natural-language content of the extracted fact */
  content: string
  /** Memory classification */
  type: MemoryType
  /** Importance score, 0–1 (higher = more salient) */
  importance: number
  /** Parent entity ID (matches Strand.entityId) */
  entityId: string
  /** ISO-8601 date of the session in which this bead was created */
  sessionDate?: string
  /** ISO-8601 date of the real-world event described by the bead (may differ from session) */
  eventDate?: string
  /** Free-form tags, e.g. ["sport", "hobby", "preference"] */
  tags: string[]
}

/**
 * A strand groups all beads that belong to a single entity.
 * Entities can be people, pets, places, or named topics.
 */
export interface Strand {
  /** Unique entity identifier */
  entityId: string
  /** Human-readable label */
  label: string
  /** Ordered list of beads (typically by sessionDate asc) */
  beads: Bead[]
}

/**
 * A directed edge connecting two beads across entity strands.
 * Crossings encode coreference and alias relationships discovered
 * during materialisation.
 */
export interface Crossing {
  /** Bead ID in the source strand */
  sourceBeadId: string
  /** Bead ID in the target strand */
  targetBeadId: string
  /** Semantic type of the crossing */
  edgeType: 'pronoun-resolves-to' | 'group-alias' | 'coreference'
  /** Optional display label for the edge */
  label?: string
}

/** Root document for the KnotView component */
export interface KnotData {
  /** Conversation identifier this knot snapshot belongs to */
  conversationId: string
  /** All entity strands in this snapshot */
  strands: Strand[]
  /** Cross-strand edges */
  crossings: Crossing[]
}
