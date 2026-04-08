"""Inline test data for the benchmark suite.

All data is self-contained — no external JSON files required.
The persona is a generic senior backend engineer (not domain-specific).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# S1 — Profile Retrieval
# ---------------------------------------------------------------------------

USER_PROFILE_FACTS: list[str] = [
    "User is a senior backend engineer at Meridian Technologies, based in Berlin.",
    "User's primary language is Python 3.12; uses Rust for latency-critical paths.",
    "User runs all services on Kubernetes 1.29 with Helm 3 charts.",
    "User prefers synchronous code and avoids async unless strictly necessary.",
    "User uses pytest with --tb=short for all test runs.",
    "User's team follows trunk-based development; no long-lived feature branches.",
    "User's work hours are 10:00-19:00 CET; standup at 10:15.",
    "User has a standing rule: never use print() for debugging, only logging.",
    "User's database of choice is PostgreSQL 16 with asyncpg driver.",
    "User participates in on-call rotation; next on-call: 2026-04-10.",
    "User's manager is Keiko Tanaka; skip-level is Arjan de Vries.",
    "User recently migrated from Jenkins to GitHub Actions.",
    "User keeps secrets in HashiCorp Vault; never in environment variables.",
    "User requires type annotations on all public APIs (mypy strict).",
    "User's side project is an open-source BM25 library called ranklib.",
]

PROFILE_QUERIES: list[str] = [
    "What language does the user prefer?",
    "How does the user handle secrets and credentials?",
    "What are the user's debugging preferences?",
    "What database does the user work with?",
    "What CI/CD system does the user use?",
]

# ---------------------------------------------------------------------------
# S2 — Avoid Repeats
# ---------------------------------------------------------------------------

PUBLISHED_TITLES: list[str] = [
    "How we cut API latency by 40% with connection pooling",
    "Migrating from Jenkins to GitHub Actions: lessons learned",
    "PostgreSQL 16 features every backend dev should know",
    "Why we switched from threads to asyncio",
    "Type annotations at scale: mypy strict in a 200k-line codebase",
    "Kubernetes Helm charts: pitfalls and best practices",
    "On-call culture: building a sustainable rotation",
    "Trunk-based development: our 6-month retrospective",
    "Secrets management with HashiCorp Vault in production",
    "BM25 vs dense retrieval: a practical comparison",
    "Rust in a Python shop: when it makes sense",
    "Distributed tracing with OpenTelemetry on Kubernetes",
    "Database migrations without downtime",
    "Load testing your API with Locust",
    "From monolith to microservices: the hidden costs",
    "Python 3.12 performance improvements in practice",
    "Feature flags without the complexity",
    "Code review practices that actually improve quality",
    "Observability: logs, metrics, and traces",
    "gRPC vs REST: choosing the right protocol",
    "Building a developer platform from scratch",
    "On writing runbooks that people actually read",
    "Incident postmortems: format and facilitation",
    "How we run 10,000 tests in under 5 minutes",
    "Dependency injection in Python without frameworks",
    "API versioning strategies that won't bite you later",
    "Zero-downtime deploys with Kubernetes rolling updates",
    "Structured logging: why and how",
    "The case for boring technology",
    "Profiling Python applications in production",
    "Database connection pool sizing: the math behind it",
    "Writing documentation developers will actually read",
    "How we reduced our Docker image size by 80%",
    "Service mesh: do you actually need it?",
    "Rate limiting: algorithms and implementation",
    "Testing database migrations with pytest",
    "Error budgets and SLOs in practice",
    "Python packaging in 2024: pyproject.toml guide",
    "Secrets rotation without service interruption",
    "The 12-factor app revisited in 2025",
    "Distributed locks: when and how to use them",
    "Health checks: what to actually check",
    "Idempotency in API design",
    "Graceful shutdown patterns in Python services",
    "Caching strategies: CDN, application, database layers",
    "Async Python pitfalls and how to avoid them",
    "Monitoring Kubernetes with Prometheus and Grafana",
    "Multi-tenancy patterns for SaaS backends",
    "GraphQL vs REST vs gRPC: a 2025 decision guide",
    "Benchmarking Python: tools and methodology",
]

AVOID_REPEATS_QUERIES: list[str] = [
    "Write an article about Python performance",
    "Suggest a topic about Kubernetes deployment",
    "What should I write about database management?",
]

# Each query has a set of titles from PUBLISHED_TITLES that should NOT be repeated.
AVOID_REPEATS_EXPECTED_SEEN: dict[str, list[str]] = {
    "Write an article about Python performance": [
        "How we cut API latency by 40% with connection pooling",
        "Python 3.12 performance improvements in practice",
        "Profiling Python applications in production",
        "Benchmarking Python: tools and methodology",
    ],
    "Suggest a topic about Kubernetes deployment": [
        "Kubernetes Helm charts: pitfalls and best practices",
        "Zero-downtime deploys with Kubernetes rolling updates",
        "Monitoring Kubernetes with Prometheus and Grafana",
    ],
    "What should I write about database management?": [
        "PostgreSQL 16 features every backend dev should know",
        "Database migrations without downtime",
        "Testing database migrations with pytest",
        "Database connection pool sizing: the math behind it",
    ],
}

# ---------------------------------------------------------------------------
# S3 — Feedback Learning
# ---------------------------------------------------------------------------

FEEDBACK_HISTORY: list[tuple[str, str]] = [
    (
        "How long should posts be?",
        "Posts were too long last time — keep it under 300 words for LinkedIn.",
    ),
    (
        "Should I use questions in posts?",
        "Avoid ending with a question, it looks clickbait-y. State things directly.",
    ),
    (
        "What about emojis?",
        "No emojis in technical posts. They distract from the content.",
    ),
    (
        "How should I format code examples?",
        "Always use fenced code blocks with language tags, never inline backticks for multi-line.",
    ),
    (
        "What tone to use?",
        "Conversational but precise. Avoid marketing language and superlatives.",
    ),
    (
        "Should I mention competitors?",
        "Never compare to competitors by name. Focus on our approach.",
    ),
    (
        "How to end an article?",
        "End with a concrete takeaway or action item, not a summary of what was covered.",
    ),
    (
        "How many sections?",
        "Three to five sections max. More than that and readers drop off.",
    ),
    (
        "What about CTAs?",
        "One CTA per post, at the end. No mid-article CTAs.",
    ),
    (
        "Image guidelines?",
        "Use diagrams over screenshots. Screenshots go stale; diagrams stay relevant.",
    ),
]

FEEDBACK_QUERIES: list[str] = [
    "What mistakes should I avoid in technical posts?",
    "What are the formatting rules for articles?",
    "How should I handle calls-to-action?",
]

# Full feedback texts semantically reachable for each query.
# Used by S3 semantic_coverage to give dense backends a fair evaluation
# (keyword matching misses paraphrased extractions from mem0/ai-knot).
# Keyed by FEEDBACK_QUERIES[i] to prevent string-key drift.
FEEDBACK_EXPECTED_RULES: dict[str, list[str]] = {
    FEEDBACK_QUERIES[0]: [
        "Posts were too long last time — keep it under 300 words for LinkedIn.",
        "Avoid ending with a question, it looks clickbait-y. State things directly.",
        "No emojis in technical posts. They distract from the content.",
        "Never compare to competitors by name. Focus on our approach.",
    ],
    FEEDBACK_QUERIES[1]: [
        "Always use fenced code blocks with language tags, never inline backticks for multi-line.",
        "Three to five sections max. More than that and readers drop off.",
        "Use diagrams over screenshots. Screenshots go stale; diagrams stay relevant.",
    ],
    FEEDBACK_QUERIES[2]: [
        "One CTA per post, at the end. No mid-article CTAs.",
        "End with a concrete takeaway or action item, not a summary of what was covered.",
    ],
}

# ---------------------------------------------------------------------------
# S4 — Deduplication
# ---------------------------------------------------------------------------

DEDUP_CANONICAL_RULE = "Always add type annotations to all public functions and methods."

DEDUP_PARAPHRASES: list[str] = [
    "Always add type annotations to all public functions and methods.",
    "All public functions and methods must have type annotations.",
    "Type hints are required on every public function and method.",
    "Every exported function should include type annotations.",
    "Public APIs must be annotated with Python type hints.",
    "Require type annotations on all public-facing functions.",
    "Add static type hints to all functions in public interfaces.",
    "Every public method must carry a type annotation.",
    "Public functions should always have their types annotated.",
    "Type annotations are mandatory for all public methods and functions.",
    "All externally visible functions must include type hints.",
    "Annotate return types and parameters on all public functions.",
    "Public functions require explicit type declarations.",
    "You must annotate the types of all public function signatures.",
    "Every public function must be annotated with Python types.",
    "Static types are required on public function definitions.",
    "Always annotate public function arguments and return values with types.",
    "Public methods and functions need type annotation coverage.",
    "Ensure type hints exist on all public functions.",
    "Type annotations must be present on every public API.",
    "All functions that are part of the public API need type annotations.",
    "Type-annotate all public methods without exception.",
    "Every function exposed publicly must have typed parameters and returns.",
    "Functions in the public interface must be type-annotated.",
    "Public function signatures must include type hints.",
    "All public callable objects must have type annotations.",
    "Write type annotations for every function accessible from outside.",
    "Public-facing functions need full type annotation coverage.",
    "Annotate all parameters and return types on public functions.",
    "Type hints are required for every method visible to callers.",
    "Always declare types on public function parameters.",
    "Every method on a public class must have type annotations.",
    "Provide type hints for all publicly accessible functions.",
    "Type annotations are compulsory for public API methods.",
    "Mark the types of arguments and return values on public functions.",
    "All API functions must have complete type annotations.",
    "Require type hints on every function that is part of the public interface.",
    "Annotate public functions with Python typing module constructs.",
    "Apply type annotations consistently across all public methods.",
    "No public function should be missing type hints.",
    "Type annotations must cover all public function definitions.",
    "Add full type coverage to every publicly accessible method.",
    "Explicit types are required for public method signatures.",
    "Always use type annotations on public API endpoints.",
    "Public class methods must include typed arguments and return types.",
    "Enforce type hints on all functions accessible externally.",
    "Type annotation is required for every public-facing callable.",
    "Annotate each public function with the correct Python types.",
    "Type annotations are obligatory for functions in the public API.",
    "All exposed functions must carry correct type annotations.",
]

DEDUP_DISTINCT_RULES: list[str] = [
    "Always add type annotations to all public functions and methods.",
    "Use pytest with --tb=short for all test runs.",
    "Never use print() for debugging; use the logging module instead.",
    "Keep secrets in HashiCorp Vault, never in environment variables.",
    "Follow trunk-based development with no long-lived feature branches.",
    "Document every public API with a one-line docstring.",
    "Run database migrations in a separate deploy step from application code.",
    "Limit function length to 40 lines; extract helpers beyond that.",
    "Every service must expose a /health endpoint.",
    "Use structured JSON logging in all production services.",
    "Write integration tests for every external API boundary.",
    "Require code review from at least one other engineer before merging.",
    "Pin all dependency versions in pyproject.toml.",
    "Tag every Docker image with the git commit SHA.",
    "Set resource requests and limits on every Kubernetes container.",
    "Use connection pooling for all database connections.",
    "Instrument all HTTP endpoints with latency histograms.",
    "All configuration must be injectable via environment variables.",
    "Enforce a maximum response time of 200 ms at the 99th percentile.",
    "Store all date/time values in UTC with explicit timezone info.",
]

# ---------------------------------------------------------------------------
# S5 — Decay
# ---------------------------------------------------------------------------

# Subset of profile facts split into "frequently accessed" vs "rarely accessed"
DECAY_FREQUENT_FACT = (
    "User's primary language is Python 3.12; uses Rust for latency-critical paths."
)
DECAY_RARE_FACT = "User participates in on-call rotation; next on-call: 2026-04-10."

# ---------------------------------------------------------------------------
# S6 — Load
# ---------------------------------------------------------------------------


# 200 semantically distinct facts for load testing.
# Spread across 10 domains × ~18 patterns each so that BM25 and TF-IDF do NOT
# collapse them. Repetitive "Service N runs on port N" facts were replaced here
# because identical structure causes near-dedup in sparse retrieval, making
# p95 latency artificially low and defeating the purpose of the load test.
_LOAD_DOMAINS: list[tuple[str, list[str]]] = [
    (
        "security",
        [
            "All API endpoints require Bearer token authentication.",
            "Secrets are rotated automatically every 90 days via Vault.",
            "mTLS is enforced between all internal services.",
            "RBAC policies are defined in OPA, not in application code.",
            "Vulnerability scans run on every Docker image push.",
            "JWT tokens expire after 15 minutes; refresh tokens after 7 days.",
            "SQL queries use parameterized statements; no string interpolation.",
            "Dependency audits run weekly via `pip-audit` and `npm audit`.",
            "Rate limiting is enforced at the API gateway, not per-service.",
            "CORS is configured explicitly; wildcard origins are forbidden.",
            "All 4xx/5xx errors are logged with request context for audit.",
            "CSP headers are set on all web responses.",
            "Password hashing uses argon2id with minimum cost parameters.",
            "Service-to-service calls are authenticated with short-lived SPIFFE SVIDs.",
            "Container images run as non-root user (uid 10001).",
            "Network policies restrict egress to known CIDR ranges.",
            "Sensitive fields in logs are masked before emission.",
            "2FA is mandatory for all admin panel accounts.",
        ],
    ),
    (
        "observability",
        [
            "All services emit structured JSON logs to stdout.",
            "Distributed traces use OpenTelemetry with Jaeger backend.",
            "Prometheus scrapes metrics on /metrics every 15 seconds.",
            "Alerts fire when p99 latency exceeds 500ms for 5 minutes.",
            "SLO burn-rate alerts use multi-window (1h + 6h) thresholds.",
            "Error budget is tracked weekly; 99.9% availability target.",
            "Health probes: /healthz (liveness) and /readyz (readiness).",
            "Spans include db.statement for all SQL queries.",
            "Custom business metrics use counter and histogram types.",
            "Log sampling at 10% for INFO, 100% for WARN/ERROR.",
            "Dashboards are versioned in Git alongside the service code.",
            "On-call rotation uses PagerDuty with 5-minute escalation.",
            "Anomaly detection runs on request-rate time series.",
            "Canary deployments are monitored with automated rollback.",
            "Memory and CPU usage are tracked per-pod, not just per-service.",
            "Database slow query logs are shipped to the central log store.",
            "Correlation IDs are propagated through all async message queues.",
            "Synthetic uptime checks run every minute from 3 regions.",
        ],
    ),
    (
        "deployment",
        [
            "Deployments use blue-green strategy with 5-minute soak period.",
            "Rollbacks are triggered automatically if error rate exceeds 1%.",
            "Helm values files are environment-specific; no manual kubectl.",
            "CI pipeline runs lint, tests, and security scan before push.",
            "Docker images are multi-stage; final stage is distroless.",
            "Kubernetes resource requests must be set; limits are optional.",
            "ConfigMaps hold non-secret configuration; Secrets hold credentials.",
            "Pod disruption budgets ensure at least 1 replica stays up.",
            "Deployments use `maxSurge: 1, maxUnavailable: 0` rolling update.",
            "Service mesh handles circuit breaking, not application code.",
            "Staging environment mirrors production topology exactly.",
            "Release tags follow semver; no 'latest' tag in production.",
            "GitOps via ArgoCD; no direct kubectl apply in production.",
            "Namespace per team; cross-namespace traffic requires explicit policy.",
            "Ingress rules are managed declaratively via annotations.",
            "Persistent volumes use ReadWriteOnce for stateful workloads.",
            "CronJobs use concurrencyPolicy: Forbid to prevent overlapping runs.",
            "Node affinity spreads replicas across availability zones.",
        ],
    ),
    (
        "auth",
        [
            "OAuth2 authorization code flow with PKCE for browser clients.",
            "Client credentials flow for machine-to-machine authentication.",
            "Token introspection endpoint validates third-party tokens.",
            "Refresh token rotation is enabled; old tokens are immediately revoked.",
            "Session cookies use SameSite=Strict and Secure flags.",
            "Device authorization flow for CLI tools and IoT devices.",
            "User impersonation requires separate admin token scope.",
            "Token claims include tenant ID for multi-tenant authorization.",
            "OIDC discovery document served at /.well-known/openid-configuration.",
            "Logout invalidates all active sessions across devices.",
            "Brute-force protection: account locked after 10 failed attempts.",
            "API keys are hashed before storage; prefix shown in UI.",
            "Scopes follow least-privilege: read and write are separate.",
            "Service accounts have no interactive login capability.",
            "SSO via SAML 2.0 for enterprise customers.",
            "Passwordless email login supported via magic links.",
            "Role assignments are audited and reviewed quarterly.",
            "Public endpoints are explicitly allowlisted in the gateway config.",
        ],
    ),
    (
        "caching",
        [
            "Redis 7 is the primary cache; TTL is always set explicitly.",
            "Cache keys include API version to avoid stale responses after deploys.",
            "Write-through cache pattern for user profile data.",
            "Cache aside pattern for expensive aggregation queries.",
            "Distributed locks use Redlock algorithm with 3 Redis nodes.",
            "Hot cache items are pre-warmed on deployment via a warmup job.",
            "Cache hit rate target is 85%; alerts fire below 70%.",
            "Large objects (>1MB) are not cached; use object storage instead.",
            "Cache eviction policy is allkeys-lru.",
            "Read-through caching is handled in a shared repository layer.",
            "Circuit breaker on cache: fall through to DB on Redis timeout.",
            "Stampede protection via probabilistic early expiration.",
            "Session data uses separate Redis instance from application cache.",
            "Cache invalidation is event-driven via a pub/sub channel.",
            "Per-tenant cache namespacing prevents cross-tenant data leaks.",
            "Batch cache population reduces DB load during peak periods.",
            "CDN caches static assets with 1-year max-age headers.",
            "Vary header is set on responses that differ by Accept-Language.",
        ],
    ),
    (
        "messaging",
        [
            "Kafka is used for event streaming between bounded contexts.",
            "Events are schema-validated with Avro + Confluent Schema Registry.",
            "Dead letter queues capture failed messages after 3 retries.",
            "Consumers use manual offset commit after successful processing.",
            "Outbox pattern ensures atomicity between DB write and event publish.",
            "Idempotency keys prevent duplicate processing of retried messages.",
            "Message ordering is guaranteed within a partition key.",
            "Consumer lag is tracked; alert fires when lag exceeds 10k messages.",
            "RabbitMQ handles task queues for short-lived background jobs.",
            "Poison messages are quarantined and flagged for manual review.",
            "Fanout exchange delivers audit events to multiple consumers.",
            "Message TTL of 24 hours prevents unbounded queue growth.",
            "Priority queues used for user-facing vs background workloads.",
            "AMQP confirms are enabled for all critical message publishing.",
            "Exactly-once semantics achieved via transactional outbox.",
            "Schema evolution follows backward-compatible rules.",
            "Consumer groups are named by service, not by deployment instance.",
            "Message payload includes correlation ID for distributed tracing.",
        ],
    ),
    (
        "storage",
        [
            "PostgreSQL 16 is the primary relational database.",
            "All database migrations are reversible and tested in CI.",
            "Connection pool size is set to max_connections / num_replicas.",
            "Read replicas serve analytics queries; writes go to primary.",
            "Table partitioning by created_at for time-series data.",
            "Partial indexes used for frequently-queried filtered subsets.",
            "VACUUM ANALYZE runs on high-churn tables nightly.",
            "Long-running transactions are killed after 30 seconds.",
            "Foreign keys are enforced in the database, not just in code.",
            "JSONB columns used for semi-structured data with GIN indexes.",
            "Row-level security enforces tenant isolation in PostgreSQL.",
            "Object storage (S3-compatible) for blobs; paths stored in DB.",
            "Database backups tested monthly with a full restore drill.",
            "Point-in-time recovery window is 7 days.",
            "Timescale extension for time-series metrics storage.",
            "Logical replication streams changes to the analytics warehouse.",
            "Database schema is owned by the application service, not shared.",
            "Statement timeout of 5s on OLTP queries prevents runaway scans.",
        ],
    ),
    (
        "testing",
        [
            "Unit tests mock all external dependencies.",
            "Integration tests run against real PostgreSQL and Redis in Docker.",
            "Contract tests verify API compatibility between services.",
            "Load tests run weekly against a staging environment.",
            "Mutation testing scores must exceed 60% for critical modules.",
            "Property-based testing used for serialization round-trips.",
            "Test fixtures are stored in dedicated factory classes.",
            "Database tests use transactions that are rolled back after each test.",
            "Snapshot tests catch unexpected changes to API response shapes.",
            "End-to-end tests run on every merge to main.",
            "Flaky test rate is tracked; any test failing >2% is quarantined.",
            "Test coverage report is generated but not used as a gate.",
            "Security regression tests run OWASP ZAP scanner on staging.",
            "Performance tests assert p99 latency SLOs as pass/fail.",
            "Chaos engineering experiments run monthly in staging.",
            "API backward compatibility is validated with Pact broker.",
            "Test data is generated programmatically, not loaded from fixtures.",
            "Browser tests use Playwright with visual diff comparison.",
        ],
    ),
    (
        "ci_cd",
        [
            "GitHub Actions runs CI on every pull request.",
            "Build artifacts are cached by dependency hash.",
            "Docker layer caching reduces average build time by 60%.",
            "Branch protection requires 1 approval and passing CI.",
            "Automated changelog generation from conventional commits.",
            "Dependency updates are automated via Renovate.",
            "Secret scanning runs on every commit via pre-commit hooks.",
            "Release pipeline requires manual approval for production.",
            "Smoke tests run immediately after each production deployment.",
            "Feature flags decouple deployment from feature release.",
            "PR size limit enforced: >400 lines triggers a warning.",
            "Parallel test execution across 4 shards reduces CI time.",
            "Preview environments created automatically for each PR.",
            "SBOM is generated and signed for each release.",
            "Deployment frequency and DORA metrics tracked in Datadog.",
            "Trunk-based development; feature branches live < 24 hours.",
            "Automated rollback triggered on deployment health check failure.",
            "Code owners file enforces review routing by module.",
        ],
    ),
    (
        "networking",
        [
            "All external traffic terminates TLS at the load balancer.",
            "Internal service communication uses HTTP/2.",
            "Timeouts are set at every network boundary: connect, read, write.",
            "Retry logic uses exponential backoff with jitter.",
            "Circuit breaker opens after 5 consecutive 5xx responses.",
            "Service discovery via Kubernetes DNS, not hardcoded IPs.",
            "Egress traffic is proxied through a controlled gateway.",
            "DNS TTL of 30 seconds for service endpoints.",
            "TCP keepalive enabled to detect stale connections early.",
            "Network policy default-deny with explicit allowlist.",
            "IPv6 support enabled on all services and load balancers.",
            "gRPC health checking protocol used for backend probes.",
            "Websocket connections have a 5-minute idle timeout.",
            "API gateway enforces request body size limit of 10MB.",
            "Connection draining waits 30 seconds before pod termination.",
            "Bandwidth throttling applied to bulk export endpoints.",
            "IP allowlist for administrative endpoints.",
            "HSTS preloading enabled with max-age of 1 year.",
        ],
    ),
]


def _build_load_facts() -> list[str]:
    """Build 200 semantically distinct facts across 10 domains.

    Each domain provides 18 facts = 180 total. We fill to 200 by cycling
    through the base USER_PROFILE_FACTS to reach exactly 200 entries.
    """
    facts: list[str] = []
    for _domain, domain_facts in _LOAD_DOMAINS:
        facts.extend(domain_facts)
    # Fill to 200 with profile facts (they are semantically different from domains)
    idx = 0
    while len(facts) < 200:
        facts.append(USER_PROFILE_FACTS[idx % len(USER_PROFILE_FACTS)])
        idx += 1
    return facts[:200]


LOAD_FACTS: list[str] = _build_load_facts()

LOAD_QUERIES: list[str] = [
    "What programming language is preferred?",
    "How are secrets managed?",
    "What database is used?",
    "What is the deployment platform?",
    "What are the debugging guidelines?",
    "How is CI/CD handled?",
    "What are the working hours?",
    "Who is the manager?",
    "What testing framework is used?",
    "What branching strategy is followed?",
]


# ---------------------------------------------------------------------------
# Typed accessors
# ---------------------------------------------------------------------------


@dataclass
class DeduplicationFixture:
    paraphrases: list[str]
    distinct_rules: list[str]
    canonical_rule: str


@dataclass
class ProfileFixture:
    raw_facts: list[str]
    queries: list[str]


PROFILE = ProfileFixture(raw_facts=USER_PROFILE_FACTS, queries=PROFILE_QUERIES)
DEDUP = DeduplicationFixture(
    paraphrases=DEDUP_PARAPHRASES,
    distinct_rules=DEDUP_DISTINCT_RULES,
    canonical_rule=DEDUP_CANONICAL_RULE,
)

# ---------------------------------------------------------------------------
# S7 — Temporal Consolidation
# ---------------------------------------------------------------------------
# 5 topics × 5 temporal versions = 25 facts, inserted interleaved (v1 of all
# topics first, then v2, …, v5) to stress ordering assumptions in backends.
# Each topic evolves from an initial state to a "latest" (v5) state.

CONSOLIDATION_FACTS: list[str] = [
    # Round 1 (v1 of each topic)
    "User edits code in vim with a minimal vimrc.",
    "User is the sole engineer on the project.",
    "Production deployments happen once a month after sprint review.",
    "User uses SQLite for data storage during prototyping.",
    "User merges their own pull requests without external review.",
    # Round 2 (v2)
    "User migrated to neovim and uses lazy.nvim for plugins.",
    "User hired one junior engineer; team is now 2 people.",
    "Team moved to weekly releases on every Friday.",
    "User migrated to PostgreSQL for the production environment.",
    "Team adopted peer review: one approval required before merge.",
    # Round 3 (v3)
    "User switched to Cursor IDE for AI-assisted pair programming.",
    "User's team grew to 5 engineers after seed funding.",
    "Deployment frequency increased to daily releases after CI improvements.",
    "User upgraded the database cluster to PostgreSQL 14.",
    "Code review policy updated to require two approvals.",
    # Round 4 (v4)
    "User uses Cursor with vim keybindings mode enabled.",
    "User leads a team of 8 engineers.",
    "Team ships multiple times per day to production.",
    "Production database runs on PostgreSQL 16.",
    "Senior engineer approval required for changes to core modules.",
    # Round 5 (v5 — latest)
    "User's primary editor is Cursor with GitHub Copilot integration enabled.",
    "User manages a 10-person engineering team across two time zones.",
    "Every merged PR triggers automatic deployment via continuous delivery pipeline.",
    "User's stack uses PostgreSQL 16 with the TimescaleDB extension for time-series data.",
    "PR policy: two approvals, passing CI, and an automated security scan are required.",
]

CONSOLIDATION_QUERIES: list[str] = [
    "What editor or IDE does the user currently use?",
    "How large is the user's engineering team?",
    "How often does the team deploy to production?",
    "What database technology does the user currently use?",
    "What is the team's code review and PR merge policy?",
]

# Keywords from the v5 (latest) fact for each query — used for latest_recall.
# All lowercase for case-insensitive matching.
# The v5 (latest) fact for each topic — ground truth for semantic_latest_recall.
# These are CONSOLIDATION_FACTS[20:25] (last round), listed in query order.
CONSOLIDATION_LATEST_FACTS: list[str] = [
    "User's primary editor is Cursor with GitHub Copilot integration enabled.",
    "User manages a 10-person engineering team across two time zones.",
    "Every merged PR triggers automatic deployment via continuous delivery pipeline.",
    "User's stack uses PostgreSQL 16 with the TimescaleDB extension for time-series data.",
    "PR policy: two approvals, passing CI, and an automated security scan are required.",
]

N_CONSOLIDATION_TOPICS = 5
N_CONSOLIDATION_VERSIONS = 5  # versions per topic
N_CONSOLIDATION_FACTS = N_CONSOLIDATION_TOPICS * N_CONSOLIDATION_VERSIONS  # 25


@dataclass
class ConsolidationFixture:
    facts: list[str]
    queries: list[str]
    latest_facts: list[str]  # v5 ground-truth fact per query, for semantic recall
    n_topics: int
    n_versions: int


CONSOLIDATION = ConsolidationFixture(
    facts=CONSOLIDATION_FACTS,
    queries=CONSOLIDATION_QUERIES,
    latest_facts=CONSOLIDATION_LATEST_FACTS,
    n_topics=N_CONSOLIDATION_TOPICS,
    n_versions=N_CONSOLIDATION_VERSIONS,
)

# ===========================================================================
# Bilingual bundle infrastructure
# ===========================================================================
# Two independent personas used for EN and RU benchmark runs:
#   EN — Alex Chen, Staff Data Engineer @ FinServe Capital, San Francisco
#   RU — Максим Петров, Senior Backend Engineer @ Яндекс, Москва
# Each bundle provides all fixture data needed for S1–S5 + S7.
# ---------------------------------------------------------------------------


@dataclass
class AvoidRepeatsFixture:
    titles: list[str]
    queries: list[str]
    expected_seen: dict[str, list[str]]


@dataclass
class FeedbackFixture:
    history: list[tuple[str, str]]
    queries: list[str]
    expected_rules: dict[str, list[str]]
    expected_keywords: dict[str, list[str]]


@dataclass
class LanguageBundle:
    language: str  # "en" | "ru"
    profile: ProfileFixture
    avoid_repeats: AvoidRepeatsFixture
    feedback: FeedbackFixture
    dedup: DeduplicationFixture
    consolidation: ConsolidationFixture


@dataclass
class MultiAgentFixture:
    """Fixture data for multi-agent benchmark scenarios (S8–S15).

    Contains four semantically distinct fact sets (DevOps / Python / Frontend / Data)
    for isolation testing, shared pool facts distributed across three publishers,
    and structured entity+attribute data for MESI CAS (4 versions) and lazy-sync
    testing with three independent sync consumers.
    """

    # S8 — private namespace isolation (4 agents × 4 domains)
    agent_a_facts: list[str]  # DevOps / infra domain
    agent_b_facts: list[str]  # Python / coding domain
    agent_c_facts: list[str]  # Frontend / UI domain
    agent_d_facts: list[str]  # Data / ML domain
    agent_a_queries: list[str]
    agent_b_queries: list[str]
    agent_c_queries: list[str]
    agent_d_queries: list[str]

    # S9 — shared pool publish + recall (3 publishers, 1 querier)
    # pool_facts[0:2] → agent_a, [2:4] → agent_b, [4:6] → agent_c
    pool_facts: list[str]
    pool_queries: list[tuple[str, str]]  # (query_text, expected_keyword_in_result)

    # S10 — MESI entity CAS (4 agents publish same entity+attribute sequentially)
    cas_entity: str
    cas_attribute: str
    cas_fact_v1: str  # agent_a publishes first
    cas_fact_v2: str  # agent_b supersedes
    cas_fact_v3: str  # agent_c supersedes
    cas_fact_v4: str  # agent_d supersedes — final winner
    cas_query: str
    cas_v2_keyword: str  # present only in v2
    cas_v4_keyword: str  # present only in v4 — used to confirm v4 won

    # S11 — MESI lazy sync (1 publisher, 3 independent sync consumers)
    sync_initial_facts: list[str]  # initial pool population (5 facts)
    sync_update_entity: str
    sync_update_attribute: str
    sync_fact_v1: str  # initial version of the updatable fact
    sync_fact_v2: str  # updated version (triggers dirty flag for all consumers)
    sync_v2_keyword: str  # word present only in v2

    # S14 — trust recovery (agent_a publishes corrected values after trust floor)
    agent_a_corrected_values: list[str]  # 9 corrected facts matching _SLOTS order

    # S16 — knowledge relay (3 rounds: A → B builds on A → C builds on B)
    relay_infra_facts: list[str]  # agent_a layer: argocd, grafana, redis, istio, helm
    relay_api_facts: list[str]  # agent_b layer: references A's service names
    relay_frontend_facts: list[str]  # agent_c layer: references B's API patterns
    relay_queries: list[tuple[str, str]]  # (query, expected_keyword) × 3

    # S17 — self-correction (agent_a detects supersession via sync_dirty, self-corrects)
    self_corr_entity: str
    self_corr_attribute: str
    self_corr_v1: str  # A's wrong initial value
    self_corr_v2: str  # B's correction (supersedes A)
    self_corr_v3: str  # A's self-corrected value (after sync_dirty)
    self_corr_v3_keyword: str  # keyword present only in v3
    self_corr_query: str

    # S18 — trust calibration (10 rounds: reliable agent_a vs unreliable agent_b)
    trust_calib_n_rounds: int
    trust_calib_reliable_facts: list[str]  # one unique fact per round for agent_a
    trust_calib_unreliable_entity_tpl: str  # format with {i} → entity name per round
    trust_calib_queries: list[tuple[str, str]]  # D's final queries (query, keyword)

    # S19 — incident reconstruction (3-phase: alert → investigation → root cause)
    incident_alert_fact: str
    incident_deploy_fact: str
    incident_migration_fact: str
    incident_alert_keyword: str
    incident_deploy_keyword: str
    incident_migration_keyword: str
    incident_queries: list[tuple[str, str]]  # (query, keyword) × 3

    # S20 — belief revision (5 rounds: contradiction → authoritative → consensus)
    belief_entity: str
    belief_attribute: str
    belief_v_a: str  # agent_a's initial claim
    belief_v_b: str  # agent_b's initial claim (conflicts with A)
    belief_v_c: str  # agent_c's authoritative version (round 2)
    belief_v_final: str  # agent_c's round-5 correction
    belief_keyword_round4: str  # keyword in v_c
    belief_keyword_round5: str  # keyword in v_final
    belief_query: str

    # S19 upgrade — red herrings (noise facts that should NOT appear in top results)
    incident_red_herring_facts: list[str]  # 5 noise facts injected before phase 1
    incident_relevant_keywords: list[str]  # keywords present ONLY in relevant facts

    # S21 — distributed product knowledge assembly (5 specialists + 1 querier)
    assembly_technical_facts: list[str]  # agent_a: API, architecture, performance
    assembly_business_facts: list[str]  # agent_b: pricing, tiers, contracts
    assembly_ops_facts: list[str]  # agent_c: SLAs, regions, maintenance
    assembly_historical_facts: list[str]  # agent_d: legacy, deprecations, evolution
    assembly_integration_facts: list[str]  # agent_e: SDKs, connectors, OAuth
    assembly_queries: list[tuple[str, list[str]]]  # (query, [relevant_kws from ≥2 agents])

    # S22 — temporal staleness detection (3 rounds of updates, query must find LATEST)
    staleness_v1_facts: list[tuple[str, str, str, str]]  # (entity, attr, content, old_kw)
    staleness_v2_updates: list[tuple[str, str, str, str]]  # (entity, attr, content, new_kw)
    staleness_v3_updates: list[tuple[str, str, str, str]]  # (entity, attr, content, latest_kw)
    staleness_queries: list[tuple[str, str, str]]  # (query, latest_kw, stale_kw_to_avoid)

    # S23 — adversarial noise injection (trust-weighted suppression)
    adversarial_slots: list[tuple[str, str]]  # (entity, attr) × 5 CAS slots
    adversarial_correct_values: list[tuple[str, str, str]]  # (entity, attr, content) × 5
    adversarial_wrong_values: list[tuple[str, str, str]]  # adversary's wrong versions
    adversarial_freestanding_correct: list[str]  # reliable agents' free-standing facts
    adversarial_freestanding_wrong: list[str]  # adversary's free-standing wrong facts
    adversarial_queries: list[tuple[str, str, str]]  # (query, correct_kw, wrong_kw)

    # S24 — multi-round onboarding (knowledge absorption into private KB)
    onboarding_pool_facts: list[str]  # facts already in pool from team agents
    onboarding_round1_queries: list[tuple[str, str]]  # round 1: empty KB, pool queries
    onboarding_absorb_facts: list[str]  # facts D inserts after round 1
    onboarding_round3_kb_queries: list[tuple[str, str]]  # queries D can answer from KB

    # S25 — knowledge conflict resolution at scale (10 slots, 4 wrong agents, 1 canonical)
    conflict_slots: list[tuple[str, str]]  # 10 (entity, attribute) pairs
    conflict_agent_a_values: list[str]  # agent_a's wrong values
    conflict_agent_b_values: list[str]  # agent_b's wrong values
    conflict_agent_c_values: list[str]  # agent_c's wrong values
    conflict_agent_d_values: list[str]  # agent_d's wrong values
    conflict_canonical_values: list[str]  # agent_e's correct canonical values
    conflict_queries: list[tuple[str, str]]  # (query, canonical_keyword)

    # S8v2 — multi-team knowledge commons (replaces S8 isolation)
    commons_platform_facts: list[str]  # agent_a: 5 infra + 2 shared-API-limit facts
    commons_backend_facts: list[str]  # agent_b: 5 service + 2 API-limit facts (different angle)
    commons_frontend_facts: list[str]  # agent_c: 5 UI + 2 deployment facts
    commons_data_facts: list[str]  # agent_d: 5 ML + 2 monitoring facts
    commons_overlap_queries: list[tuple[str, list[str]]]  # expect ≥2 agent sources
    commons_exclusive_queries: list[tuple[str, str]]  # (query, exclusive_keyword)

    # S9v2 — competing documentation sources (replaces S9)
    competing_facts_a: list[str]  # agent_a: 4 facts (some outdated)
    competing_facts_b: list[str]  # agent_b: 4 facts (corrections + new)
    competing_facts_c: list[str]  # agent_c: 4 facts (corrections + new)
    competing_slot_entity: str  # entity for CAS supersession test
    competing_slot_attribute: str
    competing_slot_v1: str  # agent_a's outdated claim (superseded by self)
    competing_slot_v2: str  # agent_a's corrected version
    competing_queries: list[tuple[str, str, str]]  # (query, correct_kw, wrong_kw_to_avoid)

    # S11v2 — progressive knowledge catchup (replaces S11 lazy sync)
    catchup_initial_facts: list[str]  # 8 facts from agent_a at T=0
    catchup_agent_b_facts: list[str]  # 5 facts agent_b publishes after first sync
    catchup_agent_c_facts: list[str]  # 6 facts agent_c publishes concurrently
    catchup_agent_a_updates: list[tuple[str, str, str]]  # 3 CAS updates while B offline
    catchup_agent_c_extra_facts: list[str]  # 4 more facts from C while B offline
    catchup_agent_d_urgent_facts: list[str]  # 3 urgent incident facts while B offline
    catchup_agent_b_response_facts: list[str]  # 2 facts B publishes based on delta
    catchup_b_response_queries: list[tuple[str, str]]  # (query, expected_keyword)

    # S12v2 — priority triage under load (replaces S12)
    triage_critical_facts: list[str]  # 8 critical facts (2 per agent, importance=0.85)
    triage_routine_facts: list[str]  # 8 routine facts (2 per agent, importance=0.45)
    triage_noise_facts: list[str]  # 8 noise facts (2 per agent, importance=0.15)
    triage_incident_queries: list[tuple[str, str]]  # queries during incident (critical only)
    triage_routine_queries: list[tuple[str, str]]  # queries after resolution

    # S15v2 — cross-team signal contamination (replaces S15)
    contamination_devops_facts: list[str]  # 6 devops facts (2 with shared terms)
    contamination_backend_facts: list[str]  # 6 backend facts (2 with shared terms)
    contamination_data_facts: list[str]  # 6 data facts (2 with shared terms)
    contamination_channel_queries: list[tuple[str, str, str]]  # (query, channel, excl_kw)
    contamination_shared_term_queries: list[tuple[str, str, str]]  # (query, channel, excl_kw)


# ---------------------------------------------------------------------------
# Multi-agent fixture data (EN)
# ---------------------------------------------------------------------------

MULTI_AGENT_FIXTURE: MultiAgentFixture = MultiAgentFixture(
    # S8: Four agents with distinct domains — Networking, Go/backend, Mobile, Analytics.
    # Queries are unambiguous: only the owning agent's facts can answer them.
    agent_a_facts=[
        "All traffic is routed through Envoy proxy with automatic circuit breaking at the edge.",
        "Consul service mesh handles service-to-service authentication using mTLS certificates.",
        "Network policies in Calico restrict pod-to-pod traffic to explicitly allowed namespaces.",
        "DNS resolution uses CoreDNS with a local cache TTL of 30 seconds per pod.",
        "WireGuard VPN tunnels connect the three regional data centers with sub-2ms latency.",
    ],
    agent_b_facts=[
        "All Go services must pass golangci-lint with the strictest preset before merge.",
        "Table-driven tests are mandatory for all public functions in Go packages.",
        "Context propagation must use context.Context as the first parameter in every RPC handler.",
        "Protobuf is the only allowed serialization format for inter-service communication.",
        "Error wrapping must use fmt.Errorf with %w verb; sentinel errors are banned.",
    ],
    agent_c_facts=[
        "SwiftUI is the required framework for all new iOS screens since Q1 2025.",
        "Kotlin Multiplatform shares business logic between Android and iOS builds.",
        "Firebase Crashlytics tracks crash-free rates for every mobile release.",
        "Fastlane automates App Store and Google Play submission pipelines.",
        "Detox handles end-to-end mobile testing on real devices in the CI farm.",
    ],
    agent_d_facts=[
        "All analytics events are ingested through Apache Flink with exactly-once semantics.",
        "ClickHouse serves as the OLAP engine for all real-time dashboard queries.",
        "dbt manages the transformation layer with daily model refreshes at 03:00 UTC.",
        "Data contracts are enforced by Schemata registry before any schema evolution.",
        "Looker is the self-service BI tool available to all non-engineering teams.",
    ],
    agent_a_queries=[
        "How is traffic routed at the edge of the network?",
        "What VPN solution connects the regional data centers?",
    ],
    agent_b_queries=[
        "What linting tool is required for Go code?",
        "What serialization format is used between services?",
    ],
    agent_c_queries=[
        "What framework is required for new iOS screens?",
        "How is end-to-end mobile testing performed?",
    ],
    agent_d_queries=[
        "What OLAP engine powers the real-time dashboards?",
        "What tool manages the analytics transformation layer?",
    ],
    # S9: pool_facts are distributed across 3 publishing agents.
    # pool_facts[0:2] → agent_a, [2:4] → agent_b, [4:6] → agent_c.
    # agent_d (empty private KB) queries the pool and must find all 3 answers.
    pool_facts=[
        "The incident commander rotation follows a 14-day cycle; handover happens every other Wednesday at 10:00.",
        "Production alerts are graded Sev1–Sev4; Sev1 requires acknowledgement within 3 minutes.",
        "All gRPC proto changes must maintain wire compatibility for at least three major versions.",
        "Index migrations on the primary cluster run behind a feature gate before global activation.",
        "SAST analysis (Semgrep) runs on every merge request targeting the release branch.",
        "Every commit must be signed with a verified GPG key before it can land on the trunk.",
    ],
    pool_queries=[
        ("How fast must Sev1 alerts be acknowledged?", "3 minutes"),
        ("How long must gRPC protos stay wire-compatible?", "three major versions"),
        ("When does SAST analysis run?", "merge request"),
    ],
    # S10: MESI CAS — 4 agents publish conflicting salary facts sequentially.
    # Each publish supersedes the previous; only v4 should remain active.
    cas_entity="Mia Torres",
    cas_attribute="annual_salary",
    cas_fact_v1="Mia Torres earns $88,000 per year as a site reliability engineer.",
    cas_fact_v2="Mia Torres's salary was adjusted to $105,000 after the mid-cycle calibration.",
    cas_fact_v3="Mia Torres's salary was raised to $120,000 following the Q3 talent review.",
    cas_fact_v4="Mia Torres's salary is now $145,000 following promotion to Principal SRE.",
    cas_query="Mia Torres salary compensation",
    cas_v2_keyword="105",
    cas_v4_keyword="Principal",  # present only in v4
    # S11: MESI lazy sync — initial pool published by agent_a, then 1 fact updated.
    # Agents b, c, d each call sync_dirty() independently; all should see the same delta.
    sync_initial_facts=[
        "The release train has three gates: integration, staging, and general availability.",
        "All microservices emit structured traces via OpenTelemetry to the Tempo backend.",
        "Experiment toggles are controlled centrally through the Unleash feature server.",
        "Cache warming runs a pre-heat job every 90 seconds after a cold restart.",
        "Endpoint resolution uses Consul DNS with health-check-aware failover.",
    ],
    sync_update_entity="release_train",
    sync_update_attribute="gate_count",
    sync_fact_v1="The release train has three gates: integration, staging, and general availability.",
    sync_fact_v2="The release train was expanded to four gates: integration, staging, shadow, and general availability.",
    sync_v2_keyword="shadow",
    # S14 — trust recovery: agent_a publishes corrected versions of all 9 slots.
    agent_a_corrected_values=[
        "Mia Torres's annual salary is $120,000 (confirmed by People Ops).",
        "Mia Torres's job title is Senior SRE (People-Ops-verified).",
        "Mia Torres now works in the Toronto office (People-Ops-verified).",
        "Liam Chen's annual salary is $145,000 (confirmed by People Ops).",
        "Liam Chen's job title is Principal Engineer (People-Ops-verified).",
        "Liam Chen now works in the Seoul office (People-Ops-verified).",
        "Liam Chen is on the Platform Reliability team (People-Ops-verified).",
        "Liam Chen joined the company in June 2019 (People-Ops-verified).",
        "Liam Chen is the tech lead for Platform Reliability (People-Ops-verified).",
    ],
    # S16 — knowledge relay facts.
    relay_infra_facts=[
        "All workloads are deployed via FluxCD with automated reconciliation every 60 seconds.",
        "Datadog collects metrics and traces from every Nomad allocation in the cluster.",
        "Memcached is the shared look-aside cache for all backend query results.",
        "Linkerd service mesh provides transparent mTLS and retries between pods.",
        "Kustomize overlays are pinned per environment with a signed commit gate.",
    ],
    relay_api_facts=[
        "The /readyz endpoint on every FluxCD-managed workload returns liveness status.",
        "Memcached stores API throttle counters with a per-token sliding window.",
        "The API gateway routes traffic to services via Nomad service discovery.",
        "GraphQL schemas are auto-generated from Go structs and published to the portal.",
        "All API responses carry X-Trace-ID correlated with Datadog APM traces.",
    ],
    relay_frontend_facts=[
        "The Vue app calls the API gateway using the GraphQL-generated TypeScript client.",
        "Server-sent event connections to the API gateway are load-balanced via Nomad.",
        "Frontend experiment flags are cached in Memcached with a 3-minute TTL.",
        "Datadog Real User Monitoring tracks Core Web Vitals for every page load.",
        "The component library is rebuilt whenever a new Kustomize overlay is promoted.",
    ],
    relay_queries=[
        ("What tool manages workload deployments?", "fluxcd"),
        ("How is API throttling implemented?", "memcached"),
        ("How does the frontend consume backend APIs?", "graphql"),
    ],
    # S17 — self-correction data.
    self_corr_entity="release_train",
    self_corr_attribute="rollback_strategy",
    self_corr_v1="Rollback is manual: the on-call engineer triggers a Nomad job revert from the CLI.",
    self_corr_v2="Rollback is automated via FluxCD revert with the --prune flag enabled.",
    self_corr_v3="Rollback is automated via FluxCD revert --prune with a PagerDuty alert fired to the on-call.",
    self_corr_v3_keyword="PagerDuty",
    self_corr_query="How does the system roll back a failed release?",
    # S18 — trust calibration: 10 rounds of reliable (A) vs unreliable (B superseded by C).
    trust_calib_n_rounds=10,
    trust_calib_reliable_facts=[
        "Incident commander rotation follows a 14-day cycle with Wednesday 10:00 handover.",
        "Sev1 alerts require acknowledgement within 3 minutes of the first page.",
        "All gRPC proto changes must maintain wire compatibility for three major versions.",
        "Index migrations run behind a feature gate before global cluster activation.",
        "SAST scanning via Semgrep runs on every merge request to the release branch.",
        "Every commit must be signed with a verified GPG key before landing on trunk.",
        "Service error budgets are tracked with a 28-day rolling window in Datadog.",
        "Experiment toggles are managed via Unleash with per-region overrides.",
        "Playbooks for every Sev1 alert are stored in Notion under /playbooks.",
        "Incident reviews are required for every Sev1 event within 72 hours.",
    ],
    trust_calib_unreliable_entity_tpl="cluster_config_{i}",
    trust_calib_queries=[
        ("What is the incident commander rotation schedule?", "wednesday"),
        ("What are the commit signing requirements?", "gpg"),
    ],
    # S19 — incident reconstruction data.
    incident_alert_fact="OrderService returns gRPC UNAVAILABLE on all methods since 09:47 UTC.",
    incident_deploy_fact="Release R-83 was promoted to OrderService at 09:41 UTC.",
    incident_migration_fact="Schema migration S-19 executed on the OrderService primary cluster at 09:38 UTC.",
    incident_alert_keyword="09:47",
    incident_deploy_keyword="09:41",
    incident_migration_keyword="migration",
    incident_queries=[
        ("What errors is OrderService returning?", "unavailable"),
        ("What changed on OrderService before the incident?", "release"),
        ("Was there schema activity before the outage?", "migration"),
    ],
    # S20 — belief revision data.
    belief_entity="platform_squad",
    belief_attribute="headcount",
    belief_v_a="The platform squad has 4 members.",
    belief_v_b="The platform squad has 9 members.",
    belief_v_c="The platform squad has 8 members (confirmed by People Ops headcount system).",
    belief_v_final="The platform squad has 7 members after one internal transfer.",
    belief_keyword_round4="8",
    belief_keyword_round5="7",
    belief_query="How many people are on the platform squad?",
    # S19 upgrade — red herring noise facts
    incident_red_herring_facts=[
        "InventoryService experienced intermittent cache timeouts at 10:15 UTC due to eviction storms.",
        "OrderService underwent routine certificate rotation at 09:20 UTC as part of weekly ops cycle.",
        "Release R-84 rolled out to ShippingService at 09:35 UTC; no issues reported by the logistics team.",
        "Memcached cluster in us-west-2 showed elevated miss rate at 09:30 UTC (resolved within 3 minutes).",
        "OrderService had a similar gRPC UNAVAILABLE event 5 days ago, which was resolved by restarting the proxy.",
    ],
    incident_relevant_keywords=["09:47", "09:41", "migration"],
    # S21 — distributed product knowledge assembly
    assembly_technical_facts=[
        "Pulse Observability Platform exposes a gRPC API with protobuf schemas and versioned service definitions.",
        "Pulse ingests up to 50,000 spans per second per tenant in the growth tier configuration.",
        "Pulse uses NATS JetStream as the underlying message bus for all real-time trace pipelines.",
        "The Pulse API gateway enforces mTLS authentication on all endpoints with Ed25519 signing.",
        "Pulse supports alert webhooks for real-time incident delivery to external HTTP callbacks.",
    ],
    assembly_business_facts=[
        "Pulse growth tier is $59 per host per month; enterprise pricing starts at $199 per month.",
        "Enterprise tier includes dedicated collectors, a named solutions architect, and SLA guarantees.",
        "Pulse offers a 21-day free trial with full feature access and no payment method required.",
        "Pulse counts active hosts monthly; idle hosts in a billing period are not charged.",
        "Enterprise annual contracts receive a 25% discount compared to monthly billing.",
    ],
    assembly_ops_facts=[
        "Pulse guarantees 99.95% monthly uptime for all paying customers; enterprise gets 99.99%.",
        "Pulse is deployed in GCP us-central1, eu-west4, and asia-northeast1 regions.",
        "Enterprise customers can request a dedicated collector fleet in their preferred GCP region.",
        "SLA violations result in service credits equal to 15x the duration of unplanned downtime.",
        "Scheduled maintenance windows occur on Saturdays between 04:00 and 06:00 UTC.",
    ],
    assembly_historical_facts=[
        "Pulse v1 collector protocol was deprecated in May 2023 and is still reachable but completely unsupported.",
        "Pulse migrated from an Elasticsearch backend to a custom time-series storage engine in 2022.",
        "The legacy alert payload format (v1) was replaced by the OpenTelemetry standard in November 2023.",
        "Pulse originally launched under the brand name 'TraceHawk' before rebranding in 2021.",
        "The Pulse batch export API reached end-of-life in January 2025, replaced by streaming export.",
    ],
    assembly_integration_facts=[
        "Pulse provides official Go and Ruby SDKs published on pkg.go.dev and RubyGems respectively.",
        "Pulse integrates natively with PagerDuty, Opsgenie, and Jira via pre-built connectors.",
        "The Pulse Go SDK uses goroutines for non-blocking I/O and requires Go 1.22 or higher.",
        "Pulse provides an official Pulumi provider for infrastructure-as-code provisioning of workspaces.",
        "Pulse supports OIDC federated identity for secure third-party app integrations.",
    ],
    assembly_queries=[
        ("What is the enterprise tier pricing and what SLA does it include?", ["199", "99.9"]),
        ("Which cloud regions host the Pulse platform?", ["us-central", "eu-west"]),
        ("Is the legacy v1 collector protocol still available?", ["deprecated", "unsupported"]),
        ("How do I integrate Pulse with a Go application?", ["go", "sdk"]),
        ("What happens if Pulse violates its uptime SLA?", ["credits", "15x"]),
    ],
    # S22 — temporal staleness detection
    staleness_v1_facts=[
        (
            "pulse_api",
            "ingestion_limit",
            "Pulse API allows up to 5,000 spans per second per workspace.",
            "5,000",
        ),
        (
            "pulse_billing",
            "monthly_price",
            "Pulse growth tier costs $89 per host per month.",
            "89",
        ),
        (
            "pulse_platform",
            "max_active_alerts",
            "Pulse supports up to 200 active alert rules per workspace.",
            "200",
        ),
        (
            "pulse_platform",
            "supported_regions",
            "Pulse is available in US-Central and EU-West regions.",
            "eu-west",
        ),
        (
            "pulse_sla",
            "query_latency",
            "Pulse guarantees P99 query latency below 300 ms under normal load.",
            "300 ms",
        ),
    ],
    staleness_v2_updates=[
        (
            "pulse_api",
            "ingestion_limit",
            "Pulse API ingestion limit upgraded to 20,000 spans per second after collector expansion.",
            "20,000",
        ),
        (
            "pulse_billing",
            "monthly_price",
            "Pulse growth tier reduced to $59 per host per month (Q1 promotion).",
            "59",
        ),
        (
            "pulse_platform",
            "max_active_alerts",
            "Pulse now supports up to 500 active alert rules after alerting engine upgrade.",
            "500",
        ),
    ],
    staleness_v3_updates=[
        (
            "pulse_api",
            "ingestion_limit",
            "Pulse API ingestion limit is now 50,000 spans per second following the Q2 infrastructure overhaul.",
            "50,000",
        ),
    ],
    staleness_queries=[
        ("What is the current ingestion limit for Pulse?", "50,000", "5,000"),
        ("How much does Pulse cost per month?", "59", "89"),
        ("How many active alert rules can Pulse support?", "500", "200"),
        ("Which regions are available for Pulse?", "eu-west", ""),
        ("What is the Pulse query latency SLA guarantee?", "300 ms", ""),
    ],
    # S23 — adversarial noise injection
    adversarial_slots=[
        ("pulse_config", "collector_version"),
        ("pulse_config", "auth_method"),
        ("pulse_config", "ingestion_limit"),
        ("pulse_config", "trace_retention"),
        ("pulse_config", "tls_provider"),
    ],
    adversarial_correct_values=[
        (
            "pulse_config",
            "collector_version",
            "Pulse current collector version is v4.1 with full backward compatibility.",
        ),
        (
            "pulse_config",
            "auth_method",
            "Pulse uses mTLS with Ed25519 certificates for all collector authentication.",
        ),
        (
            "pulse_config",
            "ingestion_limit",
            "Pulse enforces an ingestion limit of 50,000 spans per second per workspace.",
        ),
        (
            "pulse_config",
            "trace_retention",
            "Pulse retains trace data for 90 days on growth tier.",
        ),
        (
            "pulse_config",
            "tls_provider",
            "Pulse uses ZeroSSL certificates with automatic 60-day renewal.",
        ),
    ],
    adversarial_wrong_values=[
        (
            "pulse_config",
            "collector_version",
            "Pulse collector version is v1.0 with no backward compatibility guarantees.",
        ),
        (
            "pulse_config",
            "auth_method",
            "Pulse uses shared API keys transmitted in query parameters without encryption.",
        ),
        (
            "pulse_config",
            "ingestion_limit",
            "Pulse has no ingestion limits and allows unlimited span submission.",
        ),
        (
            "pulse_config",
            "trace_retention",
            "Pulse deletes all trace data after 24 hours automatically.",
        ),
        (
            "pulse_config",
            "tls_provider",
            "Pulse uses self-signed TLS certificates for all collector endpoints.",
        ),
    ],
    adversarial_freestanding_correct=[
        "Pulse platform achieves 99.99% uptime through multi-region active-active deployment.",
        "Pulse encrypts all traces at rest using ChaCha20 and in transit using TLS 1.3.",
        "Pulse provides SOC 2 Type II and HIPAA compliance certifications.",
        "Pulse supports automatic horizontal scaling up to 100,000 spans per second.",
        "Pulse offers 24/7 enterprise support with 10-minute response time SLA.",
    ],
    adversarial_freestanding_wrong=[
        "Pulse platform has 90% uptime and frequent data loss events every month.",
        "Pulse stores all trace data in plaintext without any encryption at rest.",
        "Pulse has no compliance certifications and has failed multiple penetration tests.",
        "Pulse is limited to 500 spans per second and cannot scale horizontally.",
        "Pulse offers email-only support with 7 business day response time.",
    ],
    adversarial_queries=[
        ("What is the Pulse uptime guarantee?", "99.99", "90%"),
        ("How does Pulse handle trace encryption?", "chacha20", "plaintext"),
        ("What compliance certifications does Pulse have?", "soc 2", "failed"),
        ("How many spans per second can Pulse handle?", "100,000", "500"),
        ("What support options does Pulse offer?", "24/7", "7 business"),
    ],
    # S24 — multi-round onboarding (KB absorption)
    onboarding_pool_facts=[
        "Pulse collectors use rolling restart strategy with automatic fallback on readiness probe failure.",
        "The Pulse Go SDK is installed via go get github.com/pulse-obs/sdk-go (requires Go 1.22+).",
        "Enterprise trial runs for 21 days with full feature access and dedicated solutions architect.",
        "Pulse primary deployment region is GCP us-central1 with failover to europe-west4.",
        "Pulse dashboard supports custom Grafana plugin panels for trace visualization.",
        "Akamai WAF provides DDoS protection and edge throttling for the Pulse collector gateway.",
        "Pulse streaming export delivers spans with sub-50ms end-to-end processing latency.",
        "Volume pricing: 15% discount for 100+ hosts, 25% for 500+ hosts.",
        "Pulse gRPC API supports protobuf and JSON transcoding response formats.",
        "Trace lineage tracking is available for Professional and Enterprise tier customers.",
        "Pulse GDPR Data Processing Agreement is downloadable from the workspace compliance settings.",
        "Incident response uses Opsgenie with automatic escalation after 10 minutes.",
        "Trace storage snapshots run every 4 hours with point-in-time recovery for Enterprise.",
        "Pulse Ruby SDK supports Ruby 3.2+ with native Ractor-safe instrumentation hooks.",
        "Pulse Tines integration provides 30 triggers and 8 actions for no-code incident automation.",
    ],
    onboarding_round1_queries=[
        ("How do I install the Pulse Go SDK?", "go get"),
        ("What cloud regions does Pulse use?", "us-central"),
        ("How long is the enterprise trial?", "21 days"),
        ("What visualization tool does the dashboard support?", "grafana"),
        ("How does Pulse handle DDoS protection?", "akamai"),
    ],
    onboarding_absorb_facts=[
        "Install the Pulse Go SDK via go get github.com/pulse-obs/sdk-go (Go 1.22+ required).",
        "Pulse runs primarily in GCP us-central1 with europe-west4 failover.",
        "Enterprise trial is 21 days with full features and solutions architect.",
        "Pulse dashboard uses Grafana plugin panels for custom trace visualization.",
        "Akamai WAF provides DDoS protection for the Pulse collector gateway.",
    ],
    onboarding_round3_kb_queries=[
        ("How do I install the Go SDK for Pulse?", "go get"),
        ("Where is Pulse hosted?", "us-central"),
        ("What is the trial period for enterprise?", "21 days"),
    ],
    # S25 — knowledge conflict resolution at scale
    conflict_slots=[
        ("pulse_config", "collector_version"),
        ("pulse_config", "auth_method"),
        ("pulse_config", "ingestion_limit"),
        ("pulse_config", "trace_retention"),
        ("pulse_config", "tls_provider"),
        ("pulse_config", "max_span_size_kb"),
        ("pulse_config", "session_timeout_min"),
        ("pulse_config", "mfa_required"),
        ("pulse_config", "default_timezone"),
        ("pulse_config", "audit_log_retention"),
    ],
    conflict_agent_a_values=[
        "Pulse collector version is v2.0.",
        "Auth uses static tokens only.",
        "Ingestion limit is 1,000 spans/s.",
        "Traces retained for 14 days.",
        "TLS via DigiCert.",
        "Max span size 64 KB.",
        "Session timeout 15 min.",
        "MFA not required.",
        "Default timezone PST.",
        "Audit logs kept 30 days.",
    ],
    conflict_agent_b_values=[
        "Pulse collector version is v3.0.",
        "Auth uses SAML only.",
        "Ingestion limit is 10,000 spans/s.",
        "Traces retained for 60 days.",
        "TLS via Comodo.",
        "Max span size 128 KB.",
        "Session timeout 30 min.",
        "MFA optional.",
        "Default timezone EST.",
        "Audit logs kept 90 days.",
    ],
    conflict_agent_c_values=[
        "Pulse collector version is v1.8.",
        "Auth uses LDAP.",
        "Ingestion limit is 2,500 spans/s.",
        "Traces retained for 30 days.",
        "TLS via GoDaddy.",
        "Max span size 32 KB.",
        "Session timeout 10 min.",
        "MFA enforced.",
        "Default timezone CET.",
        "Audit logs kept 45 days.",
    ],
    conflict_agent_d_values=[
        "Pulse collector version is v3.5.",
        "Auth uses mTLS.",
        "Ingestion limit is 25,000 spans/s.",
        "Traces retained for 45 days.",
        "TLS via AWS ACM.",
        "Max span size 256 KB.",
        "Session timeout 60 min.",
        "MFA disabled.",
        "Default timezone JST.",
        "Audit logs kept 180 days.",
    ],
    conflict_canonical_values=[
        "Pulse collector is at version v4.1 with full backward compatibility to v3.x.",
        "Authentication uses mTLS with Ed25519 certificates issued per workspace.",
        "Ingestion limit is 50,000 spans per second per workspace.",
        "Trace data is retained for 90 days on growth tier.",
        "TLS certificates are issued by ZeroSSL with 60-day auto-renewal.",
        "Maximum span payload size is 512 KB per span.",
        "Session timeout is 45 minutes with sliding window refresh.",
        "Multi-factor authentication is required for all workspace admins.",
        "Default timezone is UTC for all timestamps and audit entries.",
        "Audit logs are retained for 365 days (1 year) for compliance.",
    ],
    conflict_queries=[
        ("What is the current Pulse collector version?", "v4.1"),
        ("How does Pulse handle authentication?", "mtls"),
        ("What is the ingestion limit?", "50,000"),
        ("How long are traces retained?", "90 days"),
        ("What TLS certificates does Pulse use?", "zerossl"),
        ("What is the max span payload size?", "512 kb"),
        ("How long before a session times out?", "45 minutes"),
        ("Is MFA required on Pulse?", "required"),
        ("What timezone does Pulse use by default?", "utc"),
        ("How long are audit logs retained?", "365"),
    ],
    # S8v2 — multi-team knowledge commons
    commons_platform_facts=[
        "All workloads are deployed via FluxCD with automated reconciliation every 90 seconds.",
        "Datadog collects metrics from all Nomad allocations using StatsD and DogStatsD exporters.",
        "Memcached cluster provides shared look-aside caching with 12 nodes and 8 GB RAM per node.",
        "Linkerd service mesh provides transparent mTLS between all workload pods in the cluster.",
        "Kustomize overlays are pinned per environment with signed commit gates and drift alerts.",
        "Platform team sets the ingress throttle to 2000 req/min using Traefik middleware.",
        "Platform monitors request latency P99 via Datadog dashboards with 150ms alert threshold.",
    ],
    commons_backend_facts=[
        "Backend services are built in Go 1.22 with Chi router for all REST endpoints.",
        "All database queries use pgx driver with connection pooling via PgBouncer.",
        "Backend team manages the identity service using OIDC with Dex.",
        "NATS JetStream is used for inter-service messaging within the backend cluster.",
        "Backend implements circuit breakers using gobreaker for downstream service calls.",
        "Backend team enforces request throttling at 800 req/min per client at the application layer.",
        "Backend measures request latency per endpoint and reports to the central Datadog instance.",
    ],
    commons_frontend_facts=[
        "Vue 3 with TypeScript is the standard for all customer-facing UI components.",
        "Turbopack is the build tool with instant hot module replacement for development.",
        "UnoCSS handles all styling with a custom atomic design token system.",
        "Cypress runs end-to-end browser tests against staging before every promotion.",
        "Histoire documents every shared component in the cross-team design system.",
        "Frontend deploys independently via FluxCD to a separate Nomad namespace.",
        "Frontend team uses Cloudflare Workers for server-side rendering and edge caching.",
    ],
    commons_data_facts=[
        "All ML models are versioned in Weights & Biases with automatic experiment tracking.",
        "Feature engineering runs on Apache Flink with Iceberg tables for ACID transactions.",
        "Model inference is served via vLLM on GPU-enabled Nomad nodes.",
        "Soda Core validates data quality at every pipeline stage.",
        "Neptune.ai tracks hyperparameter sweeps for all model training jobs.",
        "Data pipelines publish metrics to Datadog using the same DogStatsD exporter stack.",
        "Data team monitors pipeline latency through custom Datadog dashboards with 5-min granularity.",
    ],
    commons_overlap_queries=[
        ("What are the request throttle limits in our system?", ["2000", "800"]),
        ("How do teams deploy to production?", ["fluxcd", "reconciliation"]),
        ("How is request latency monitored?", ["datadog", "latency"]),
    ],
    commons_exclusive_queries=[
        ("Who manages the OIDC identity service?", "dex"),
        ("What ML model serving infrastructure do we use?", "vllm"),
    ],
    # S9v2 — competing documentation sources
    competing_facts_a=[
        "Sev1 alert SLA requires acknowledgement within 10 minutes of the initial page.",
        "The collector API supports both gRPC and REST endpoints for backward compatibility.",
        "Incident commander rotation covers weekdays during business hours only.",
        "Collector config changes require a full restart with a 60-second drain period.",
    ],
    competing_facts_b=[
        "Sev1 alert SLA was tightened to 4 minutes acknowledgement in Q1 2025 policy update.",
        "REST collector endpoint introduced throttling at 5000 spans/s; gRPC remains unlimited.",
        "Incident commander coverage was expanded to 24/7 after the February 2025 weekend outage.",
        "Collector config changes now apply via hot reload with zero-downtime propagation.",
    ],
    competing_facts_c=[
        "All config changes require two approvers before promotion (updated from one approver).",
        "REST collector endpoint has been officially deprecated since April 2025; migration guide in Notion.",
        "SAST scanning via Semgrep and CodeQL runs on every merge request to the release branch.",
        "Incident reviews must be published within 72 hours of resolution.",
    ],
    competing_slot_entity="alert_sla",
    competing_slot_attribute="sev1_ack_time",
    competing_slot_v1="Sev1 alert acknowledgement SLA is 10 minutes from initial Opsgenie page.",
    competing_slot_v2="Sev1 alert acknowledgement SLA is 4 minutes (updated Q1 2025 policy revision).",
    competing_queries=[
        ("What is our Sev1 alert acknowledgement SLA?", "4 minutes", "10 minutes"),
        ("Is the REST collector endpoint still supported?", "deprecated", "backward compatibility"),
        ("When is incident commander coverage required?", "24/7", "weekdays"),
    ],
    # S11v2 — progressive knowledge catchup
    catchup_initial_facts=[
        "Pulse collector gateway routes all spans through Traefik proxy with automatic retry logic.",
        "Trace storage connection pooling is managed by Odyssey with a max of 150 connections.",
        "Collector binaries are built with Bazel and cached in Artifactory registry.",
        "Datadog agent collects metrics from all allocations at 10-second scrape intervals.",
        "Experiment toggles are managed through Unleash with per-region targeting rules.",
        "TLS certificates are provisioned automatically by Smallstep CA with ACME protocol.",
        "Log aggregation uses Vector with 45-day retention and structured CBOR format.",
        "CI/CD pipeline runs in Buildkite with mandatory status checks on the release branch.",
    ],
    catchup_agent_b_facts=[
        "Backend services use connection pooling with a 45-second idle timeout.",
        "All gRPC endpoints implement health checking via the gRPC health protocol v1.",
        "Throttling middleware tracks per-workspace usage via Memcached sliding window counters.",
        "Distributed tracing uses OpenTelemetry with Tempo backend for trace collection.",
        "Schema migrations are versioned with Atlas and reviewed before production deploy.",
    ],
    catchup_agent_c_facts=[
        "Frontend builds are deployed via Cloudflare Pages with preview deployments for every MR.",
        "Vue component library is published to the internal Verdaccio registry daily.",
        "End-to-end tests run against a dedicated staging environment with synthetic trace data.",
        "Performance budgets enforce maximum 150KB JS bundle size per route.",
        "Accessibility audits run automatically using axe-core in the CI pipeline.",
        "Design tokens are synced from Figma to code via Style Dictionary v4.",
    ],
    catchup_agent_a_updates=[
        (
            "storage_config",
            "max_connections",
            "Odyssey max connections increased to 400 after load test results.",
        ),
        (
            "observability",
            "scrape_interval",
            "Datadog scrape interval reduced to 5 seconds for critical collector services.",
        ),
        (
            "ci_cd",
            "pipeline_platform",
            "CI/CD migrated from Buildkite to Dagger for faster reproducible pipeline execution.",
        ),
    ],
    catchup_agent_c_extra_facts=[
        "Web vitals monitoring added: INP target under 200ms, CLS under 0.05.",
        "Service worker enabled for offline-first caching of dashboard assets.",
        "A/B testing framework integrated with Unleash for frontend experiments.",
        "Bundle analyzer reports generated on every MR to track size regressions.",
    ],
    catchup_agent_d_urgent_facts=[
        "URGENT: Billing service experiencing 25% timeout rate since 08:30 UTC.",
        "URGENT: Root cause identified as trace storage connection exhaustion on billing-tsdb-primary.",
        "URGENT: Temporary fix deployed — Odyssey pool size doubled to 300 for billing service.",
    ],
    catchup_agent_b_response_facts=[
        "Acknowledged billing service incident; increasing client-side retry budget from 2 to 4 attempts.",
        "Verified that Dagger migration does not affect our backend deployment workflow.",
    ],
    catchup_b_response_queries=[
        ("What actions were taken for the billing service incident?", "retry"),
        ("Is the CI/CD migration affecting backend deployments?", "dagger"),
    ],
    # S12v2 — priority triage under load
    triage_critical_facts=[
        "Collector gateway returning 502 on all POST /v1/traces routes since 07:15 UTC.",
        "Billing service is completely unresponsive; no successful invoice generation in 20 minutes.",
        "Trace storage primary node CPU at 97% utilization; read replicas lagging by 60 seconds.",
        "Identity service rejecting all OIDC token refresh requests with 503 errors.",
        "Edge cache invalidation failed globally; serving stale dashboards for the last 45 minutes.",
        "Nomad cluster scheduler stuck; unable to place new allocations in us-central1.",
        "Message bus (NATS) consumer backlog exceeding 2 million spans on the ingest stream.",
        "DNS resolution failures affecting 35% of internal collector-to-storage calls.",
    ],
    triage_routine_facts=[
        "Release R-52 scheduled for next Wednesday; includes trace storage schema migration.",
        "Quarterly penetration test by external firm begins next Tuesday.",
        "Team offsite planned for April 20-22; reduced incident commander coverage during that period.",
        "New hire onboarding: two SRE engineers starting on May 1st.",
        "Dependency update: upgrading Go from 1.22 to 1.23 for security patches.",
        "Documentation sprint: collector SDK reference refresh planned for next sprint.",
        "Cost optimization review: GCP committed use discount renewals due in 45 days.",
        "Performance testing scheduled: load test for the new alerting engine next week.",
    ],
    triage_noise_facts=[
        "There are 4 open Linear tickets about minor tooltip alignment issues in the dashboard.",
        "The office espresso machine on floor 2 is still broken; facilities ticket pending.",
        "Team lunch scheduled for Thursday at 12:00 in the rooftop lounge.",
        "Garage level C will be closed for repainting this weekend.",
        "The annual company demo day theme will be announced next Friday.",
        "Product requested a logo color change from indigo to violet on the marketing site.",
        "The internal Notion workspace has 62 pages flagged as potentially outdated.",
        "Hot desk booking system shows 58% office occupancy for next week.",
    ],
    triage_incident_queries=[
        ("What services are currently experiencing outages?", "502"),
        ("Are there any billing processing issues?", "unresponsive"),
        ("What is the current trace storage health status?", "97%"),
        ("Are there identity service problems right now?", "rejecting"),
    ],
    triage_routine_queries=[
        ("What releases are scheduled this week?", "wednesday"),
        ("When is the penetration test?", "tuesday"),
    ],
    # S15v2 — cross-team signal contamination
    contamination_devops_facts=[
        "Nomad clusters are provisioned with Pulumi and managed via FluxCD reconciliation.",
        "All promotions go through a shadow phase with 5% mirrored traffic before full rollout.",
        "Datadog and OpsGenie handle infrastructure monitoring and alert routing.",
        "Infrastructure secrets are stored in CyberArk Conjur with automatic rotation.",
        "Promotion rollbacks are automated via FluxCD when Datadog health checks fail.",
        "Ingress latency is monitored by the platform team using Datadog dashboards.",
    ],
    contamination_backend_facts=[
        "Chi router handles all REST endpoints with custom middleware validation on request models.",
        "Backend services use Memcached for query result caching and distributed locks.",
        "Database connection pooling via Odyssey ensures sub-3ms query latency under load.",
        "NATS streams are used for real-time backend-to-backend event propagation.",
        "Backend promotion pipeline includes integration tests against a staging cluster.",
        "Endpoint latency is tracked per-route with P99 alerts above 120ms.",
    ],
    contamination_data_facts=[
        "Apache Flink processes streaming ETL jobs on a continuous schedule with 10-second windows.",
        "Weights & Biases tracks all model versions with automatic champion/challenger promotion criteria.",
        "Data pipeline monitoring uses custom Datadog dashboards with 5-minute granularity.",
        "Iceberg tables provide ACID transactions for all data lake write operations.",
        "Feature store serves pre-computed features with sub-8ms latency via Memcached.",
        "Pipeline promotion uses Dagster jobs managed through the data team's GitOps workflow.",
    ],
    contamination_channel_queries=[
        ("How are Nomad clusters provisioned?", "devops", "pulumi"),
        ("What router handles REST endpoints?", "backend", "chi"),
        ("What processes streaming ETL jobs?", "data", "flink"),
    ],
    contamination_shared_term_queries=[
        ("How is promotion done?", "devops", "shadow"),
        ("How is latency monitored?", "backend", "per-route"),
        ("How is monitoring done?", "data", "pipeline"),
    ],
)


# ===========================================================================
# EN Bundle — Alex Chen, Staff Data Engineer @ FinServe Capital
# ===========================================================================

_EN_PROFILE_FACTS: list[str] = [
    "Alex Chen is a Staff Data Engineer at FinServe Capital, based in San Francisco.",
    "Alex's primary language is Python 3.12; uses Go for high-throughput pipeline components.",
    "Alex uses Apache Kafka for event streaming between data domains.",
    "Alex's transformation layer is dbt on Databricks; pipeline orchestration runs on Prefect 2.",
    "Alex runs all infrastructure on Kubernetes 1.30 with Helm charts.",
    "Alex's analytical warehouse is Snowflake; operational database is PostgreSQL 16.",
    "Alex requires type annotations on all public functions and methods (mypy strict mode).",
    "Alex's team uses trunk-based development with feature flags for in-progress work.",
    "Alex's work hours are 09:00–18:00 PST; weekly data review meeting on Thursdays.",
    "Alex enforces schema versioning for all Avro and Protobuf event schemas in Confluent Schema Registry.",
    "Alex keeps secrets in AWS Secrets Manager; never in environment variables or config files.",
    "Alex uses pytest with pytest-datadir for all pipeline unit tests.",
    "Alex recently migrated orchestration from Apache Airflow 2.6 to Prefect 2.",
    "Alex's team follows a 'data contract first' policy: schema agreed before implementation.",
    "Alex's manager is Sarah Lin; skip-level is David Park, VP of Engineering.",
]

_EN_PROFILE_QUERIES: list[str] = [
    "What programming language does Alex prefer?",
    "How does Alex manage secrets and credentials?",
    "What orchestration tool does the team use?",
    "What is the team's primary analytical warehouse?",
    "What is the team's branching and development strategy?",
]

_EN_AVOID_REPEATS_TITLES: list[str] = [
    "Building fault-tolerant Kafka consumers with Python",
    "dbt best practices for large-scale data transformations",
    "Migrating from Apache Airflow to Prefect 2: lessons learned",
    "Confluent Schema Registry with Avro: a practical guide",
    "Snowflake cost optimization: 10 strategies that work",
    "Delta Lake on Databricks: ACID transactions for your data lake",
    "Incremental dbt models: from full refresh to stream-like updates",
    "Data contracts in practice: schema agreements before code",
    "Exactly-once Kafka delivery: guarantees and trade-offs",
    "Partition pruning in Snowflake: write queries that skip data",
    "Prefect 2 deployments: from local runs to production flows",
    "How we cut Databricks job costs by 40% with spot instances",
    "Writing testable dbt macros",
    "Kafka consumer group rebalancing: causes and mitigations",
    "PostgreSQL 16 for analytics: when to stay and when to move to Snowflake",
    "Data quality with dbt-expectations: beyond not-null tests",
    "Schema evolution strategies: backward and forward compatibility",
    "Dead letter queues for Kafka: design patterns",
    "Orchestrating dbt with Prefect: a step-by-step guide",
    "TimescaleDB vs InfluxDB: choosing a time-series database",
    "Streaming ingestion into Snowflake with Kafka connectors",
    "Building a data observability platform from scratch",
    "Fivetran vs Airbyte: a 2025 ELT comparison",
    "Column-level lineage in dbt: tracking data provenance",
    "Spark vs dbt: choosing the right transformation engine",
    "Kubernetes for data pipelines: stateful workloads done right",
    "AWS Glue vs Databricks: total cost of ownership analysis",
    "Event-driven data pipelines with Kafka and Flink",
    "Materialized views in Snowflake: when they help and when they hurt",
    "Data mesh principles applied to a fintech data platform",
    "Testing Kafka consumers without a running broker",
    "Go for data engineering: when Python is too slow",
    "dbt snapshots: tracking slowly changing dimensions",
    "Iceberg vs Delta Lake vs Hudi: open table format comparison",
    "Pipeline retries and idempotency: getting it right",
    "Monitoring dbt pipeline SLAs with Prefect and PagerDuty",
    "ClickHouse for real-time analytics: a fintech case study",
    "Airflow to Prefect migration: what nobody tells you",
    "The hidden costs of Kafka: partition sizing and storage",
    "Data versioning with lakeFS: git for your data lake",
    "Building a CI pipeline for dbt models",
    "PostgreSQL logical replication for real-time data sync",
    "Trino vs Spark SQL for ad-hoc analytics",
    "Schema-on-read vs schema-on-write in modern data platforms",
    "Prefect agents on Kubernetes: scaling horizontally",
    "Handling late-arriving events in streaming pipelines",
    "Avro vs Parquet vs ORC: when to use which format",
    "Data platform reliability: SLOs for pipelines",
    "Type-safe data pipelines with Pydantic and dbt",
    "Zero-downtime schema migrations in Snowflake",
]

_EN_AVOID_REPEATS_QUERIES: list[str] = [
    "Write an article about Apache Kafka and event streaming",
    "Suggest a topic about dbt and data transformation",
    "What should I write about data pipeline orchestration?",
]

_EN_AVOID_REPEATS_EXPECTED_SEEN: dict[str, list[str]] = {
    _EN_AVOID_REPEATS_QUERIES[0]: [
        "Building fault-tolerant Kafka consumers with Python",
        "Exactly-once Kafka delivery: guarantees and trade-offs",
        "Kafka consumer group rebalancing: causes and mitigations",
        "Dead letter queues for Kafka: design patterns",
        "Streaming ingestion into Snowflake with Kafka connectors",
    ],
    _EN_AVOID_REPEATS_QUERIES[1]: [
        "dbt best practices for large-scale data transformations",
        "Incremental dbt models: from full refresh to stream-like updates",
        "Writing testable dbt macros",
        "Data quality with dbt-expectations: beyond not-null tests",
        "Column-level lineage in dbt: tracking data provenance",
    ],
    _EN_AVOID_REPEATS_QUERIES[2]: [
        "Migrating from Apache Airflow to Prefect 2: lessons learned",
        "Orchestrating dbt with Prefect: a step-by-step guide",
        "Prefect 2 deployments: from local runs to production flows",
        "Airflow to Prefect migration: what nobody tells you",
    ],
}

_EN_FEEDBACK_HISTORY: list[tuple[str, str]] = [
    (
        "How long should data engineering posts be?",
        "Keep posts under 400 words — data engineers scan, not read.",
    ),
    (
        "Should I include code snippets?",
        "Always show code. No theory without a concrete implementation example.",
    ),
    (
        "What about SQL formatting in articles?",
        "Format SQL with uppercase keywords and consistent indentation; never inline snippets for multi-line queries.",
    ),
    (
        "Should I compare tools by name?",
        "Compare tools with benchmarks and numbers, not opinions. Show query plans and timing.",
    ),
    (
        "How should I end a post?",
        "End with one practical takeaway — a command or snippet the reader can run today.",
    ),
    (
        "What tone to use?",
        "Technical and direct. Skip adjectives like 'powerful', 'amazing', and 'seamless'.",
    ),
    (
        "What kind of diagrams?",
        "Data flow diagrams only — no vague architecture boxes or generic cloud icons.",
    ),
    (
        "Should I mention cloud providers?",
        "Stay cloud-agnostic unless the post is specifically about that provider.",
    ),
    (
        "How many examples should I use?",
        "One deep example beats three shallow ones. Pick the hardest edge case to illustrate.",
    ),
    (
        "What about performance numbers?",
        "Always include query plan output or benchmark results, never just narrative claims.",
    ),
]

_EN_FEEDBACK_QUERIES: list[str] = [
    "What are the content and style rules for writing technical posts?",
    "How should code and SQL be formatted in data engineering articles?",
    "What makes a good ending for a data engineering article?",
]

_EN_FEEDBACK_EXPECTED_RULES: dict[str, list[str]] = {
    _EN_FEEDBACK_QUERIES[0]: [
        "Keep posts under 400 words — data engineers scan, not read.",
        "Technical and direct. Skip adjectives like 'powerful', 'amazing', and 'seamless'.",
        "Stay cloud-agnostic unless the post is specifically about that provider.",
        "One deep example beats three shallow ones. Pick the hardest edge case to illustrate.",
    ],
    _EN_FEEDBACK_QUERIES[1]: [
        "Always show code. No theory without a concrete implementation example.",
        "Format SQL with uppercase keywords and consistent indentation; never inline snippets for multi-line queries.",
        "Always include query plan output or benchmark results, never just narrative claims.",
    ],
    _EN_FEEDBACK_QUERIES[2]: [
        "End with one practical takeaway — a command or snippet the reader can run today.",
        "One deep example beats three shallow ones. Pick the hardest edge case to illustrate.",
    ],
}

_EN_FEEDBACK_EXPECTED_KEYWORDS: dict[str, list[str]] = {
    _EN_FEEDBACK_QUERIES[0]: ["400", "scan", "adjective", "cloud", "agnostic"],
    _EN_FEEDBACK_QUERIES[1]: ["sql", "code", "uppercase", "benchmark", "query"],
    _EN_FEEDBACK_QUERIES[2]: ["takeaway", "command", "snippet", "example"],
}

_EN_DEDUP_CANONICAL_RULE = (
    "Always version all Avro and Protobuf schemas in the Schema Registry before deployment."
)

_EN_DEDUP_PARAPHRASES: list[str] = [
    "Always version all Avro and Protobuf schemas in the Schema Registry before deployment.",
    "All Avro and Protobuf schemas must be versioned in the Schema Registry prior to deployment.",
    "Register and version every Avro and Protobuf schema in the Schema Registry before deploying.",
    "Schema versioning in the Schema Registry is mandatory for Avro and Protobuf before any deployment.",
    "Before deploying, ensure all Avro and Protobuf schemas are versioned in the registry.",
    "Every Avro and Protobuf schema must be stored with a version in the Schema Registry before going live.",
    "Versioned Avro and Protobuf schemas are required in the Schema Registry ahead of deployment.",
    "Register all Avro and Protobuf schemas with explicit versions in the Schema Registry prior to rollout.",
    "Schema Registry versioning is required for all Avro and Protobuf schemas before deployment.",
    "Do not deploy until all Avro and Protobuf schemas are versioned in the Schema Registry.",
    "Always add a versioned entry to the Schema Registry for every Avro and Protobuf schema.",
    "Avro and Protobuf schemas must carry a Schema Registry version number before any production deployment.",
    "You must version your Avro and Protobuf schemas in the registry before deploying.",
    "Require Schema Registry versioning for Avro and Protobuf schemas as a deployment prerequisite.",
    "Bump the schema version in the Schema Registry for every Avro and Protobuf change before deploying.",
    "Schema Registry entries for Avro and Protobuf schemas must be versioned before release.",
    "Enforce version registration in the Schema Registry for all Avro and Protobuf schemas pre-deploy.",
    "All schema changes for Avro and Protobuf must be committed to the Schema Registry with a new version.",
    "New Avro and Protobuf schema versions must be registered before the corresponding code is deployed.",
    "Schema Registry versioning is a hard requirement for Avro and Protobuf schemas before any deploy.",
    "Never deploy Avro or Protobuf schema changes without first versioning them in the Schema Registry.",
    "Avro and Protobuf schema versions must exist in the Schema Registry prior to service deployment.",
    "Each Avro and Protobuf schema change requires a new version registered in the Schema Registry.",
    "Pre-deployment checklist: Avro and Protobuf schemas versioned in the Schema Registry.",
    "Version registration in the Schema Registry is mandatory for Avro and Protobuf before deployment.",
    "Always register a schema version for Avro and Protobuf in the Schema Registry before deploying services.",
    "Schema Registry version bump required for all Avro and Protobuf schema updates before deploying.",
    "Enforce Schema Registry versioning on every Avro and Protobuf schema ahead of production rollout.",
    "Avro and Protobuf schemas must be versioned in the Schema Registry as part of the deploy process.",
    "No deployment proceeds until all Avro and Protobuf schemas have been versioned in the Schema Registry.",
    "Register schema versions in the Schema Registry for Avro and Protobuf before deploying pipeline code.",
    "Schema versioning for Avro and Protobuf in the Schema Registry is a deployment gate.",
    "All new Avro and Protobuf schemas need a version in the Schema Registry before any rollout.",
    "Schema Registry: always add a version for Avro and Protobuf schemas before you deploy.",
    "Avro and Protobuf schema versioning in the registry is required; check before every deployment.",
    "Bump and register the schema version for Avro and Protobuf before promoting to production.",
    "Deploying without versioning Avro and Protobuf schemas in the Schema Registry is not allowed.",
    "Schema version registration must precede deployment for all Avro and Protobuf schemas.",
    "Every Avro and Protobuf schema deployed to production must have a Schema Registry version.",
    "Schema Registry version entry for Avro and Protobuf is required before merging deployment PRs.",
    "Add schema version to the Schema Registry for Avro and Protobuf before triggering deploy.",
    "Avro and Protobuf schemas must have explicit version records in the Schema Registry before deployment.",
    "A Schema Registry version is mandatory for Avro and Protobuf schemas before every production deploy.",
    "Register the schema version in the Schema Registry for all Avro and Protobuf schemas pre-deployment.",
    "Schema Registry versioning must be completed for Avro and Protobuf schemas before deployment runs.",
    "No Avro or Protobuf schema should go to production without a version in the Schema Registry.",
    "Always ensure Avro and Protobuf schema versions are committed to the Schema Registry before deployment.",
    "Versioning Avro and Protobuf schemas in the Schema Registry is required as a deploy prerequisite.",
    "All Avro and Protobuf schemas require a Schema Registry version before the deploy gate passes.",
    "Schema Registry: version every Avro and Protobuf schema before deploying the consuming service.",
]

_EN_DEDUP_DISTINCT_RULES: list[str] = [
    "Always version all Avro and Protobuf schemas in the Schema Registry before deployment.",
    "Use incremental dbt models with unique keys instead of full refreshes on large tables.",
    "All Kafka consumers must have a dedicated dead letter queue configured.",
    "Run EXPLAIN ANALYZE on all Snowflake queries before promoting to production.",
    "Pin dbt package versions in packages.yml; never use the 'latest' tag.",
    "Partition all large fact tables by created_at for efficient time-range queries.",
    "Write dbt-expectations data quality tests for every critical dbt model.",
    "All pipeline code must be testable in isolation with mock data fixtures.",
    "Use connection pooling on all PostgreSQL connections; default pool size is 10.",
    "Log structured JSON with correlation IDs in every pipeline step.",
    "Validate input schemas at pipeline entry points; reject malformed records early.",
    "Store raw ingested data in an immutable landing zone before any transformation.",
    "Use Delta Lake format for all large fact tables on Databricks.",
    "Pipeline SLAs are defined per dataset; alert when p95 processing latency exceeds the target.",
    "All secrets are fetched from AWS Secrets Manager at runtime, never baked into images.",
    "Idempotent writes require a unique pipeline run ID in every upsert statement.",
    "Data contracts are reviewed and signed off by consuming teams before schema changes.",
    "Never write raw SQL in application code; use dbt models or parameterized queries.",
    "Tag all Kafka events with schema version and producer service name.",
    "Document every dbt model with a description field in schema.yml.",
]

_EN_CONSOLIDATION_FACTS: list[str] = [
    # Round 1 (v1 — starting state)
    # Entity names are stable per topic so entity-addressed CAS can match across versions.
    "Alex's team uses Apache Airflow 2.2 for pipeline orchestration on a self-hosted cluster.",
    "Alex is the sole data engineer on the team.",
    "Alex's team runs the analytical warehouse on Amazon Redshift with manual VACUUM scheduling.",
    "Alex's team deploys pipeline changes manually after team sign-off.",
    "Alex's team stores event schemas as plain JSON files in a Git repository.",
    # Round 2 (v2)
    "Alex's team migrated orchestration to Apache Airflow 2.6 with the Celery executor.",
    "Alex hired a junior data engineer; Alex now leads a team of 2 people.",
    "Alex's team evaluated Snowflake as a replacement for Redshift.",
    "Alex's team moved deployments to a CI-gated shell script triggered on merge.",
    "Alex's team moved schemas to a shared Confluence wiki page.",
    # Round 3 (v3)
    "Alex's team piloted Prefect 2 for new pipelines alongside legacy Airflow DAGs.",
    "Alex grew the data team to 4 engineers after Series A funding.",
    "Alex's team now uses Snowflake as the primary warehouse for all analytics workloads.",
    "Alex's team runs deployments via a GitHub Actions workflow on every merge to main.",
    "Alex's team adopted Confluent Schema Registry with Avro for all Kafka topics.",
    # Round 4 (v4)
    "Alex's team builds all new pipelines in Prefect 2; Airflow handles only legacy DAGs.",
    "Alex leads a team of 6 data engineers.",
    "Alex's team runs all analytical workloads on Snowflake; Delta Lake on Databricks handles streaming.",
    "Alex's team deploys to production automatically via CI after integration tests pass.",
    "Alex's team enforces backward compatibility checks on all Avro schema updates in Schema Registry.",
    # Round 5 (v5 — latest state)
    "Alex's team runs all orchestration on Prefect 2 with Temporal for long-running workflows.",
    "Alex manages an 8-person data engineering team spread across San Francisco and London.",
    "Alex's team uses Snowflake for OLAP and PostgreSQL 16 with TimescaleDB for operational analytics.",
    "Alex's team deploys every merged PR to production automatically via GitHub Actions.",
    "Alex's team governs all event schemas via a data contract registry; breaking changes require a 30-day deprecation period.",
]

_EN_CONSOLIDATION_QUERIES: list[str] = [
    "What orchestration tool does Alex's team currently use?",
    "How large is Alex's data engineering team?",
    "What data warehouse and storage technology does the team use?",
    "How are pipeline changes deployed to production?",
    "How does the team manage event schemas and data contracts?",
]

_EN_CONSOLIDATION_LATEST_FACTS: list[str] = [
    "Alex's team runs all orchestration on Prefect 2 with Temporal for long-running workflows.",
    "Alex manages an 8-person data engineering team spread across San Francisco and London.",
    "Alex's team uses Snowflake for OLAP and PostgreSQL 16 with TimescaleDB for operational analytics.",
    "Alex's team deploys every merged PR to production automatically via GitHub Actions.",
    "Alex's team governs all event schemas via a data contract registry; breaking changes require a 30-day deprecation period.",
]

BUNDLE_EN = LanguageBundle(
    language="en",
    profile=ProfileFixture(raw_facts=_EN_PROFILE_FACTS, queries=_EN_PROFILE_QUERIES),
    avoid_repeats=AvoidRepeatsFixture(
        titles=_EN_AVOID_REPEATS_TITLES,
        queries=_EN_AVOID_REPEATS_QUERIES,
        expected_seen=_EN_AVOID_REPEATS_EXPECTED_SEEN,
    ),
    feedback=FeedbackFixture(
        history=_EN_FEEDBACK_HISTORY,
        queries=_EN_FEEDBACK_QUERIES,
        expected_rules=_EN_FEEDBACK_EXPECTED_RULES,
        expected_keywords=_EN_FEEDBACK_EXPECTED_KEYWORDS,
    ),
    dedup=DeduplicationFixture(
        paraphrases=_EN_DEDUP_PARAPHRASES,
        distinct_rules=_EN_DEDUP_DISTINCT_RULES,
        canonical_rule=_EN_DEDUP_CANONICAL_RULE,
    ),
    consolidation=ConsolidationFixture(
        facts=_EN_CONSOLIDATION_FACTS,
        queries=_EN_CONSOLIDATION_QUERIES,
        latest_facts=_EN_CONSOLIDATION_LATEST_FACTS,
        n_topics=5,
        n_versions=5,
    ),
)

# ===========================================================================
# RU Bundle — Максим Петров, Senior Backend Engineer @ Яндекс
# ===========================================================================

_RU_PROFILE_FACTS: list[str] = [
    "Максим Петров — senior backend-инженер в Яндексе, Москва.",
    "Максим пишет преимущественно на Python 3.11; использует Go для высоконагруженных сервисов.",
    "Максим использует ClickHouse для аналитических запросов и YDB для транзакционных данных.",
    "Вся инфраструктура команды работает на Kubernetes в Яндекс.Облаке.",
    "Максим использует Kafka для стриминга событий между сервисами.",
    "Основной фреймворк для Python-сервисов — FastAPI; для Go — стандартная библиотека с роутером chi.",
    "Максим требует полную типизацию всех публичных функций: mypy strict для Python, go vet для Go.",
    "Команда Максима использует trunk-based development с feature-флагами через LaunchDarkly.",
    "Рабочие часы Максима: 10:00–19:00 МСК; ежедневный стендап в 10:15.",
    "Максим хранит секреты в HashiCorp Vault; в переменных окружения только ссылки на секреты.",
    "Максим использует pytest с parametrize для unit-тестов и testcontainers для интеграционных.",
    "Команда мигрировала с Jenkins на GitHub Actions шесть месяцев назад.",
    "Максим ведёт внутренний технический блог на Habr на тему системного дизайна и highload.",
    "Правило ревью в команде: минимум два аппрувала для изменений в core-сервисах.",
    "Менеджер Максима — Сергей Иванов; skip-level — Елена Смирнова, директор разработки.",
]

_RU_PROFILE_QUERIES: list[str] = [
    "Какой язык программирования предпочитает Максим?",
    "Как в команде хранятся секреты?",
    "Какую базу данных использует Максим для аналитики?",
    "Какой CI/CD использует команда?",
    "Какой фреймворк используется для Python-сервисов?",
]

_RU_AVOID_REPEATS_TITLES: list[str] = [
    "ClickHouse vs PostgreSQL: когда стоит переходить на аналитическую базу",
    "YDB в продакшене: транзакции в распределённой системе",
    "Kafka для новичков: от топиков до consumer groups",
    "Как мы мигрировали с Jenkins на GitHub Actions без боли",
    "FastAPI в highload: оптимизация async-эндпоинтов",
    "Kubernetes в Яндекс.Облаке: наш опыт за два года",
    "Trunk-based development: почему мы отказались от feature-веток",
    "HashiCorp Vault: управление секретами в микросервисной архитектуре",
    "Типизация Python-кода: mypy strict в реальном проекте",
    "Go vs Python в highload: сравниваем производительность на реальных задачах",
    "Testcontainers для интеграционных тестов: никаких моков для баз данных",
    "Feature-флаги через LaunchDarkly: опыт внедрения",
    "gRPC vs REST в внутренних сервисах: что выбрать в 2025",
    "ClickHouse: оптимизация запросов с материализованными вью",
    "Мониторинг микросервисов: от логов до трейсов",
    "Kubernetes HPA и VPA: автоскейлинг на практике",
    "Kafka Schema Registry: как мы перестали ломать консьюмеры",
    "YDB vs ClickHouse: когда нужна аналитика, а когда — транзакции",
    "Как писать идемпотентные обработчики Kafka-сообщений",
    "GitHub Actions: кэширование зависимостей для ускорения CI",
    "Graceful shutdown в Go: паттерны и подводные камни",
    "Python asyncio в продакшене: ловушки и как их избежать",
    "ClickHouse MergeTree: выбор ключа сортировки под запросы",
    "Деплой без даунтайма: rolling update в Kubernetes",
    "Observability в 2025: OpenTelemetry как стандарт",
    "Distributed tracing с Jaeger: от инструментации до анализа",
    "Как мы снизили latency Kafka-консьюмеров на 60%",
    "Параллельное тестирование в Python: pytest-xdist в CI",
    "Секреты Kubernetes: Vault Agent Injector vs External Secrets Operator",
    "Go routines и каналы: паттерны конкурентности",
    "ClickHouse ReplicatedMergeTree: репликация без боли",
    "Как мы организовали внутреннюю документацию для 50+ сервисов",
    "Структурированное логирование: почему JSON лучше plain text",
    "FastAPI Depends: инъекция зависимостей без фреймворков",
    "Профилирование Python в продакшене: py-spy и scalene",
    "YDB: работа с таблицами через Python SDK",
    "Rate limiting на уровне API Gateway: алгоритмы и реализация",
    "Как тестировать Kafka без реального брокера",
    "PostgreSQL vs YDB: выбор хранилища для транзакционных данных",
    "Мониторинг ClickHouse: метрики, которые важны на самом деле",
    "Kubernetes Network Policies: минимум привилегий для подов",
    "CI/CD для монорепо: стратегии и инструменты",
    "Pydantic v2: типизация данных на уровне рантайма",
    "Graceful degradation в микросервисах: circuit breaker на практике",
    "ClickHouse Kafka engine: потоковая запись без Flink",
    "Как мы избавились от технического долга в core-сервисах",
    "GitHub Actions матрицы: параллельный запуск тестов по несколько окружений",
    "Оптимизация Docker-образов: от 2 ГБ до 200 МБ",
    "Эффективные код-ревью: правила, которые экономят время",
    "Секреты высоконагруженных FastAPI-приложений",
]

_RU_AVOID_REPEATS_QUERIES: list[str] = [
    "Напиши статью про ClickHouse и аналитические базы данных",
    "Предложи тему про Kafka и потоковую обработку",
    "О чём написать статью про Kubernetes и деплой?",
]

_RU_AVOID_REPEATS_EXPECTED_SEEN: dict[str, list[str]] = {
    _RU_AVOID_REPEATS_QUERIES[0]: [
        "ClickHouse vs PostgreSQL: когда стоит переходить на аналитическую базу",
        "ClickHouse: оптимизация запросов с материализованными вью",
        "ClickHouse MergeTree: выбор ключа сортировки под запросы",
        "ClickHouse ReplicatedMergeTree: репликация без боли",
        "Мониторинг ClickHouse: метрики, которые важны на самом деле",
    ],
    _RU_AVOID_REPEATS_QUERIES[1]: [
        "Kafka для новичков: от топиков до consumer groups",
        "Kafka Schema Registry: как мы перестали ломать консьюмеры",
        "Как мы снизили latency Kafka-консьюмеров на 60%",
        "Как писать идемпотентные обработчики Kafka-сообщений",
        "Как тестировать Kafka без реального брокера",
    ],
    _RU_AVOID_REPEATS_QUERIES[2]: [
        "Kubernetes в Яндекс.Облаке: наш опыт за два года",
        "Kubernetes HPA и VPA: автоскейлинг на практике",
        "Деплой без даунтайма: rolling update в Kubernetes",
        "Kubernetes Network Policies: минимум привилегий для подов",
    ],
}

_RU_FEEDBACK_HISTORY: list[tuple[str, str]] = [
    (
        "Какой должна быть длина статьи на Habr?",
        "Оптимальная длина — 1500–2500 слов. Короче — нет пользы читателю, длиннее — теряют.",
    ),
    (
        "Стоит ли использовать заголовки?",
        "Обязательно H2 и H3. Читатели Habr сканируют статью по заголовкам, не читают линейно.",
    ),
    (
        "Как оформлять код в статьях?",
        "Всегда fenced code blocks с указанием языка. Никаких инлайн-блоков для многострочного кода.",
    ),
    (
        "Сравнивать ли инструменты по именам?",
        "Сравнения только с цифрами и бенчмарками, не мнениями. Покажи числа, а не 'лучше'.",
    ),
    (
        "Как заканчивать статью?",
        "Заканчивай конкретным выводом или командой, которую читатель может запустить сразу.",
    ),
    (
        "Какой тон использовать?",
        "Технический и прямой. Без маркетинговых слов: 'мощный', 'удобный', 'революционный'.",
    ),
    (
        "Какие схемы добавлять?",
        "Только диаграммы потоков данных. Никаких размытых 'архитектурных' квадратиков без деталей.",
    ),
    (
        "Упоминать ли облачных провайдеров?",
        "Сохраняй нейтральность, если статья не специфична для провайдера. Не пиши рекламу.",
    ),
    (
        "Сколько примеров использовать?",
        "Один глубокий пример лучше трёх поверхностных. Разбери самый сложный кейс.",
    ),
    (
        "Что делать с метриками производительности?",
        "Всегда публикуй EXPLAIN ANALYZE или benchmark-вывод, а не просто утверждения.",
    ),
]

_RU_FEEDBACK_QUERIES: list[str] = [
    "Какие правила стиля и содержания для технических статей на Habr?",
    "Как оформлять код и SQL в статьях по бэкенду?",
    "Как правильно завершить техническую статью?",
]

_RU_FEEDBACK_EXPECTED_RULES: dict[str, list[str]] = {
    _RU_FEEDBACK_QUERIES[0]: [
        "Оптимальная длина — 1500–2500 слов. Короче — нет пользы читателю, длиннее — теряют.",
        "Технический и прямой. Без маркетинговых слов: 'мощный', 'удобный', 'революционный'.",
        "Сохраняй нейтральность, если статья не специфична для провайдера. Не пиши рекламу.",
        "Один глубокий пример лучше трёх поверхностных. Разбери самый сложный кейс.",
    ],
    _RU_FEEDBACK_QUERIES[1]: [
        "Всегда fenced code blocks с указанием языка. Никаких инлайн-блоков для многострочного кода.",
        "Сравнения только с цифрами и бенчмарками, не мнениями. Покажи числа, а не 'лучше'.",
        "Всегда публикуй EXPLAIN ANALYZE или benchmark-вывод, а не просто утверждения.",
    ],
    _RU_FEEDBACK_QUERIES[2]: [
        "Заканчивай конкретным выводом или командой, которую читатель может запустить сразу.",
        "Один глубокий пример лучше трёх поверхностных. Разбери самый сложный кейс.",
    ],
}

_RU_FEEDBACK_EXPECTED_KEYWORDS: dict[str, list[str]] = {
    _RU_FEEDBACK_QUERIES[0]: ["2500", "слов", "маркетинговых", "нейтральность", "пример"],
    _RU_FEEDBACK_QUERIES[1]: ["code", "block", "языка", "benchmark", "explain"],
    _RU_FEEDBACK_QUERIES[2]: ["вывод", "команд", "пример", "сложный"],
}

_RU_DEDUP_CANONICAL_RULE = "Все публичные функции и методы должны иметь полную типизацию аргументов и возвращаемых значений."

_RU_DEDUP_PARAPHRASES: list[str] = [
    "Все публичные функции и методы должны иметь полную типизацию аргументов и возвращаемых значений.",
    "Все публичные функции и методы обязаны быть типизированы — параметры и возвращаемые типы.",
    "Полная типизация обязательна для всех публичных функций и методов.",
    "Каждая публичная функция и метод должны иметь аннотации типов для всех параметров и возвращаемого значения.",
    "Аннотации типов обязательны для публичных функций: параметры и возвращаемое значение.",
    "Каждая публичная функция должна быть полностью типизирована.",
    "Типизируй все параметры и возвращаемые значения публичных функций без исключений.",
    "Все методы и функции публичного API требуют полной типизации.",
    "Публичные функции без аннотаций типов недопустимы.",
    "Обязательная типизация параметров и возвращаемых значений для всех публичных функций.",
    "Каждая экспортируемая функция должна иметь типы параметров и возвращаемого значения.",
    "Без полной типизации не должно быть ни одной публичной функции.",
    "Все сигнатуры публичных функций должны включать аннотации типов.",
    "Требуй аннотации типов на всех публичных функциях и методах.",
    "Публичные методы классов обязаны иметь типизированные параметры и возвращаемые значения.",
    "Аннотации типов для всех параметров и возвращаемых значений — обязательное требование для публичных функций.",
    "Каждая публично доступная функция должна иметь полные аннотации типов.",
    "Все функции публичного интерфейса должны быть полностью типизированы.",
    "Присваивай аннотации типов всем параметрам и возвращаемым значениям публичных функций.",
    "Никакой публичной функции без полной типизации параметров и возвращаемого типа.",
    "Публичный API: обязательная аннотация типов для каждой функции и метода.",
    "Все параметры и возвращаемые значения публичных функций должны быть явно типизированы.",
    "Типизация — обязательна для всех публичных функций и методов.",
    "Каждая публичная функция должна иметь явные аннотации типов без исключений.",
    "Указывай типы для всех параметров и возвращаемых значений публичных функций.",
    "Публичные функции без явных типов — нарушение стандарта кодирования.",
    "Все публично видимые функции обязаны иметь полную типизацию.",
    "Всегда добавляй аннотации типов к параметрам и возвращаемым значениям публичных функций.",
    "Требование: каждая публичная функция должна быть полностью аннотирована типами.",
    "Аннотируй типы для всех публичных методов и функций без исключений.",
    "Полная аннотация типов — обязательное условие для публичных функций.",
    "Типы параметров и возвращаемые типы обязательны для публичных функций и методов.",
    "Все публичные функции должны иметь типы: параметры и возвращаемое значение.",
    "Типизация параметров и возвращаемых значений обязательна для публичного API.",
    "Каждый публичный метод должен иметь полную типизацию без исключений.",
    "Типы для всех параметров и возвращаемых значений публичных функций — не опция, а требование.",
    "Всегда указывай типы параметров и возвращаемых значений для публичных функций.",
    "Без аннотации типов нет публичных функций.",
    "Публичные функции требуют полной типизации параметров и возвращаемых значений.",
    "Все публичные методы и функции должны быть покрыты аннотациями типов.",
    "Обязательная аннотация типов для всех параметров и возвращаемых значений публичных функций.",
    "Типизируй все публичные функции и методы без исключений.",
    "Аннотации типов обязательны на всех публичных функциях — аргументы и возвращаемое значение.",
    "Все публичные функции проекта должны иметь полные аннотации типов.",
    "Требуй полную типизацию для каждой публичной функции и метода.",
    "Аннотации типов должны охватывать все параметры и возвращаемые значения публичных функций.",
    "Публичные функции без аннотаций типов — нарушение правил проекта.",
    "Полная типизация публичных функций — обязательное условие.",
    "Все экспортируемые функции должны иметь типы параметров и возвращаемое значение.",
    "Каждый публичный метод обязан иметь аннотации типов на аргументах и возвращаемом значении.",
]

_RU_DEDUP_DISTINCT_RULES: list[str] = [
    "Все публичные функции и методы должны иметь полную типизацию аргументов и возвращаемых значений.",
    "Используй pytest с parametrize для всех юнит-тестов.",
    "Никогда не используй print() для отладки; только logging.",
    "Храни секреты в HashiCorp Vault; никогда в переменных окружения напрямую.",
    "Trunk-based development: ветки живут не дольше одного рабочего дня.",
    "Документируй каждый публичный метод однострочным docstring.",
    "Запускай миграции базы данных отдельным шагом деплоя, не вместе с кодом приложения.",
    "Максимальная длина функции — 40 строк; выноси логику в хелперы.",
    "Каждый сервис должен предоставлять эндпоинт /healthz.",
    "Используй структурированное JSON-логирование во всех продакшен-сервисах.",
    "Пиши интеграционные тесты для каждой внешней зависимости через testcontainers.",
    "Требуй минимум два аппрувала для мержа в main в core-сервисах.",
    "Закрепляй версии зависимостей в pyproject.toml.",
    "Тегируй Docker-образы хешем git-коммита.",
    "Устанавливай resource requests и limits для всех контейнеров в Kubernetes.",
    "Используй пулы соединений для всех подключений к базам данных.",
    "Инструментируй все HTTP-эндпоинты гистограммами латентности.",
    "Вся конфигурация должна передаваться через переменные окружения.",
    "P99 latency не должна превышать 200 мс.",
    "Все datetime хранятся в UTC с явным указанием timezone.",
]

_RU_CONSOLIDATION_FACTS: list[str] = [
    # Round 1 (v1 — начальное состояние)
    "Максим использует vim с минимальным vimrc для редактирования кода.",
    "Максим — единственный backend-инженер в своей команде.",
    "Деплой в продакшен происходит раз в месяц после спринт-ревью.",
    "Максим использует SQLite для хранения данных в прототипах.",
    "Максим мержит свои пулл-реквесты без внешнего ревью.",
    # Round 2 (v2)
    "Максим переехал на Neovim и использует lazy.nvim для плагинов.",
    "Максим нанял джуниор-разработчика; команда теперь 2 человека.",
    "Команда перешла на еженедельные релизы по пятницам.",
    "Максим мигрировал с SQLite на PostgreSQL для продакшен-окружения.",
    "Команда ввела ревью: один аппрувал обязателен перед мержем.",
    # Round 3 (v3)
    "Максим перешёл на VS Code с плагинами для Python и Go.",
    "Команда Максима выросла до 5 инженеров после раунда финансирования.",
    "Частота деплоев увеличилась до ежедневных релизов после настройки CI.",
    "Максим перевёл команду с PostgreSQL 13 на PostgreSQL 15.",
    "Политика ревью обновилась: теперь требуется два аппрувала.",
    # Round 4 (v4)
    "Максим использует VS Code с расширением GitHub Copilot.",
    "Максим руководит командой из 8 инженеров.",
    "Команда деплоится несколько раз в день в продакшен.",
    "Продакшен-база данных работает на PostgreSQL 16 с репликацией.",
    "Для изменений в core-сервисах требуется аппрувал senior-инженера.",
    # Round 5 (v5 — актуальное состояние)
    "Основной редактор Максима — Cursor с интегрированным GitHub Copilot и vim-режимом.",
    "Максим руководит командой из 10 инженеров в двух часовых поясах.",
    "Каждый смерженный PR автоматически деплоится в продакшен через GitHub Actions.",
    "Продакшен-стек использует PostgreSQL 16 с расширением TimescaleDB для временных рядов.",
    "Политика PR: два аппрувала, зелёный CI и автоматический security-скан обязательны.",
]

_RU_CONSOLIDATION_QUERIES: list[str] = [
    "Какой редактор или IDE сейчас использует Максим?",
    "Какой размер инженерной команды у Максима?",
    "Как часто команда деплоится в продакшен?",
    "Какую базу данных использует команда?",
    "Какова политика ревью и мержа пулл-реквестов?",
]

_RU_CONSOLIDATION_LATEST_FACTS: list[str] = [
    "Основной редактор Максима — Cursor с интегрированным GitHub Copilot и vim-режимом.",
    "Максим руководит командой из 10 инженеров в двух часовых поясах.",
    "Каждый смерженный PR автоматически деплоится в продакшен через GitHub Actions.",
    "Продакшен-стек использует PostgreSQL 16 с расширением TimescaleDB для временных рядов.",
    "Политика PR: два аппрувала, зелёный CI и автоматический security-скан обязательны.",
]

BUNDLE_RU = LanguageBundle(
    language="ru",
    profile=ProfileFixture(raw_facts=_RU_PROFILE_FACTS, queries=_RU_PROFILE_QUERIES),
    avoid_repeats=AvoidRepeatsFixture(
        titles=_RU_AVOID_REPEATS_TITLES,
        queries=_RU_AVOID_REPEATS_QUERIES,
        expected_seen=_RU_AVOID_REPEATS_EXPECTED_SEEN,
    ),
    feedback=FeedbackFixture(
        history=_RU_FEEDBACK_HISTORY,
        queries=_RU_FEEDBACK_QUERIES,
        expected_rules=_RU_FEEDBACK_EXPECTED_RULES,
        expected_keywords=_RU_FEEDBACK_EXPECTED_KEYWORDS,
    ),
    dedup=DeduplicationFixture(
        paraphrases=_RU_DEDUP_PARAPHRASES,
        distinct_rules=_RU_DEDUP_DISTINCT_RULES,
        canonical_rule=_RU_DEDUP_CANONICAL_RULE,
    ),
    consolidation=ConsolidationFixture(
        facts=_RU_CONSOLIDATION_FACTS,
        queries=_RU_CONSOLIDATION_QUERIES,
        latest_facts=_RU_CONSOLIDATION_LATEST_FACTS,
        n_topics=5,
        n_versions=5,
    ),
)

# ===========================================================================
# BUNDLE_EN_FAST — mini slice of BUNDLE_EN for ≤5-min dev iteration cycles
#
# Reuses existing EN content (no new data):
#   S2: first 15 titles instead of 50  (3× fewer LLM inserts)
#   S3: first 5 feedback pairs instead of 10
#   S7: 3 topics × 3 versions = 9 facts instead of 25
# ===========================================================================

# S7 fast: pick topics 0,1,2 × versions 1,2,3
# _EN_CONSOLIDATION_FACTS layout: [v1_t0..v1_t4, v2_t0..v2_t4, ..., v5_t0..v5_t4]
_FAST_CONSOLIDATION_FACTS: list[str] = [
    *_EN_CONSOLIDATION_FACTS[0:3],  # v1: topics 0,1,2
    *_EN_CONSOLIDATION_FACTS[5:8],  # v2: topics 0,1,2
    *_EN_CONSOLIDATION_FACTS[10:13],  # v3: topics 0,1,2
]

BUNDLE_EN_FAST = LanguageBundle(
    language="en",
    profile=BUNDLE_EN.profile,
    avoid_repeats=AvoidRepeatsFixture(
        titles=BUNDLE_EN.avoid_repeats.titles[:15],
        queries=BUNDLE_EN.avoid_repeats.queries,
        expected_seen=BUNDLE_EN.avoid_repeats.expected_seen,
    ),
    feedback=FeedbackFixture(
        history=BUNDLE_EN.feedback.history[:5],
        queries=BUNDLE_EN.feedback.queries,
        expected_rules=BUNDLE_EN.feedback.expected_rules,
        expected_keywords=BUNDLE_EN.feedback.expected_keywords,
    ),
    dedup=BUNDLE_EN.dedup,
    consolidation=ConsolidationFixture(
        facts=_FAST_CONSOLIDATION_FACTS,
        queries=_EN_CONSOLIDATION_QUERIES[:3],
        latest_facts=_EN_CONSOLIDATION_LATEST_FACTS[:3],
        n_topics=3,
        n_versions=3,
    ),
)

# ===========================================================================
# Professional benchmark fixtures (new S1–S8 scenarios)
# ===========================================================================


@dataclass
class RetrievalAccuracyFixture:
    """Ground truth for S1 MRR & Precision@k."""

    facts: list[str]  # facts to insert verbatim
    queries: list[str]  # queries to evaluate
    relevant_fact_per_query: dict[str, str]  # query → exact relevant fact text


@dataclass
class ParaphraseFixture:
    """Ground truth for S2 Semantic Gap. paraphrase_queries[i] ↔ verbatim_facts[i]."""

    verbatim_facts: list[str]
    paraphrase_queries: list[str]


@dataclass
class NoiseToleranceFixture:
    """Ground truth for S5 Noise Tolerance."""

    signal_facts: list[str]  # 5 target facts
    signal_queries: list[str]  # 5 queries to signal facts
    noise_facts: list[str]  # 200 semantically unrelated facts


# ---------------------------------------------------------------------------
# EN data — Alex Chen, Staff Data Engineer @ FinServe Capital
# ---------------------------------------------------------------------------

_EN_RETRIEVAL_QUERIES: list[str] = [
    "What programming language does Alex primarily use?",
    "How does Alex store secrets and credentials?",
    "What is Alex's pipeline orchestration tool?",
    "What is Alex's primary analytical warehouse?",
    "What branching strategy does Alex's team use?",
    "What testing framework does Alex use for pipelines?",
    "What event streaming technology does Alex use?",
    "What schema management does Alex enforce?",
    "Who is Alex's manager?",
    "What orchestration migration did Alex recently complete?",
]

_EN_RETRIEVAL_RELEVANT: dict[str, str] = {
    _EN_RETRIEVAL_QUERIES[
        0
    ]: "Alex's primary language is Python 3.12; uses Go for high-throughput pipeline components.",
    _EN_RETRIEVAL_QUERIES[
        1
    ]: "Alex keeps secrets in AWS Secrets Manager; never in environment variables or config files.",
    _EN_RETRIEVAL_QUERIES[
        2
    ]: "Alex's transformation layer is dbt on Databricks; pipeline orchestration runs on Prefect 2.",
    _EN_RETRIEVAL_QUERIES[
        3
    ]: "Alex's analytical warehouse is Snowflake; operational database is PostgreSQL 16.",
    _EN_RETRIEVAL_QUERIES[
        4
    ]: "Alex's team uses trunk-based development with feature flags for in-progress work.",
    _EN_RETRIEVAL_QUERIES[5]: "Alex uses pytest with pytest-datadir for all pipeline unit tests.",
    _EN_RETRIEVAL_QUERIES[6]: "Alex uses Apache Kafka for event streaming between data domains.",
    _EN_RETRIEVAL_QUERIES[
        7
    ]: "Alex enforces schema versioning for all Avro and Protobuf event schemas in Confluent Schema Registry.",
    _EN_RETRIEVAL_QUERIES[
        8
    ]: "Alex's manager is Sarah Lin; skip-level is David Park, VP of Engineering.",
    _EN_RETRIEVAL_QUERIES[
        9
    ]: "Alex recently migrated orchestration from Apache Airflow 2.6 to Prefect 2.",
}

EN_RETRIEVAL_ACCURACY = RetrievalAccuracyFixture(
    facts=_EN_PROFILE_FACTS,
    queries=_EN_RETRIEVAL_QUERIES,
    relevant_fact_per_query=_EN_RETRIEVAL_RELEVANT,
)

_EN_PARAPHRASE_FACTS: list[str] = [
    "Alex's primary language is Python 3.12; uses Go for high-throughput pipeline components.",
    "Alex keeps secrets in AWS Secrets Manager; never in environment variables or config files.",
    "Alex uses Apache Kafka for event streaming between data domains.",
    "Alex's analytical warehouse is Snowflake; operational database is PostgreSQL 16.",
    "Alex's team uses trunk-based development with feature flags for in-progress work.",
    "Alex uses pytest with pytest-datadir for all pipeline unit tests.",
    "Alex's transformation layer is dbt on Databricks; pipeline orchestration runs on Prefect 2.",
    "Alex requires type annotations on all public functions and methods (mypy strict mode).",
    "Alex's team follows a 'data contract first' policy: schema agreed before implementation.",
    "Alex enforces schema versioning for all Avro and Protobuf event schemas in Confluent Schema Registry.",
]

_EN_PARAPHRASE_QUERIES: list[str] = [
    "Which coding language does Alex mainly work with?",
    "Where does Alex's team store sensitive credentials?",
    "What messaging system does Alex rely on for data streaming?",
    "What database does Alex use for analytics workloads?",
    "How does Alex's team manage code branches?",
    "What testing tool does Alex run for unit tests?",
    "What does Alex use to orchestrate data pipelines?",
    "What are Alex's requirements for Python type checking?",
    "What is Alex's team's policy when implementing new data schemas?",
    "How does Alex manage event schema versions?",
]

EN_PARAPHRASE = ParaphraseFixture(
    verbatim_facts=_EN_PARAPHRASE_FACTS,
    paraphrase_queries=_EN_PARAPHRASE_QUERIES,
)

_EN_SIGNAL_FACTS: list[str] = [
    "Alex's primary language is Python 3.12; uses Go for high-throughput pipeline components.",
    "Alex keeps secrets in AWS Secrets Manager; never in environment variables or config files.",
    "Alex uses Apache Kafka for event streaming between data domains.",
    "Alex's analytical warehouse is Snowflake; operational database is PostgreSQL 16.",
    "Alex's team uses trunk-based development with feature flags for in-progress work.",
]

_EN_SIGNAL_QUERIES: list[str] = [
    "What programming language does Alex use?",
    "How does Alex manage secrets?",
    "What event streaming technology does Alex use?",
    "What is Alex's data warehouse?",
    "What branching strategy does Alex's team follow?",
]

EN_NOISE_TOLERANCE = NoiseToleranceFixture(
    signal_facts=_EN_SIGNAL_FACTS,
    signal_queries=_EN_SIGNAL_QUERIES,
    noise_facts=LOAD_FACTS,  # 200 semantically distinct infra facts
)
