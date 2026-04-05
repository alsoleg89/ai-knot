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
CONSOLIDATION_LATEST_KEYWORDS: dict[str, list[str]] = {
    CONSOLIDATION_QUERIES[0]: ["copilot"],
    CONSOLIDATION_QUERIES[1]: ["10-person", "10 person", "ten"],
    CONSOLIDATION_QUERIES[2]: ["continuous", "every merged", "automatic"],
    CONSOLIDATION_QUERIES[3]: ["timescaledb", "time-series"],
    CONSOLIDATION_QUERIES[4]: ["security scan"],
}

N_CONSOLIDATION_TOPICS = 5
N_CONSOLIDATION_VERSIONS = 5  # versions per topic
N_CONSOLIDATION_FACTS = N_CONSOLIDATION_TOPICS * N_CONSOLIDATION_VERSIONS  # 25


@dataclass
class ConsolidationFixture:
    facts: list[str]
    queries: list[str]
    latest_keywords: dict[str, list[str]]
    n_topics: int
    n_versions: int


CONSOLIDATION = ConsolidationFixture(
    facts=CONSOLIDATION_FACTS,
    queries=CONSOLIDATION_QUERIES,
    latest_keywords=CONSOLIDATION_LATEST_KEYWORDS,
    n_topics=N_CONSOLIDATION_TOPICS,
    n_versions=N_CONSOLIDATION_VERSIONS,
)
