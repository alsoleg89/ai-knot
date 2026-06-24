"""Golden retrieval dataset for ai-knot eval (105 cases)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalCase:
    query: str
    facts: list[dict]  # each has 'id', 'content', optionally 'importance', 'access_count'
    relevant_ids: list[str]


# 30 cases covering: semantic (user info), procedural (preferences), episodic (events)
RETRIEVAL_DATASET: list[RetrievalCase] = [
    # --- Semantic: user personal info ---
    RetrievalCase(
        query="Where does the user work?",
        facts=[
            {
                "id": "s001",
                "content": "User works at Sber as a senior software engineer.",
                "tags": ["career", "team"],
            },
            {
                "id": "s002",
                "content": "User prefers dark mode in all code editors.",
                "tags": ["personal"],
            },
            {
                "id": "s003",
                "content": "User lives in Moscow and commutes by metro.",
                "tags": ["lifestyle"],
            },
            {"id": "s004", "content": "User has a dog named Pushkin.", "tags": ["personal"]},
            {
                "id": "s005",
                "content": "User's primary programming language is Python.",
                "tags": ["personal", "python"],
            },
        ],
        relevant_ids=["s001"],
    ),
    RetrievalCase(
        query="What programming language does the user prefer?",
        facts=[
            {
                "id": "s006",
                "content": "User's primary programming language is Python.",
                "tags": ["personal", "python"],
            },
            {
                "id": "s007",
                "content": "User also knows TypeScript and uses it for frontend work.",
                "tags": ["frontend"],
            },
            {
                "id": "s008",
                "content": "User studied mathematics at Moscow State University.",
                "tags": ["education"],
            },
            {
                "id": "s009",
                "content": "User's team uses Git with a trunk-based development model.",
                "tags": ["git", "team"],
            },
            {"id": "s010", "content": "User enjoys hiking on weekends.", "tags": ["personal"]},
        ],
        relevant_ids=["s006", "s007"],
    ),
    RetrievalCase(
        query="What city does the user live in?",
        facts=[
            {
                "id": "s011",
                "content": "User lives in Moscow and commutes by metro.",
                "tags": ["lifestyle"],
            },
            {
                "id": "s012",
                "content": "User's office is near Lubyanka metro station.",
                "tags": ["lifestyle"],
            },
            {
                "id": "s013",
                "content": "User has visited Saint Petersburg twice this year.",
                "tags": ["personal"],
            },
            {
                "id": "s014",
                "content": "User uses macOS for personal work and Linux for servers.",
                "tags": ["tools"],
            },
            {
                "id": "s015",
                "content": "User's favorite IDE is VS Code with vim keybindings.",
                "tags": ["tools"],
            },
        ],
        relevant_ids=["s011"],
    ),
    RetrievalCase(
        query="What is the user's educational background?",
        facts=[
            {
                "id": "s016",
                "content": "User studied mathematics at Moscow State University.",
                "tags": ["education"],
            },
            {
                "id": "s017",
                "content": "User completed an online ML course from Coursera in 2022.",
                "tags": ["education"],
            },
            {
                "id": "s018",
                "content": "User's team has 6 engineers and 2 product managers.",
                "tags": ["team"],
            },
            {
                "id": "s019",
                "content": "User prefers to write code with type hints always.",
                "tags": ["personal"],
            },
            {
                "id": "s020",
                "content": "User reads technical books in English and Russian.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["s016", "s017"],
    ),
    RetrievalCase(
        query="What pets does the user have?",
        facts=[
            {
                "id": "s021",
                "content": "User has a dog named Pushkin, a labrador.",
                "tags": ["personal"],
            },
            {
                "id": "s022",
                "content": "User had a cat but it moved to live with their parents.",
                "tags": ["personal"],
            },
            {
                "id": "s023",
                "content": "User's apartment is 65 square meters.",
                "tags": ["personal"],
            },
            {
                "id": "s024",
                "content": "User enjoys cooking Italian food on weekends.",
                "tags": ["food"],
            },
            {
                "id": "s025",
                "content": "User subscribes to several technical newsletters.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["s021", "s022"],
    ),
    # --- Procedural: user preferences and workflows ---
    RetrievalCase(
        query="How does the user want code formatted?",
        facts=[
            {
                "id": "p001",
                "content": "Always use ruff for Python formatting and linting.",
                "tags": ["python"],
            },
            {
                "id": "p002",
                "content": "Always add type hints to all function signatures.",
                "tags": ["personal"],
            },
            {
                "id": "p003",
                "content": "Use 4 spaces for indentation, never tabs.",
                "tags": ["personal"],
            },
            {"id": "p004", "content": "User prefers dark mode in editors.", "tags": ["personal"]},
            {
                "id": "p005",
                "content": "User checks email at 9am and 3pm only.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["p001", "p002", "p003"],
    ),
    RetrievalCase(
        query="What testing framework does the user use?",
        facts=[
            {
                "id": "p006",
                "content": "User uses pytest for all Python testing.",
                "tags": ["python"],
            },
            {
                "id": "p007",
                "content": "User writes unit tests before submitting PRs.",
                "tags": ["personal"],
            },
            {
                "id": "p008",
                "content": "User's team requires 80% code coverage.",
                "tags": ["team", "testing"],
            },
            {
                "id": "p009",
                "content": "User prefers to use fixtures over setUp/tearDown.",
                "tags": ["personal"],
            },
            {
                "id": "p010",
                "content": "User has a standing desk at the office.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["p006", "p007", "p008", "p009"],
    ),
    RetrievalCase(
        query="How does the user handle code reviews?",
        facts=[
            {
                "id": "p011",
                "content": "User reviews PRs the same day they are opened.",
                "tags": ["personal"],
            },
            {
                "id": "p012",
                "content": "User leaves comments with suggested alternatives, not just issues.",
                "tags": ["personal"],
            },
            {
                "id": "p013",
                "content": "User requires at least two approvals before merging.",
                "tags": ["personal"],
            },
            {
                "id": "p014",
                "content": "User uses GitHub for version control.",
                "tags": ["personal"],
            },
            {
                "id": "p015",
                "content": "User prefers asynchronous communication over meetings.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["p011", "p012", "p013"],
    ),
    RetrievalCase(
        query="What is the user's preferred commit message style?",
        facts=[
            {
                "id": "p016",
                "content": "User writes commit messages in imperative mood.",
                "tags": ["git"],
            },
            {
                "id": "p017",
                "content": "User's commit subject lines are under 72 characters.",
                "tags": ["git"],
            },
            {
                "id": "p018",
                "content": "User uses conventional commits format: feat, fix, chore.",
                "tags": ["code-quality"],
            },
            {
                "id": "p019",
                "content": "User does not include ticket numbers in commit messages.",
                "tags": ["git"],
            },
            {
                "id": "p020",
                "content": "User uses poetry for dependency management.",
                "tags": ["python"],
            },
        ],
        relevant_ids=["p016", "p017", "p018"],
    ),
    RetrievalCase(
        query="How does the user prefer to document code?",
        facts=[
            {
                "id": "p021",
                "content": "User writes docstrings for all public functions and classes.",
                "tags": ["personal"],
            },
            {"id": "p022", "content": "User uses Google-style docstrings.", "tags": ["personal"]},
            {
                "id": "p023",
                "content": "User documents architecture decisions in DECISIONS.md files.",
                "tags": ["personal"],
            },
            {
                "id": "p024",
                "content": "User does not write inline comments unless logic is non-obvious.",
                "tags": ["personal"],
            },
            {
                "id": "p025",
                "content": "User's favorite keyboard shortcut is Ctrl+Shift+P.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["p021", "p022", "p023"],
    ),
    # --- Episodic: specific past events ---
    RetrievalCase(
        query="What happened at the last sprint retrospective?",
        facts=[
            {
                "id": "e001",
                "content": "Sprint retrospective on Monday revealed deployment pipeline was slow.",
                "tags": ["ci-cd", "workflow"],
            },
            {
                "id": "e002",
                "content": "Team agreed to add caching to CI/CD pipeline in next sprint.",
                "tags": ["caching", "ci-cd", "team"],
            },
            {
                "id": "e003",
                "content": "User presented metrics showing 40% test suite speedup.",
                "tags": ["testing"],
            },
            {
                "id": "e004",
                "content": "User is reading Clean Architecture by Robert Martin.",
                "tags": ["personal"],
            },
            {
                "id": "e005",
                "content": "User's standup is at 10am every weekday.",
                "tags": ["workflow"],
            },
        ],
        relevant_ids=["e001", "e002", "e003"],
    ),
    RetrievalCase(
        query="When did the production deployment fail?",
        facts=[
            {
                "id": "e006",
                "content": "Production deployment failed last Tuesday due to a config error.",
                "tags": ["ci-cd", "configuration", "error-handling"],
            },
            {
                "id": "e007",
                "content": "Deploy was rolled back within 15 minutes by the on-call engineer.",
                "tags": ["ci-cd", "incident-response", "team"],
            },
            {
                "id": "e008",
                "content": "Post-mortem identified missing environment variable as root cause.",
                "tags": ["configuration", "incident-response"],
            },
            {
                "id": "e009",
                "content": "User completed the Python advanced course last month.",
                "tags": ["education", "python"],
            },
            {
                "id": "e010",
                "content": "User's team uses Slack for communication.",
                "tags": ["communication", "team"],
            },
        ],
        relevant_ids=["e006", "e007", "e008"],
    ),
    RetrievalCase(
        query="What did the user discuss with their manager last week?",
        facts=[
            {
                "id": "e011",
                "content": "User met with manager last Thursday to discuss promotion timeline.",
                "tags": ["career", "team"],
            },
            {
                "id": "e012",
                "content": "Manager suggested user take on a tech lead role in Q3.",
                "tags": ["career", "team"],
            },
            {
                "id": "e013",
                "content": "User expressed interest in moving to the ML platform team.",
                "tags": ["team"],
            },
            {
                "id": "e014",
                "content": "User drinks two cups of coffee per day.",
                "tags": ["personal"],
            },
            {
                "id": "e015",
                "content": "User's team lunch is every Friday.",
                "tags": ["food", "team"],
            },
        ],
        relevant_ids=["e011", "e012", "e013"],
    ),
    RetrievalCase(
        query="What new library did the user try recently?",
        facts=[
            {
                "id": "e016",
                "content": "User tried Polars for data processing last week and was impressed.",
                "tags": ["data"],
            },
            {
                "id": "e017",
                "content": "User benchmarked Polars against Pandas and found 3x speedup.",
                "tags": ["data"],
            },
            {
                "id": "e018",
                "content": "User is considering migrating the data pipeline to Polars.",
                "tags": ["ci-cd"],
            },
            {
                "id": "e019",
                "content": "User's team has a book club that meets monthly.",
                "tags": ["hobby", "team"],
            },
            {"id": "e020", "content": "User prefers tea in the afternoon.", "tags": ["personal"]},
        ],
        relevant_ids=["e016", "e017", "e018"],
    ),
    RetrievalCase(
        query="What bug was fixed in the authentication module?",
        facts=[
            {
                "id": "e021",
                "content": "Fixed a race condition in the authentication token refresh logic.",
                "tags": ["error-handling"],
            },
            {
                "id": "e022",
                "content": "The bug caused occasional 401 errors for users after 1 hour.",
                "tags": ["error-handling"],
            },
            {
                "id": "e023",
                "content": "Fix added a mutex lock around the token refresh operation.",
                "tags": ["personal"],
            },
            {
                "id": "e024",
                "content": "User attended a conference on distributed systems last month.",
                "tags": ["education"],
            },
            {
                "id": "e025",
                "content": "User's favorite restaurant near the office serves Georgian food.",
                "tags": ["food"],
            },
        ],
        relevant_ids=["e021", "e022", "e023"],
    ),
    # --- Mixed: multi-type queries ---
    RetrievalCase(
        query="What are the user's Python best practices?",
        facts=[
            {"id": "m001", "content": "Always use type hints in Python code.", "tags": ["python"]},
            {
                "id": "m002",
                "content": "Use dataclasses for simple data containers.",
                "tags": ["personal"],
            },
            {
                "id": "m003",
                "content": "Prefer composition over inheritance in Python.",
                "tags": ["python"],
            },
            {
                "id": "m004",
                "content": "User likes to go running in Gorky Park.",
                "tags": ["personal"],
            },
            {
                "id": "m005",
                "content": "User uses mypy in strict mode for type checking.",
                "tags": ["python", "typing"],
            },
        ],
        relevant_ids=["m001", "m002", "m003", "m005"],
    ),
    RetrievalCase(
        query="How does the user manage project dependencies?",
        facts=[
            {
                "id": "m006",
                "content": "User uses poetry for Python dependency management.",
                "tags": ["python"],
            },
            {
                "id": "m007",
                "content": "User pins all transitive dependencies in production.",
                "tags": ["personal"],
            },
            {
                "id": "m008",
                "content": "User runs dependency audits monthly using safety check.",
                "tags": ["personal"],
            },
            {
                "id": "m009",
                "content": "User's home router uses OpenWRT firmware.",
                "tags": ["personal"],
            },
            {
                "id": "m010",
                "content": "User contributes to open source in spare time.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["m006", "m007", "m008"],
    ),
    RetrievalCase(
        query="What happened at the system design interview?",
        facts=[
            {
                "id": "m011",
                "content": "User did a system design interview at a tech company last Tuesday.",
                "tags": ["career"],
            },
            {
                "id": "m012",
                "content": "Interview focused on designing a rate limiter for an API.",
                "tags": ["api"],
            },
            {
                "id": "m013",
                "content": "User explained token bucket and sliding window algorithms.",
                "tags": ["personal"],
            },
            {
                "id": "m014",
                "content": "User received positive feedback and awaits second round.",
                "tags": ["personal"],
            },
            {
                "id": "m015",
                "content": "User prefers minimalist desk setup with a single monitor.",
                "tags": ["monitoring"],
            },
        ],
        relevant_ids=["m011", "m012", "m013", "m014"],
    ),
    RetrievalCase(
        query="What database does the user's project use?",
        facts=[
            {
                "id": "m016",
                "content": "Project uses PostgreSQL 15 as the primary database.",
                "tags": ["database"],
            },
            {
                "id": "m017",
                "content": "Redis is used for caching and session storage.",
                "tags": ["caching", "database"],
            },
            {
                "id": "m018",
                "content": "User migrated from SQLite to PostgreSQL in January.",
                "tags": ["database"],
            },
            {
                "id": "m019",
                "content": "User's favorite podcast is Lex Fridman.",
                "tags": ["personal"],
            },
            {"id": "m020", "content": "User tracks time with Toggl.", "tags": ["personal"]},
        ],
        relevant_ids=["m016", "m017", "m018"],
    ),
    RetrievalCase(
        query="How does the user handle secrets and credentials?",
        facts=[
            {
                "id": "m021",
                "content": "User stores all secrets in environment variables, never in code.",
                "tags": ["personal"],
            },
            {
                "id": "m022",
                "content": "User uses HashiCorp Vault for secrets management in production.",
                "tags": ["security"],
            },
            {"id": "m023", "content": "User rotates API keys every 90 days.", "tags": ["api"]},
            {
                "id": "m024",
                "content": "User uses a password manager with a strong master password.",
                "tags": ["security", "team"],
            },
            {"id": "m025", "content": "User likes jazz music while coding.", "tags": ["hobby"]},
        ],
        relevant_ids=["m021", "m022", "m023"],
    ),
    # --- Additional semantic cases ---
    RetrievalCase(
        query="What is the user's team size?",
        facts=[
            {
                "id": "a001",
                "content": "User's engineering team has 6 developers and 2 product managers.",
                "tags": ["team"],
            },
            {
                "id": "a002",
                "content": "Team is split between Moscow and remote workers.",
                "tags": ["team"],
            },
            {
                "id": "a003",
                "content": "User leads a subteam of 3 engineers on the backend.",
                "tags": ["team"],
            },
            {"id": "a004", "content": "User's laptop is a MacBook Pro M2.", "tags": ["personal"]},
            {"id": "a005", "content": "User takes a daily walk after lunch.", "tags": ["food"]},
        ],
        relevant_ids=["a001", "a002", "a003"],
    ),
    RetrievalCase(
        query="What monitoring tools does the user use?",
        facts=[
            {
                "id": "a006",
                "content": "User's team uses Grafana and Prometheus for metrics monitoring.",
                "tags": ["monitoring", "team"],
            },
            {
                "id": "a007",
                "content": "Sentry is used for error tracking and alerting.",
                "tags": ["error-handling", "monitoring"],
            },
            {
                "id": "a008",
                "content": "User set up PagerDuty for on-call rotations last year.",
                "tags": ["incident-response", "monitoring"],
            },
            {
                "id": "a009",
                "content": "User prefers 9-hour workdays with a long lunch break.",
                "tags": ["food"],
            },
            {
                "id": "a010",
                "content": "User does not use social media except LinkedIn.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["a006", "a007", "a008"],
    ),
    RetrievalCase(
        query="What did the user learn at the conference?",
        facts=[
            {
                "id": "a011",
                "content": "User attended PyCon Russia and learned about asyncio best practices.",
                "tags": ["education"],
            },
            {
                "id": "a012",
                "content": "A talk on Python performance profiling was the highlight.",
                "tags": ["python"],
            },
            {
                "id": "a013",
                "content": "User connected with maintainers of several open source libraries.",
                "tags": ["personal"],
            },
            {
                "id": "a014",
                "content": "User brought conference swag back for the team.",
                "tags": ["education", "team"],
            },
            {
                "id": "a015",
                "content": "User's home office has a mechanical keyboard.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["a011", "a012", "a013"],
    ),
    RetrievalCase(
        query="How does the user prefer to receive feedback?",
        facts=[
            {
                "id": "a016",
                "content": "User prefers written feedback over verbal to have time to process it.",
                "tags": ["personal"],
            },
            {
                "id": "a017",
                "content": "User asks for specific examples when receiving critical feedback.",
                "tags": ["personal"],
            },
            {
                "id": "a018",
                "content": "User schedules 1:1 with manager every two weeks.",
                "tags": ["team"],
            },
            {
                "id": "a019",
                "content": "User's morning routine starts with 20 minutes of reading.",
                "tags": ["personal"],
            },
            {
                "id": "a020",
                "content": "User tracks personal goals in Notion.",
                "tags": ["documentation"],
            },
        ],
        relevant_ids=["a016", "a017", "a018"],
    ),
    RetrievalCase(
        query="What CI/CD system does the team use?",
        facts=[
            {
                "id": "a021",
                "content": "Team uses GitHub Actions for continuous integration.",
                "tags": ["ci-cd", "team"],
            },
            {
                "id": "a022",
                "content": "Deployments are automated via ArgoCD to Kubernetes.",
                "tags": ["ci-cd", "devops"],
            },
            {
                "id": "a023",
                "content": "CI pipeline runs linting, type checks, and tests on every PR.",
                "tags": ["ci-cd", "code-review"],
            },
            {
                "id": "a024",
                "content": "User has a personal wiki written in Markdown.",
                "tags": ["documentation", "knowledge-sharing"],
            },
            {
                "id": "a025",
                "content": "User's favorite book is SICP.",
                "tags": ["hobby"],
            },
        ],
        relevant_ids=["a021", "a022", "a023"],
    ),
    RetrievalCase(
        query="What is the user's approach to refactoring?",
        facts=[
            {
                "id": "b001",
                "content": "User refactors code only when covered by tests.",
                "tags": ["personal"],
            },
            {
                "id": "b002",
                "content": "User applies the boy scout rule: leave code cleaner than found.",
                "tags": ["engineering-practices"],
            },
            {
                "id": "b003",
                "content": "User avoids large refactoring PRs, prefers incremental changes.",
                "tags": ["personal"],
            },
            {
                "id": "b004",
                "content": "User drinks matcha latte in the morning.",
                "tags": ["personal"],
            },
            {
                "id": "b005",
                "content": "User's commute takes 35 minutes each way.",
                "tags": ["lifestyle"],
            },
        ],
        relevant_ids=["b001", "b002", "b003"],
    ),
    RetrievalCase(
        query="How does the user stay up to date with technology?",
        facts=[
            {
                "id": "b006",
                "content": "User reads Hacker News daily for tech news.",
                "tags": ["personal"],
            },
            {
                "id": "b007",
                "content": "User follows several Python core developers on social media.",
                "tags": ["python"],
            },
            {
                "id": "b008",
                "content": "User subscribes to Real Python and Python Weekly newsletters.",
                "tags": ["python"],
            },
            {
                "id": "b009",
                "content": "User listens to technical podcasts during commute.",
                "tags": ["lifestyle"],
            },
            {"id": "b010", "content": "User's desk plant is a succulent.", "tags": ["personal"]},
        ],
        relevant_ids=["b006", "b007", "b008", "b009"],
    ),
    RetrievalCase(
        query="What happened during the on-call incident last week?",
        facts=[
            {
                "id": "b011",
                "content": "On-call incident: DB connection pool exhausted last Wednesday at 2am.",
                "tags": ["incident-response"],
            },
            {
                "id": "b012",
                "content": "User was paged and resolved the incident by increasing pool size.",
                "tags": ["incident-response"],
            },
            {
                "id": "b013",
                "content": "Root cause analysis showed a sudden traffic spike from a batch job.",
                "tags": ["personal"],
            },
            {
                "id": "b014",
                "content": "User updated runbook to include connection pool tuning steps.",
                "tags": ["incident-response"],
            },
            {
                "id": "b015",
                "content": "User ordered a new mechanical keyboard last week.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["b011", "b012", "b013", "b014"],
    ),
    RetrievalCase(
        query="What are the user's goals for this quarter?",
        facts=[
            {
                "id": "b016",
                "content": "User's Q1 goal is to ship the new memory indexing feature.",
                "tags": ["personal"],
            },
            {
                "id": "b017",
                "content": "User wants to improve test coverage from 75% to 90% this quarter.",
                "tags": ["testing"],
            },
            {
                "id": "b018",
                "content": "User plans to give one internal tech talk this quarter.",
                "tags": ["knowledge-sharing"],
            },
            {
                "id": "b019",
                "content": "User's personal goal is to run a 10k race in under 55 minutes.",
                "tags": ["health"],
            },
            {
                "id": "b020",
                "content": "User checks personal finances every Sunday.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["b016", "b017", "b018"],
    ),
    RetrievalCase(
        query="How does the user handle technical debt?",
        facts=[
            {
                "id": "b021",
                "content": "User tracks technical debt in a dedicated GitHub project board.",
                "tags": ["personal"],
            },
            {
                "id": "b022",
                "content": "Team allocates 20% of each sprint to debt reduction.",
                "tags": ["team", "workflow"],
            },
            {
                "id": "b023",
                "content": "User documents debt items with estimated effort and business impact.",
                "tags": ["personal"],
            },
            {
                "id": "b024",
                "content": "User prefers to address debt proactively rather than reactively.",
                "tags": ["personal"],
            },
            {
                "id": "b025",
                "content": "User's team celebrates when a long-standing debt item is closed.",
                "tags": ["team"],
            },
        ],
        relevant_ids=["b021", "b022", "b023", "b024"],
    ),
    # ================================================================
    # Variable haystack: 10-fact (medium) — prefix d
    # ================================================================
    RetrievalCase(
        query="What container tools does the user's team rely on?",
        facts=[
            {
                "id": "d001",
                "content": "Team runs services in Docker containers.",
                "tags": ["devops", "team"],
            },
            {
                "id": "d002",
                "content": "Kubernetes orchestrates all production pods.",
                "tags": ["devops"],
            },
            {
                "id": "d003",
                "content": "Helm charts manage Kubernetes deployments.",
                "tags": ["devops"],
            },
            {"id": "d004", "content": "User jogs in the park every morning.", "tags": ["personal"]},
            {"id": "d005", "content": "User reads sci-fi novels before bed.", "tags": ["personal"]},
            {"id": "d006", "content": "User's cat sleeps on the keyboard.", "tags": ["personal"]},
            {"id": "d007", "content": "User drinks green tea at lunch.", "tags": ["food"]},
            {"id": "d008", "content": "User's monitor is 27 inches.", "tags": ["monitoring"]},
            {"id": "d009", "content": "User has a library card.", "tags": ["personal"]},
            {"id": "d010", "content": "User collects vinyl records.", "tags": ["personal"]},
        ],
        relevant_ids=["d001", "d002", "d003"],
    ),
    RetrievalCase(
        query="What cloud provider does the team use?",
        facts=[
            {
                "id": "d011",
                "content": "Team deploys to AWS using EKS clusters.",
                "tags": ["ci-cd", "cloud", "team"],
            },
            {"id": "d012", "content": "S3 buckets store all build artifacts.", "tags": ["cloud"]},
            {
                "id": "d013",
                "content": "CloudWatch handles logging and metrics.",
                "tags": ["cloud", "logging"],
            },
            {"id": "d014", "content": "User takes the bus to work.", "tags": ["lifestyle"]},
            {"id": "d015", "content": "User enjoys photography on weekends.", "tags": ["personal"]},
            {"id": "d016", "content": "User's desk has a cactus plant.", "tags": ["personal"]},
            {"id": "d017", "content": "User plays chess online.", "tags": ["personal"]},
            {"id": "d018", "content": "User bought a new backpack.", "tags": ["personal"]},
            {"id": "d019", "content": "User's phone runs Android.", "tags": ["tools"]},
            {"id": "d020", "content": "User subscribes to Netflix.", "tags": ["personal"]},
        ],
        relevant_ids=["d011", "d012", "d013"],
    ),
    RetrievalCase(
        query="How does the user handle logging?",
        facts=[
            {
                "id": "d021",
                "content": "User uses structlog for structured logging.",
                "tags": ["logging"],
            },
            {
                "id": "d022",
                "content": "Log level is INFO in production, DEBUG in dev.",
                "tags": ["error-handling", "logging"],
            },
            {
                "id": "d023",
                "content": "Logs are shipped to Elasticsearch via Fluentd.",
                "tags": ["logging"],
            },
            {"id": "d024", "content": "User's bicycle is a fixie.", "tags": ["personal"]},
            {"id": "d025", "content": "User meal-preps on Sundays.", "tags": ["personal"]},
            {"id": "d026", "content": "User has three siblings.", "tags": ["personal"]},
            {
                "id": "d027",
                "content": "User's wallpaper is a mountain photo.",
                "tags": ["personal"],
            },
            {"id": "d028", "content": "User dislikes cold weather.", "tags": ["personal"]},
            {
                "id": "d029",
                "content": "User wears glasses for computer work.",
                "tags": ["personal"],
            },
            {"id": "d030", "content": "User follows NBA basketball.", "tags": ["personal"]},
        ],
        relevant_ids=["d021", "d022", "d023"],
    ),
    RetrievalCase(
        query="What message queue does the project use?",
        facts=[
            {
                "id": "d031",
                "content": "Project uses RabbitMQ for async messaging.",
                "tags": ["messaging"],
            },
            {
                "id": "d032",
                "content": "Celery workers consume tasks from RabbitMQ.",
                "tags": ["messaging"],
            },
            {
                "id": "d033",
                "content": "Dead-letter queues catch failed messages.",
                "tags": ["messaging"],
            },
            {
                "id": "d034",
                "content": "User enjoys board games with friends.",
                "tags": ["personal"],
            },
            {
                "id": "d035",
                "content": "User's headphones are noise-cancelling.",
                "tags": ["personal"],
            },
            {"id": "d036", "content": "User prefers window seats on planes.", "tags": ["personal"]},
            {"id": "d037", "content": "User's favorite color is blue.", "tags": ["personal"]},
            {"id": "d038", "content": "User went skiing last winter.", "tags": ["personal"]},
            {"id": "d039", "content": "User has a standing desk converter.", "tags": ["personal"]},
            {"id": "d040", "content": "User likes dark chocolate.", "tags": ["personal"]},
        ],
        relevant_ids=["d031", "d032", "d033"],
    ),
    RetrievalCase(
        query="What frontend framework does the user know?",
        facts=[
            {
                "id": "d041",
                "content": "User builds frontends with React and Next.js.",
                "tags": ["frontend"],
            },
            {"id": "d042", "content": "User uses Tailwind CSS for styling.", "tags": ["frontend"]},
            {"id": "d043", "content": "User's favorite IDE is VS Code.", "tags": ["tools"]},
            {
                "id": "d044",
                "content": "User takes cold showers in the morning.",
                "tags": ["personal"],
            },
            {"id": "d045", "content": "User's apartment has two bedrooms.", "tags": ["personal"]},
            {
                "id": "d046",
                "content": "User donates to open-source projects.",
                "tags": ["personal"],
            },
            {
                "id": "d047",
                "content": "User meditates for ten minutes daily.",
                "tags": ["personal"],
            },
            {"id": "d048", "content": "User's car is a Toyota.", "tags": ["lifestyle"]},
            {
                "id": "d049",
                "content": "User subscribes to a coffee delivery.",
                "tags": ["personal"],
            },
            {"id": "d050", "content": "User walks 8000 steps per day.", "tags": ["personal"]},
        ],
        relevant_ids=["d041", "d042"],
    ),
    RetrievalCase(
        query="What API design style does the user prefer?",
        facts=[
            {
                "id": "d051",
                "content": "User designs REST APIs with OpenAPI specs.",
                "tags": ["api"],
            },
            {
                "id": "d052",
                "content": "User prefers JSON:API format for responses.",
                "tags": ["api", "code-quality"],
            },
            {
                "id": "d053",
                "content": "User versions APIs via URL path, not headers.",
                "tags": ["personal"],
            },
            {"id": "d054", "content": "User keeps a gratitude journal.", "tags": ["personal"]},
            {"id": "d055", "content": "User's mouse is ergonomic.", "tags": ["personal"]},
            {"id": "d056", "content": "User likes to visit art museums.", "tags": ["personal"]},
            {"id": "d057", "content": "User prefers aisle seats in cinemas.", "tags": ["personal"]},
            {"id": "d058", "content": "User's alarm is set to 6:30am.", "tags": ["personal"]},
            {"id": "d059", "content": "User streams music on Spotify.", "tags": ["hobby"]},
            {"id": "d060", "content": "User's favorite season is autumn.", "tags": ["personal"]},
        ],
        relevant_ids=["d051", "d052", "d053"],
    ),
    RetrievalCase(
        query="What security practices does the team follow?",
        facts=[
            {
                "id": "d061",
                "content": "Team runs SAST scans on every pull request.",
                "tags": ["git", "security", "team"],
            },
            {
                "id": "d062",
                "content": "Dependencies are audited weekly with Snyk.",
                "tags": ["security"],
            },
            {
                "id": "d063",
                "content": "Secrets are injected at runtime, never baked.",
                "tags": ["personal"],
            },
            {"id": "d064", "content": "User swims twice a week.", "tags": ["personal"]},
            {
                "id": "d065",
                "content": "User's partner is a graphic designer.",
                "tags": ["personal"],
            },
            {"id": "d066", "content": "User built a bookshelf last month.", "tags": ["personal"]},
            {
                "id": "d067",
                "content": "User reads the morning news over breakfast.",
                "tags": ["food"],
            },
            {"id": "d068", "content": "User has a gym membership.", "tags": ["health"]},
            {"id": "d069", "content": "User uses a Kindle for e-books.", "tags": ["personal"]},
            {"id": "d070", "content": "User prefers trains over planes.", "tags": ["personal"]},
        ],
        relevant_ids=["d061", "d062", "d063"],
    ),
    RetrievalCase(
        query="What caching strategy does the project use?",
        facts=[
            {
                "id": "d071",
                "content": "Redis caches hot database queries with a 5m TTL.",
                "tags": ["caching", "database"],
            },
            {"id": "d072", "content": "CDN caches static assets at the edge.", "tags": ["caching"]},
            {
                "id": "d073",
                "content": "User installed a new shelf in the garage.",
                "tags": ["personal"],
            },
            {
                "id": "d074",
                "content": "User watches cooking shows on YouTube.",
                "tags": ["personal"],
            },
            {
                "id": "d075",
                "content": "User's keyboard has Cherry MX switches.",
                "tags": ["personal"],
            },
            {
                "id": "d076",
                "content": "User's dentist appointment is next Friday.",
                "tags": ["personal"],
            },
            {"id": "d077", "content": "User planted tomatoes in the garden.", "tags": ["personal"]},
            {"id": "d078", "content": "User paints watercolors as a hobby.", "tags": ["personal"]},
            {"id": "d079", "content": "User keeps a sourdough starter.", "tags": ["personal"]},
            {"id": "d080", "content": "User walks the dog before work.", "tags": ["personal"]},
        ],
        relevant_ids=["d071", "d072"],
    ),
    RetrievalCase(
        query="How does the user handle database migrations?",
        facts=[
            {
                "id": "d081",
                "content": "Alembic manages all database schema migrations.",
                "tags": ["database"],
            },
            {
                "id": "d082",
                "content": "Migrations run automatically during deployment.",
                "tags": ["ci-cd"],
            },
            {
                "id": "d083",
                "content": "Rollback scripts are required for every migration.",
                "tags": ["database"],
            },
            {"id": "d084", "content": "User bakes sourdough bread on weekends.", "tags": ["food"]},
            {
                "id": "d085",
                "content": "User's favorite movie genre is thriller.",
                "tags": ["personal"],
            },
            {
                "id": "d086",
                "content": "User uses a whiteboard for brainstorming.",
                "tags": ["personal"],
            },
            {"id": "d087", "content": "User's router has a VPN configured.", "tags": ["personal"]},
            {"id": "d088", "content": "User brings lunch to the office daily.", "tags": ["food"]},
            {"id": "d089", "content": "User prefers hardcover books.", "tags": ["personal"]},
            {"id": "d090", "content": "User's apartment faces south.", "tags": ["personal"]},
        ],
        relevant_ids=["d081", "d082", "d083"],
    ),
    RetrievalCase(
        query="What error tracking does the team use?",
        facts=[
            {
                "id": "d091",
                "content": "Sentry captures all unhandled exceptions.",
                "tags": ["monitoring"],
            },
            {
                "id": "d092",
                "content": "PagerDuty alerts trigger for P1 incidents.",
                "tags": ["monitoring"],
            },
            {
                "id": "d093",
                "content": "Error budgets are reviewed each sprint.",
                "tags": ["error-handling", "workflow"],
            },
            {"id": "d094", "content": "User uses a fountain pen for notes.", "tags": ["personal"]},
            {
                "id": "d095",
                "content": "User's commute playlist is 45 minutes.",
                "tags": ["hobby", "lifestyle"],
            },
            {"id": "d096", "content": "User ice skates in winter.", "tags": ["personal"]},
            {"id": "d097", "content": "User prefers large coffee mugs.", "tags": ["personal"]},
            {"id": "d098", "content": "User's desk lamp is LED.", "tags": ["personal"]},
            {"id": "d099", "content": "User has a balcony garden.", "tags": ["personal"]},
            {
                "id": "d100",
                "content": "User does crossword puzzles on Sundays.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["d091", "d092", "d093"],
    ),
    # ================================================================
    # Variable haystack: 20-fact (large) — prefix l
    # ================================================================
    RetrievalCase(
        query="What version control workflow does the team follow?",
        facts=[
            {
                "id": "l001",
                "content": "Team uses Git with trunk-based development.",
                "tags": ["git", "team"],
            },
            {
                "id": "l002",
                "content": "Feature branches are short-lived, under 2 days.",
                "tags": ["personal"],
            },
            {"id": "l003", "content": "PRs require two approvals before merge.", "tags": ["git"]},
            {
                "id": "l004",
                "content": "User's coffee machine is a Nespresso.",
                "tags": ["personal"],
            },
            {"id": "l005", "content": "User reads manga on weekends.", "tags": ["personal"]},
            {
                "id": "l006",
                "content": "User's shower takes exactly 7 minutes.",
                "tags": ["personal"],
            },
            {
                "id": "l007",
                "content": "User subscribes to a meal kit service.",
                "tags": ["personal"],
            },
            {"id": "l008", "content": "User's phone case is blue.", "tags": ["personal"]},
            {"id": "l009", "content": "User watches F1 races on Sundays.", "tags": ["personal"]},
            {"id": "l010", "content": "User's favorite fruit is mango.", "tags": ["personal"]},
            {
                "id": "l011",
                "content": "User goes to the dentist every 6 months.",
                "tags": ["personal"],
            },
            {"id": "l012", "content": "User's winter coat is down-filled.", "tags": ["personal"]},
            {
                "id": "l013",
                "content": "User keeps a piggy bank on the shelf.",
                "tags": ["personal"],
            },
            {"id": "l014", "content": "User's calendar app is Fantastical.", "tags": ["personal"]},
            {"id": "l015", "content": "User's gym is a 10-minute walk away.", "tags": ["health"]},
            {"id": "l016", "content": "User plays guitar in a cover band.", "tags": ["hobby"]},
            {"id": "l017", "content": "User buys organic vegetables.", "tags": ["personal"]},
            {"id": "l018", "content": "User's favorite emoji is thumbs up.", "tags": ["personal"]},
            {"id": "l019", "content": "User naps for 20 minutes after lunch.", "tags": ["food"]},
            {"id": "l020", "content": "User wears sneakers to the office.", "tags": ["personal"]},
        ],
        relevant_ids=["l001", "l002", "l003"],
    ),
    RetrievalCase(
        query="How does the user manage task priorities?",
        facts=[
            {
                "id": "l021",
                "content": "User triages tasks using Eisenhower matrix.",
                "tags": ["personal"],
            },
            {"id": "l022", "content": "Critical bugs are fixed same day.", "tags": ["personal"]},
            {
                "id": "l023",
                "content": "Backlog grooming happens every Wednesday.",
                "tags": ["workflow"],
            },
            {
                "id": "l024",
                "content": "User's umbrella is always in the bag.",
                "tags": ["personal"],
            },
            {"id": "l025", "content": "User takes vitamins every morning.", "tags": ["personal"]},
            {
                "id": "l026",
                "content": "User has a poster of Tokyo on the wall.",
                "tags": ["personal"],
            },
            {
                "id": "l027",
                "content": "User cleans the apartment on Saturdays.",
                "tags": ["personal"],
            },
            {"id": "l028", "content": "User's toothbrush is electric.", "tags": ["personal"]},
            {"id": "l029", "content": "User walks to the supermarket.", "tags": ["personal"]},
            {"id": "l030", "content": "User likes pineapple on pizza.", "tags": ["food"]},
            {"id": "l031", "content": "User has a rubber duck on the desk.", "tags": ["personal"]},
            {"id": "l032", "content": "User's password manager is Bitwarden.", "tags": ["team"]},
            {"id": "l033", "content": "User wears a smartwatch.", "tags": ["personal"]},
            {"id": "l034", "content": "User's pillow is memory foam.", "tags": ["personal"]},
            {
                "id": "l035",
                "content": "User prefers paper towels over cloth.",
                "tags": ["personal"],
            },
            {"id": "l036", "content": "User's bike helmet is neon green.", "tags": ["personal"]},
            {
                "id": "l037",
                "content": "User keeps snacks in the desk drawer.",
                "tags": ["personal"],
            },
            {"id": "l038", "content": "User's bookshelf is sorted by color.", "tags": ["personal"]},
            {
                "id": "l039",
                "content": "User uses a standing mat at the desk.",
                "tags": ["personal"],
            },
            {"id": "l040", "content": "User's favorite number is 7.", "tags": ["personal"]},
        ],
        relevant_ids=["l021", "l022", "l023"],
    ),
    RetrievalCase(
        query="What does the user think about microservices?",
        facts=[
            {
                "id": "l041",
                "content": "User prefers microservices over monoliths.",
                "tags": ["personal"],
            },
            {
                "id": "l042",
                "content": "Each microservice owns its own database.",
                "tags": ["architecture", "database"],
            },
            {
                "id": "l043",
                "content": "gRPC is used for inter-service communication.",
                "tags": ["api"],
            },
            {
                "id": "l044",
                "content": "User's favorite ice cream is pistachio.",
                "tags": ["personal"],
            },
            {
                "id": "l045",
                "content": "User vacuums the apartment every Thursday.",
                "tags": ["personal"],
            },
            {"id": "l046", "content": "User watches nature documentaries.", "tags": ["personal"]},
            {"id": "l047", "content": "User's socks are always mismatched.", "tags": ["personal"]},
            {"id": "l048", "content": "User tips generously at restaurants.", "tags": ["personal"]},
            {
                "id": "l049",
                "content": "User prefers window seats on the train.",
                "tags": ["personal"],
            },
            {
                "id": "l050",
                "content": "User stores leftovers in glass containers.",
                "tags": ["personal"],
            },
            {"id": "l051", "content": "User's ringtone is a classic bell.", "tags": ["personal"]},
            {"id": "l052", "content": "User hangs laundry to dry.", "tags": ["personal"]},
            {"id": "l053", "content": "User waters the plants on Mondays.", "tags": ["personal"]},
            {"id": "l054", "content": "User's tea is always without sugar.", "tags": ["personal"]},
            {"id": "l055", "content": "User listens to lo-fi while coding.", "tags": ["hobby"]},
            {
                "id": "l056",
                "content": "User checks the weather app every morning.",
                "tags": ["personal"],
            },
            {"id": "l057", "content": "User floss daily.", "tags": ["personal"]},
            {"id": "l058", "content": "User's shower gel is unscented.", "tags": ["personal"]},
            {"id": "l059", "content": "User's keys are on a carabiner clip.", "tags": ["personal"]},
            {"id": "l060", "content": "User prefers sparkling water.", "tags": ["personal"]},
        ],
        relevant_ids=["l041", "l042", "l043"],
    ),
    RetrievalCase(
        query="How does the team do on-call rotations?",
        facts=[
            {
                "id": "l061",
                "content": "On-call rotation is one week per engineer.",
                "tags": ["incident-response", "team"],
            },
            {
                "id": "l062",
                "content": "On-call engineer gets a comp day after shift.",
                "tags": ["incident-response", "team"],
            },
            {
                "id": "l063",
                "content": "Runbooks are stored in Confluence.",
                "tags": ["documentation"],
            },
            {
                "id": "l064",
                "content": "Escalation goes to team lead after 30 minutes.",
                "tags": ["team"],
            },
            {"id": "l065", "content": "User's yoga mat is purple.", "tags": ["health"]},
            {"id": "l066", "content": "User prefers whole wheat bread.", "tags": ["food"]},
            {"id": "l067", "content": "User's sunglasses are polarized.", "tags": ["personal"]},
            {"id": "l068", "content": "User uses bamboo chopsticks.", "tags": ["personal"]},
            {
                "id": "l069",
                "content": "User's doorbell is a smart Ring device.",
                "tags": ["personal"],
            },
            {
                "id": "l070",
                "content": "User folds laundry while watching TV.",
                "tags": ["personal"],
            },
            {"id": "l071", "content": "User's trash day is Tuesday.", "tags": ["personal"]},
            {"id": "l072", "content": "User uses reusable grocery bags.", "tags": ["personal"]},
            {
                "id": "l073",
                "content": "User's thermostat is set to 22 degrees.",
                "tags": ["personal"],
            },
            {"id": "l074", "content": "User prefers ebooks over audiobooks.", "tags": ["personal"]},
            {"id": "l075", "content": "User's dishwasher runs every night.", "tags": ["personal"]},
            {"id": "l076", "content": "User hangs a calendar on the fridge.", "tags": ["personal"]},
            {"id": "l077", "content": "User's belt is brown leather.", "tags": ["personal"]},
            {
                "id": "l078",
                "content": "User drinks water from a metal bottle.",
                "tags": ["personal"],
            },
            {"id": "l079", "content": "User's wallet is minimalist.", "tags": ["personal"]},
            {"id": "l080", "content": "User irons shirts on Sunday evening.", "tags": ["personal"]},
        ],
        relevant_ids=["l061", "l062", "l063", "l064"],
    ),
    RetrievalCase(
        query="What is the team's sprint cadence?",
        facts=[
            {"id": "l081", "content": "Sprints are two weeks long.", "tags": ["personal"]},
            {
                "id": "l082",
                "content": "Sprint planning is on Monday morning.",
                "tags": ["workflow"],
            },
            {
                "id": "l083",
                "content": "Retrospective happens every other Friday.",
                "tags": ["personal"],
            },
            {
                "id": "l084",
                "content": "User's laundry detergent is fragrance-free.",
                "tags": ["personal"],
            },
            {"id": "l085", "content": "User uses a silk pillowcase.", "tags": ["personal"]},
            {"id": "l086", "content": "User's favorite soup is tomato.", "tags": ["personal"]},
            {"id": "l087", "content": "User prefers bar soap over liquid.", "tags": ["personal"]},
            {
                "id": "l088",
                "content": "User hangs pictures with command strips.",
                "tags": ["personal"],
            },
            {"id": "l089", "content": "User's vacuum is a robot.", "tags": ["personal"]},
            {
                "id": "l090",
                "content": "User keeps a flashlight in the drawer.",
                "tags": ["personal"],
            },
            {"id": "l091", "content": "User recycles paper and plastic.", "tags": ["personal"]},
            {"id": "l092", "content": "User's mouse pad has a wrist rest.", "tags": ["api"]},
            {"id": "l093", "content": "User charges phone overnight.", "tags": ["personal"]},
            {"id": "l094", "content": "User's couch is grey.", "tags": ["personal"]},
            {"id": "l095", "content": "User owns a portable charger.", "tags": ["personal"]},
            {"id": "l096", "content": "User's shampoo is sulfate-free.", "tags": ["personal"]},
            {"id": "l097", "content": "User keeps a spare tire in the trunk.", "tags": ["git"]},
            {"id": "l098", "content": "User's curtains are blackout.", "tags": ["personal"]},
            {"id": "l099", "content": "User uses a French press for coffee.", "tags": ["personal"]},
            {"id": "l100", "content": "User's shoe rack holds eight pairs.", "tags": ["personal"]},
        ],
        relevant_ids=["l081", "l082", "l083"],
    ),
    RetrievalCase(
        query="How does the team handle feature flags?",
        facts=[
            {
                "id": "l101",
                "content": "Feature flags are managed via LaunchDarkly.",
                "tags": ["feature-flags"],
            },
            {
                "id": "l102",
                "content": "Flags are cleaned up within 30 days of launch.",
                "tags": ["personal"],
            },
            {
                "id": "l103",
                "content": "Kill switches exist for every critical path.",
                "tags": ["personal"],
            },
            {
                "id": "l104",
                "content": "User folds shirts using the KonMari method.",
                "tags": ["personal"],
            },
            {"id": "l105", "content": "User's rain boots are yellow.", "tags": ["personal"]},
            {"id": "l106", "content": "User prefers crunchy peanut butter.", "tags": ["personal"]},
            {"id": "l107", "content": "User stores spices alphabetically.", "tags": ["personal"]},
            {"id": "l108", "content": "User's headband is navy blue.", "tags": ["personal"]},
            {"id": "l109", "content": "User prefers cold brew coffee.", "tags": ["personal"]},
            {"id": "l110", "content": "User's welcome mat says 'Hello'.", "tags": ["personal"]},
            {"id": "l111", "content": "User uses a lint roller daily.", "tags": ["code-quality"]},
            {"id": "l112", "content": "User's fan is tower style.", "tags": ["personal"]},
            {"id": "l113", "content": "User keeps sunscreen in the bag.", "tags": ["personal"]},
            {"id": "l114", "content": "User's notebook is dotted grid.", "tags": ["personal"]},
            {"id": "l115", "content": "User prefers mechanical pencils.", "tags": ["personal"]},
            {"id": "l116", "content": "User's cutting board is bamboo.", "tags": ["personal"]},
            {"id": "l117", "content": "User hangs keys by the door.", "tags": ["personal"]},
            {"id": "l118", "content": "User's blanket is fleece.", "tags": ["personal"]},
            {"id": "l119", "content": "User keeps a first-aid kit at home.", "tags": ["personal"]},
            {"id": "l120", "content": "User's bathrobe is cotton.", "tags": ["personal"]},
        ],
        relevant_ids=["l101", "l102", "l103"],
    ),
    RetrievalCase(
        query="What load testing tools does the team use?",
        facts=[
            {
                "id": "l121",
                "content": "Team uses Locust for load testing APIs.",
                "tags": ["performance", "team"],
            },
            {
                "id": "l122",
                "content": "Load tests run before every major release.",
                "tags": ["personal"],
            },
            {
                "id": "l123",
                "content": "P99 latency target is under 200ms.",
                "tags": ["performance"],
            },
            {"id": "l124", "content": "User's desk organizer is acrylic.", "tags": ["personal"]},
            {"id": "l125", "content": "User uses a timer when cooking.", "tags": ["personal"]},
            {"id": "l126", "content": "User's favorite card game is UNO.", "tags": ["personal"]},
            {"id": "l127", "content": "User sleeps with white noise.", "tags": ["personal"]},
            {"id": "l128", "content": "User's laundry basket is wicker.", "tags": ["personal"]},
            {
                "id": "l129",
                "content": "User keeps a stain remover pen handy.",
                "tags": ["personal"],
            },
            {"id": "l130", "content": "User prefers cotton towels.", "tags": ["personal"]},
            {"id": "l131", "content": "User's toaster has four slots.", "tags": ["personal"]},
            {"id": "l132", "content": "User uses a paper planner too.", "tags": ["personal"]},
            {"id": "l133", "content": "User's mailbox is at the lobby.", "tags": ["personal"]},
            {"id": "l134", "content": "User's measuring cups are stainless.", "tags": ["personal"]},
            {
                "id": "l135",
                "content": "User keeps a magnet board in the kitchen.",
                "tags": ["personal"],
            },
            {"id": "l136", "content": "User's shower curtain is clear.", "tags": ["personal"]},
            {"id": "l137", "content": "User prefers loose-leaf tea.", "tags": ["personal"]},
            {"id": "l138", "content": "User's bicycle lock is U-shaped.", "tags": ["personal"]},
            {"id": "l139", "content": "User stores batteries in a drawer.", "tags": ["personal"]},
            {"id": "l140", "content": "User's coasters are cork.", "tags": ["personal"]},
        ],
        relevant_ids=["l121", "l122", "l123"],
    ),
    RetrievalCase(
        query="How does the team share knowledge internally?",
        facts=[
            {
                "id": "l141",
                "content": "Weekly tech talks are held on Thursdays.",
                "tags": ["personal"],
            },
            {
                "id": "l142",
                "content": "Team wiki is hosted on Notion.",
                "tags": ["documentation", "knowledge-sharing", "team"],
            },
            {
                "id": "l143",
                "content": "New hires do a code walkthrough in week one.",
                "tags": ["knowledge-sharing"],
            },
            {
                "id": "l144",
                "content": "Pair programming sessions happen twice a week.",
                "tags": ["knowledge-sharing"],
            },
            {"id": "l145", "content": "User's shower head is rain-style.", "tags": ["personal"]},
            {"id": "l146", "content": "User uses a tongue scraper.", "tags": ["personal"]},
            {"id": "l147", "content": "User's pan collection is cast iron.", "tags": ["personal"]},
            {"id": "l148", "content": "User drinks oat milk in coffee.", "tags": ["personal"]},
            {"id": "l149", "content": "User's yoga class is on Wednesdays.", "tags": ["health"]},
            {"id": "l150", "content": "User prefers bar deodorant.", "tags": ["personal"]},
            {
                "id": "l151",
                "content": "User's fire extinguisher is in the kitchen.",
                "tags": ["personal"],
            },
            {
                "id": "l152",
                "content": "User's bird feeder is on the balcony.",
                "tags": ["personal"],
            },
            {
                "id": "l153",
                "content": "User stores flour in airtight containers.",
                "tags": ["personal"],
            },
            {"id": "l154", "content": "User's doormat is coir.", "tags": ["personal"]},
            {"id": "l155", "content": "User keeps a shoe horn by the door.", "tags": ["personal"]},
            {"id": "l156", "content": "User's oven mitts are silicone.", "tags": ["personal"]},
            {"id": "l157", "content": "User uses a garlic press.", "tags": ["personal"]},
            {"id": "l158", "content": "User's bath mat is microfiber.", "tags": ["personal"]},
            {
                "id": "l159",
                "content": "User keeps a plant mister on the shelf.",
                "tags": ["personal"],
            },
            {
                "id": "l160",
                "content": "User's sponge holder is stainless steel.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["l141", "l142", "l143", "l144"],
    ),
    RetrievalCase(
        query="What does the user's CI pipeline look like?",
        facts=[
            {
                "id": "l161",
                "content": "CI runs lint, type checks, and tests in parallel.",
                "tags": ["ci-cd", "code-quality"],
            },
            {
                "id": "l162",
                "content": "Build artifacts are cached between runs.",
                "tags": ["caching"],
            },
            {
                "id": "l163",
                "content": "Pipeline fails fast on the first error.",
                "tags": ["ci-cd", "error-handling"],
            },
            {"id": "l164", "content": "User's spatula is heat-resistant.", "tags": ["personal"]},
            {
                "id": "l165",
                "content": "User keeps napkins in a holder on the table.",
                "tags": ["personal"],
            },
            {"id": "l166", "content": "User's comforter is goose down.", "tags": ["personal"]},
            {"id": "l167", "content": "User prefers glass Tupperware.", "tags": ["personal"]},
            {"id": "l168", "content": "User's teapot is ceramic.", "tags": ["personal"]},
            {
                "id": "l169",
                "content": "User keeps a step stool in the closet.",
                "tags": ["personal"],
            },
            {"id": "l170", "content": "User's ironing board folds flat.", "tags": ["personal"]},
            {"id": "l171", "content": "User uses a mesh laundry bag.", "tags": ["personal"]},
            {"id": "l172", "content": "User's can opener is manual.", "tags": ["personal"]},
            {
                "id": "l173",
                "content": "User keeps a box of baking soda in the fridge.",
                "tags": ["personal"],
            },
            {"id": "l174", "content": "User's soap dispenser is touchless.", "tags": ["personal"]},
            {"id": "l175", "content": "User hangs a clock above the desk.", "tags": ["personal"]},
            {
                "id": "l176",
                "content": "User's extension cord has surge protection.",
                "tags": ["personal"],
            },
            {"id": "l177", "content": "User keeps a dustpan under the sink.", "tags": ["personal"]},
            {"id": "l178", "content": "User's hand towels are grey.", "tags": ["personal"]},
            {"id": "l179", "content": "User prefers twist-off bottle caps.", "tags": ["personal"]},
            {"id": "l180", "content": "User's coat rack holds five hooks.", "tags": ["personal"]},
        ],
        relevant_ids=["l161", "l162", "l163"],
    ),
    RetrievalCase(
        query="How does the user approach performance optimization?",
        facts=[
            {
                "id": "l181",
                "content": "User profiles code with py-spy before optimizing.",
                "tags": ["performance"],
            },
            {
                "id": "l182",
                "content": "User avoids premature optimization.",
                "tags": ["performance"],
            },
            {
                "id": "l183",
                "content": "Hot paths are benchmarked with pytest-benchmark.",
                "tags": ["performance", "python"],
            },
            {"id": "l184", "content": "User's potato peeler is Y-shaped.", "tags": ["personal"]},
            {
                "id": "l185",
                "content": "User keeps a lint brush in the closet.",
                "tags": ["code-quality"],
            },
            {
                "id": "l186",
                "content": "User's blender makes smoothies daily.",
                "tags": ["personal"],
            },
            {"id": "l187", "content": "User prefers glass water bottles.", "tags": ["personal"]},
            {"id": "l188", "content": "User's bookmarks are leather.", "tags": ["personal"]},
            {"id": "l189", "content": "User keeps stamps in the desk.", "tags": ["personal"]},
            {"id": "l190", "content": "User's pencil case is canvas.", "tags": ["personal"]},
            {
                "id": "l191",
                "content": "User uses a wooden spoon for cooking.",
                "tags": ["personal"],
            },
            {"id": "l192", "content": "User's clothespins are wooden.", "tags": ["personal"]},
            {"id": "l193", "content": "User keeps a thermos at the office.", "tags": ["personal"]},
            {"id": "l194", "content": "User's scissors are titanium coated.", "tags": ["personal"]},
            {"id": "l195", "content": "User prefers unlined notebooks.", "tags": ["personal"]},
            {"id": "l196", "content": "User's trash can has a foot pedal.", "tags": ["personal"]},
            {
                "id": "l197",
                "content": "User keeps a power strip under the desk.",
                "tags": ["personal"],
            },
            {"id": "l198", "content": "User's key ring has a bottle opener.", "tags": ["personal"]},
            {"id": "l199", "content": "User prefers bar shampoo.", "tags": ["personal"]},
            {"id": "l200", "content": "User's dish rack is foldable.", "tags": ["personal"]},
        ],
        relevant_ids=["l181", "l182", "l183"],
    ),
    # ================================================================
    # Variable haystack: 3-fact (tiny) — prefix t
    # ================================================================
    RetrievalCase(
        query="What text editor does the user prefer?",
        facts=[
            {
                "id": "t001",
                "content": "User's primary editor is Neovim with LazyVim.",
                "tags": ["tools"],
            },
            {"id": "t002", "content": "User drinks espresso after lunch.", "tags": ["food"]},
            {"id": "t003", "content": "User's desk is walnut wood.", "tags": ["personal"]},
        ],
        relevant_ids=["t001"],
    ),
    RetrievalCase(
        query="What is the user's morning routine?",
        facts=[
            {
                "id": "t004",
                "content": "User wakes at 6am and does 15 minutes of yoga.",
                "tags": ["health"],
            },
            {
                "id": "t005",
                "content": "User eats oatmeal with berries for breakfast.",
                "tags": ["food"],
            },
            {"id": "t006", "content": "User's favorite band is Radiohead.", "tags": ["personal"]},
        ],
        relevant_ids=["t004", "t005"],
    ),
    RetrievalCase(
        query="What language does the user speak at home?",
        facts=[
            {
                "id": "t007",
                "content": "User speaks Russian at home with family.",
                "tags": ["personal"],
            },
            {
                "id": "t008",
                "content": "User also speaks fluent English at work.",
                "tags": ["personal"],
            },
            {"id": "t009", "content": "User's shoes are size 43.", "tags": ["personal"]},
        ],
        relevant_ids=["t007", "t008"],
    ),
    RetrievalCase(
        query="What keyboard does the user have?",
        facts=[
            {
                "id": "t010",
                "content": "User types on a Keychron K2 mechanical keyboard.",
                "tags": ["personal"],
            },
            {
                "id": "t011",
                "content": "User's favorite holiday is New Year.",
                "tags": ["lifestyle"],
            },
            {
                "id": "t012",
                "content": "User subscribes to a cheese delivery.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["t010"],
    ),
    RetrievalCase(
        query="Where did the user go on vacation?",
        facts=[
            {
                "id": "t013",
                "content": "User vacationed in Georgia last summer.",
                "tags": ["lifestyle"],
            },
            {
                "id": "t014",
                "content": "User hiked in the Caucasus mountains for a week.",
                "tags": ["lifestyle"],
            },
            {
                "id": "t015",
                "content": "User's dentist is near the metro station.",
                "tags": ["lifestyle"],
            },
        ],
        relevant_ids=["t013", "t014"],
    ),
    RetrievalCase(
        query="What shell does the user use?",
        facts=[
            {
                "id": "t016",
                "content": "User uses zsh with oh-my-zsh and Powerlevel10k.",
                "tags": ["tools"],
            },
            {"id": "t017", "content": "User's rain jacket is Gore-Tex.", "tags": ["personal"]},
            {
                "id": "t018",
                "content": "User keeps a spare charger at the office.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["t016"],
    ),
    RetrievalCase(
        query="What is the user's favorite food?",
        facts=[
            {
                "id": "t019",
                "content": "User loves Georgian khachapuri and khinkali.",
                "tags": ["food"],
            },
            {
                "id": "t020",
                "content": "User also enjoys Japanese ramen on cold days.",
                "tags": ["food"],
            },
            {"id": "t021", "content": "User's mattress is orthopedic.", "tags": ["personal"]},
        ],
        relevant_ids=["t019", "t020"],
    ),
    RetrievalCase(
        query="What is the user's operating system?",
        facts=[
            {
                "id": "t022",
                "content": "User runs Arch Linux on the personal laptop.",
                "tags": ["tools"],
            },
            {"id": "t023", "content": "Work machine runs macOS Sonoma.", "tags": ["tools"]},
            {"id": "t024", "content": "User's wallet is RFID-blocking.", "tags": ["personal"]},
        ],
        relevant_ids=["t022", "t023"],
    ),
    RetrievalCase(
        query="How tall is the user?",
        facts=[
            {"id": "t025", "content": "User is 182cm tall.", "tags": ["personal"]},
            {"id": "t026", "content": "User weighs about 78 kilograms.", "tags": ["personal"]},
            {"id": "t027", "content": "User's guitar is acoustic.", "tags": ["hobby"]},
        ],
        relevant_ids=["t025"],
    ),
    RetrievalCase(
        query="What car does the user drive?",
        facts=[
            {"id": "t028", "content": "User drives a Skoda Octavia.", "tags": ["lifestyle"]},
            {
                "id": "t029",
                "content": "User parks in the underground garage.",
                "tags": ["personal"],
            },
            {"id": "t030", "content": "User's scarf is cashmere.", "tags": ["personal"]},
        ],
        relevant_ids=["t028", "t029"],
    ),
    # ================================================================
    # Adversarial: near-miss / keyword-overlap distractors — prefix n
    # ================================================================
    RetrievalCase(
        query="What database does the user's project use?",
        facts=[
            {
                "id": "n001",
                "content": "Production backend runs on PostgreSQL 15.",
                "tags": ["database"],
            },
            {
                "id": "n002",
                "content": "User studied database theory in university.",
                "tags": ["database", "education"],
            },
            {
                "id": "n003",
                "content": "User read a blog post about database sharding.",
                "tags": ["database"],
            },
            {
                "id": "n004",
                "content": "Team discussed migrating the database last month.",
                "tags": ["database", "team"],
            },
            {
                "id": "n005",
                "content": "User bookmarked a database optimization guide.",
                "tags": ["database", "performance"],
            },
        ],
        relevant_ids=["n001"],
    ),
    RetrievalCase(
        query="What testing framework does the user prefer?",
        facts=[
            {
                "id": "n006",
                "content": "User runs all Python tests with pytest.",
                "tags": ["python"],
            },
            {
                "id": "n007",
                "content": "User read an article comparing testing frameworks.",
                "tags": ["personal"],
            },
            {
                "id": "n008",
                "content": "User attended a talk about testing best practices.",
                "tags": ["personal"],
            },
            {
                "id": "n009",
                "content": "Testing new features takes extra sprint time.",
                "tags": ["workflow"],
            },
            {
                "id": "n010",
                "content": "User bookmarked a testing patterns cheat sheet.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["n006"],
    ),
    RetrievalCase(
        query="Where does the user deploy applications?",
        facts=[
            {
                "id": "n011",
                "content": "All services deploy to AWS EKS clusters.",
                "tags": ["ci-cd", "cloud"],
            },
            {
                "id": "n012",
                "content": "User wrote a blog post about deployment strategies.",
                "tags": ["ci-cd"],
            },
            {"id": "n013", "content": "Deploy pipeline takes about 12 minutes.", "tags": ["ci-cd"]},
            {
                "id": "n014",
                "content": "User presented on blue-green deployment patterns.",
                "tags": ["ci-cd"],
            },
            {
                "id": "n015",
                "content": "Team debated deploy frequency at the retro.",
                "tags": ["ci-cd", "metrics", "team"],
            },
        ],
        relevant_ids=["n011"],
    ),
    RetrievalCase(
        query="What monitoring does the team have?",
        facts=[
            {
                "id": "n016",
                "content": "Grafana dashboards show service health metrics.",
                "tags": ["monitoring"],
            },
            {
                "id": "n017",
                "content": "Prometheus scrapes metrics every 15 seconds.",
                "tags": ["monitoring"],
            },
            {
                "id": "n018",
                "content": "User compared monitoring tools at a conference.",
                "tags": ["education"],
            },
            {
                "id": "n019",
                "content": "Team discussed expanding monitoring coverage.",
                "tags": ["team", "testing"],
            },
            {
                "id": "n020",
                "content": "Monitoring alert fatigue was raised at the retro.",
                "tags": ["monitoring", "workflow"],
            },
        ],
        relevant_ids=["n016", "n017"],
    ),
    RetrievalCase(
        query="How does the user handle code reviews?",
        facts=[
            {
                "id": "n021",
                "content": "User reviews every PR within one business day.",
                "tags": ["code-review"],
            },
            {
                "id": "n022",
                "content": "User read a guide about effective code reviews.",
                "tags": ["personal"],
            },
            {
                "id": "n023",
                "content": "Code review backlog was discussed at standup.",
                "tags": ["code-review", "workflow"],
            },
            {
                "id": "n024",
                "content": "User suggested pairing instead of async reviews.",
                "tags": ["personal"],
            },
            {
                "id": "n025",
                "content": "Review turnaround is tracked as a team metric.",
                "tags": ["code-review", "monitoring", "team"],
            },
        ],
        relevant_ids=["n021"],
    ),
    RetrievalCase(
        query="What CI system does the team use?",
        facts=[
            {
                "id": "n026",
                "content": "GitHub Actions runs all CI checks on every PR.",
                "tags": ["ci-cd", "code-review"],
            },
            {
                "id": "n027",
                "content": "CI build times have increased by 30% this quarter.",
                "tags": ["ci-cd"],
            },
            {
                "id": "n028",
                "content": "Team wants to migrate CI to a faster platform.",
                "tags": ["ci-cd", "team"],
            },
            {
                "id": "n029",
                "content": "CI flakiness was the top retro complaint.",
                "tags": ["ci-cd", "workflow"],
            },
            {
                "id": "n030",
                "content": "User investigated CI caching improvements.",
                "tags": ["caching", "ci-cd"],
            },
        ],
        relevant_ids=["n026"],
    ),
    RetrievalCase(
        query="What Python version does the project require?",
        facts=[
            {"id": "n031", "content": "Project requires Python 3.11 or newer.", "tags": ["python"]},
            {
                "id": "n032",
                "content": "User tested Python 3.12 but found a compatibility bug.",
                "tags": ["error-handling", "python"],
            },
            {
                "id": "n033",
                "content": "Python packaging ecosystem has many tools.",
                "tags": ["python"],
            },
            {
                "id": "n034",
                "content": "User's Python learning started ten years ago.",
                "tags": ["python"],
            },
            {
                "id": "n035",
                "content": "Python 3.13 release notes look promising.",
                "tags": ["python"],
            },
        ],
        relevant_ids=["n031"],
    ),
    RetrievalCase(
        query="What is the user's approach to error handling?",
        facts=[
            {
                "id": "n036",
                "content": "User wraps external API calls in try-except blocks.",
                "tags": ["api", "error-handling"],
            },
            {
                "id": "n037",
                "content": "Custom exception hierarchy inherits from AppError.",
                "tags": ["error-handling"],
            },
            {
                "id": "n038",
                "content": "Error messages should include context for debugging.",
                "tags": ["error-handling"],
            },
            {
                "id": "n039",
                "content": "User read about error handling in Go vs Python.",
                "tags": ["error-handling", "python"],
            },
            {
                "id": "n040",
                "content": "Error rates spiked during the last deploy.",
                "tags": ["ci-cd", "error-handling"],
            },
        ],
        relevant_ids=["n036", "n037", "n038"],
    ),
    RetrievalCase(
        query="What linter does the user use for Python?",
        facts=[
            {
                "id": "n041",
                "content": "User runs ruff for both linting and formatting.",
                "tags": ["python"],
            },
            {
                "id": "n042",
                "content": "User considered switching linters last year.",
                "tags": ["personal"],
            },
            {
                "id": "n043",
                "content": "Linting rules are configured in pyproject.toml.",
                "tags": ["packaging"],
            },
            {
                "id": "n044",
                "content": "User blogged about linter comparison results.",
                "tags": ["personal"],
            },
            {
                "id": "n045",
                "content": "Lint warnings are treated as errors in CI.",
                "tags": ["ci-cd", "code-quality"],
            },
        ],
        relevant_ids=["n041", "n043"],
    ),
    RetrievalCase(
        query="How does the user structure Python packages?",
        facts=[
            {
                "id": "n046",
                "content": "User uses src-layout with pyproject.toml.",
                "tags": ["packaging"],
            },
            {
                "id": "n047",
                "content": "Each module has an __init__.py with public API.",
                "tags": ["api", "packaging"],
            },
            {
                "id": "n048",
                "content": "User discussed package structure at a meetup.",
                "tags": ["packaging"],
            },
            {
                "id": "n049",
                "content": "Package naming follows PEP 8 conventions.",
                "tags": ["packaging", "python"],
            },
            {
                "id": "n050",
                "content": "User reviewed several package templates on GitHub.",
                "tags": ["packaging"],
            },
        ],
        relevant_ids=["n046", "n047"],
    ),
    # ================================================================
    # Adversarial: synonym / paraphrase — prefix n (continued)
    # ================================================================
    RetrievalCase(
        query="Where is the user employed?",
        facts=[
            {
                "id": "n051",
                "content": "User works as an engineer at a large bank.",
                "tags": ["team"],
            },
            {
                "id": "n052",
                "content": "User attended an employment law workshop.",
                "tags": ["education"],
            },
            {"id": "n053", "content": "User's employer provides free lunch.", "tags": ["food"]},
            {
                "id": "n054",
                "content": "User helped a friend with a job search.",
                "tags": ["personal"],
            },
            {
                "id": "n055",
                "content": "Employment market for developers is strong.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["n051"],
    ),
    RetrievalCase(
        query="What does the user do for exercise?",
        facts=[
            {"id": "n056", "content": "User runs 5km three times a week.", "tags": ["health"]},
            {
                "id": "n057",
                "content": "User does strength training on Tuesdays.",
                "tags": ["health"],
            },
            {
                "id": "n058",
                "content": "Exercise equipment takes up half the balcony.",
                "tags": ["health"],
            },
            {
                "id": "n059",
                "content": "User read about exercise and sleep quality.",
                "tags": ["health"],
            },
            {
                "id": "n060",
                "content": "User's gym buddy moved to another city.",
                "tags": ["health"],
            },
        ],
        relevant_ids=["n056", "n057"],
    ),
    RetrievalCase(
        query="What is the user's job title?",
        facts=[
            {
                "id": "n061",
                "content": "User holds the position of senior engineer.",
                "tags": ["career", "team"],
            },
            {
                "id": "n062",
                "content": "User's previous role was mid-level developer.",
                "tags": ["career"],
            },
            {
                "id": "n063",
                "content": "Job title conventions differ across companies.",
                "tags": ["career"],
            },
            {
                "id": "n064",
                "content": "User discussed career titles with the manager.",
                "tags": ["career", "team"],
            },
            {
                "id": "n065",
                "content": "User updated LinkedIn profile with new title.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["n061", "n062"],
    ),
    RetrievalCase(
        query="How does the user manage time?",
        facts=[
            {
                "id": "n066",
                "content": "User blocks focused work in 90-minute slots.",
                "tags": ["personal"],
            },
            {
                "id": "n067",
                "content": "User batches meetings on Tuesday and Thursday.",
                "tags": ["personal"],
            },
            {
                "id": "n068",
                "content": "Time tracking shows 6 hours of deep work daily.",
                "tags": ["personal"],
            },
            {
                "id": "n069",
                "content": "User read a book about managing time better.",
                "tags": ["hobby"],
            },
            {
                "id": "n070",
                "content": "Time zone differences complicate team syncs.",
                "tags": ["team"],
            },
        ],
        relevant_ids=["n066", "n067", "n068"],
    ),
    RetrievalCase(
        query="What does the user think about AI coding tools?",
        facts=[
            {
                "id": "n071",
                "content": "User uses Copilot for autocomplete daily.",
                "tags": ["ai-tools"],
            },
            {
                "id": "n072",
                "content": "User finds AI tools helpful for boilerplate.",
                "tags": ["personal"],
            },
            {
                "id": "n073",
                "content": "AI coding assistants were a conference topic.",
                "tags": ["ai-tools", "education"],
            },
            {
                "id": "n074",
                "content": "User worries about AI-generated code quality.",
                "tags": ["personal"],
            },
            {"id": "n075", "content": "AI tools market is growing rapidly.", "tags": ["personal"]},
        ],
        relevant_ids=["n071", "n072", "n074"],
    ),
    RetrievalCase(
        query="What does the user read for fun?",
        facts=[
            {
                "id": "n076",
                "content": "User enjoys science fiction by Lem and Asimov.",
                "tags": ["hobby"],
            },
            {
                "id": "n077",
                "content": "User reads one non-fiction book per month.",
                "tags": ["hobby"],
            },
            {
                "id": "n078",
                "content": "User's reading list is tracked in Goodreads.",
                "tags": ["personal"],
            },
            {
                "id": "n079",
                "content": "Reading comprehension research is fascinating.",
                "tags": ["personal"],
            },
            {"id": "n080", "content": "User's reading lamp is adjustable.", "tags": ["personal"]},
        ],
        relevant_ids=["n076", "n077"],
    ),
    RetrievalCase(
        query="What is the user's salary expectation?",
        facts=[
            {
                "id": "n081",
                "content": "User expects compensation above market median.",
                "tags": ["career"],
            },
            {"id": "n082", "content": "User discussed salary bands with HR.", "tags": ["career"]},
            {
                "id": "n083",
                "content": "Salary transparency law passed in the region.",
                "tags": ["career"],
            },
            {
                "id": "n084",
                "content": "User compared salaries on levels.fyi.",
                "tags": ["personal"],
            },
            {
                "id": "n085",
                "content": "Salary negotiations happen during annual review.",
                "tags": ["career", "code-review"],
            },
        ],
        relevant_ids=["n081", "n082"],
    ),
    RetrievalCase(
        query="What music does the user listen to while coding?",
        facts=[
            {
                "id": "n086",
                "content": "User plays lo-fi hip-hop playlists while coding.",
                "tags": ["hobby"],
            },
            {
                "id": "n087",
                "content": "User switches to classical music for deep focus.",
                "tags": ["hobby"],
            },
            {
                "id": "n088",
                "content": "Music streaming costs about $10 per month.",
                "tags": ["hobby"],
            },
            {
                "id": "n089",
                "content": "User discussed coding music in the team chat.",
                "tags": ["hobby", "team"],
            },
            {
                "id": "n090",
                "content": "Music taste varies widely across the team.",
                "tags": ["hobby", "team"],
            },
        ],
        relevant_ids=["n086", "n087"],
    ),
    RetrievalCase(
        query="How does the user learn new technologies?",
        facts=[
            {
                "id": "n091",
                "content": "User builds small prototype projects to learn.",
                "tags": ["education"],
            },
            {
                "id": "n092",
                "content": "User watches conference talks on YouTube.",
                "tags": ["education"],
            },
            {
                "id": "n093",
                "content": "Learning new tech is part of the career plan.",
                "tags": ["career"],
            },
            {
                "id": "n094",
                "content": "User set a learning budget of 5 hours per week.",
                "tags": ["personal"],
            },
            {
                "id": "n095",
                "content": "Team discusses new technologies at Friday demos.",
                "tags": ["team"],
            },
        ],
        relevant_ids=["n091", "n092", "n094"],
    ),
    RetrievalCase(
        query="What does the user eat for lunch?",
        facts=[
            {
                "id": "n096",
                "content": "User usually brings a homemade salad to work.",
                "tags": ["food"],
            },
            {
                "id": "n097",
                "content": "On Fridays the team orders sushi for lunch.",
                "tags": ["food", "team"],
            },
            {"id": "n098", "content": "Lunch break is from 1pm to 2pm.", "tags": ["food"]},
            {
                "id": "n099",
                "content": "Lunch delivery services are popular at the office.",
                "tags": ["food"],
            },
            {
                "id": "n100",
                "content": "User discussed lunch options with the new hire.",
                "tags": ["food", "team"],
            },
        ],
        relevant_ids=["n096", "n097"],
    ),
    # ================================================================
    # Adversarial: very short queries (1-3 words) — prefix q
    # ================================================================
    RetrievalCase(
        query="Python",
        facts=[
            {
                "id": "q001",
                "content": "User's primary language is Python 3.11.",
                "tags": ["personal", "python"],
            },
            {
                "id": "q002",
                "content": "User enjoys cooking pasta on Saturdays.",
                "tags": ["personal"],
            },
            {"id": "q003", "content": "User's cat naps on the windowsill.", "tags": ["personal"]},
            {"id": "q004", "content": "User reads the news on an iPad.", "tags": ["personal"]},
            {"id": "q005", "content": "User's backpack is waterproof.", "tags": ["personal"]},
        ],
        relevant_ids=["q001"],
    ),
    RetrievalCase(
        query="dog name",
        facts=[
            {
                "id": "q006",
                "content": "User's dog is called Pushkin, a labrador.",
                "tags": ["personal"],
            },
            {"id": "q007", "content": "User walks the dog twice a day.", "tags": ["personal"]},
            {"id": "q008", "content": "User's neighbor also has a labrador.", "tags": ["personal"]},
            {
                "id": "q009",
                "content": "User bought dog food in bulk.",
                "tags": ["food", "personal"],
            },
            {"id": "q010", "content": "User's vet is on the next street.", "tags": ["personal"]},
        ],
        relevant_ids=["q006"],
    ),
    RetrievalCase(
        query="salary",
        facts=[
            {
                "id": "q011",
                "content": "User's annual salary is above market median.",
                "tags": ["career"],
            },
            {"id": "q012", "content": "User negotiated a raise last quarter.", "tags": ["career"]},
            {
                "id": "q013",
                "content": "User tracks expenses in a spreadsheet.",
                "tags": ["personal"],
            },
            {"id": "q014", "content": "User's bonus is performance-based.", "tags": ["career"]},
            {"id": "q015", "content": "User donates to charity monthly.", "tags": ["personal"]},
        ],
        relevant_ids=["q011", "q012", "q014"],
    ),
    RetrievalCase(
        query="IDE",
        facts=[
            {
                "id": "q016",
                "content": "User codes in VS Code with vim keybindings.",
                "tags": ["tools"],
            },
            {
                "id": "q017",
                "content": "User tried IntelliJ but returned to VS Code.",
                "tags": ["tools"],
            },
            {"id": "q018", "content": "User's monitor is ultrawide.", "tags": ["monitoring"]},
            {
                "id": "q019",
                "content": "User prefers dark themes in all editors.",
                "tags": ["personal"],
            },
            {
                "id": "q020",
                "content": "User uses split panes when reviewing code.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["q016", "q017"],
    ),
    RetrievalCase(
        query="vacation",
        facts=[
            {
                "id": "q021",
                "content": "User took vacation in Tbilisi last August.",
                "tags": ["lifestyle"],
            },
            {
                "id": "q022",
                "content": "User plans a ski trip for the winter.",
                "tags": ["lifestyle"],
            },
            {"id": "q023", "content": "User's passport expires next year.", "tags": ["personal"]},
            {"id": "q024", "content": "User prefers direct flights.", "tags": ["personal"]},
            {"id": "q025", "content": "User's suitcase is carry-on size.", "tags": ["personal"]},
        ],
        relevant_ids=["q021", "q022"],
    ),
    RetrievalCase(
        query="team",
        facts=[
            {"id": "q026", "content": "User's team has 6 backend engineers.", "tags": ["team"]},
            {
                "id": "q027",
                "content": "Team standup is at 10am daily.",
                "tags": ["team", "workflow"],
            },
            {
                "id": "q028",
                "content": "Team uses Slack for daily communication.",
                "tags": ["communication", "team"],
            },
            {"id": "q029", "content": "User's desk is near the window.", "tags": ["personal"]},
            {
                "id": "q030",
                "content": "User's calendar is always color-coded.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["q026", "q027", "q028"],
    ),
    RetrievalCase(
        query="Docker",
        facts=[
            {"id": "q031", "content": "All services run in Docker containers.", "tags": ["devops"]},
            {
                "id": "q032",
                "content": "Docker Compose is used for local development.",
                "tags": ["devops"],
            },
            {
                "id": "q033",
                "content": "User's bike tire went flat yesterday.",
                "tags": ["personal"],
            },
            {
                "id": "q034",
                "content": "User prefers vinyl records over streaming.",
                "tags": ["personal"],
            },
            {"id": "q035", "content": "User's plant needs more sunlight.", "tags": ["personal"]},
        ],
        relevant_ids=["q031", "q032"],
    ),
    RetrievalCase(
        query="meetings",
        facts=[
            {
                "id": "q036",
                "content": "User keeps meetings under 30 minutes.",
                "tags": ["personal"],
            },
            {
                "id": "q037",
                "content": "User declines meetings without an agenda.",
                "tags": ["personal"],
            },
            {
                "id": "q038",
                "content": "User batches meetings on two days a week.",
                "tags": ["personal"],
            },
            {"id": "q039", "content": "User's headset is Jabra Evolve2.", "tags": ["personal"]},
            {
                "id": "q040",
                "content": "User's screen background is a forest.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["q036", "q037", "q038"],
    ),
    # ================================================================
    # Adversarial: very long queries (20+ words) — prefix q continued
    # ================================================================
    RetrievalCase(
        query=(
            "I remember the user mentioned something about how they"
            " manage configuration and environment variables in their"
            " production deployment pipeline"
        ),
        facts=[
            {
                "id": "q041",
                "content": "Env vars are injected via Vault at deploy time.",
                "tags": ["ci-cd", "security"],
            },
            {
                "id": "q042",
                "content": "Config files live in a separate Git repo.",
                "tags": ["configuration", "git"],
            },
            {
                "id": "q043",
                "content": "User prefers decaf coffee in the evening.",
                "tags": ["personal"],
            },
            {"id": "q044", "content": "User's lunch is usually leftovers.", "tags": ["food"]},
            {
                "id": "q045",
                "content": "User keeps an umbrella at the office.",
                "tags": ["personal"],
            },
            {
                "id": "q046",
                "content": "Pipeline deploys to staging first automatically.",
                "tags": ["ci-cd"],
            },
            {"id": "q047", "content": "User's phone wallpaper is a sunset.", "tags": ["personal"]},
        ],
        relevant_ids=["q041", "q042", "q046"],
    ),
    RetrievalCase(
        query=(
            "Can you tell me everything the user has said about"
            " their approach to writing automated tests, including"
            " unit tests, integration tests, and end-to-end tests?"
        ),
        facts=[
            {
                "id": "q048",
                "content": "User writes pytest unit tests before every PR.",
                "tags": ["code-review", "python"],
            },
            {
                "id": "q049",
                "content": "Integration tests run against a local Postgres.",
                "tags": ["database"],
            },
            {
                "id": "q050",
                "content": "E2E tests use Playwright for browser checks.",
                "tags": ["frontend", "testing"],
            },
            {
                "id": "q051",
                "content": "User prefers almond milk in smoothies.",
                "tags": ["personal"],
            },
            {
                "id": "q052",
                "content": "User's running shoes are Nike Pegasus.",
                "tags": ["personal"],
            },
            {
                "id": "q053",
                "content": "Test coverage must exceed 80% for merge.",
                "tags": ["git", "testing"],
            },
            {"id": "q054", "content": "User does not like cilantro.", "tags": ["personal"]},
        ],
        relevant_ids=["q048", "q049", "q050", "q053"],
    ),
    RetrievalCase(
        query=(
            "What was the situation when the user's team experienced"
            " that major production outage that affected customers"
            " and required an emergency fix late at night?"
        ),
        facts=[
            {
                "id": "q055",
                "content": "P1 outage on March 5th lasted three hours.",
                "tags": ["incident-response"],
            },
            {
                "id": "q056",
                "content": "Root cause was a failed database migration.",
                "tags": ["database"],
            },
            {
                "id": "q057",
                "content": "User rolled back the migration at 2am.",
                "tags": ["database"],
            },
            {
                "id": "q058",
                "content": "Post-mortem led to mandatory rollback scripts.",
                "tags": ["incident-response"],
            },
            {
                "id": "q059",
                "content": "User's favorite snack is dried mango.",
                "tags": ["personal"],
            },
            {"id": "q060", "content": "User subscribes to a magazine.", "tags": ["hobby"]},
        ],
        relevant_ids=["q055", "q056", "q057", "q058"],
    ),
    RetrievalCase(
        query=(
            "I want to know about the user's opinions and preferences"
            " regarding the use of type annotations and static type"
            " checking in Python codebases"
        ),
        facts=[
            {
                "id": "q061",
                "content": "User enables mypy strict mode on all projects.",
                "tags": ["python", "typing"],
            },
            {
                "id": "q062",
                "content": "Type hints are required on every function.",
                "tags": ["personal"],
            },
            {
                "id": "q063",
                "content": "User prefers Protocol over ABC for interfaces.",
                "tags": ["typing"],
            },
            {"id": "q064", "content": "User's tea brand is Ahmad.", "tags": ["personal"]},
            {"id": "q065", "content": "User walks to the bakery on Sundays.", "tags": ["personal"]},
            {
                "id": "q066",
                "content": "User prefers explicit types over type inference.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["q061", "q062", "q063", "q066"],
    ),
    RetrievalCase(
        query=(
            "How does the user typically go about debugging a"
            " difficult issue in production that is hard to"
            " reproduce locally in the development environment?"
        ),
        facts=[
            {
                "id": "q067",
                "content": "User checks Sentry traces for the error context.",
                "tags": ["error-handling", "monitoring"],
            },
            {
                "id": "q068",
                "content": "User reproduces bugs using production log replay.",
                "tags": ["personal"],
            },
            {
                "id": "q069",
                "content": "User adds temporary debug logging to staging.",
                "tags": ["error-handling", "logging"],
            },
            {
                "id": "q070",
                "content": "User's favorite park bench is under an oak.",
                "tags": ["personal"],
            },
            {
                "id": "q071",
                "content": "User prefers wired internet over WiFi.",
                "tags": ["personal"],
            },
            {
                "id": "q072",
                "content": "User pairs with a colleague for tricky bugs.",
                "tags": ["team"],
            },
        ],
        relevant_ids=["q067", "q068", "q069", "q072"],
    ),
    RetrievalCase(
        query=(
            "What are all the things the user has mentioned about"
            " how their team communicates, collaborates, and shares"
            " information on a daily and weekly basis?"
        ),
        facts=[
            {
                "id": "q073",
                "content": "Daily standup is async in a Slack thread.",
                "tags": ["communication", "workflow"],
            },
            {
                "id": "q074",
                "content": "Weekly sync is a 30-minute video call.",
                "tags": ["communication"],
            },
            {
                "id": "q075",
                "content": "Design docs are shared via Google Docs.",
                "tags": ["documentation"],
            },
            {
                "id": "q076",
                "content": "User prefers text over voice messages.",
                "tags": ["personal"],
            },
            {"id": "q077", "content": "User's desk plant is a fern.", "tags": ["personal"]},
            {
                "id": "q078",
                "content": "Decisions are recorded in an ADR log.",
                "tags": ["documentation"],
            },
        ],
        relevant_ids=["q073", "q074", "q075", "q076", "q078"],
    ),
    RetrievalCase(
        query=(
            "Tell me about the user's experience, thoughts, and"
            " current usage of various cloud services including"
            " compute, storage, networking, and managed databases"
        ),
        facts=[
            {
                "id": "q079",
                "content": "EC2 instances run behind an ALB load balancer.",
                "tags": ["cloud"],
            },
            {
                "id": "q080",
                "content": "S3 stores all media uploads and backups.",
                "tags": ["cloud"],
            },
            {"id": "q081", "content": "RDS manages the PostgreSQL database.", "tags": ["database"]},
            {
                "id": "q082",
                "content": "VPC peering connects staging and production.",
                "tags": ["cloud"],
            },
            {
                "id": "q083",
                "content": "User's umbrella is compact and black.",
                "tags": ["personal"],
            },
            {
                "id": "q084",
                "content": "User's favorite museum is Tretyakov Gallery.",
                "tags": ["personal"],
            },
            {
                "id": "q085",
                "content": "CloudFront CDN serves static assets.",
                "tags": ["caching", "cloud"],
            },
        ],
        relevant_ids=["q079", "q080", "q081", "q082", "q085"],
    ),
    # ================================================================
    # Adversarial: high-noise topical clusters — prefix h
    # All facts in the same domain; only 1-2 truly relevant.
    # ================================================================
    RetrievalCase(
        query="What Python formatter does the user prefer?",
        facts=[
            {
                "id": "h001",
                "content": "User formats Python code exclusively with ruff.",
                "tags": ["python"],
            },
            {
                "id": "h002",
                "content": "User runs Python type checks with mypy strict.",
                "tags": ["python"],
            },
            {
                "id": "h003",
                "content": "User writes Python tests using pytest framework.",
                "tags": ["python"],
            },
            {
                "id": "h004",
                "content": "User manages Python packages with poetry.",
                "tags": ["python"],
            },
            {
                "id": "h005",
                "content": "User documents Python code with Google docstrings.",
                "tags": ["python"],
            },
            {
                "id": "h006",
                "content": "User profiles Python code using py-spy.",
                "tags": ["performance", "python"],
            },
            {
                "id": "h007",
                "content": "User lints Python code with ruff check.",
                "tags": ["code-quality", "python"],
            },
            {
                "id": "h008",
                "content": "User deploys Python apps in Docker containers.",
                "tags": ["ci-cd", "devops", "python"],
            },
            {
                "id": "h009",
                "content": "User structures Python projects with src-layout.",
                "tags": ["packaging", "python"],
            },
            {
                "id": "h010",
                "content": "User debugs Python with print statements first.",
                "tags": ["python"],
            },
        ],
        relevant_ids=["h001"],
    ),
    RetrievalCase(
        query="What is the user's test coverage target?",
        facts=[
            {
                "id": "h011",
                "content": "Team requires 80% minimum test coverage to merge.",
                "tags": ["git", "team", "testing"],
            },
            {
                "id": "h012",
                "content": "User writes unit tests before submitting PRs.",
                "tags": ["personal"],
            },
            {
                "id": "h013",
                "content": "User prefers fixtures over setup/teardown.",
                "tags": ["personal"],
            },
            {
                "id": "h014",
                "content": "Integration tests use a local database.",
                "tags": ["database"],
            },
            {"id": "h015", "content": "E2E tests run nightly on staging.", "tags": ["testing"]},
            {
                "id": "h016",
                "content": "Test execution is parallelized with pytest-xdist.",
                "tags": ["python", "testing"],
            },
            {
                "id": "h017",
                "content": "Flaky tests are quarantined in a separate suite.",
                "tags": ["testing"],
            },
            {
                "id": "h018",
                "content": "User measures mutation testing coverage quarterly.",
                "tags": ["testing"],
            },
            {
                "id": "h019",
                "content": "Test data is generated with factory_boy.",
                "tags": ["testing"],
            },
            {
                "id": "h020",
                "content": "Snapshot tests validate API response schemas.",
                "tags": ["api"],
            },
        ],
        relevant_ids=["h011"],
    ),
    RetrievalCase(
        query="What does the user use for container orchestration?",
        facts=[
            {
                "id": "h021",
                "content": "Kubernetes orchestrates all production workloads.",
                "tags": ["devops"],
            },
            {
                "id": "h022",
                "content": "Docker builds images in the CI pipeline.",
                "tags": ["ci-cd", "devops"],
            },
            {
                "id": "h023",
                "content": "Helm charts template Kubernetes manifests.",
                "tags": ["devops"],
            },
            {
                "id": "h024",
                "content": "ArgoCD syncs manifests from the Git repo.",
                "tags": ["ci-cd", "git"],
            },
            {
                "id": "h025",
                "content": "Istio service mesh handles inter-pod traffic.",
                "tags": ["architecture", "devops"],
            },
            {
                "id": "h026",
                "content": "Pods auto-scale based on CPU utilization.",
                "tags": ["personal"],
            },
            {"id": "h027", "content": "Container images are scanned by Trivy.", "tags": ["devops"]},
            {
                "id": "h028",
                "content": "Secrets are mounted from Vault via CSI driver.",
                "tags": ["security"],
            },
            {
                "id": "h029",
                "content": "Namespace per team isolates resources.",
                "tags": ["devops", "team"],
            },
            {"id": "h030", "content": "Kustomize patches per environment.", "tags": ["devops"]},
        ],
        relevant_ids=["h021"],
    ),
    RetrievalCase(
        query="What is the team's incident response process?",
        facts=[
            {
                "id": "h031",
                "content": "P1 incidents trigger an immediate war room call.",
                "tags": ["incident-response"],
            },
            {
                "id": "h032",
                "content": "On-call engineer acknowledges alerts within 5 min.",
                "tags": ["incident-response", "team"],
            },
            {
                "id": "h033",
                "content": "Incident commander coordinates the response.",
                "tags": ["incident-response"],
            },
            {
                "id": "h034",
                "content": "Post-mortems are blameless and written within 48h.",
                "tags": ["personal"],
            },
            {
                "id": "h035",
                "content": "Runbooks cover the top 20 failure scenarios.",
                "tags": ["personal"],
            },
            {
                "id": "h036",
                "content": "Status page is updated during major incidents.",
                "tags": ["personal"],
            },
            {
                "id": "h037",
                "content": "User was on-call when the DB went down.",
                "tags": ["incident-response"],
            },
            {
                "id": "h038",
                "content": "Incident timeline is logged in PagerDuty.",
                "tags": ["incident-response", "monitoring"],
            },
            {
                "id": "h039",
                "content": "SLOs are reviewed after every P1 post-mortem.",
                "tags": ["incident-response"],
            },
            {
                "id": "h040",
                "content": "Team runs chaos engineering drills quarterly.",
                "tags": ["team"],
            },
        ],
        relevant_ids=["h031", "h032", "h033", "h034"],
    ),
    RetrievalCase(
        query="How does the user write API documentation?",
        facts=[
            {
                "id": "h041",
                "content": "API docs are auto-generated from OpenAPI specs.",
                "tags": ["api", "documentation"],
            },
            {
                "id": "h042",
                "content": "User writes docstrings on all public endpoints.",
                "tags": ["personal"],
            },
            {
                "id": "h043",
                "content": "Swagger UI is hosted at /docs for each service.",
                "tags": ["api", "documentation"],
            },
            {
                "id": "h044",
                "content": "API changelog is maintained in CHANGELOG.md.",
                "tags": ["api", "documentation"],
            },
            {"id": "h045", "content": "API versioning uses URL path prefixes.", "tags": ["api"]},
            {
                "id": "h046",
                "content": "Rate limit headers are documented per endpoint.",
                "tags": ["api"],
            },
            {
                "id": "h047",
                "content": "Error response formats follow RFC 7807.",
                "tags": ["error-handling"],
            },
            {
                "id": "h048",
                "content": "API authentication uses Bearer JWT tokens.",
                "tags": ["api", "security"],
            },
            {
                "id": "h049",
                "content": "SDK examples are included in the docs.",
                "tags": ["documentation"],
            },
            {
                "id": "h050",
                "content": "API deprecation notices are sent 90 days ahead.",
                "tags": ["api"],
            },
        ],
        relevant_ids=["h041", "h042", "h043"],
    ),
    RetrievalCase(
        query="What Git branching model does the team follow?",
        facts=[
            {
                "id": "h051",
                "content": "Team follows trunk-based development on main.",
                "tags": ["git", "team"],
            },
            {
                "id": "h052",
                "content": "Feature branches live less than two days.",
                "tags": ["personal"],
            },
            {
                "id": "h053",
                "content": "Release branches are cut from main for tags.",
                "tags": ["personal"],
            },
            {
                "id": "h054",
                "content": "Hotfix branches merge to main and release.",
                "tags": ["git"],
            },
            {
                "id": "h055",
                "content": "Branch protection requires passing CI checks.",
                "tags": ["ci-cd", "git"],
            },
            {"id": "h056", "content": "Stale branches are deleted after merge.", "tags": ["git"]},
            {
                "id": "h057",
                "content": "User rebases feature branches before merge.",
                "tags": ["git"],
            },
            {
                "id": "h058",
                "content": "Merge commits preserve full branch history.",
                "tags": ["git"],
            },
            {"id": "h059", "content": "Draft PRs signal work-in-progress.", "tags": ["personal"]},
            {
                "id": "h060",
                "content": "Auto-merge is enabled for green Dependabot PRs.",
                "tags": ["git"],
            },
        ],
        relevant_ids=["h051", "h052", "h053", "h054"],
    ),
    RetrievalCase(
        query="How does the user handle database backups?",
        facts=[
            {
                "id": "h061",
                "content": "PostgreSQL backups run nightly via pg_dump.",
                "tags": ["database"],
            },
            {
                "id": "h062",
                "content": "Backups are stored in S3 with 30-day retention.",
                "tags": ["cloud"],
            },
            {
                "id": "h063",
                "content": "Point-in-time recovery uses WAL archiving.",
                "tags": ["backup"],
            },
            {
                "id": "h064",
                "content": "Database replica handles read-heavy queries.",
                "tags": ["backup", "database"],
            },
            {"id": "h065", "content": "Connection pooling uses PgBouncer.", "tags": ["database"]},
            {"id": "h066", "content": "Schema migrations run via Alembic.", "tags": ["database"]},
            {
                "id": "h067",
                "content": "Database indexes are reviewed quarterly.",
                "tags": ["database"],
            },
            {"id": "h068", "content": "Slow query log threshold is 500ms.", "tags": ["personal"]},
            {
                "id": "h069",
                "content": "Vacuum runs automatically on a schedule.",
                "tags": ["personal"],
            },
            {
                "id": "h070",
                "content": "Table partitioning splits data by month.",
                "tags": ["personal"],
            },
        ],
        relevant_ids=["h061", "h062", "h063"],
    ),
    RetrievalCase(
        query="What authentication method does the API use?",
        facts=[
            {
                "id": "h071",
                "content": "API authenticates requests via JWT Bearer tokens.",
                "tags": ["api", "security"],
            },
            {
                "id": "h072",
                "content": "OAuth2 handles third-party integrations.",
                "tags": ["personal"],
            },
            {
                "id": "h073",
                "content": "API keys are used for service-to-service calls.",
                "tags": ["api"],
            },
            {
                "id": "h074",
                "content": "Rate limiting is per API key, 100 req/min.",
                "tags": ["api", "security"],
            },
            {
                "id": "h075",
                "content": "CORS policy allows specific frontend origins.",
                "tags": ["frontend", "security"],
            },
            {
                "id": "h076",
                "content": "API gateway handles TLS termination.",
                "tags": ["api", "security"],
            },
            {"id": "h077", "content": "Request validation uses Pydantic models.", "tags": ["api"]},
            {
                "id": "h078",
                "content": "Response pagination uses cursor-based tokens.",
                "tags": ["personal"],
            },
            {"id": "h079", "content": "API health check endpoint is at /healthz.", "tags": ["api"]},
            {
                "id": "h080",
                "content": "GraphQL layer sits in front of REST services.",
                "tags": ["api"],
            },
        ],
        relevant_ids=["h071", "h072", "h073"],
    ),
    RetrievalCase(
        query="How does the team manage infrastructure as code?",
        facts=[
            {
                "id": "h081",
                "content": "Terraform provisions all AWS infrastructure.",
                "tags": ["cloud", "infrastructure"],
            },
            {
                "id": "h082",
                "content": "State files are stored in S3 with DynamoDB lock.",
                "tags": ["cloud"],
            },
            {
                "id": "h083",
                "content": "Modules are versioned in a shared registry.",
                "tags": ["personal"],
            },
            {
                "id": "h084",
                "content": "Plan output is reviewed before apply.",
                "tags": ["personal"],
            },
            {"id": "h085", "content": "Drift detection runs daily.", "tags": ["personal"]},
            {
                "id": "h086",
                "content": "Networking is defined in a dedicated VPC module.",
                "tags": ["cloud", "packaging"],
            },
            {
                "id": "h087",
                "content": "IAM policies follow least-privilege principle.",
                "tags": ["cloud"],
            },
            {
                "id": "h088",
                "content": "Cost alerts trigger when budget exceeds 80%.",
                "tags": ["personal"],
            },
            {"id": "h089", "content": "Tags are enforced on all resources.", "tags": ["personal"]},
            {
                "id": "h090",
                "content": "Terragrunt wraps Terraform for DRY configs.",
                "tags": ["infrastructure"],
            },
        ],
        relevant_ids=["h081", "h082", "h083"],
    ),
    RetrievalCase(
        query="How does the team track technical metrics?",
        facts=[
            {
                "id": "h091",
                "content": "DORA metrics are tracked in a team dashboard.",
                "tags": ["metrics", "team"],
            },
            {"id": "h092", "content": "Deployment frequency target is daily.", "tags": ["ci-cd"]},
            {
                "id": "h093",
                "content": "Lead time for changes is under 24 hours.",
                "tags": ["metrics"],
            },
            {"id": "h094", "content": "Change failure rate is below 5%.", "tags": ["metrics"]},
            {
                "id": "h095",
                "content": "Mean time to recovery target is under 1 hour.",
                "tags": ["backup"],
            },
            {
                "id": "h096",
                "content": "Sprint velocity is charted per iteration.",
                "tags": ["workflow"],
            },
            {
                "id": "h097",
                "content": "Bug count trend is reviewed at retros.",
                "tags": ["error-handling"],
            },
            {
                "id": "h098",
                "content": "Code churn is monitored for hotspot files.",
                "tags": ["personal"],
            },
            {
                "id": "h099",
                "content": "PR review time SLA is under 4 hours.",
                "tags": ["code-review"],
            },
            {
                "id": "h100",
                "content": "Uptime SLO is 99.9% for core services.",
                "tags": ["monitoring"],
            },
        ],
        relevant_ids=["h091", "h092", "h093", "h094", "h095"],
    ),
]
