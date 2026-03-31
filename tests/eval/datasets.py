"""Golden retrieval dataset for ai-knot eval."""

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
            {"id": "s001", "content": "User works at Sber as a senior software engineer."},
            {"id": "s002", "content": "User prefers dark mode in all code editors."},
            {"id": "s003", "content": "User lives in Moscow and commutes by metro."},
            {"id": "s004", "content": "User has a dog named Pushkin."},
            {"id": "s005", "content": "User's primary programming language is Python."},
        ],
        relevant_ids=["s001"],
    ),
    RetrievalCase(
        query="What programming language does the user prefer?",
        facts=[
            {"id": "s006", "content": "User's primary programming language is Python."},
            {"id": "s007", "content": "User also knows TypeScript and uses it for frontend work."},
            {"id": "s008", "content": "User studied mathematics at Moscow State University."},
            {"id": "s009", "content": "User's team uses Git with a trunk-based development model."},
            {"id": "s010", "content": "User enjoys hiking on weekends."},
        ],
        relevant_ids=["s006", "s007"],
    ),
    RetrievalCase(
        query="What city does the user live in?",
        facts=[
            {"id": "s011", "content": "User lives in Moscow and commutes by metro."},
            {"id": "s012", "content": "User's office is near Lubyanka metro station."},
            {"id": "s013", "content": "User has visited Saint Petersburg twice this year."},
            {"id": "s014", "content": "User uses macOS for personal work and Linux for servers."},
            {"id": "s015", "content": "User's favorite IDE is VS Code with vim keybindings."},
        ],
        relevant_ids=["s011"],
    ),
    RetrievalCase(
        query="What is the user's educational background?",
        facts=[
            {"id": "s016", "content": "User studied mathematics at Moscow State University."},
            {"id": "s017", "content": "User completed an online ML course from Coursera in 2022."},
            {"id": "s018", "content": "User's team has 6 engineers and 2 product managers."},
            {"id": "s019", "content": "User prefers to write code with type hints always."},
            {"id": "s020", "content": "User reads technical books in English and Russian."},
        ],
        relevant_ids=["s016", "s017"],
    ),
    RetrievalCase(
        query="What pets does the user have?",
        facts=[
            {"id": "s021", "content": "User has a dog named Pushkin, a labrador."},
            {"id": "s022", "content": "User had a cat but it moved to live with their parents."},
            {"id": "s023", "content": "User's apartment is 65 square meters."},
            {"id": "s024", "content": "User enjoys cooking Italian food on weekends."},
            {"id": "s025", "content": "User subscribes to several technical newsletters."},
        ],
        relevant_ids=["s021", "s022"],
    ),
    # --- Procedural: user preferences and workflows ---
    RetrievalCase(
        query="How does the user want code formatted?",
        facts=[
            {"id": "p001", "content": "Always use ruff for Python formatting and linting."},
            {"id": "p002", "content": "Always add type hints to all function signatures."},
            {"id": "p003", "content": "Use 4 spaces for indentation, never tabs."},
            {"id": "p004", "content": "User prefers dark mode in editors."},
            {"id": "p005", "content": "User checks email at 9am and 3pm only."},
        ],
        relevant_ids=["p001", "p002", "p003"],
    ),
    RetrievalCase(
        query="What testing framework does the user use?",
        facts=[
            {"id": "p006", "content": "User uses pytest for all Python testing."},
            {"id": "p007", "content": "User writes unit tests before submitting PRs."},
            {"id": "p008", "content": "User's team requires 80% code coverage."},
            {"id": "p009", "content": "User prefers to use fixtures over setUp/tearDown."},
            {"id": "p010", "content": "User has a standing desk at the office."},
        ],
        relevant_ids=["p006", "p007", "p008", "p009"],
    ),
    RetrievalCase(
        query="How does the user handle code reviews?",
        facts=[
            {"id": "p011", "content": "User reviews PRs the same day they are opened."},
            {
                "id": "p012",
                "content": "User leaves comments with suggested alternatives, not just issues.",
            },
            {"id": "p013", "content": "User requires at least two approvals before merging."},
            {"id": "p014", "content": "User uses GitHub for version control."},
            {"id": "p015", "content": "User prefers asynchronous communication over meetings."},
        ],
        relevant_ids=["p011", "p012", "p013"],
    ),
    RetrievalCase(
        query="What is the user's preferred commit message style?",
        facts=[
            {"id": "p016", "content": "User writes commit messages in imperative mood."},
            {"id": "p017", "content": "User's commit subject lines are under 72 characters."},
            {"id": "p018", "content": "User uses conventional commits format: feat, fix, chore."},
            {"id": "p019", "content": "User does not include ticket numbers in commit messages."},
            {"id": "p020", "content": "User uses poetry for dependency management."},
        ],
        relevant_ids=["p016", "p017", "p018"],
    ),
    RetrievalCase(
        query="How does the user prefer to document code?",
        facts=[
            {
                "id": "p021",
                "content": "User writes docstrings for all public functions and classes.",
            },
            {"id": "p022", "content": "User uses Google-style docstrings."},
            {
                "id": "p023",
                "content": "User documents architecture decisions in DECISIONS.md files.",
            },
            {
                "id": "p024",
                "content": "User does not write inline comments unless logic is non-obvious.",
            },
            {"id": "p025", "content": "User's favorite keyboard shortcut is Ctrl+Shift+P."},
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
            },
            {
                "id": "e002",
                "content": "Team agreed to add caching to CI/CD pipeline in next sprint.",
            },
            {"id": "e003", "content": "User presented metrics showing 40% test suite speedup."},
            {"id": "e004", "content": "User is reading Clean Architecture by Robert Martin."},
            {"id": "e005", "content": "User's standup is at 10am every weekday."},
        ],
        relevant_ids=["e001", "e002", "e003"],
    ),
    RetrievalCase(
        query="When did the production deployment fail?",
        facts=[
            {
                "id": "e006",
                "content": "Production deployment failed last Tuesday due to a config error.",
            },
            {
                "id": "e007",
                "content": "Deploy was rolled back within 15 minutes by the on-call engineer.",
            },
            {
                "id": "e008",
                "content": "Post-mortem identified missing environment variable as root cause.",
            },
            {"id": "e009", "content": "User completed the Python advanced course last month."},
            {"id": "e010", "content": "User's team uses Slack for communication."},
        ],
        relevant_ids=["e006", "e007", "e008"],
    ),
    RetrievalCase(
        query="What did the user discuss with their manager last week?",
        facts=[
            {
                "id": "e011",
                "content": "User met with manager last Thursday to discuss promotion timeline.",
            },
            {"id": "e012", "content": "Manager suggested user take on a tech lead role in Q3."},
            {"id": "e013", "content": "User expressed interest in moving to the ML platform team."},
            {"id": "e014", "content": "User drinks two cups of coffee per day."},
            {"id": "e015", "content": "User's team lunch is every Friday."},
        ],
        relevant_ids=["e011", "e012", "e013"],
    ),
    RetrievalCase(
        query="What new library did the user try recently?",
        facts=[
            {
                "id": "e016",
                "content": "User tried Polars for data processing last week and was impressed.",
            },
            {
                "id": "e017",
                "content": "User benchmarked Polars against Pandas and found 3x speedup.",
            },
            {"id": "e018", "content": "User is considering migrating the data pipeline to Polars."},
            {"id": "e019", "content": "User's team has a book club that meets monthly."},
            {"id": "e020", "content": "User prefers tea in the afternoon."},
        ],
        relevant_ids=["e016", "e017", "e018"],
    ),
    RetrievalCase(
        query="What bug was fixed in the authentication module?",
        facts=[
            {
                "id": "e021",
                "content": "Fixed a race condition in the authentication token refresh logic.",
            },
            {
                "id": "e022",
                "content": "The bug caused occasional 401 errors for users after 1 hour.",
            },
            {"id": "e023", "content": "Fix added a mutex lock around the token refresh operation."},
            {
                "id": "e024",
                "content": "User attended a conference on distributed systems last month.",
            },
            {
                "id": "e025",
                "content": "User's favorite restaurant near the office serves Georgian food.",
            },
        ],
        relevant_ids=["e021", "e022", "e023"],
    ),
    # --- Mixed: multi-type queries ---
    RetrievalCase(
        query="What are the user's Python best practices?",
        facts=[
            {"id": "m001", "content": "Always use type hints in Python code."},
            {"id": "m002", "content": "Use dataclasses for simple data containers."},
            {"id": "m003", "content": "Prefer composition over inheritance in Python."},
            {"id": "m004", "content": "User likes to go running in Gorky Park."},
            {"id": "m005", "content": "User uses mypy in strict mode for type checking."},
        ],
        relevant_ids=["m001", "m002", "m003", "m005"],
    ),
    RetrievalCase(
        query="How does the user manage project dependencies?",
        facts=[
            {"id": "m006", "content": "User uses poetry for Python dependency management."},
            {"id": "m007", "content": "User pins all transitive dependencies in production."},
            {"id": "m008", "content": "User runs dependency audits monthly using safety check."},
            {"id": "m009", "content": "User's home router uses OpenWRT firmware."},
            {"id": "m010", "content": "User contributes to open source in spare time."},
        ],
        relevant_ids=["m006", "m007", "m008"],
    ),
    RetrievalCase(
        query="What happened at the system design interview?",
        facts=[
            {
                "id": "m011",
                "content": "User did a system design interview at a tech company last Tuesday.",
            },
            {"id": "m012", "content": "Interview focused on designing a rate limiter for an API."},
            {"id": "m013", "content": "User explained token bucket and sliding window algorithms."},
            {"id": "m014", "content": "User received positive feedback and awaits second round."},
            {"id": "m015", "content": "User prefers minimalist desk setup with a single monitor."},
        ],
        relevant_ids=["m011", "m012", "m013", "m014"],
    ),
    RetrievalCase(
        query="What database does the user's project use?",
        facts=[
            {"id": "m016", "content": "Project uses PostgreSQL 15 as the primary database."},
            {"id": "m017", "content": "Redis is used for caching and session storage."},
            {"id": "m018", "content": "User migrated from SQLite to PostgreSQL in January."},
            {"id": "m019", "content": "User's favorite podcast is Lex Fridman."},
            {"id": "m020", "content": "User tracks time with Toggl."},
        ],
        relevant_ids=["m016", "m017", "m018"],
    ),
    RetrievalCase(
        query="How does the user handle secrets and credentials?",
        facts=[
            {
                "id": "m021",
                "content": "User stores all secrets in environment variables, never in code.",
            },
            {
                "id": "m022",
                "content": "User uses HashiCorp Vault for secrets management in production.",
            },
            {"id": "m023", "content": "User rotates API keys every 90 days."},
            {
                "id": "m024",
                "content": "User uses a password manager with a strong master password.",
            },
            {"id": "m025", "content": "User likes jazz music while coding."},
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
            },
            {"id": "a002", "content": "Team is split between Moscow and remote workers."},
            {"id": "a003", "content": "User leads a subteam of 3 engineers on the backend."},
            {"id": "a004", "content": "User's laptop is a MacBook Pro M2."},
            {"id": "a005", "content": "User takes a daily walk after lunch."},
        ],
        relevant_ids=["a001", "a002", "a003"],
    ),
    RetrievalCase(
        query="What monitoring tools does the user use?",
        facts=[
            {
                "id": "a006",
                "content": "User's team uses Grafana and Prometheus for metrics monitoring.",
            },
            {"id": "a007", "content": "Sentry is used for error tracking and alerting."},
            {"id": "a008", "content": "User set up PagerDuty for on-call rotations last year."},
            {"id": "a009", "content": "User prefers 9-hour workdays with a long lunch break."},
            {"id": "a010", "content": "User does not use social media except LinkedIn."},
        ],
        relevant_ids=["a006", "a007", "a008"],
    ),
    RetrievalCase(
        query="What did the user learn at the conference?",
        facts=[
            {
                "id": "a011",
                "content": "User attended PyCon Russia and learned about asyncio best practices.",
            },
            {"id": "a012", "content": "A talk on Python performance profiling was the highlight."},
            {
                "id": "a013",
                "content": "User connected with maintainers of several open source libraries.",
            },
            {"id": "a014", "content": "User brought conference swag back for the team."},
            {"id": "a015", "content": "User's home office has a mechanical keyboard."},
        ],
        relevant_ids=["a011", "a012", "a013"],
    ),
    RetrievalCase(
        query="How does the user prefer to receive feedback?",
        facts=[
            {
                "id": "a016",
                "content": "User prefers written feedback over verbal to have time to process it.",
            },
            {
                "id": "a017",
                "content": "User asks for specific examples when receiving critical feedback.",
            },
            {"id": "a018", "content": "User schedules 1:1 with manager every two weeks."},
            {"id": "a019", "content": "User's morning routine starts with 20 minutes of reading."},
            {"id": "a020", "content": "User tracks personal goals in Notion."},
        ],
        relevant_ids=["a016", "a017", "a018"],
    ),
    RetrievalCase(
        query="What CI/CD system does the team use?",
        facts=[
            {"id": "a021", "content": "Team uses GitHub Actions for continuous integration."},
            {"id": "a022", "content": "Deployments are automated via ArgoCD to Kubernetes."},
            {
                "id": "a023",
                "content": "CI pipeline runs linting, type checks, and tests on every PR.",
            },
            {"id": "a024", "content": "User has a personal wiki written in Markdown."},
            {
                "id": "a025",
                "content": "User's favorite book is SICP.",
            },
        ],
        relevant_ids=["a021", "a022", "a023"],
    ),
    RetrievalCase(
        query="What is the user's approach to refactoring?",
        facts=[
            {"id": "b001", "content": "User refactors code only when covered by tests."},
            {
                "id": "b002",
                "content": "User applies the boy scout rule: leave code cleaner than found.",
            },
            {
                "id": "b003",
                "content": "User avoids large refactoring PRs, prefers incremental changes.",
            },
            {"id": "b004", "content": "User drinks matcha latte in the morning."},
            {"id": "b005", "content": "User's commute takes 35 minutes each way."},
        ],
        relevant_ids=["b001", "b002", "b003"],
    ),
    RetrievalCase(
        query="How does the user stay up to date with technology?",
        facts=[
            {"id": "b006", "content": "User reads Hacker News daily for tech news."},
            {
                "id": "b007",
                "content": "User follows several Python core developers on social media.",
            },
            {
                "id": "b008",
                "content": "User subscribes to Real Python and Python Weekly newsletters.",
            },
            {"id": "b009", "content": "User listens to technical podcasts during commute."},
            {"id": "b010", "content": "User's desk plant is a succulent."},
        ],
        relevant_ids=["b006", "b007", "b008", "b009"],
    ),
    RetrievalCase(
        query="What happened during the on-call incident last week?",
        facts=[
            {
                "id": "b011",
                "content": "On-call incident: DB connection pool exhausted last Wednesday at 2am.",
            },
            {
                "id": "b012",
                "content": "User was paged and resolved the incident by increasing pool size.",
            },
            {
                "id": "b013",
                "content": "Root cause analysis showed a sudden traffic spike from a batch job.",
            },
            {
                "id": "b014",
                "content": "User updated runbook to include connection pool tuning steps.",
            },
            {"id": "b015", "content": "User ordered a new mechanical keyboard last week."},
        ],
        relevant_ids=["b011", "b012", "b013", "b014"],
    ),
    RetrievalCase(
        query="What are the user's goals for this quarter?",
        facts=[
            {"id": "b016", "content": "User's Q1 goal is to ship the new memory indexing feature."},
            {
                "id": "b017",
                "content": "User wants to improve test coverage from 75% to 90% this quarter.",
            },
            {"id": "b018", "content": "User plans to give one internal tech talk this quarter."},
            {
                "id": "b019",
                "content": "User's personal goal is to run a 10k race in under 55 minutes.",
            },
            {"id": "b020", "content": "User checks personal finances every Sunday."},
        ],
        relevant_ids=["b016", "b017", "b018"],
    ),
    RetrievalCase(
        query="How does the user handle technical debt?",
        facts=[
            {
                "id": "b021",
                "content": "User tracks technical debt in a dedicated GitHub project board.",
            },
            {"id": "b022", "content": "Team allocates 20% of each sprint to debt reduction."},
            {
                "id": "b023",
                "content": "User documents debt items with estimated effort and business impact.",
            },
            {
                "id": "b024",
                "content": "User prefers to address debt proactively rather than reactively.",
            },
            {
                "id": "b025",
                "content": "User's team celebrates when a long-standing debt item is closed.",
            },
        ],
        relevant_ids=["b021", "b022", "b023", "b024"],
    ),
]
