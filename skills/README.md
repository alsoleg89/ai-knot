# ai-knot skills

`ai-knot` ships a repo-native skill for coding assistants that support the
skills standard.

## Available skill

| Skill | What it covers | Install |
|---|---|---|
| `ai-knot` | deterministic agent memory across the core Python API, plain function-calling helpers, MCP, CrewAI, LlamaIndex, AutoGen, the OpenAI Agents SDK, PydanticAI, explicit LangGraph tool helpers, LangChain retrievers, TypeScript routing, by-id inspection, and first-run troubleshooting | `npx skills add https://github.com/alsoleg89/ai-knot --skill ai-knot` |

## When to use it

Use the skill when you want an assistant to:

- wire `ai-knot` into an existing agent or app,
- choose the right surface without guessing,
- know the exact adapter names, `get(...)` inspection verbs, and setup commands,
- load the right repo-native examples and troubleshooting path.

## See also

- [docs/integrations.md](../docs/integrations.md)
- [docs/usage.md](../docs/usage.md)
- [docs/deployment.md](../docs/deployment.md)
- [docs/troubleshooting.md](../docs/troubleshooting.md)
