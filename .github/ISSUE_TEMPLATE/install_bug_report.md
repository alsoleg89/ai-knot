---
name: Install bug report
about: pip / npm / MCP setup / first-run path is broken or unclear
labels: bug
---

## Which path failed?

- [ ] `pip install ai-knot`
- [ ] `pip install "ai-knot[...]"`
- [ ] `npm install ai-knot`
- [ ] `ai-knot setup claude`
- [ ] `ai-knot setup openclaw`
- [ ] Codespaces / devcontainer
- [ ] Other

## Exact command

```bash
# Paste the command that failed
```

## `ai-knot doctor --json` output

```bash
ai-knot doctor --json
```

```json
{}
```

## If the public package / repo looked stale, `scripts/check_public_release.py` output

```bash
./.venv/bin/python scripts/check_public_release.py
```

```text
paste the relevant failing lines here
```

## Environment

- OS:
- Python version:
- Node version:
- ai-knot version:

## Expected result

<!-- What did you expect to happen? -->

## Actual result

```text
paste the full output / traceback here
```

## Surface you were trying to use after install

- [ ] Core Python API
- [ ] CrewAI
- [ ] OpenClaw
- [ ] Claude Desktop / Claude Code
- [ ] AutoGen
- [ ] OpenAI Agents SDK
- [ ] LangChain / LangGraph
- [ ] TypeScript / npm
- [ ] HTTP sidecar

## If you found a workaround, what was it?

<!-- This is especially useful for release-day install friction. -->
