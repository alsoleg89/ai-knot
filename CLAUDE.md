# Dev Notes

## Git workflow

**Never push directly to `main`.** No exceptions.

- All changes go to a feature branch
- To merge into `main`: open a pull request, wait for maintainer approval
- Force-pushing `main` is forbidden
- To fix something on `main`: branch → commit → PR → merge

## Commit conventions

**Author**: commits must be attributed to the project maintainer:
`alsoleg89 <155813332+alsoleg89@users.noreply.github.com>`

Verify before pushing:
```bash
git log -1 --format="%an"   # must be: alsoleg89
```

**Subject line** — keep it short and factual. Avoid:
- Tool or service names (editor, AI assistant, CI bot names)
- Internal workflow jargon (audit names, pipeline step names)
- Auto-generated session or trace URLs

**Body** — optional. If present: one blank line after subject, then context.
No auto-generated footers or URLs.

**Merge commits** — subject must not expose internal branch naming conventions.

## Sanity check before push

```bash
# Confirm author
git log --format="%an" | sort -u

# Confirm no leaked URLs
git log --format="%B" | grep "https://claude\|session_"
```
