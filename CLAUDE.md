# Claude Code Instructions

## Git workflow

**NEVER push directly to `main`.** No exceptions.

- All changes go to a feature branch
- To get code into `main`: create a pull request, wait for explicit user approval
- Force push to `main` is also forbidden
- If a fix is needed on `main`: commit to feature branch → create PR → user merges

## Commit hygiene — before every commit

**Author**: always use `--author="alsoleg89 <155813332+alsoleg89@users.noreply.github.com>"`
Verify with: `git log -1 --format="%an"` — must show `alsoleg89`, never `Claude`

**Commit message subject** — forbidden words:
- `claude.ai`, `Claude`, `CLAUDE`
- `AI-indicator`, `AI-flagged`, `AI vocab`, `ZeroGPT`
- `sentence rhythm`, `commit density`, `pre-launch-audit`

**Commit message body** — keep minimal or empty.
Never add `https://claude.ai/code/session_...` URLs.

**Merge commits** — must not expose branch names with `claude/` in the message.

## Before pushing

Run this sanity check:
```bash
# No Claude author
git log --format="%an" | sort -u

# No AI-related words in commit messages
git log --format="%s %b" | grep -i "claude\.ai\|AI-indicator\|zerogpt\|sentence rhythm"
```

## Files

`CLAUDE.md` is gitignored — never commit it.
