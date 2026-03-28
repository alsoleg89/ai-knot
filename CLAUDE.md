# Claude Code Instructions

## Git workflow

**NEVER push directly to `main`.** No exceptions.

- All changes go to a feature branch
- To get code into `main`: create a pull request and wait for explicit user approval
- Force push to `main` is also forbidden
- If a fix is needed on `main`: commit to feature branch → create PR → user merges

This rule applies even when the technical reason seems justified (e.g. "the workflow only runs from main").
