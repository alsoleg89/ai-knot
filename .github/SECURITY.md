# Security Policy

## Supported Versions

ai-knot is pre-1.0. Security fixes land on the **latest release only** — upgrade to
the current version before reporting.

| Version | Supported |
|---------|-----------|
| Latest release | ✅ |
| Older releases | ❌ (upgrade first) |

## Reporting a Vulnerability

If you discover a security vulnerability in ai-knot, please report it responsibly:

1. **Do not** open a public issue.
2. Email the maintainers or open a private security advisory via
   [GitHub Security Advisories](https://github.com/alsoleg89/ai-knot/security/advisories/new).
3. Include a description of the vulnerability, steps to reproduce, and potential impact.

We will acknowledge receipt within 48 hours and aim to release a fix within 7 days
for critical issues.

## Scope

- ai-knot library code (`src/ai_knot/`)
- CLI interface
- Storage backends (YAML, SQLite, PostgreSQL)
- LLM provider integrations

## Out of Scope

- Vulnerabilities in third-party dependencies (report upstream)
- Issues requiring physical access to the machine
- Social engineering attacks
