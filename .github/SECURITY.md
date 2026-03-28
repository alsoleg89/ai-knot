# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in agentmemo, please report it responsibly:

1. **Do not** open a public issue.
2. Email the maintainers or open a private security advisory via
   [GitHub Security Advisories](https://github.com/alsoleg89/agentmemo/security/advisories/new).
3. Include a description of the vulnerability, steps to reproduce, and potential impact.

We will acknowledge receipt within 48 hours and aim to release a fix within 7 days
for critical issues.

## Scope

- agentmemo library code (`src/agentmemo/`)
- CLI interface
- Storage backends (YAML, SQLite, PostgreSQL)
- LLM provider integrations

## Out of Scope

- Vulnerabilities in third-party dependencies (report upstream)
- Issues requiring physical access to the machine
- Social engineering attacks
