# Security Policy

## Supported Use

This repository is intended for personal or internal bridge automation. Do not use it to bypass GitHub, OpenAI, WeChat, or provider rate limits, access controls, anti-abuse systems, or platform terms.

## Secrets

Keep all credentials outside git:

- Use `.env`, `.env.local`, or a user-level service env file for provider API keys and bridge credentials.
- Store GitHub Actions credentials as repository or organization secrets.
- Never commit QR login output, account state, service state, session caches, database files, private keys, tokens, or debug logs.

Before pushing to a public repository, run a local secret scan and review `git status --short`.

## GitHub Actions

The default CI workflow runs isolated tests only. Live provider tests are manual via `workflow_dispatch` and require explicit GitHub Actions secrets.

## Reporting

If you discover a security issue in your fork, handle it privately first: rotate exposed credentials, remove the secret from git history if needed, and only then publish a fix.
