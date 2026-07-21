# Fork Maintenance Notes

This working tree was prepared from:

<https://github.com/Gan-Xing/CodexBridge>

## Before Publishing Your Fork

- Confirm that you have permission to redistribute the upstream code. No root `LICENSE` file was present in the downloaded source, so do not assume it is open source.
- Keep the repository private unless you have confirmed the license and redistribution rights.
- Set your own GitHub remote instead of reusing the upstream remote.
- Review `.github/workflows/` before enabling Actions in your repository.
- Add only your own secrets to GitHub Actions. Do not commit `.env`, account state, QR login files, service env files, or local runtime directories.
- If you later rename the project, update package names, README references, service names, and docs together so users are not misled about ownership.

## Suggested Pre-Push Checks

```bash
git status --short
npm ci
npm run typecheck
npm test
```

Optional secret scans:

```bash
git grep -n -I -E "(api[_-]?key|secret|token|password|authorization|bearer|sk-[A-Za-z0-9]|ghp_|github_pat_)"
```
