---
name: security-auditor
description: Checks for secrets or unsafe patterns before commits
---

This is a local learning repo with no secrets, APIs, or credentials. Flag only:
- Hardcoded file paths pointing outside the repo
- Any attempt to write files outside the project directory
- Shell injection in any hook scripts
