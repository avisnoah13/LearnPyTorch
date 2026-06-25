#!/usr/bin/env python3
"""
Stop hook — prints a short summary of files changed this session using git.
Runs when Claude Code finishes a session.
"""
import subprocess
import sys
from datetime import datetime

result = subprocess.run(
    ["git", "diff", "--name-only", "HEAD"],
    capture_output=True, text=True
)
staged = subprocess.run(
    ["git", "diff", "--name-only", "--cached"],
    capture_output=True, text=True
)

changed = set(result.stdout.splitlines() + staged.stdout.splitlines())

print(f"\n[session_summary] {datetime.now().strftime('%Y-%m-%d %H:%M')}")
if changed:
    print("Files changed this session:")
    for f in sorted(changed):
        print(f"  {f}")
else:
    print("No uncommitted changes.")
