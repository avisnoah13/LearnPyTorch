#!/usr/bin/env python3
"""
PreToolUse hook — blocks shell commands that could damage the repo or environment.
Reads the pending tool input from stdin as JSON and exits non-zero to block.
"""
import json
import sys

BLOCKED_PATTERNS = [
    "rm -rf",
    "rmdir /s",
    "git reset --hard",
    "git push --force",
    "pip uninstall",
    "conda remove",
    "del /f",
    "format ",
]

try:
    tool_input = json.load(sys.stdin)
    command = tool_input.get("command", "")
    for pattern in BLOCKED_PATTERNS:
        if pattern in command:
            print(f"[block_dangerous] Blocked: '{pattern}' found in command.", file=sys.stderr)
            sys.exit(1)
except Exception:
    pass  # Don't block on parse errors
