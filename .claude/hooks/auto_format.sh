#!/usr/bin/env bash
# PostToolUse hook — auto-formats any .py file that was just written or edited.
# Reads tool result from stdin; extracts the file path and runs autopep8 if available.

INPUT=$(cat)
FILE=$(echo "$INPUT" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    path = data.get('file_path', '')
    if path.endswith('.py'):
        print(path)
except Exception:
    pass
" 2>/dev/null)

if [ -n "$FILE" ] && [ -f "$FILE" ]; then
    if command -v autopep8 &>/dev/null; then
        autopep8 --in-place "$FILE"
    fi
fi
