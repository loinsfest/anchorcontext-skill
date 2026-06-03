#!/usr/bin/env bash
# Anchor Context Skill — Install Script (macOS / Linux / Git Bash)
#
# 5-step install:
#   1. Check Python 3.9+
#   2. Copy skill to ~/.claude/skills/anchor-context/
#   3. Merge hooks into ~/.claude/settings.json
#   4. Verify installation
#   5. Print usage instructions

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "============================================"
echo " Anchor Context Skill — Installer"
echo "============================================"
echo ""

SKILL_DIR="${HOME}/.claude/skills/anchor-context"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETTINGS_FILE="${HOME}/.claude/settings.json"

# ── Step 1: Check Python ─────────────────────────────────────────────────

echo -n "[1/5] Checking Python... "

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}FAIL${NC}"
    echo "  Python 3.9+ is required but not found."
    echo "  Install Python from https://python.org and try again."
    exit 1
fi

echo -e "${GREEN}OK${NC} (using $PYTHON $version)"

# ── Step 2: Copy skill ───────────────────────────────────────────────────

echo -n "[2/5] Installing skill to ~/.claude/skills/anchor-context/... "

mkdir -p "${SKILL_DIR}"

# Copy skill files (use rsync if available, cp otherwise)
if command -v rsync &>/dev/null; then
    rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        "${SCRIPT_DIR}/anchor-context/" "${SKILL_DIR}/"
else
    cp -r "${SCRIPT_DIR}/anchor-context/"* "${SKILL_DIR}/" 2>/dev/null || true
fi

echo -e "${GREEN}OK${NC}"

# ── Step 3: Register hooks ───────────────────────────────────────────────

echo -n "[3/5] Configuring hooks in settings.json... "

mkdir -p "${HOME}/.claude"

# Merge hooks into settings.json using Python
"$PYTHON" -c "
import json, os

settings_path = os.path.expanduser('${SETTINGS_FILE}')
os.makedirs(os.path.dirname(settings_path), exist_ok=True)

# Load existing settings or create default
if os.path.exists(settings_path):
    with open(settings_path, 'r') as f:
        try:
            settings = json.load(f)
        except json.JSONDecodeError:
            settings = {}
else:
    settings = {}

if 'hooks' not in settings:
    settings['hooks'] = {}

# PreCompact hook — saves anchors before compaction
settings['hooks']['PreCompact'] = settings['hooks'].get('PreCompact', [])
precompact_exists = any(
    'anchor-context' in str(h) for h in settings['hooks']['PreCompact']
)

if not precompact_exists:
    settings['hooks']['PreCompact'].append({
        'matcher': '',
        'hooks': [{
            'type': 'command',
            'command': f'python \"{os.path.expanduser(\"~/.claude/skills/anchor-context\")}/scripts/pre_compact.py\" save',
            'async': True
        }]
    })

# SessionStart[compact] hook — injects anchors after compaction
settings['hooks']['SessionStart'] = settings['hooks'].get('SessionStart', [])
inject_exists = any(
    'anchor-context' in str(h) for h in settings['hooks']['SessionStart']
)

if not inject_exists:
    settings['hooks']['SessionStart'].append({
        'matcher': 'compact',
        'hooks': [{
            'type': 'command',
            'command': f'python \"{os.path.expanduser(\"~/.claude/skills/anchor-context\")}/scripts/inject.py\"',
            'async': False
        }]
    })

# Stop hook — saves anchors on session exit (backup if PreCompact never fired)
settings['hooks']['Stop'] = settings['hooks'].get('Stop', [])
stop_exists = any(
    'anchor-context' in str(h) for h in settings['hooks']['Stop']
)

if not stop_exists:
    settings['hooks']['Stop'].append({
        'matcher': '',
        'hooks': [{
            'type': 'command',
            'command': f'python \"{os.path.expanduser(\"~/.claude/skills/anchor-context\")}/scripts/stop_backup.py\"',
            'async': True
        }]
    })

# Write back
with open(settings_path, 'w') as f:
    json.dump(settings, f, indent=2, ensure_ascii=False)
"

echo -e "${GREEN}OK${NC}"

# ── Step 4: Verify ───────────────────────────────────────────────────────

echo -n "[4/5] Verifying installation... "

ERRORS=0

if [ ! -f "${SKILL_DIR}/SKILL.md" ]; then
    echo -e "${RED}FAIL${NC}"
    echo "  SKILL.md not found at ${SKILL_DIR}/SKILL.md"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -f "${SKILL_DIR}/scripts/anchor/models.py" ]; then
    echo -e "${RED}FAIL${NC}"
    echo "  Core library not found at ${SKILL_DIR}/scripts/anchor/models.py"
    ERRORS=$((ERRORS + 1))
fi

# Quick import test — use os.path for cross-platform path resolution
"$PYTHON" -c "
import sys, os
skill_dir = os.path.expanduser('${SKILL_DIR}')
sys.path.insert(0, os.path.join(skill_dir, 'scripts'))
from anchor import Anchor, AnchorType, EntityClass, AnchorSequence
print('OK')
" 2>/dev/null || {
    echo -e "${YELLOW}WARN${NC}"
    echo "  Python import test failed. The skill may still work for extraction only."
    ERRORS=$((ERRORS + 1))
}

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}OK${NC}"
fi

# ── Step 5: Done ─────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Anchor Context Skill installed!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  Skill location: ${SKILL_DIR}"
echo "  Anchor storage: ~/.claude/anchors/"
echo ""
echo "  How to use:"
echo "    1. Open Claude Code in any project"
echo "    2. Say '锚点上下文' or 'anchor context'"
echo "    3. Anchors are auto-saved during compaction"
echo ""
echo "  Manual commands:"
echo "    python ${SKILL_DIR}/scripts/inject.py --format   # View anchors"
echo ""
echo "  To uninstall:"
echo "    rm -rf ${SKILL_DIR}"
echo "    (then remove hooks from ~/.claude/settings.json)"
echo ""
