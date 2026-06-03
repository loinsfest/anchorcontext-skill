# Anchor Context Skill — Install Script (Windows PowerShell)
#
# 5-step install:
#   1. Check Python 3.9+
#   2. Copy skill to ~\.claude\skills\anchor-context\
#   3. Merge hooks into ~\.claude\settings.json
#   4. Verify installation
#   5. Print usage instructions

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Anchor Context Skill — Installer (Windows)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$SkillDir = "$env:USERPROFILE\.claude\skills\anchor-context"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SettingsFile = "$env:USERPROFILE\.claude\settings.json"

# ── Step 1: Check Python ─────────────────────────────────────────────────

Write-Host -NoNewline "[1/5] Checking Python... "

$PythonCmd = $null
foreach ($cmd in @("python", "python3")) {
    try {
        $version = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        $major = & $cmd -c "import sys; print(sys.version_info.major)" 2>$null
        $minor = & $cmd -c "import sys; print(sys.version_info.minor)" 2>$null
        if ($LASTEXITCODE -eq 0 -and [int]$major -ge 3 -and [int]$minor -ge 9) {
            $PythonCmd = $cmd
            break
        }
    } catch {
        continue
    }
}

if (-not $PythonCmd) {
    Write-Host "FAIL" -ForegroundColor Red
    Write-Host "  Python 3.9+ is required but not found."
    Write-Host "  Install Python from https://python.org and try again."
    exit 1
}

Write-Host "OK (using $PythonCmd $version)" -ForegroundColor Green

# ── Step 2: Copy skill ───────────────────────────────────────────────────

Write-Host -NoNewline "[2/5] Installing skill to ~\.claude\skills\anchor-context\... "

New-Item -ItemType Directory -Force -Path $SkillDir | Out-Null

$SourceDir = Join-Path $ScriptDir "anchor-context"
Copy-Item -Path "$SourceDir\*" -Destination $SkillDir -Recurse -Force

Write-Host "OK" -ForegroundColor Green

# ── Step 3: Register hooks ───────────────────────────────────────────────

Write-Host -NoNewline "[3/5] Configuring hooks in settings.json... "

New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.claude" | Out-Null

# Merge hooks using Python
$mergeScript = @"
import json, os

settings_path = os.path.expanduser(r'$SettingsFile')
os.makedirs(os.path.dirname(settings_path), exist_ok=True)

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

settings['hooks']['PreCompact'] = settings['hooks'].get('PreCompact', [])
precompact_exists = any('anchor-context' in str(h) for h in settings['hooks']['PreCompact'])

if not precompact_exists:
    settings['hooks']['PreCompact'].append({
        'matcher': '',
        'hooks': [{
            'type': 'command',
            'command': f'python \"{os.path.expanduser(\"~/.claude/skills/anchor-context\")}/scripts/pre_compact.py\" save',
            'async': True
        }]
    })

settings['hooks']['SessionStart'] = settings['hooks'].get('SessionStart', [])
inject_exists = any('anchor-context' in str(h) for h in settings['hooks']['SessionStart'])

if not inject_exists:
    settings['hooks']['SessionStart'].append({
        'matcher': 'compact',
        'hooks': [{
            'type': 'command',
            'command': f'python \"{os.path.expanduser(\"~/.claude/skills/anchor-context\")}/scripts/inject.py\"',
            'async': False
        }]
    })

with open(settings_path, 'w') as f:
    json.dump(settings, f, indent=2, ensure_ascii=False)
"@

$mergeResult = & $PythonCmd -c $mergeScript 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL" -ForegroundColor Red
    Write-Host "  $mergeResult"
    exit 1
}

Write-Host "OK" -ForegroundColor Green

# ── Step 4: Verify ───────────────────────────────────────────────────────

Write-Host -NoNewline "[4/5] Verifying installation... "

$errors = 0

if (-not (Test-Path "$SkillDir\SKILL.md")) {
    Write-Host "FAIL" -ForegroundColor Red
    Write-Host "  SKILL.md not found at $SkillDir\SKILL.md"
    $errors++
}

if (-not (Test-Path "$SkillDir\scripts\anchor\models.py")) {
    Write-Host "FAIL" -ForegroundColor Red
    Write-Host "  Core library not found at $SkillDir\scripts\anchor\models.py"
    $errors++
}

# Quick import test
$importResult = & $PythonCmd -c "import sys; sys.path.insert(0, r'$SkillDir\scripts'); from anchor import Anchor, AnchorType, EntityClass, AnchorSequence; print('OK')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARN" -ForegroundColor Yellow
    Write-Host "  Python import test had issues. Skill may still work for extraction."
    $errors++
}

if ($errors -eq 0) {
    Write-Host "OK" -ForegroundColor Green
}

# ── Step 5: Done ─────────────────────────────────────────────────────────

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Anchor Context Skill installed!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Skill location: $SkillDir"
Write-Host "  Anchor storage: ~\.claude\anchors\"
Write-Host ""
Write-Host "  How to use:"
Write-Host "    1. Open Claude Code in any project"
Write-Host "    2. Say '锚点上下文' or 'anchor context'"
Write-Host "    3. Anchors are auto-saved during compaction"
Write-Host ""
Write-Host "  Manual commands:"
Write-Host "    python $SkillDir\scripts\inject.py --format   # View anchors"
Write-Host ""
Write-Host "  To uninstall:"
Write-Host "    Remove-Item -Recurse -Force $SkillDir"
Write-Host "    (then remove hooks from ~\.claude\settings.json)"
Write-Host ""
