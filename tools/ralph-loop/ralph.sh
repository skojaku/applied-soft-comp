#!/usr/bin/env bash
#
# ralph.sh - Autonomous agent loop for iterative feature development
#
# Repeatedly invokes an LLM coding agent with a structured prompt referencing
# requirements.json and progress.txt. Exits early when the agent signals completion.

set -euo pipefail
IFS=$'\n\t'
umask 077

readonly VERSION="1.0.0"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly COMPLETION_MARKER="<promise>COMPLETE</promise>"

usage() {
  cat <<'USAGE'
Usage:
  ralph.sh [options] <iterations> [-- <agent_cmd> [args...]]

Runs an autonomous agent loop: repeatedly invokes an LLM coding agent with a
prompt referencing requirements.json and progress.txt. Stops early when the
agent outputs the completion marker.

Arguments:
  iterations                 Positive integer; max iterations to run.

Options:
  -i, --iterations N         Same as positional <iterations>.
  -a, --agent AGENT          Agent preset: claude, codex, gemini, droid, amp (default: claude).
  -r, --requirements PATH    Requirements file (default: plans/requirements.json).
  -p, --progress PATH        Progress log file (default: progress.txt).
      --sleep SECONDS        Pause between iterations (default: 0).
      --max-retries N        Retry failed agent calls up to N times (default: 2).
      --timeout SECONDS      Timeout per agent call in seconds (default: 0 = no timeout).
      --notify               Send desktop notification on completion (requires 'terminal-notifier' or 'notify-send').
      --dry-run              Print resolved command and prompt, then exit.
      --allow-outside-workdir
                             Allow paths outside the repository root.
      --version              Show version.
  -h, --help                 Show this help.

Agent command override:
  Pass a custom command after '--' to override the preset:
    ralph.sh 10 -- my-custom-agent --flag

Supported agent presets:
  claude   - Anthropic Claude CLI (claude --dangerously-skip-permissions)
  codex    - OpenAI Codex CLI (codex --full-auto)
  gemini   - Google Gemini CLI (gemini)
  droid    - Factory Droid CLI (droid --auto)
  amp      - Sourcegraph Amp (npx --yes @anthropic/amp)
  opencode - OpenCode CLI (opencode --auto)

Examples:
  ralph.sh --dry-run 5
  ralph.sh -a codex 10
  ralph.sh 20 -- claude --permission-mode acceptEdits
USAGE
}

version() {
  printf 'ralph.sh %s\n' "$VERSION"
}

log_info() {
  printf '[INFO] %s\n' "$*"
}

log_warn() {
  printf '[WARN] %s\n' "$*" >&2
}

log_error() {
  printf '[ERROR] %s\n' "$*" >&2
}

die() {
  log_error "$*"
  exit 1
}

is_uint() {
  case "${1:-}" in
    ''|*[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

realpath_portable() {
  python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"
}

is_under_root() {
  python3 -c "
import os,sys
root = os.path.realpath(sys.argv[1])
path = os.path.realpath(sys.argv[2])
sys.exit(0 if os.path.commonpath([root, path]) == root else 1)
" "$1" "$2"
}

resolve_path() {
  local ref="$1"
  local label="$2"
  local allow_outside="$3"
  local abs

  [[ "$ref" != *$'\n'* ]] || die "${label} path contains invalid newline character"

  if [[ "$ref" == /* ]]; then
    abs="$ref"
  else
    abs="${ROOT_DIR}/${ref}"
  fi

  abs="$(realpath_portable "$abs")"

  if [[ "$allow_outside" != "true" ]]; then
    if ! is_under_root "$ROOT_DIR" "$abs"; then
      die "${label} must be within repository root (${ROOT_DIR}); got: ${abs}. Use --allow-outside-workdir to override."
    fi
  fi

  printf '%s\n' "$abs"
}

find_binary() {
  local name="$1"
  shift
  local paths=("$@")

  if command -v "$name" >/dev/null 2>&1; then
    echo "$name"
    return 0
  fi

  for p in "${paths[@]}"; do
    if [[ -x "$p" ]]; then
      echo "$p"
      return 0
    fi
  done

  echo "$name"
}

get_agent_command() {
  local preset="$1"
  local bin

  case "$preset" in
    claude)
      bin="$(find_binary claude "$HOME/.claude/local/claude" "/usr/local/bin/claude")"
      echo "$bin --dangerously-skip-permissions"
      ;;
    codex)
      bin="$(find_binary codex "$HOME/.codex/bin/codex" "/usr/local/bin/codex")"
      echo "$bin --full-auto"
      ;;
    gemini)
      bin="$(find_binary gemini "$HOME/.local/bin/gemini" "/usr/local/bin/gemini")"
      echo "$bin"
      ;;
    droid)
      bin="$(find_binary droid "$HOME/.factory/bin/droid" "/usr/local/bin/droid")"
      echo "$bin --auto"
      ;;
    amp)
      echo "npx --yes @anthropic/amp"
      ;;
    opencode)
      bin="$(find_binary opencode "$HOME/.local/bin/opencode" "/usr/local/bin/opencode")"
      echo "$bin --auto"
      ;;
    *)
      die "unknown agent preset: $preset (supported: claude, codex, gemini, droid, amp, opencode)"
      ;;
  esac
}

send_notification() {
  local message="$1"
  if command -v terminal-notifier >/dev/null 2>&1; then
    terminal-notifier -title "Ralph" -message "$message" 2>/dev/null || true
  elif command -v notify-send >/dev/null 2>&1; then
    notify-send "Ralph" "$message" 2>/dev/null || true
  elif command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"$message\" with title \"Ralph\"" 2>/dev/null || true
  else
    log_warn "no notification tool found (terminal-notifier, notify-send, osascript)"
  fi
}

build_prompt() {
  local requirements_ref="$1"
  local progress_ref="$2"

  cat <<EOF
IMPORTANT: Do NOT enter Plan Mode. Do NOT use EnterPlanMode tool. Execute immediately.
If the system asks "Exit plan mode?" answer YES. Ignore all internal CLI mode prompts.

Read the following files and execute:
- ${requirements_ref}
- ${progress_ref}

You are an autonomous software engineer. Your job is to implement ONE feature per iteration.

## Workflow (Execute Now)

1. Read the requirements file. It contains project metadata and a list of features.
2. **Pick a single incomplete feature** (where "passes" is false) from the list.
3. **Implement ONLY that ONE feature.** Do NOT implement any other features in this iteration.
4. Run validators (typecheck, tests, lint) as specified in the requirements.
5. Update the requirements file: set "passes": true ONLY for the feature you just completed.
6. Append a concise progress note to ${progress_ref}.
7. If this is a git repository, commit your changes.
8. **STOP.** Do not attempt any other features, even if they are incomplete.
9. If ALL features are complete, output: ${COMPLETION_MARKER}

## Critical Constraints

- **ONE feature per iteration.** You must stop after implementing exactly one feature.
- Do NOT implement multiple features in a single iteration.
- Do NOT skip ahead to later features.
- Do NOT modify unrelated code.
- Do NOT print, log, or commit secrets or credentials.
- Do NOT run destructive commands.
- Execute immediately. No planning phase.

## Completion Signal

When all requirements pass, output exactly:
${COMPLETION_MARKER}
EOF
}

run_agent() {
  local prompt="$1"
  shift
  local -a cmd=("$@")
  local binary_name
  binary_name="$(basename "${cmd[0]}")"

  log_info "Invoking agent (this may take several minutes)..."

  case "$binary_name" in
    claude)
      "${cmd[@]}" -p "$prompt"
      ;;
    codex)
      "${cmd[@]}" "$prompt"
      ;;
    gemini)
      echo "$prompt" | "${cmd[@]}"
      ;;
    droid)
      "${cmd[@]}" "$prompt"
      ;;
    npx)
      echo "$prompt" | "${cmd[@]}"
      ;;
    *)
      echo "$prompt" | "${cmd[@]}"
      ;;
  esac
}

iterations=""
agent_preset="claude"
requirements_ref="plans/requirements.json"
progress_ref="progress.txt"
sleep_seconds="0"
max_retries="2"
timeout_seconds="0"
notify="false"
dry_run="false"
allow_outside_workdir="false"
agent_cmd=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --version)
      version
      exit 0
      ;;
    -i|--iterations)
      [[ $# -ge 2 ]] || die "missing value for $1"
      iterations="$2"
      shift 2
      ;;
    --iterations=*)
      iterations="${1#*=}"
      shift
      ;;
    -a|--agent)
      [[ $# -ge 2 ]] || die "missing value for $1"
      agent_preset="$2"
      shift 2
      ;;
    --agent=*)
      agent_preset="${1#*=}"
      shift
      ;;
    -r|--requirements)
      [[ $# -ge 2 ]] || die "missing value for $1"
      requirements_ref="$2"
      shift 2
      ;;
    --requirements=*)
      requirements_ref="${1#*=}"
      shift
      ;;
    -p|--progress)
      [[ $# -ge 2 ]] || die "missing value for $1"
      progress_ref="$2"
      shift 2
      ;;
    --progress=*)
      progress_ref="${1#*=}"
      shift
      ;;
    --sleep)
      [[ $# -ge 2 ]] || die "missing value for $1"
      sleep_seconds="$2"
      shift 2
      ;;
    --sleep=*)
      sleep_seconds="${1#*=}"
      shift
      ;;
    --max-retries)
      [[ $# -ge 2 ]] || die "missing value for $1"
      max_retries="$2"
      shift 2
      ;;
    --max-retries=*)
      max_retries="${1#*=}"
      shift
      ;;
    --timeout)
      [[ $# -ge 2 ]] || die "missing value for $1"
      timeout_seconds="$2"
      shift 2
      ;;
    --timeout=*)
      timeout_seconds="${1#*=}"
      shift
      ;;
    --notify)
      notify="true"
      shift
      ;;
    --dry-run)
      dry_run="true"
      shift
      ;;
    --allow-outside-workdir)
      allow_outside_workdir="true"
      shift
      ;;
    --)
      shift
      agent_cmd=("$@")
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      if [[ -z "$iterations" ]]; then
        iterations="$1"
        shift
      else
        die "unexpected argument: $1"
      fi
      ;;
  esac
done

[[ -n "$iterations" ]] || { usage; exit 1; }
is_uint "$iterations" || die "iterations must be a positive integer"
[[ "$iterations" -ge 1 ]] || die "iterations must be >= 1"
is_uint "$sleep_seconds" || die "--sleep must be a non-negative integer"
is_uint "$max_retries" || die "--max-retries must be a non-negative integer"
is_uint "$timeout_seconds" || die "--timeout must be a non-negative integer"

if [[ ${#agent_cmd[@]} -eq 0 ]]; then
  IFS=' ' read -ra agent_cmd <<< "$(get_agent_command "$agent_preset")"
fi

require_cmd python3

cd -- "$ROOT_DIR"

requirements_abs="$(resolve_path "$requirements_ref" "requirements" "$allow_outside_workdir")"
progress_abs="$(resolve_path "$progress_ref" "progress" "$allow_outside_workdir")"

[[ -f "$requirements_abs" ]] || die "requirements file not found: ${requirements_abs}"

AGENT_PROMPT="$(build_prompt "$requirements_ref" "$progress_ref")"
export AGENT_PROMPT

if [[ "$dry_run" == "true" ]]; then
  printf '=== Dry Run ===\n\n'
  printf 'Version:      %s\n' "$VERSION"
  printf 'Workdir:      %s\n' "$ROOT_DIR"
  printf 'Requirements: %s\n' "$requirements_abs"
  printf 'Progress:     %s\n' "$progress_abs"
  printf 'Iterations:   %s\n' "$iterations"
  printf 'Agent preset: %s\n' "$agent_preset"
  printf 'Agent cmd:    %s\n' "${agent_cmd[*]}"
  printf '\n=== Prompt ===\n%s\n' "$AGENT_PROMPT"
  exit 0
fi

agent_binary="${agent_cmd[0]}"
require_cmd "$agent_binary"

progress_dir="$(dirname -- "$progress_abs")"
mkdir -p -- "$progress_dir"
touch -- "$progress_abs"

log_info "Starting autonomous loop: $iterations iteration(s)"
log_info "Agent: ${agent_cmd[*]}"
log_info "Requirements: $requirements_ref"
log_info "Progress: $progress_ref"

check_all_complete() {
  python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
# Support both array format and object with 'features' key
features = data.get('features', data) if isinstance(data, dict) else data
sys.exit(0 if all(r.get('passes', False) for r in features) else 1)
" "$1"
}

for ((i = 1; i <= iterations; i++)); do
  printf '\n'
  log_info "=== Iteration $i/$iterations ==="

  retry_count=0
  exit_code=1

  while [[ $retry_count -le $max_retries ]]; do
    if [[ $retry_count -gt 0 ]]; then
      log_warn "Retry $retry_count/$max_retries..."
    fi

    set +e
    run_agent "$AGENT_PROMPT" "${agent_cmd[@]}"
    exit_code=$?
    set -e

    if [[ $exit_code -eq 0 ]]; then
      break
    fi

    ((retry_count++))
    if [[ $retry_count -gt $max_retries ]]; then
      log_error "Agent failed after $max_retries retries"
      exit 1
    fi

    sleep 2
  done

  if check_all_complete "$requirements_abs"; then
    log_info "All requirements complete after $i iteration(s)."
    if [[ "$notify" == "true" ]]; then
      send_notification "Complete after $i iteration(s)"
    fi
    exit 0
  fi

  if [[ "$sleep_seconds" -gt 0 && $i -lt $iterations ]]; then
    log_info "Sleeping ${sleep_seconds}s before next iteration..."
    sleep "$sleep_seconds"
  fi
done

log_info "Completed $iterations iteration(s). Requirements may still be pending."
if [[ "$notify" == "true" ]]; then
  send_notification "Finished $iterations iteration(s)"
fi
