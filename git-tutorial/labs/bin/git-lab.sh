#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACES_ROOT="$LABS_ROOT/workspaces"

SCENARIOS=("basics" "collaboration" "recovery" "advanced")
FORCE=0

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '%s\n' "$*"
}

usage() {
  cat <<'EOF'
Usage:
  ./bin/git-lab.sh --list
  ./bin/git-lab.sh <scenario> [--force]

Scenarios:
  basics
  collaboration
  recovery
  advanced

Options:
  --list   List all scenarios
  --force  Recreate the target workspace if it already exists
  -h       Show this help
EOF
}

is_known_scenario() {
  local candidate="$1"
  local item
  for item in "${SCENARIOS[@]}"; do
    if [[ "$item" == "$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

ensure_workspace_dir() {
  mkdir -p "$WORKSPACES_ROOT"
}

prepare_target() {
  local target="$1"
  ensure_workspace_dir
  case "$target" in
    "$WORKSPACES_ROOT"/*) ;;
    *) die "Refusing to touch path outside labs/workspaces: $target" ;;
  esac

  if [[ -e "$target" ]]; then
    if [[ "$FORCE" -eq 1 ]]; then
      rm -rf "$target"
    else
      die "Target already exists: $target (use --force to recreate)"
    fi
  fi

  mkdir -p "$target"
}

configure_repo() {
  local repo="$1"
  git -C "$repo" config user.name "Git Tutorial Lab"
  git -C "$repo" config user.email "lab@example.com"
}

commit_all() {
  local repo="$1"
  local message="$2"
  git -C "$repo" add -A
  git -C "$repo" commit -m "$message" >/dev/null
}

create_basics() {
  local target="$WORKSPACES_ROOT/basics"
  local repo="$target/git-lab-foundations"

  prepare_target "$target"
  mkdir -p "$repo"
  git -C "$repo" init -b main >/dev/null
  configure_repo "$repo"

  cat >"$repo/README.md" <<'EOF'
# Git Lab Foundations

Use this repo for chapters 1-6.
EOF

  cat >"$repo/.gitignore" <<'EOF'
dist/
.env.local
*.log
EOF

  mkdir -p "$repo/src"
  cat >"$repo/src/app.txt" <<'EOF'
hello git tutorial
EOF
  commit_all "$repo" "chore: initialize basics lab"

  cat >"$repo/notes.md" <<'EOF'
# Notes

- observe status
- observe diff
EOF
  commit_all "$repo" "docs: add note template"

  printf '\n- already staged line\n' >>"$repo/notes.md"
  git -C "$repo" add notes.md
  printf -- '- worktree only line\n' >>"$repo/notes.md"

  mkdir -p "$repo/dist"
  cat >"$repo/dist/app.js" <<'EOF'
compiled output
EOF
  cat >"$repo/.env.local" <<'EOF'
SECRET=demo
EOF
  cat >"$repo/scratch.txt" <<'EOF'
untracked working note
EOF

  info "Created basics lab at: $repo"
  info "Next commands:"
  info "  cd $repo"
  info "  git status -sb"
  info "  git diff"
  info "  git diff --cached"
  info "  git ls-files --stage"
  info "  git check-ignore -v .env.local"
}

create_collaboration() {
  local target="$WORKSPACES_ROOT/collaboration"
  local origin="$target/origin.git"
  local alice="$target/alice"
  local bob="$target/bob"

  prepare_target "$target"

  git init --bare --initial-branch=main "$origin" >/dev/null

  git clone "$origin" "$alice" >/dev/null
  configure_repo "$alice"

  cat >"$alice/README.md" <<'EOF'
# Collaboration Lab

Use this repo for chapters 7-12.
EOF
  cat >"$alice/app.txt" <<'EOF'
base line
EOF
  commit_all "$alice" "chore: initialize collaboration lab"
  git -C "$alice" push -u origin main >/dev/null

  git clone "$origin" "$bob" >/dev/null
  configure_repo "$bob"

  printf '\nchange from bob on main\n' >>"$bob/app.txt"
  commit_all "$bob" "feat: bob advances main"
  git -C "$bob" push >/dev/null

  git -C "$alice" switch -c feature/login >/dev/null
  cat >"$alice/feature.txt" <<'EOF'
login form
EOF
  commit_all "$alice" "feat: add login form"
  printf '\nclient validation\n' >>"$alice/feature.txt"
  commit_all "$alice" "feat: add login validation"

  git -C "$alice" switch main >/dev/null
  printf '\nlocal change from alice\n' >>"$alice/app.txt"
  commit_all "$alice" "feat: alice local main change"

  info "Created collaboration lab at: $target"
  info "Important repos:"
  info "  $alice"
  info "  $bob"
  info "  $origin"
  info "Suggested next commands in alice:"
  info "  cd $alice"
  info "  git status -sb"
  info "  git fetch"
  info "  git branch -vv"
  info "  git log --oneline --graph --decorate --all"
  info "  git push    # should be rejected before syncing"
}

create_recovery() {
  local target="$WORKSPACES_ROOT/recovery"
  local repo="$target/recovery-lab"

  prepare_target "$target"
  mkdir -p "$repo"
  git -C "$repo" init -b main >/dev/null
  configure_repo "$repo"

  cat >"$repo/calc.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

amount="${1:-100}"
tax="${2:-20}"
echo $((amount + tax))
EOF
  chmod +x "$repo/calc.sh"

  cat >"$repo/verify.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

output="$(./calc.sh 100 20)"
if [[ "$output" == "120" ]]; then
  exit 0
fi

echo "Expected 120, got $output" >&2
exit 1
EOF
  chmod +x "$repo/verify.sh"
  commit_all "$repo" "feat: add calculator"

  cat >"$repo/README.md" <<'EOF'
# Recovery Lab

Use this repo for chapters 13-18.
EOF
  commit_all "$repo" "docs: add recovery lab readme"
  git -C "$repo" tag -a known-good -m "Known good state" >/dev/null

  cat >"$repo/calc.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

amount="${1:-100}"
tax="${2:-20}"
echo $((amount - tax))
EOF
  chmod +x "$repo/calc.sh"
  commit_all "$repo" "fix: refactor calculator arithmetic"

  cat >"$repo/notes.md" <<'EOF'
Bug intentionally introduced for bisect and recovery labs.
EOF
  commit_all "$repo" "docs: add bisect note"
  git -C "$repo" tag -a known-bad -m "Known bad state" >/dev/null

  git -C "$repo" branch release/1.0 known-good >/dev/null

  info "Created recovery lab at: $repo"
  info "Suggested next commands:"
  info "  cd $repo"
  info "  git log --oneline --graph --decorate --all"
  info "  ./verify.sh"
  info "  git bisect start"
  info "  git bisect bad"
  info "  git bisect good known-good"
  info "  git bisect run ./verify.sh"
  info "  git switch release/1.0"
}

create_advanced() {
  local target="$WORKSPACES_ROOT/advanced"
  local repo="$target/advanced-repo"
  local hotfix_worktree="$target/advanced-hotfix"

  prepare_target "$target"
  mkdir -p "$repo"
  git -C "$repo" init -b main >/dev/null
  configure_repo "$repo"

  cat >"$repo/service.txt" <<'EOF'
service initialized
EOF
  commit_all "$repo" "chore: initialize advanced lab"

  mkdir -p "$repo/.githooks"
  cat >"$repo/.githooks/pre-commit" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if git diff --cached | grep -q "BLOCK_ME"; then
  echo "Detected BLOCK_ME in staged changes" >&2
  exit 1
fi
EOF
  chmod +x "$repo/.githooks/pre-commit"

  cat >"$repo/README.md" <<'EOF'
# Advanced Lab

Use this repo for chapters 19-24.
EOF
  commit_all "$repo" "docs: add advanced lab readme"
  git -C "$repo" config core.hooksPath .githooks
  git -C "$repo" tag -a v1.0.0 -m "Release v1.0.0" >/dev/null
  git -C "$repo" branch release/1.0 >/dev/null

  printf '\nperf tuning notes\n' >>"$repo/service.txt"
  commit_all "$repo" "perf: add tuning note"

  git -C "$repo" worktree add "$hotfix_worktree" -b hotfix/urgent release/1.0 >/dev/null
  configure_repo "$hotfix_worktree"

  info "Created advanced lab at: $target"
  info "Important paths:"
  info "  repo: $repo"
  info "  hotfix worktree: $hotfix_worktree"
  info "Suggested next commands:"
  info "  cd $repo"
  info "  git config --show-origin --list"
  info "  git show-ref --heads --tags"
  info "  git cat-file -p HEAD"
  info "  git worktree list"
}

main() {
  local scenario=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --list)
        printf '%s\n' "${SCENARIOS[@]}"
        exit 0
        ;;
      --force)
        FORCE=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        if is_known_scenario "$1"; then
          if [[ -n "$scenario" ]]; then
            die "Only one scenario can be created at a time"
          fi
          scenario="$1"
          shift
        else
          die "Unknown argument: $1"
        fi
        ;;
    esac
  done

  if [[ -z "$scenario" ]]; then
    usage
    exit 1
  fi

  case "$scenario" in
    basics) create_basics ;;
    collaboration) create_collaboration ;;
    recovery) create_recovery ;;
    advanced) create_advanced ;;
    *) die "Unhandled scenario: $scenario" ;;
  esac
}

main "$@"
