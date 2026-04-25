#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACES_ROOT="$LABS_ROOT/workspaces"
FORCE=0

LAB_IDS=(
  LAB-BRANCH-CONFLICT-01
  LAB-BRANCH-REBASE-01
  LAB-BRANCH-TASK-01
  LAB-COLLAB-PR-01
  LAB-COLLAB-PUSH-REJECTED-01
  LAB-COLLAB-REMOTE-01
  LAB-DAILY-CLEAN-COMMIT-01
  LAB-DAILY-COMMIT-01
  LAB-DAILY-DIFF-REVIEW-01
  LAB-DAILY-IGNORE-01
  LAB-DEBUG-BISECT-01
  LAB-GOV-DISASTER-01
  LAB-GOV-HOOKS-01
  LAB-GOV-LARGE-REPO-01
  LAB-MODEL-HISTORY-01
  LAB-MODEL-INDEX-01
  LAB-MODEL-STATE-01
  LAB-ORIENT-STATUS-01
  LAB-RECOVERY-BAD-COMMIT-01
  LAB-RECOVERY-REFLOG-01
  LAB-RECOVERY-RESET-01
  LAB-RECOVERY-UNDO-01
  LAB-RELEASE-BISECT-01
  LAB-RELEASE-HOTFIX-TAG-01
  LAB-RELEASE-STASH-WORKTREE-01
  LAB-SETUP-STATE-01
)

usage() {
  cat <<'EOF'
Usage:
  ./bin/git-lab.sh --list
  ./bin/git-lab.sh <LAB-ID> [--force]

Examples:
  ./bin/git-lab.sh LAB-SETUP-STATE-01 --force
  ./bin/git-lab.sh LAB-GOV-HOOKS-01

Workspaces are created under labs/workspaces/<lab-slug>/.
EOF
}

info() { printf '%s\n' "$*"; }
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

slug_for() {
  case "$1" in
    LAB-BRANCH-CONFLICT-01) printf 'branch_conflict_01' ;;
    LAB-BRANCH-REBASE-01) printf 'branch_rebase_01' ;;
    LAB-BRANCH-TASK-01) printf 'branch_task_01' ;;
    LAB-COLLAB-PR-01) printf 'colpr_01' ;;
    LAB-COLLAB-PUSH-REJECTED-01) printf 'colpush_rejected_01' ;;
    LAB-COLLAB-REMOTE-01) printf 'colremote_01' ;;
    LAB-DAILY-CLEAN-COMMIT-01) printf 'daily_clean_commit_01' ;;
    LAB-DAILY-COMMIT-01) printf 'daily_commit_01' ;;
    LAB-DAILY-DIFF-REVIEW-01) printf 'daily_diff_review_01' ;;
    LAB-DAILY-IGNORE-01) printf 'daily_ignore_01' ;;
    LAB-DEBUG-BISECT-01) printf 'debug_bisect_01' ;;
    LAB-GOV-DISASTER-01) printf 'gov_disaster_01' ;;
    LAB-GOV-HOOKS-01) printf 'gov_hooks_01' ;;
    LAB-GOV-LARGE-REPO-01) printf 'gov_large_repo_01' ;;
    LAB-MODEL-HISTORY-01) printf 'model_history_01' ;;
    LAB-MODEL-INDEX-01) printf 'model_index_01' ;;
    LAB-MODEL-STATE-01) printf 'model_state_01' ;;
    LAB-ORIENT-STATUS-01) printf 'orient_status_01' ;;
    LAB-RECOVERY-BAD-COMMIT-01) printf 'recovery_bad_commit_01' ;;
    LAB-RECOVERY-REFLOG-01) printf 'recovery_reflog_01' ;;
    LAB-RECOVERY-RESET-01) printf 'recovery_reset_01' ;;
    LAB-RECOVERY-UNDO-01) printf 'recovery_undo_01' ;;
    LAB-RELEASE-BISECT-01) printf 'release_bisect_01' ;;
    LAB-RELEASE-HOTFIX-TAG-01) printf 'release_hotfix_tag_01' ;;
    LAB-RELEASE-STASH-WORKTREE-01) printf 'release_stash_worktree_01' ;;
    LAB-SETUP-STATE-01) printf 'setup_state_01' ;;
    *) return 1 ;;
  esac
}

list_labs() {
  local id
  for id in "${LAB_IDS[@]}"; do
    printf '%s\n' "$id"
  done
}

parse_args() {
  if [[ $# -eq 0 ]]; then usage; exit 0; fi
  if [[ "$1" == "--list" ]]; then list_labs; exit 0; fi
  if [[ "$1" == "-h" || "$1" == "--help" ]]; then usage; exit 0; fi
  LAB_ID="$1"; shift
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --force) FORCE=1 ;;
      -h|--help) usage; exit 0 ;;
      *) die "Unknown option: $1" ;;
    esac
    shift
  done
}

prepare_target() {
  local slug="$1"
  mkdir -p "$WORKSPACES_ROOT"
  TARGET="$WORKSPACES_ROOT/$slug"
  case "$TARGET" in
    "$WORKSPACES_ROOT"/*) ;;
    *) die "Refusing to touch path outside labs/workspaces: $TARGET" ;;
  esac
  if [[ -e "$TARGET" ]]; then
    if [[ "$FORCE" -eq 1 ]]; then
      rm -rf "$TARGET"
    else
      die "Target already exists: $TARGET (use --force to recreate)"
    fi
  fi
  mkdir -p "$TARGET"
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

init_repo() {
  local repo="$1"
  mkdir -p "$repo"
  git -C "$repo" init -b main >/dev/null
  configure_repo "$repo"
}

create_setup_state() {
  local repo="$TARGET/state-lab"
  init_repo "$repo"
  printf '# State Lab\n' > "$repo/README.md"
  printf 'alpha\nbeta\n' > "$repo/notes.md"
  commit_all "$repo" "chore: initialize state lab"
  printf 'staged line\n' >> "$repo/notes.md"
  git -C "$repo" add notes.md
  printf 'unstaged line\n' >> "$repo/notes.md"
  printf 'scratch\n' > "$repo/scratch.txt"
}

create_daily_commit() {
  local repo="$TARGET/daily-lab"
  init_repo "$repo"
  printf '# Daily Lab\n' > "$repo/README.md"
  printf 'total=1\nmode=demo\n' > "$repo/app.txt"
  commit_all "$repo" "chore: initialize daily lab"
  printf 'total=2\nmode=demo\nfeature=true\n' > "$repo/app.txt"
  printf 'debug.log\n' > "$repo/debug.log"
  printf '*.log\n' > "$repo/.gitignore"
}

create_branch_conflict() {
  local origin="$TARGET/origin.git" alice="$TARGET/alice" bob="$TARGET/bob"
  git init --bare --initial-branch=main "$origin" >/dev/null
  git clone "$origin" "$alice" >/dev/null 2>&1
  configure_repo "$alice"
  printf 'title: profile\ncolor: blue\n' > "$alice/profile.txt"
  commit_all "$alice" "chore: initialize profile"
  git -C "$alice" push -u origin main >/dev/null 2>&1
  git clone "$origin" "$bob" >/dev/null 2>&1
  configure_repo "$bob"
  git -C "$alice" switch -c feature/profile >/dev/null
  printf 'title: profile\ncolor: green\n' > "$alice/profile.txt"
  commit_all "$alice" "feat: choose profile green"
  printf 'title: profile\ncolor: orange\n' > "$bob/profile.txt"
  commit_all "$bob" "feat: choose profile orange"
  git -C "$bob" push >/dev/null 2>&1
}

create_collab_push_rejected() {
  local origin="$TARGET/origin.git" alice="$TARGET/alice" bob="$TARGET/bob"
  git init --bare --initial-branch=main "$origin" >/dev/null
  git clone "$origin" "$alice" >/dev/null 2>&1
  configure_repo "$alice"
  printf 'base line\n' > "$alice/app.txt"
  commit_all "$alice" "chore: initialize collaboration"
  git -C "$alice" push -u origin main >/dev/null 2>&1
  git clone "$origin" "$bob" >/dev/null 2>&1
  configure_repo "$bob"
  printf 'base line\nbob remote change\n' > "$bob/app.txt"
  commit_all "$bob" "feat: bob updates main"
  git -C "$bob" push >/dev/null 2>&1
  printf 'base line\nalice local change\n' > "$alice/app.txt"
  commit_all "$alice" "feat: alice updates main"
}

create_recovery_reset() {
  local repo="$TARGET/recovery-lab"
  init_repo "$repo"
  printf 'v1\n' > "$repo/story.txt"
  commit_all "$repo" "chore: v1"
  printf 'v2\n' >> "$repo/story.txt"
  commit_all "$repo" "feat: v2 important work"
  git -C "$repo" tag safe-point
  printf 'v3\n' >> "$repo/story.txt"
  commit_all "$repo" "feat: v3 more work"
  git -C "$repo" reset --hard HEAD~1 >/dev/null
}

create_release_bisect() {
  local repo="$TARGET/release-lab"
  init_repo "$repo"
  cat > "$repo/verify.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
grep -q 'result=ok' app.conf
EOF
  chmod +x "$repo/verify.sh"
  printf 'result=ok\n' > "$repo/app.conf"
  commit_all "$repo" "chore: release baseline"
  git -C "$repo" tag -a v1.0.0 -m "v1.0.0"
  printf 'result=ok\nfeature=on\n' > "$repo/app.conf"
  commit_all "$repo" "feat: enable feature"
  printf 'result=broken\nfeature=on\n' > "$repo/app.conf"
  commit_all "$repo" "feat: change result calculation"
  git -C "$repo" tag -a v1.1.0 -m "v1.1.0"
  git -C "$repo" switch -c release/1.1 >/dev/null
}

create_gov_hooks() {
  local repo="$TARGET/governance-lab"
  init_repo "$repo"
  mkdir -p "$repo/.githooks"
  cat > "$repo/.githooks/pre-commit" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if git diff --cached --name-only | grep -E '\.(pem|key)$' >/dev/null; then
  echo 'Refusing to commit private-key-like files.' >&2
  exit 1
fi
EOF
  chmod +x "$repo/.githooks/pre-commit"
  git -C "$repo" config core.hooksPath .githooks
  git -C "$repo" config alias.st 'status -sb'
  git -C "$repo" config alias.lg 'log --oneline --graph --decorate --all'
  printf '# Governance Lab\n' > "$repo/README.md"
  commit_all "$repo" "chore: initialize governance lab"
  printf 'demo-secret\n' > "$repo/secret.pem"
  git -C "$repo" add secret.pem
}

create_gov_large_repo() {
  local repo="$TARGET/large-repo-lab"
  init_repo "$repo"
  mkdir -p "$repo/src" "$repo/dist" "$repo/assets" "$repo/docs"
  printf 'console.log("hello")\n' > "$repo/src/app.js"
  printf 'generated bundle\n' > "$repo/dist/app.bundle.js"
  python3 - <<PY2
from pathlib import Path
Path('$repo/assets/intro.mp4').write_bytes(b'0' * 2048)
Path('$repo/assets/model.onnx').write_bytes(b'1' * 2048)
PY2
  printf '# LFS Decision\n\n| path | decision | reason |\n|---|---|---|\n' > "$repo/LFS_DECISION.md"
  commit_all "$repo" "chore: initialize large repo sample"
  printf 'dist/\n*.log\n' > "$repo/.gitignore"
}

create_gov_disaster() {
  local repo="$TARGET/incident-lab"
  init_repo "$repo"
  cat > "$repo/POLICY.md" <<'EOF'
# Team Git Policy Draft

- Main branch:
- Feature branch naming:
- Merge strategy:
- Required checks:
- History rewrite boundary:
- Release tag owner:
EOF
  cat > "$repo/INCIDENT-CARDS.md" <<'EOF'
# Incident Cards

## Main is broken
- Observe:
- Freeze:
- Backup:
- Restore:
- Follow-up:

## Remote branch was force-pushed incorrectly
- Observe:
- Freeze:
- Backup:
- Restore:
- Follow-up:

## Release tag points to wrong commit
- Observe:
- Freeze:
- Backup:
- Restore:
- Follow-up:
EOF
  commit_all "$repo" "docs: add policy and incident card templates"
  git -C "$repo" switch -c task/example >/dev/null
  printf 'example\n' > "$repo/change.txt"
  commit_all "$repo" "feat: example task"
  git -C "$repo" switch main >/dev/null
  git -C "$repo" tag -a v1.0.0 -m "v1.0.0"
}

parse_args "$@"
SLUG="$(slug_for "$LAB_ID")" || die "Unknown lab id: $LAB_ID (run --list)"
prepare_target "$SLUG"
case "$LAB_ID" in
  LAB-BRANCH-CONFLICT-01) create_branch_conflict ;;
  LAB-BRANCH-REBASE-01) create_setup_state ;;
  LAB-BRANCH-TASK-01) create_setup_state ;;
  LAB-COLLAB-PR-01) create_setup_state ;;
  LAB-COLLAB-PUSH-REJECTED-01) create_collab_push_rejected ;;
  LAB-COLLAB-REMOTE-01) create_setup_state ;;
  LAB-DAILY-CLEAN-COMMIT-01) create_setup_state ;;
  LAB-DAILY-COMMIT-01) create_daily_commit ;;
  LAB-DAILY-DIFF-REVIEW-01) create_setup_state ;;
  LAB-DAILY-IGNORE-01) create_setup_state ;;
  LAB-DEBUG-BISECT-01) create_setup_state ;;
  LAB-GOV-DISASTER-01) create_gov_disaster ;;
  LAB-GOV-HOOKS-01) create_gov_hooks ;;
  LAB-GOV-LARGE-REPO-01) create_gov_large_repo ;;
  LAB-MODEL-HISTORY-01) create_setup_state ;;
  LAB-MODEL-INDEX-01) create_setup_state ;;
  LAB-MODEL-STATE-01) create_setup_state ;;
  LAB-ORIENT-STATUS-01) create_setup_state ;;
  LAB-RECOVERY-BAD-COMMIT-01) create_setup_state ;;
  LAB-RECOVERY-REFLOG-01) create_setup_state ;;
  LAB-RECOVERY-RESET-01) create_recovery_reset ;;
  LAB-RECOVERY-UNDO-01) create_setup_state ;;
  LAB-RELEASE-BISECT-01) create_release_bisect ;;
  LAB-RELEASE-HOTFIX-TAG-01) create_setup_state ;;
  LAB-RELEASE-STASH-WORKTREE-01) create_setup_state ;;
  LAB-SETUP-STATE-01) create_setup_state ;;
esac

info "Created $LAB_ID at: $TARGET"
info "Read scenario: $LABS_ROOT/scenarios/$LAB_ID.md"
