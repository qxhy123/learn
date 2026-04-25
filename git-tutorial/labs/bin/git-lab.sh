#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACES_ROOT="$LABS_ROOT/workspaces"
FORCE=0
SMOKE=0
LAB_ID=""
TARGET=""

LAB_IDS=(
  LAB-ORIENT-STATUS-01
  LAB-MODEL-STATE-01
  LAB-MODEL-INDEX-01
  LAB-MODEL-HISTORY-01
  LAB-DAILY-CLEAN-COMMIT-01
  LAB-DAILY-DIFF-REVIEW-01
  LAB-DAILY-IGNORE-01
  LAB-BRANCH-TASK-01
  LAB-BRANCH-CONFLICT-01
  LAB-BRANCH-REBASE-01
  LAB-COLLAB-REMOTE-01
  LAB-COLLAB-PUSH-REJECTED-01
  LAB-COLLAB-PR-01
  LAB-RECOVERY-UNDO-01
  LAB-RECOVERY-BAD-COMMIT-01
  LAB-RECOVERY-REFLOG-01
  LAB-RELEASE-STASH-WORKTREE-01
  LAB-RELEASE-HOTFIX-TAG-01
  LAB-DEBUG-BISECT-01
  LAB-GOV-HOOKS-01
  LAB-GOV-LARGE-REPO-01
  LAB-GOV-DISASTER-01
)

usage() {
  cat <<'EOF'
Usage:
  ./bin/git-lab.sh --list
  ./bin/git-lab.sh --smoke [--force]
  ./bin/git-lab.sh <LAB-ID> [--force]

Examples:
  ./bin/git-lab.sh LAB-MODEL-STATE-01 --force
  ./bin/git-lab.sh LAB-GOV-HOOKS-01
  ./bin/git-lab.sh --smoke --force

Workspaces are created under labs/workspaces/<lab-slug>/.
The smoke path creates every listed lab under labs/workspaces/_smoke/ and
runs a minimal Git observation command in each generated repository.
EOF
}

info() { printf '%s\n' "$*"; }
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

lab_known() {
  local candidate="$1" id
  for id in "${LAB_IDS[@]}"; do
    [[ "$candidate" == "$id" ]] && return 0
  done
  return 1
}

slug_for() {
  local lab="$1"
  lab_known "$lab" || return 1
  case "$lab" in
    LAB-ORIENT-STATUS-01) printf 'orient-status' ;;
    LAB-MODEL-STATE-01) printf 'model-state' ;;
    LAB-MODEL-INDEX-01) printf 'model-index' ;;
    LAB-MODEL-HISTORY-01) printf 'model-history' ;;
    LAB-DAILY-CLEAN-COMMIT-01) printf 'daily-clean-commit' ;;
    LAB-DAILY-DIFF-REVIEW-01) printf 'daily-diff-review' ;;
    LAB-DAILY-IGNORE-01) printf 'daily-ignore' ;;
    LAB-BRANCH-TASK-01) printf 'branch-task' ;;
    LAB-BRANCH-CONFLICT-01) printf 'branch-conflict' ;;
    LAB-BRANCH-REBASE-01) printf 'branch-rebase' ;;
    LAB-COLLAB-REMOTE-01) printf 'collab-remote' ;;
    LAB-COLLAB-PUSH-REJECTED-01) printf 'collab-push-rejected' ;;
    LAB-COLLAB-PR-01) printf 'collab-pr' ;;
    LAB-RECOVERY-UNDO-01) printf 'recovery-undo' ;;
    LAB-RECOVERY-BAD-COMMIT-01) printf 'recovery-bad-commit' ;;
    LAB-RECOVERY-REFLOG-01) printf 'recovery-reflog' ;;
    LAB-RELEASE-STASH-WORKTREE-01) printf 'release-stash-worktree' ;;
    LAB-RELEASE-HOTFIX-TAG-01) printf 'release-hotfix-tag' ;;
    LAB-DEBUG-BISECT-01) printf 'debug-bisect' ;;
    LAB-GOV-HOOKS-01) printf 'gov-hooks' ;;
    LAB-GOV-LARGE-REPO-01) printf 'gov-large-repo' ;;
    LAB-GOV-DISASTER-01) printf 'gov-disaster' ;;
  esac
}

repo_hint_for() {
  case "$1" in
    LAB-ORIENT-STATUS-01) printf 'status-lab' ;;
    LAB-MODEL-STATE-01) printf 'state-lab' ;;
    LAB-MODEL-INDEX-01) printf 'index-lab' ;;
    LAB-MODEL-HISTORY-01) printf 'history-lab' ;;
    LAB-DAILY-CLEAN-COMMIT-01) printf 'clean-commit-lab' ;;
    LAB-DAILY-DIFF-REVIEW-01) printf 'diff-review-lab' ;;
    LAB-DAILY-IGNORE-01) printf 'ignore-lab' ;;
    LAB-BRANCH-TASK-01) printf 'task-branch-lab' ;;
    LAB-BRANCH-CONFLICT-01) printf 'alice' ;;
    LAB-BRANCH-REBASE-01) printf 'alice' ;;
    LAB-COLLAB-REMOTE-01) printf 'learner' ;;
    LAB-COLLAB-PUSH-REJECTED-01) printf 'alice' ;;
    LAB-COLLAB-PR-01) printf 'review-lab' ;;
    LAB-RECOVERY-UNDO-01) printf 'undo-lab' ;;
    LAB-RECOVERY-BAD-COMMIT-01) printf 'bad-commit-lab' ;;
    LAB-RECOVERY-REFLOG-01) printf 'reflog-lab' ;;
    LAB-RELEASE-STASH-WORKTREE-01) printf 'interrupt-lab' ;;
    LAB-RELEASE-HOTFIX-TAG-01) printf 'release-lab' ;;
    LAB-DEBUG-BISECT-01) printf 'bisect-lab' ;;
    LAB-GOV-HOOKS-01) printf 'governance-lab' ;;
    LAB-GOV-LARGE-REPO-01) printf 'large-repo-lab' ;;
    LAB-GOV-DISASTER-01) printf 'incident-lab' ;;
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
  if [[ "$1" == "--smoke" ]]; then SMOKE=1; shift; fi
  if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then usage; exit 0; fi
  if [[ "$SMOKE" -eq 0 ]]; then
    [[ $# -gt 0 ]] || die "Missing LAB-ID"
    LAB_ID="$1"; shift
  fi
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
  local repo="$1" message="$2"
  git -C "$repo" add -A
  git -C "$repo" commit -m "$message" >/dev/null
}

init_repo() {
  local repo="$1"
  mkdir -p "$repo"
  git -C "$repo" init -b main >/dev/null
  configure_repo "$repo"
}

create_state_lab() {
  local repo="${1:-$TARGET/state-lab}"
  init_repo "$repo"
  printf '# State Lab\n\nTrack working tree, index, and HEAD.\n' > "$repo/README.md"
  printf 'alpha\nbeta\n' > "$repo/notes.md"
  commit_all "$repo" "chore: initialize state lab"
  printf 'staged line\n' >> "$repo/notes.md"
  git -C "$repo" add notes.md
  printf 'unstaged line\n' >> "$repo/notes.md"
  printf 'scratch\n' > "$repo/scratch.txt"
}

create_index_lab() {
  local repo="$TARGET/index-lab"
  init_repo "$repo"
  printf 'title: index lab\nstatus: clean\n' > "$repo/card.txt"
  commit_all "$repo" "chore: initialize index lab"
  printf 'title: index lab\nstatus: staged\n' > "$repo/card.txt"
  git -C "$repo" add card.txt
  printf 'title: index lab\nstatus: staged\nnotes: unstaged after add\n' > "$repo/card.txt"
}

create_history_lab() {
  local repo="$TARGET/history-lab"
  init_repo "$repo"
  printf 'v1\n' > "$repo/story.txt"
  commit_all "$repo" "chore: start story"
  printf 'v1\nv2\n' > "$repo/story.txt"
  commit_all "$repo" "feat: add second scene"
  git -C "$repo" tag -a v1.0.0 -m "v1.0.0"
  git -C "$repo" switch -c experiment/ending >/dev/null
  printf 'v1\nv2\nexperimental ending\n' > "$repo/story.txt"
  commit_all "$repo" "experiment: draft alternate ending"
  git -C "$repo" switch main >/dev/null
  printf 'v1\nv2\nmain ending\n' > "$repo/story.txt"
  commit_all "$repo" "feat: add main ending"
}

create_daily_clean_commit() {
  local repo="$TARGET/clean-commit-lab"
  init_repo "$repo"
  printf '# Clean Commit Lab\n' > "$repo/README.md"
  printf 'enabled=false\nthreshold=10\n' > "$repo/app.conf"
  commit_all "$repo" "chore: initialize clean commit lab"
  printf 'enabled=true\nthreshold=10\n' > "$repo/app.conf"
  printf 'manual verification notes\n' > "$repo/notes.md"
}

create_diff_review() {
  local repo="$TARGET/diff-review-lab"
  init_repo "$repo"
  printf 'def price(base):\n    return base\n' > "$repo/billing.py"
  printf '# Billing\n' > "$repo/README.md"
  commit_all "$repo" "chore: initialize billing sample"
  cat > "$repo/billing.py" <<'EOF'
def price(base):
    discount = 2
    return base - discount

def format_price(value):
    return f"${value:.2f}"
EOF
  printf '\nReview checklist:\n- pricing behavior\n- formatting behavior\n' >> "$repo/README.md"
}

create_ignore_lab() {
  local repo="$TARGET/ignore-lab"
  init_repo "$repo"
  mkdir -p "$repo/build"
  printf 'tracked output\n' > "$repo/build/output.log"
  printf 'source\n' > "$repo/app.txt"
  commit_all "$repo" "chore: initialize ignore lab with tracked output"
  printf 'build/\n*.tmp\n' > "$repo/.gitignore"
  printf 'new ignored output\n' > "$repo/build/new.log"
  printf 'scratch\n' > "$repo/scratch.tmp"
  printf 'source changed\n' > "$repo/app.txt"
}

create_branch_task() {
  local repo="$TARGET/task-branch-lab"
  init_repo "$repo"
  printf '# Task Branch Lab\n' > "$repo/README.md"
  printf 'status=base\n' > "$repo/app.conf"
  commit_all "$repo" "chore: initialize task branch lab"
  git -C "$repo" switch -c feature/status-message >/dev/null
  printf 'status=feature\n' > "$repo/app.conf"
  printf 'Implement a small task, then compare main and feature/status-message.\n' > "$repo/TASK.md"
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

create_branch_rebase() {
  local origin="$TARGET/origin.git" alice="$TARGET/alice" bob="$TARGET/bob"
  git init --bare --initial-branch=main "$origin" >/dev/null
  git clone "$origin" "$alice" >/dev/null 2>&1
  configure_repo "$alice"
  printf 'line 1\n' > "$alice/guide.md"
  commit_all "$alice" "chore: initialize guide"
  git -C "$alice" push -u origin main >/dev/null 2>&1
  git -C "$alice" switch -c feature/local-notes >/dev/null
  printf 'line 1\nlocal feature note\n' > "$alice/guide.md"
  commit_all "$alice" "feat: add local notes"
  git clone "$origin" "$bob" >/dev/null 2>&1
  configure_repo "$bob"
  printf 'line 1\nmain update\n' > "$bob/guide.md"
  commit_all "$bob" "docs: update guide on main"
  git -C "$bob" push >/dev/null 2>&1
}

create_collab_remote() {
  local origin="$TARGET/origin.git" seed="$TARGET/seed" learner="$TARGET/learner" teammate="$TARGET/teammate"
  git init --bare --initial-branch=main "$origin" >/dev/null
  git clone "$origin" "$seed" >/dev/null 2>&1
  configure_repo "$seed"
  printf '# Remote Lab\n' > "$seed/README.md"
  commit_all "$seed" "chore: initialize remote lab"
  git -C "$seed" push -u origin main >/dev/null 2>&1
  git -C "$seed" switch -c docs/update >/dev/null
  printf 'draft docs\n' > "$seed/docs.txt"
  commit_all "$seed" "docs: draft remote branch"
  git -C "$seed" push -u origin docs/update >/dev/null 2>&1
  git clone "$origin" "$learner" >/dev/null 2>&1
  configure_repo "$learner"
  git clone "$origin" "$teammate" >/dev/null 2>&1
  configure_repo "$teammate"
  printf '# Remote Lab\n\nteammate update\n' > "$teammate/README.md"
  commit_all "$teammate" "docs: teammate updates main"
  git -C "$teammate" push >/dev/null 2>&1
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

create_collab_pr() {
  local repo="$TARGET/review-lab"
  init_repo "$repo"
  printf '# Review Lab\n' > "$repo/README.md"
  printf 'feature=false\n' > "$repo/app.conf"
  commit_all "$repo" "chore: initialize review lab"
  git -C "$repo" switch -c feature/reviewable-change >/dev/null
  printf 'feature=true\n' > "$repo/app.conf"
  cat > "$repo/PR-CHECKLIST.md" <<'EOF'
# PR Self-check

- [ ] I reviewed `git diff main...HEAD`.
- [ ] Commits are small and explain intent.
- [ ] Risk and rollback notes are included.
EOF
  commit_all "$repo" "feat: enable reviewable change"
}

create_recovery_undo() {
  local repo="$TARGET/undo-lab"
  init_repo "$repo"
  printf 'v1\n' > "$repo/story.txt"
  commit_all "$repo" "chore: v1"
  printf 'v2 shared safe change\n' >> "$repo/story.txt"
  commit_all "$repo" "feat: shared v2"
  printf 'staged fix\n' >> "$repo/story.txt"
  git -C "$repo" add story.txt
  printf 'unstaged scratch\n' >> "$repo/story.txt"
  printf 'untracked scratch\n' > "$repo/scratch.txt"
}

create_recovery_bad_commit() {
  local origin="$TARGET/origin.git" repo="$TARGET/bad-commit-lab"
  git init --bare --initial-branch=main "$origin" >/dev/null
  git clone "$origin" "$repo" >/dev/null 2>&1
  configure_repo "$repo"
  printf 'name=demo\n' > "$repo/app.conf"
  commit_all "$repo" "chore: initialize bad commit lab"
  git -C "$repo" push -u origin main >/dev/null 2>&1
  printf 'name=demo\nmode=broken\n' > "$repo/app.conf"
  commit_all "$repo" "oops"
  printf 'forgotten test note\n' > "$repo/test-plan.md"
  printf 'name=demo\nmode=shared-bug\n' > "$repo/app.conf"
  commit_all "$repo" "feat: shared but wrong behavior"
  git -C "$repo" push >/dev/null 2>&1
}

create_recovery_reflog() {
  local repo="$TARGET/reflog-lab"
  init_repo "$repo"
  printf 'base\n' > "$repo/work.txt"
  commit_all "$repo" "chore: base"
  printf 'important local work\n' >> "$repo/work.txt"
  commit_all "$repo" "feat: important local work"
  git -C "$repo" tag before-reset
  git -C "$repo" reset --hard HEAD~1 >/dev/null
  git -C "$repo" switch -c temp/rescue-demo >/dev/null
  printf 'detached style note\n' > "$repo/rescue.txt"
  commit_all "$repo" "feat: temporary rescue note"
  git -C "$repo" switch main >/dev/null
  git -C "$repo" branch -D temp/rescue-demo >/dev/null
}

create_release_stash_worktree() {
  local repo="$TARGET/interrupt-lab"
  init_repo "$repo"
  printf 'status=main\n' > "$repo/app.conf"
  printf '# Interrupt Lab\n' > "$repo/README.md"
  commit_all "$repo" "chore: initialize interrupt lab"
  git -C "$repo" switch -c feature/long-running >/dev/null
  printf 'status=feature\n' > "$repo/app.conf"
  printf 'draft notes\n' > "$repo/notes.md"
  git -C "$repo" add app.conf
  git -C "$repo" switch main >/dev/null
  git -C "$repo" worktree add "$TARGET/hotfix-worktree" -b hotfix/readme >/dev/null
  printf '# Interrupt Lab\n\nhotfix note\n' > "$TARGET/hotfix-worktree/README.md"
}

create_release_hotfix_tag() {
  local repo="$TARGET/release-lab"
  init_repo "$repo"
  printf 'version=1.0.0\nstatus=ok\n' > "$repo/app.conf"
  commit_all "$repo" "chore: release baseline"
  git -C "$repo" tag -a v1.0.0 -m "v1.0.0"
  printf 'version=1.1.0\nstatus=regression\n' > "$repo/app.conf"
  commit_all "$repo" "feat: prepare 1.1.0"
  git -C "$repo" tag -a v1.1.0 -m "v1.1.0"
  git -C "$repo" switch -c hotfix/v1.0.1 v1.0.0 >/dev/null
  printf 'version=1.0.1\nstatus=hotfix-needed\n' > "$repo/app.conf"
}

create_debug_bisect() {
  local repo="$TARGET/bisect-lab"
  init_repo "$repo"
  cat > "$repo/verify.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
grep -q 'result=ok' app.conf
EOF
  chmod +x "$repo/verify.sh"
  printf 'result=ok\nstep=1\n' > "$repo/app.conf"
  commit_all "$repo" "chore: good baseline"
  printf 'result=ok\nstep=2\n' > "$repo/app.conf"
  commit_all "$repo" "feat: harmless change"
  printf 'result=broken\nstep=3\n' > "$repo/app.conf"
  commit_all "$repo" "feat: introduce regression"
  printf 'result=broken\nstep=4\n' > "$repo/app.conf"
  commit_all "$repo" "docs: later noisy change"
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

create_lab() {
  case "$1" in
    LAB-ORIENT-STATUS-01) create_state_lab "$TARGET/status-lab" ;;
    LAB-MODEL-STATE-01) create_state_lab ;;
    LAB-MODEL-INDEX-01) create_index_lab ;;
    LAB-MODEL-HISTORY-01) create_history_lab ;;
    LAB-DAILY-CLEAN-COMMIT-01) create_daily_clean_commit ;;
    LAB-DAILY-DIFF-REVIEW-01) create_diff_review ;;
    LAB-DAILY-IGNORE-01) create_ignore_lab ;;
    LAB-BRANCH-TASK-01) create_branch_task ;;
    LAB-BRANCH-CONFLICT-01) create_branch_conflict ;;
    LAB-BRANCH-REBASE-01) create_branch_rebase ;;
    LAB-COLLAB-REMOTE-01) create_collab_remote ;;
    LAB-COLLAB-PUSH-REJECTED-01) create_collab_push_rejected ;;
    LAB-COLLAB-PR-01) create_collab_pr ;;
    LAB-RECOVERY-UNDO-01) create_recovery_undo ;;
    LAB-RECOVERY-BAD-COMMIT-01) create_recovery_bad_commit ;;
    LAB-RECOVERY-REFLOG-01) create_recovery_reflog ;;
    LAB-RELEASE-STASH-WORKTREE-01) create_release_stash_worktree ;;
    LAB-RELEASE-HOTFIX-TAG-01) create_release_hotfix_tag ;;
    LAB-DEBUG-BISECT-01) create_debug_bisect ;;
    LAB-GOV-HOOKS-01) create_gov_hooks ;;
    LAB-GOV-LARGE-REPO-01) create_gov_large_repo ;;
    LAB-GOV-DISASTER-01) create_gov_disaster ;;
    *) die "Unknown lab id: $1 (run --list)" ;;
  esac
}

run_lab() {
  local id="$1" slug
  slug="$(slug_for "$id")" || die "Unknown lab id: $id (run --list)"
  prepare_target "$slug"
  create_lab "$id"
  info "Created $id at: $TARGET"
  info "Read scenario: $LABS_ROOT/scenarios/$id.md"
}

smoke_all() {
  local original_root="$WORKSPACES_ROOT" id slug repo_hint repo_path
  WORKSPACES_ROOT="$LABS_ROOT/workspaces/_smoke"
  FORCE=1
  rm -rf "$WORKSPACES_ROOT"
  for id in "${LAB_IDS[@]}"; do
    slug="$(slug_for "$id")"
    run_lab "$id" >/dev/null
    repo_hint="$(repo_hint_for "$id")"
    repo_path="$WORKSPACES_ROOT/$slug/$repo_hint"
    if [[ ! -d "$repo_path/.git" ]]; then
      die "Smoke failed: expected Git repo at $repo_path for $id"
    fi
    git -C "$repo_path" status -sb >/dev/null
    git -C "$repo_path" log --oneline --max-count=1 >/dev/null
    printf 'PASS %s -> %s\n' "$id" "$repo_path"
  done
  rm -rf "$WORKSPACES_ROOT"
  WORKSPACES_ROOT="$original_root"
  info "Smoke completed for ${#LAB_IDS[@]} labs."
}

parse_args "$@"
if [[ "$SMOKE" -eq 1 ]]; then
  smoke_all
else
  run_lab "$LAB_ID"
fi
