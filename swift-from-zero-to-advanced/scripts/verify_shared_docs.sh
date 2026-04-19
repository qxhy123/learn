#!/usr/bin/env bash
set -euo pipefail

check_file() {
  local path="$1"
  [[ -f "$path" ]] || {
    echo "missing:$path"
    exit 1
  }
}

check_heading() {
  local pattern="$1"
  local path="$2"
  rg -q "$pattern" "$path" || {
    echo "missing-heading:$path:$pattern"
    exit 1
  }
}

check_file "swift-from-zero-to-advanced/references/authoring-rules.md"
check_file "swift-from-zero-to-advanced/references/bilingual-style-guide.md"
check_file "swift-from-zero-to-advanced/references/chapter-template.md"
check_file "swift-from-zero-to-advanced/references/learning-paths.md"
check_file "swift-from-zero-to-advanced/glossary/core-terms.md"

check_heading "^# Authoring Rules$" "swift-from-zero-to-advanced/references/authoring-rules.md"
check_heading "^# Bilingual Style Guide$" "swift-from-zero-to-advanced/references/bilingual-style-guide.md"
check_heading "^# Chapter Template$" "swift-from-zero-to-advanced/references/chapter-template.md"
check_heading "^# Learning Paths$" "swift-from-zero-to-advanced/references/learning-paths.md"
check_heading "^# Core Terms Glossary$" "swift-from-zero-to-advanced/glossary/core-terms.md"

echo "shared-docs-ok"
