# SwiftUI System Tutorial Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the first deliverable slice of `swiftui-system-tutorial/`: product skeleton, orientation docs, Part 1 and Part 2 chapters, `BoardFlow` starter project, appendix/lab support docs, and verification scripts.

**Architecture:** Build this as a new top-level tutorial product, not a rewrite of `swift-tutorial/`. Lock the directory contract and file inventory first with shell verifiers, then add a compileable `BoardFlow` SwiftPM starter, then author the first two tutorial parts in the same chapter style already used by `swift-tutorial/`.

**Tech Stack:** Markdown, Bash, Swift Package Manager, SwiftUI, XCTest, git

---

## Planned File Map

### Root Product Surface

- Create: `swiftui-system-tutorial/README.md`
  - Product landing page: positioning, audience, learning path, directory map.
- Create: `swiftui-system-tutorial/00-orientation.md`
  - Reading strategy, prerequisites, how `BoardFlow` evolves.
- Create: `swiftui-system-tutorial/01-learning-map.md`
  - Eight-part roadmap and capability progression.

### Tutorial Parts

- Create: `swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/01-what-swiftui-app-code-is-actually-expressing.md`
- Create: `swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/02-view-composition-and-the-three-core-layout-stacks.md`
- Create: `swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/03-fundamental-interactive-components.md`
- Create: `swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/04-state-driven-ui-fundamentals.md`
- Create: `swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/05-build-boardflow-home-shell.md`
- Create: `swiftui-system-tutorial/part2-components-navigation-and-state-ownership/06-lists-forms-and-input-contracts.md`
- Create: `swiftui-system-tutorial/part2-components-navigation-and-state-ownership/07-navigationstack-and-navigationsplitview.md`
- Create: `swiftui-system-tutorial/part2-components-navigation-and-state-ownership/08-binding-and-state-ownership.md`
- Create: `swiftui-system-tutorial/part2-components-navigation-and-state-ownership/09-observable-models-and-screen-state-coordination.md`
- Create: `swiftui-system-tutorial/part2-components-navigation-and-state-ownership/10-build-boardflow-v1-desktop-skeleton.md`

### Project Line

- Create: `swiftui-system-tutorial/projects/README.md`
  - Explains `BoardFlow` starter, checkpoints, and final structure.
- Create: `swiftui-system-tutorial/projects/boardflow/README.md`
  - Introduces the project line and where Part 1/2 fit.
- Create: `swiftui-system-tutorial/projects/boardflow/checkpoints/README.md`
  - Lists phase checkpoints and ownership.
- Create: `swiftui-system-tutorial/projects/boardflow/checkpoints/part1-shell/README.md`
  - Describes end-state after Part 1.
- Create: `swiftui-system-tutorial/projects/boardflow/checkpoints/part2-v1-workbench/README.md`
  - Describes end-state after Part 2.
- Create: `swiftui-system-tutorial/projects/boardflow/starter/README.md`
  - Build/run/test instructions for starter package.
- Create: `swiftui-system-tutorial/projects/boardflow/starter/Package.swift`
- Create: `swiftui-system-tutorial/projects/boardflow/starter/Sources/BoardFlowStarter/BoardFlowApp.swift`
- Create: `swiftui-system-tutorial/projects/boardflow/starter/Sources/BoardFlowStarter/Models/BoardDocument.swift`
- Create: `swiftui-system-tutorial/projects/boardflow/starter/Sources/BoardFlowStarter/Views/BoardHomeView.swift`
- Create: `swiftui-system-tutorial/projects/boardflow/starter/Tests/BoardFlowStarterTests/BoardDocumentTests.swift`

### Support Docs

- Create: `swiftui-system-tutorial/appendix/component-atlas.md`
- Create: `swiftui-system-tutorial/appendix/layout-playbook.md`
- Create: `swiftui-system-tutorial/appendix/state-ownership-guide.md`
- Create: `swiftui-system-tutorial/appendix/navigation-and-workbench-patterns.md`
- Create: `swiftui-system-tutorial/appendix/gesture-playbook.md`
- Create: `swiftui-system-tutorial/appendix/canvas-and-drawing-guide.md`
- Create: `swiftui-system-tutorial/appendix/animation-guide.md`
- Create: `swiftui-system-tutorial/appendix/mac-interop-guide.md`
- Create: `swiftui-system-tutorial/appendix/performance-and-identity-guide.md`
- Create: `swiftui-system-tutorial/appendix/glossary.md`
- Create: `swiftui-system-tutorial/appendix/faq.md`
- Create: `swiftui-system-tutorial/appendix/references.md`
- Create: `swiftui-system-tutorial/labs/README.md`
- Create: `swiftui-system-tutorial/labs/part1.md`
- Create: `swiftui-system-tutorial/labs/part2.md`

### Verification

- Create: `swiftui-system-tutorial/scripts/verify_layout.sh`
  - Locks root inventory and H1 headings.
- Create: `swiftui-system-tutorial/scripts/verify_appendix.sh`
  - Locks appendix/lab inventory and H1 headings.
- Create: `swiftui-system-tutorial/scripts/verify_boardflow_build.sh`
  - Builds and tests starter package.
- Create: `swiftui-system-tutorial/scripts/verify_parts.sh`
  - Locks Part 1 and Part 2 chapter inventory and H1 headings.

## Task 1: Lock the Product Skeleton and Root Reading Surface

**Files:**
- Create: `swiftui-system-tutorial/scripts/verify_layout.sh`
- Create: `swiftui-system-tutorial/README.md`
- Create: `swiftui-system-tutorial/00-orientation.md`
- Create: `swiftui-system-tutorial/01-learning-map.md`
- Create: `swiftui-system-tutorial/projects/README.md`
- Create: `swiftui-system-tutorial/projects/boardflow/README.md`
- Create: `swiftui-system-tutorial/projects/boardflow/checkpoints/README.md`
- Create: `swiftui-system-tutorial/projects/boardflow/checkpoints/part1-shell/README.md`
- Create: `swiftui-system-tutorial/projects/boardflow/checkpoints/part2-v1-workbench/README.md`

- [ ] **Step 1: Write the failing layout verifier**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

require_file() {
  local rel="$1"
  [[ -f "$ROOT_DIR/$rel" ]] || fail "missing-file: $rel"
}

require_dir() {
  local rel="$1"
  [[ -d "$ROOT_DIR/$rel" ]] || fail "missing-dir: $rel"
}

require_heading() {
  local rel="$1"
  local pattern="$2"
  grep -qE "$pattern" "$ROOT_DIR/$rel" || fail "missing-heading: $rel :: $pattern"
}

require_dir "swiftui-system-tutorial"
require_dir "swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system"
require_dir "swiftui-system-tutorial/part2-components-navigation-and-state-ownership"
require_dir "swiftui-system-tutorial/projects"
require_dir "swiftui-system-tutorial/projects/boardflow"
require_dir "swiftui-system-tutorial/projects/boardflow/checkpoints"
require_dir "swiftui-system-tutorial/projects/boardflow/starter"
require_dir "swiftui-system-tutorial/appendix"
require_dir "swiftui-system-tutorial/labs"
require_dir "swiftui-system-tutorial/scripts"

require_file "swiftui-system-tutorial/README.md"
require_file "swiftui-system-tutorial/00-orientation.md"
require_file "swiftui-system-tutorial/01-learning-map.md"
require_file "swiftui-system-tutorial/projects/README.md"
require_file "swiftui-system-tutorial/projects/boardflow/README.md"
require_file "swiftui-system-tutorial/projects/boardflow/checkpoints/README.md"
require_file "swiftui-system-tutorial/projects/boardflow/checkpoints/part1-shell/README.md"
require_file "swiftui-system-tutorial/projects/boardflow/checkpoints/part2-v1-workbench/README.md"

require_heading "swiftui-system-tutorial/README.md" '^# SwiftUI 系统教程：从零到 Mac 创作工具工程$'
require_heading "swiftui-system-tutorial/00-orientation.md" '^# 导读：如何使用这套 SwiftUI 系统教程$'
require_heading "swiftui-system-tutorial/01-learning-map.md" '^# 学习地图：BoardFlow 主线与 SwiftUI 能力图谱$'

printf 'layout-ok\n'
```

- [ ] **Step 2: Run the layout verifier and confirm the skeleton does not exist yet**

Run:

```bash
bash swiftui-system-tutorial/scripts/verify_layout.sh
```

Expected:

```text
missing-dir: swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system
```

- [ ] **Step 3: Create the root docs and checkpoint readmes with fixed headings and section structure**

```bash
mkdir -p \
  swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system \
  swiftui-system-tutorial/part2-components-navigation-and-state-ownership \
  swiftui-system-tutorial/projects/boardflow/checkpoints/part1-shell \
  swiftui-system-tutorial/projects/boardflow/checkpoints/part2-v1-workbench \
  swiftui-system-tutorial/projects/boardflow/starter \
  swiftui-system-tutorial/appendix \
  swiftui-system-tutorial/labs
```

```markdown
# SwiftUI 系统教程：从零到 Mac 创作工具工程

## 这套教程解决什么问题
- 它是独立 SwiftUI 产品，不是 swift-tutorial 的附录
- 它围绕 BoardFlow 讲普通 UI、工作台、画布、绘制、文档与互操作

## 谁适合读
- 已有基础 Swift 语法
- 想系统学 SwiftUI，而不是只背组件

## 学习路径
1. Part 1：SwiftUI 语言和基础 View 系统
2. Part 2：组件、导航与状态所有权
3. 后续 Part：工作台、画布、绘制、文档、互操作、性能

## BoardFlow 主线
- starter
- checkpoints
- final
```

```markdown
# 导读：如何使用这套 SwiftUI 系统教程

## 先修要求
- 完成 Swift 基础学习
- 能使用 Xcode 与 SwiftPM

## 如何阅读
- 每章先看机制，再看 BoardFlow 的落点
- 每个 Part 完成后执行 labs 和 starter 代码

## 和 swift-tutorial 的关系
- swift-tutorial 负责通用 Swift 主线
- 这里负责 SwiftUI 系统主线
```

```markdown
# 学习地图：BoardFlow 主线与 SwiftUI 能力图谱

## 八个 Part 的能力跃迁
- Part 1：应用壳与基础视图语言
- Part 2：组件、导航、状态所有权
- Part 3：Mac 工作台结构
- Part 4：画布空间与手势
- Part 5：绘制、动画、自定义 Layout
- Part 6：文档、持久化、撤销重做
- Part 7：AppKit 边界与工程分层
- Part 8：性能、测试、可扩展架构

## BoardFlow 检查点
- part1-shell
- part2-v1-workbench
```

```markdown
# Projects：BoardFlow 主线与检查点

## 目录说明
- `boardflow/starter`：Part 1 和 Part 2 的起始工程
- `boardflow/checkpoints`：阶段性结果说明
- `boardflow/final`：后续阶段最终项目
```

```markdown
# BoardFlow：Mac 白板创作工具主项目

## 目标
- 用一个持续生长的创作工具承载 SwiftUI 全景能力

## 当前阶段
- Part 1：最小应用壳
- Part 2：桌面应用骨架
```

```markdown
# BoardFlow Checkpoints：阶段性里程碑

## 当前已规划检查点
- `part1-shell`
- `part2-v1-workbench`
```

```markdown
# Checkpoint：part1-shell

## 目标
- 应用可启动
- 首页有欢迎区、最近白板列表、创建入口
```

```markdown
# Checkpoint：part2-v1-workbench

## 目标
- 形成 sidebar、列表、详情区、基础表单的桌面骨架
```

- [ ] **Step 4: Run the layout verifier again**

Run:

```bash
bash swiftui-system-tutorial/scripts/verify_layout.sh
```

Expected:

```text
layout-ok
```

- [ ] **Step 5: Commit the skeleton**

```bash
git add \
  swiftui-system-tutorial/scripts/verify_layout.sh \
  swiftui-system-tutorial/README.md \
  swiftui-system-tutorial/00-orientation.md \
  swiftui-system-tutorial/01-learning-map.md \
  swiftui-system-tutorial/projects/README.md \
  swiftui-system-tutorial/projects/boardflow/README.md \
  swiftui-system-tutorial/projects/boardflow/checkpoints/README.md \
  swiftui-system-tutorial/projects/boardflow/checkpoints/part1-shell/README.md \
  swiftui-system-tutorial/projects/boardflow/checkpoints/part2-v1-workbench/README.md

git commit -m "Establish the SwiftUI system tutorial product shell" -m "The new SwiftUI curriculum needs a separate product boundary before any chapter content is written. This commit locks the root docs, project checkpoints, and a layout verifier so later writing work has a fixed contract.

Constraint: Keep swift-tutorial and swiftui-system-tutorial as separate products
Rejected: Start by drafting chapters without verifier scripts | would leave the new product boundary soft
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Extend the new tutorial through explicit files and verifiers rather than silently blending it into swift-tutorial
Tested: bash swiftui-system-tutorial/scripts/verify_layout.sh
Not-tested: Content completeness beyond root headings and inventory"
```

## Task 2: Add Appendix and Lab Support Surfaces

**Files:**
- Create: `swiftui-system-tutorial/scripts/verify_appendix.sh`
- Create: `swiftui-system-tutorial/appendix/component-atlas.md`
- Create: `swiftui-system-tutorial/appendix/layout-playbook.md`
- Create: `swiftui-system-tutorial/appendix/state-ownership-guide.md`
- Create: `swiftui-system-tutorial/appendix/navigation-and-workbench-patterns.md`
- Create: `swiftui-system-tutorial/appendix/gesture-playbook.md`
- Create: `swiftui-system-tutorial/appendix/canvas-and-drawing-guide.md`
- Create: `swiftui-system-tutorial/appendix/animation-guide.md`
- Create: `swiftui-system-tutorial/appendix/mac-interop-guide.md`
- Create: `swiftui-system-tutorial/appendix/performance-and-identity-guide.md`
- Create: `swiftui-system-tutorial/appendix/glossary.md`
- Create: `swiftui-system-tutorial/appendix/faq.md`
- Create: `swiftui-system-tutorial/appendix/references.md`
- Create: `swiftui-system-tutorial/labs/README.md`
- Create: `swiftui-system-tutorial/labs/part1.md`
- Create: `swiftui-system-tutorial/labs/part2.md`

- [ ] **Step 1: Write the failing appendix/lab verifier**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

require_file() {
  local rel="$1"
  [[ -f "$ROOT_DIR/$rel" ]] || fail "missing-file: $rel"
}

require_heading() {
  local rel="$1"
  local pattern="$2"
  grep -qE "$pattern" "$ROOT_DIR/$rel" || fail "missing-heading: $rel :: $pattern"
}

files=(
  "swiftui-system-tutorial/appendix/component-atlas.md|^# 组件图谱：基础视图、输入、容器与反馈$"
  "swiftui-system-tutorial/appendix/layout-playbook.md|^# 布局手册：Stack、Split、Overlay 与空间组织$"
  "swiftui-system-tutorial/appendix/state-ownership-guide.md|^# 状态所有权指南：State、Binding、Observable 与 Environment$"
  "swiftui-system-tutorial/appendix/navigation-and-workbench-patterns.md|^# 导航与工作台模式：NavigationStack、NavigationSplitView 与多面板组织$"
  "swiftui-system-tutorial/appendix/gesture-playbook.md|^# 手势手册：点击、拖拽、缩放与冲突处理$"
  "swiftui-system-tutorial/appendix/canvas-and-drawing-guide.md|^# 绘制手册：Shape、Path 与 Canvas$"
  "swiftui-system-tutorial/appendix/animation-guide.md|^# 动画手册：过渡、交易与空间过渡$"
  "swiftui-system-tutorial/appendix/mac-interop-guide.md|^# Mac 互操作手册：AppKit 边界与桥接策略$"
  "swiftui-system-tutorial/appendix/performance-and-identity-guide.md|^# 性能与身份手册：Diffing、刷新与大画布判断$"
  "swiftui-system-tutorial/appendix/glossary.md|^# 术语表：SwiftUI 与 BoardFlow 核心概念$"
  "swiftui-system-tutorial/appendix/faq.md|^# FAQ：学习路径、工程选择与常见误区$"
  "swiftui-system-tutorial/appendix/references.md|^# 参考资料：Apple 文档与延伸阅读$"
  "swiftui-system-tutorial/labs/README.md|^# Labs：把 Part 1 和 Part 2 变成手上的代码$"
  "swiftui-system-tutorial/labs/part1.md|^# Lab 1：把 BoardFlow 首页从静态界面写出来$"
  "swiftui-system-tutorial/labs/part2.md|^# Lab 2：把 BoardFlow 变成带 Sidebar 的桌面骨架$"
)

for item in "${files[@]}"; do
  IFS='|' read -r rel pattern <<<"$item"
  require_file "$rel"
  require_heading "$rel" "$pattern"
done

printf 'appendix-ok\n'
```

- [ ] **Step 2: Run the appendix verifier and confirm the support docs are still missing**

Run:

```bash
bash swiftui-system-tutorial/scripts/verify_appendix.sh
```

Expected:

```text
missing-file: swiftui-system-tutorial/appendix/component-atlas.md
```

- [ ] **Step 3: Write the appendix shells and the first two labs**

```markdown
# 组件图谱：基础视图、输入、容器与反馈

## 当前覆盖范围
- 基础视图：Text、Image、Button、Label
- 输入组件：TextField、Toggle、Picker、Stepper
- 容器：List、Form、Section
- 反馈：alert、sheet、popover、ProgressView
```

```markdown
# 布局手册：Stack、Split、Overlay 与空间组织

## 当前覆盖范围
- VStack、HStack、ZStack
- NavigationSplitView 的桌面工作台角色
- overlay/background/safeAreaInset 的层次判断
```

```markdown
# 状态所有权指南：State、Binding、Observable 与 Environment

## 判断问题
1. 谁创建状态
2. 谁拥有状态
3. 谁允许修改状态
4. 哪些视图只是消费状态
```

```markdown
# 导航与工作台模式：NavigationStack、NavigationSplitView 与多面板组织

## 当前覆盖范围
- 常规层级导航
- Mac 多栏工作台
- sidebar / detail / inspector 的职责边界
```

```markdown
# 手势手册：点击、拖拽、缩放与冲突处理
# 绘制手册：Shape、Path 与 Canvas
# 动画手册：过渡、交易与空间过渡
# Mac 互操作手册：AppKit 边界与桥接策略
# 性能与身份手册：Diffing、刷新与大画布判断
# 术语表：SwiftUI 与 BoardFlow 核心概念
# FAQ：学习路径、工程选择与常见误区
# 参考资料：Apple 文档与延伸阅读
```

```markdown
# Labs：把 Part 1 和 Part 2 变成手上的代码

## 使用方式
- 先看章节
- 再写 starter
- 最后对照 checkpoints 自查
```

```markdown
# Lab 1：把 BoardFlow 首页从静态界面写出来

## 目标
- 做出欢迎区
- 做出最近白板列表
- 做出创建入口
```

```markdown
# Lab 2：把 BoardFlow 变成带 Sidebar 的桌面骨架

## 目标
- 加入 sidebar
- 加入白板列表和详情区
- 加入基础创建表单
```

- [ ] **Step 4: Run the appendix verifier again**

Run:

```bash
bash swiftui-system-tutorial/scripts/verify_appendix.sh
```

Expected:

```text
appendix-ok
```

- [ ] **Step 5: Commit the support docs**

```bash
git add \
  swiftui-system-tutorial/scripts/verify_appendix.sh \
  swiftui-system-tutorial/appendix \
  swiftui-system-tutorial/labs

git commit -m "Add appendix and lab surfaces for the SwiftUI tutorial" -m "The new tutorial needs support material early so chapters can refer to stable component, layout, state, and navigation guides instead of collapsing everything into one cheat sheet. The first two labs also give Part 1 and Part 2 a concrete practice lane.

Constraint: Support docs should be scoped to Part 1 and Part 2 while still reserving the advanced appendix map
Rejected: Wait to add appendices until all eight parts exist | would push the tutorial back toward scattered inline explanations
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Expand appendix files by topic ownership; do not re-merge them into a single SwiftUI cheat sheet
Tested: bash swiftui-system-tutorial/scripts/verify_appendix.sh
Not-tested: Whether later advanced parts will require appendix renames"
```

## Task 3: Create a Compileable BoardFlow Starter Package

**Files:**
- Create: `swiftui-system-tutorial/scripts/verify_boardflow_build.sh`
- Create: `swiftui-system-tutorial/projects/boardflow/starter/Package.swift`
- Create: `swiftui-system-tutorial/projects/boardflow/starter/README.md`
- Create: `swiftui-system-tutorial/projects/boardflow/starter/Sources/BoardFlowStarter/BoardFlowApp.swift`
- Create: `swiftui-system-tutorial/projects/boardflow/starter/Sources/BoardFlowStarter/Models/BoardDocument.swift`
- Create: `swiftui-system-tutorial/projects/boardflow/starter/Sources/BoardFlowStarter/Views/BoardHomeView.swift`
- Create: `swiftui-system-tutorial/projects/boardflow/starter/Tests/BoardFlowStarterTests/BoardDocumentTests.swift`

- [ ] **Step 1: Write the package manifest, smoke tests, and build verifier before the implementation exists**

```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "BoardFlowStarter",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "BoardFlowStarter", targets: ["BoardFlowStarter"])
    ],
    targets: [
        .executableTarget(name: "BoardFlowStarter"),
        .testTarget(name: "BoardFlowStarterTests", dependencies: ["BoardFlowStarter"])
    ]
)
```

```swift
import XCTest
@testable import BoardFlowStarter

final class BoardDocumentTests: XCTestCase {
    func testEmptyDocumentUsesUntitledBoardTitle() {
        XCTAssertEqual(BoardDocument.empty.title, "Untitled Board")
    }

    func testSampleBoardsContainThreeEntries() {
        XCTAssertEqual(BoardSummary.samples.count, 3)
    }
}
```

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../projects/boardflow/starter"
swift build
swift test
printf 'boardflow-build-ok\n'
```

- [ ] **Step 2: Run the tests and confirm they fail because the models do not exist yet**

Run:

```bash
cd swiftui-system-tutorial/projects/boardflow/starter
swift test
```

Expected:

```text
error: no such module 'BoardFlowStarter'
```

or a compile failure mentioning `BoardDocument` / `BoardSummary` not found.

- [ ] **Step 3: Write the minimal SwiftUI app shell and model implementation**

```swift
import Foundation

struct BoardSummary: Identifiable, Equatable {
    let id: UUID
    var title: String
    var cardCount: Int

    init(id: UUID = UUID(), title: String, cardCount: Int) {
        self.id = id
        self.title = title
        self.cardCount = cardCount
    }

    static let samples: [BoardSummary] = [
        BoardSummary(title: "Weekly Planning", cardCount: 8),
        BoardSummary(title: "Product Discovery", cardCount: 14),
        BoardSummary(title: "Research Synthesis", cardCount: 5),
    ]
}

struct BoardDocument: Equatable {
    var title: String
    var boards: [BoardSummary]

    static let empty = BoardDocument(title: "Untitled Board", boards: BoardSummary.samples)
}
```

```swift
import SwiftUI

@main
struct BoardFlowApp: App {
    var body: some Scene {
        WindowGroup {
            BoardHomeView(document: .empty)
        }
    }
}
```

```swift
import SwiftUI

struct BoardHomeView: View {
    let document: BoardDocument

    var body: some View {
        NavigationSplitView {
            List(document.boards) { board in
                Label(board.title, systemImage: "square.on.square")
            }
            .navigationTitle("Boards")
        } detail: {
            VStack(alignment: .leading, spacing: 16) {
                Text(document.title)
                    .font(.largeTitle.bold())
                Text("BoardFlow starter for Part 1 and Part 2")
                    .foregroundStyle(.secondary)
                Text("Recent boards")
                    .font(.headline)
                ForEach(document.boards) { board in
                    HStack {
                        Text(board.title)
                        Spacer()
                        Text("\(board.cardCount) cards")
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer()
            }
            .padding(24)
        }
    }
}
```

```markdown
# BoardFlow Starter：Part 1 和 Part 2 的起始工程

## 运行方式
- `swift build`
- `swift test`

## 当前覆盖
- 最小 SwiftUI App 壳
- 最近白板列表
- 基础 detail 区
```

- [ ] **Step 4: Run the build verifier and confirm the starter compiles and tests**

Run:

```bash
bash swiftui-system-tutorial/scripts/verify_boardflow_build.sh
```

Expected:

```text
boardflow-build-ok
```

- [ ] **Step 5: Commit the starter package**

```bash
git add \
  swiftui-system-tutorial/scripts/verify_boardflow_build.sh \
  swiftui-system-tutorial/projects/boardflow/starter

git commit -m "Create the initial BoardFlow starter package" -m "The tutorial needs a compileable project line from the first slice so Part 1 and Part 2 are grounded in real code, not only markdown. This starter is intentionally small: one app shell, one document model, one home view, and smoke tests.

Constraint: Keep the starter lightweight enough for Part 1 and Part 2, without prebuilding later canvas complexity
Rejected: Use an Xcode-only project without CLI verification | would remove reliable scripted validation from the tutorial repo
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Grow BoardFlow in checkpointed slices; do not preload Part 3+ concepts into the starter
Tested: bash swiftui-system-tutorial/scripts/verify_boardflow_build.sh
Not-tested: GUI runtime behavior under Xcode previews"
```

## Task 4: Author Part 1 Chapters

**Files:**
- Create: `swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/01-what-swiftui-app-code-is-actually-expressing.md`
- Create: `swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/02-view-composition-and-the-three-core-layout-stacks.md`
- Create: `swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/03-fundamental-interactive-components.md`
- Create: `swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/04-state-driven-ui-fundamentals.md`
- Create: `swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/05-build-boardflow-home-shell.md`

- [ ] **Step 1: Write Chapter 1 with the fixed tutorial section structure**

```markdown
# 第1章：SwiftUI App 到底在写什么

## 为什么这一章现在出现
- 解释为什么 BoardFlow 先从 App 壳开始
- 建立 `App`、`Scene`、`WindowGroup` 的角色分工

## 从一个较弱起点开始：把 SwiftUI 当成控件初始化脚本
- 写出“手工缓存 view / 把 body 当一次性施工”的错误心智

## 更强的理解：SwiftUI 在声明当前状态下的界面
- 解释 value-like view description

## BoardFlow 落点
- 对照 starter 里的 `BoardFlowApp`

## 双语关键词
## 常见错误
## English Recap
## Drills
## Project Handoff
```

- [ ] **Step 2: Write Chapter 2 and Chapter 3 around layout and core components**

```markdown
# 第2章：View Composition 与三大基础布局

## 为什么这一章现在出现
- 从 App 壳进入真实界面结构

## 三种基础布局的职责
- `VStack`
- `HStack`
- `ZStack`

## 按语义拆 View，而不是按截图切块
## BoardFlow 落点
## 双语关键词
## 常见错误
## English Recap
## Drills
## Project Handoff
```

```markdown
# 第3章：最基本的可交互组件

## 当前目标
- 用 `Text`、`Image`、`Button`、`Label` 做出首页入口

## 组件职责
- 展示
- 触发意图
- 辅助标签语义

## BoardFlow 落点
- 欢迎区
- 最近白板入口
- 创建按钮
```

- [ ] **Step 3: Write Chapter 4 on `@State` and state-driven redraw**

```markdown
# 第4章：状态驱动界面的第一原则

## 为什么现在讲状态
- 没有状态，按钮和输入不会形成稳定 UI 行为

## 从一个较弱起点开始：把 UI 变化写成手工 patch
- 解释为什么这会打散单一事实源

## 更强的方向：让状态变化驱动描述重算
- `@State`
- 局部状态
- 派生状态

## BoardFlow 落点
- 首页筛选开关
- 最近白板展示状态
```

- [ ] **Step 4: Write Chapter 5 as the first project integration chapter**

```markdown
# 第5章：做出 BoardFlow 的最小工作台首页

## 本章交付
- 标题区
- 最近白板列表
- 创建白板入口

## 工程结构
- `BoardFlowApp`
- `BoardDocument`
- `BoardHomeView`

## 本章怎么串起 Part 1
- App 壳
- 基础布局
- 核心组件
- 状态驱动

## 自查清单
## 常见错误
## Project Handoff
```

- [ ] **Step 5: Verify Part 1 chapter inventory and H1 headings**

Run:

```bash
test -f swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/01-what-swiftui-app-code-is-actually-expressing.md
test -f swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/02-view-composition-and-the-three-core-layout-stacks.md
test -f swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/03-fundamental-interactive-components.md
test -f swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/04-state-driven-ui-fundamentals.md
test -f swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/05-build-boardflow-home-shell.md
grep -q '^# 第1章：SwiftUI App 到底在写什么$' swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/01-what-swiftui-app-code-is-actually-expressing.md
grep -q '^# 第2章：View Composition 与三大基础布局$' swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/02-view-composition-and-the-three-core-layout-stacks.md
grep -q '^# 第3章：最基本的可交互组件$' swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/03-fundamental-interactive-components.md
grep -q '^# 第4章：状态驱动界面的第一原则$' swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/04-state-driven-ui-fundamentals.md
grep -q '^# 第5章：做出 BoardFlow 的最小工作台首页$' swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/05-build-boardflow-home-shell.md
```

Expected:

```text
no output; exit status 0
```

- [ ] **Step 6: Commit Part 1**

```bash
git add swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system

git commit -m "Author Part 1 of the SwiftUI system tutorial" -m "Part 1 establishes the new tutorial's base language: App structure, view composition, core components, and state-driven UI. This locks the first complete teaching slice around the BoardFlow home shell.

Constraint: Match the stronger chapter-writing style already used in swift-tutorial while shifting the project line to BoardFlow
Rejected: Collapse Part 1 into one long overview chapter | would weaken progression and practice checkpoints
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Keep every chapter tied to BoardFlow and a reusable SwiftUI mechanism; do not drift into API glossary writing
Tested: shell file/heading checks for all Part 1 chapter files
Not-tested: Human editorial polish of every paragraph"
```

## Task 5: Author Part 2 and Lock the First Full Deliverable

**Files:**
- Create: `swiftui-system-tutorial/scripts/verify_parts.sh`
- Create: `swiftui-system-tutorial/part2-components-navigation-and-state-ownership/06-lists-forms-and-input-contracts.md`
- Create: `swiftui-system-tutorial/part2-components-navigation-and-state-ownership/07-navigationstack-and-navigationsplitview.md`
- Create: `swiftui-system-tutorial/part2-components-navigation-and-state-ownership/08-binding-and-state-ownership.md`
- Create: `swiftui-system-tutorial/part2-components-navigation-and-state-ownership/09-observable-models-and-screen-state-coordination.md`
- Create: `swiftui-system-tutorial/part2-components-navigation-and-state-ownership/10-build-boardflow-v1-desktop-skeleton.md`
- Modify: `swiftui-system-tutorial/README.md`
- Modify: `swiftui-system-tutorial/01-learning-map.md`
- Modify: `swiftui-system-tutorial/labs/part1.md`
- Modify: `swiftui-system-tutorial/labs/part2.md`

- [ ] **Step 1: Write the full parts verifier before Part 2 exists**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

require_file() {
  local rel="$1"
  [[ -f "$ROOT_DIR/$rel" ]] || fail "missing-file: $rel"
}

require_heading() {
  local rel="$1"
  local pattern="$2"
  grep -qE "$pattern" "$ROOT_DIR/$rel" || fail "missing-heading: $rel :: $pattern"
}

chapters=(
  "swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/01-what-swiftui-app-code-is-actually-expressing.md|^# 第1章：SwiftUI App 到底在写什么$"
  "swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/02-view-composition-and-the-three-core-layout-stacks.md|^# 第2章：View Composition 与三大基础布局$"
  "swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/03-fundamental-interactive-components.md|^# 第3章：最基本的可交互组件$"
  "swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/04-state-driven-ui-fundamentals.md|^# 第4章：状态驱动界面的第一原则$"
  "swiftui-system-tutorial/part1-swiftui-language-and-basic-view-system/05-build-boardflow-home-shell.md|^# 第5章：做出 BoardFlow 的最小工作台首页$"
  "swiftui-system-tutorial/part2-components-navigation-and-state-ownership/06-lists-forms-and-input-contracts.md|^# 第6章：列表、表单与输入契约$"
  "swiftui-system-tutorial/part2-components-navigation-and-state-ownership/07-navigationstack-and-navigationsplitview.md|^# 第7章：NavigationStack 与 NavigationSplitView$"
  "swiftui-system-tutorial/part2-components-navigation-and-state-ownership/08-binding-and-state-ownership.md|^# 第8章：Binding 与状态拥有关系$"
  "swiftui-system-tutorial/part2-components-navigation-and-state-ownership/09-observable-models-and-screen-state-coordination.md|^# 第9章：Observable Model 与屏幕级状态协调$"
  "swiftui-system-tutorial/part2-components-navigation-and-state-ownership/10-build-boardflow-v1-desktop-skeleton.md|^# 第10章：做出 BoardFlow v1 桌面应用骨架$"
)

for item in "${chapters[@]}"; do
  IFS='|' read -r rel pattern <<<"$item"
  require_file "$rel"
  require_heading "$rel" "$pattern"
done

grep -q 'part1-swiftui-language-and-basic-view-system' "$ROOT_DIR/swiftui-system-tutorial/README.md" || fail "missing-readme-link: part1"
grep -q 'part2-components-navigation-and-state-ownership' "$ROOT_DIR/swiftui-system-tutorial/README.md" || fail "missing-readme-link: part2"

printf 'parts-ok\n'
```

- [ ] **Step 2: Run the parts verifier and confirm Part 2 is still missing**

Run:

```bash
bash swiftui-system-tutorial/scripts/verify_parts.sh
```

Expected:

```text
missing-file: swiftui-system-tutorial/part2-components-navigation-and-state-ownership/06-lists-forms-and-input-contracts.md
```

- [ ] **Step 3: Write Chapter 6 and Chapter 7 around containers and navigation**

```markdown
# 第6章：列表、表单与输入契约

## 为什么这一章现在出现
- BoardFlow 从首页进入桌面应用骨架

## 组件职责
- `List`：有身份的集合界面
- `Form`：输入契约
- `Section`：语义分组
- `TextField`、`Toggle`、`Picker`、`Stepper`：基础输入

## BoardFlow 落点
- 白板列表
- 新建白板表单
```

```markdown
# 第7章：NavigationStack 与 NavigationSplitView

## 为什么 Mac 教程必须讲 Split View
- 普通层级导航不够表达工作台结构

## 两种导航模型
- `NavigationStack`
- `NavigationSplitView`

## BoardFlow 落点
- sidebar
- detail
- 预留 inspector 位置
```

- [ ] **Step 4: Write Chapter 8 and Chapter 9 around state ownership**

```markdown
# 第8章：Binding 与状态拥有关系

## 什么时候需要 Binding
- 父拥有
- 子编辑

## 什么时候不能乱传 Binding
- 子视图不该持有全局真源
- 编辑缓冲和最终状态必须分开

## BoardFlow 落点
- 白板草稿标题
- 选择态与编辑区
```

```markdown
# 第9章：Observable Model 与屏幕级状态协调

## 为什么 Part 2 需要可观察模型
- 白板列表、选中项、创建草稿已经超出局部状态

## 两类可观察模型
- `@Observable`
- `ObservableObject`

## BoardFlow 落点
- screen model
- single source of truth
- derived state
```

- [ ] **Step 5: Write Chapter 10 and update the root docs/labs to reference the finished first slice**

```markdown
# 第10章：做出 BoardFlow v1 桌面应用骨架

## 本章交付
- sidebar
- 白板列表
- detail 区
- 创建白板入口

## 本章如何串起 Part 2
- 输入组件
- 容器组件
- 导航结构
- 状态所有权

## 自查清单
## 常见错误
## Project Handoff
```

```markdown
## 学习路径
1. [Part 1](part1-swiftui-language-and-basic-view-system/)
2. [Part 2](part2-components-navigation-and-state-ownership/)
```

```markdown
## Part 1 Lab
- 对照第1章到第5章完成 starter 首页

## Part 2 Lab
- 对照第6章到第10章把 starter 发展成 v1 workbench
```

- [ ] **Step 6: Run all verifiers for the first delivery slice**

Run:

```bash
bash swiftui-system-tutorial/scripts/verify_layout.sh
bash swiftui-system-tutorial/scripts/verify_appendix.sh
bash swiftui-system-tutorial/scripts/verify_boardflow_build.sh
bash swiftui-system-tutorial/scripts/verify_parts.sh
```

Expected:

```text
layout-ok
appendix-ok
boardflow-build-ok
parts-ok
```

- [ ] **Step 7: Commit Part 2 and the integrated first slice**

```bash
git add \
  swiftui-system-tutorial/scripts/verify_parts.sh \
  swiftui-system-tutorial/part2-components-navigation-and-state-ownership \
  swiftui-system-tutorial/README.md \
  swiftui-system-tutorial/01-learning-map.md \
  swiftui-system-tutorial/labs/part1.md \
  swiftui-system-tutorial/labs/part2.md

git commit -m "Finish the first BoardFlow teaching slice through Part 2" -m "This commit closes the first executable slice of the new tutorial: root product, support docs, starter package, Part 1, and Part 2 now line up around one coherent learning path. The result is enough to validate the new product direction before tackling Mac workbench depth and canvas-specific topics.

Constraint: Phase 1 must end with a coherent Part 1/2 deliverable, not an unfinished Part 3 teaser
Rejected: Start drafting Part 3 immediately in the same batch | would weaken the checkpoint and make review harder
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Treat Part 1/2 as a stable baseline; later work should build new capability layers rather than rewriting this slice casually
Tested: bash swiftui-system-tutorial/scripts/verify_layout.sh; bash swiftui-system-tutorial/scripts/verify_appendix.sh; bash swiftui-system-tutorial/scripts/verify_boardflow_build.sh; bash swiftui-system-tutorial/scripts/verify_parts.sh
Not-tested: Human editorial review of pacing and language consistency across all files"
```

## Self-Review Checklist

- Spec coverage:
  - new top-level product: Task 1
  - appendix and labs: Task 2
  - `BoardFlow` starter: Task 3
  - Part 1 chapters: Task 4
  - Part 2 chapters and first-slice integration: Task 5
- Placeholder scan:
  - no placeholder markers or deferred-test language should remain
  - every verification command must exist by the time its task runs
- Type consistency:
  - `BoardDocument`, `BoardSummary`, and `BoardHomeView` names must match between tests, models, and docs
  - chapter numbers and titles must match the headings expected by `verify_parts.sh`

## Notes for the Implementer

- Do not modify `swift-tutorial/` in this phase.
- Do not create Part 3+ chapter files yet.
- Keep all new prose aligned with the current Chinese-first tutorial tone already used in `swift-tutorial/`.
- If the starter package requires small path or access-control fixes to make `swift test` pass, keep the diff local to `projects/boardflow/starter/`.
