# Modern Swift App-First Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `swift-tutorial/` with a brand-new app-first Swift tutorial product centered on the `FocusList` project line, complete with 24 chapters, project assets, labs, appendix materials, and verification scripts.

**Architecture:** Work in slices that each leave the new product more real than before. First replace the old tutorial tree with the new directory contract and baseline verifiers, then build the `FocusList` project line and root docs, then author Parts 1 through 6 in order, then fill labs and appendix materials, and finally tighten verification and cross-links.

**Tech Stack:** Markdown, Bash, Swift Package Manager, SwiftUI, Observation, SwiftData, Swift Testing, Swift Concurrency, `rg`, `bash`, `git`

---

## File Structure

### Delete

- All currently tracked files under `swift-tutorial/`
  - The old tutorial is intentionally removed in full before new content lands.

### Create

Root product surface:

- `swift-tutorial/README.md`
- `swift-tutorial/00-preface.md`
- `swift-tutorial/projects/README.md`
- `swift-tutorial/labs/README.md`
- `swift-tutorial/appendix/environment-setup.md`
- `swift-tutorial/appendix/glossary.md`
- `swift-tutorial/appendix/swiftui-cheatsheet.md`
- `swift-tutorial/appendix/swift-testing-cheatsheet.md`
- `swift-tutorial/appendix/faq.md`
- `swift-tutorial/appendix/answers.md`
- `swift-tutorial/scripts/verify_layout.sh`
- `swift-tutorial/scripts/verify_parts.sh`
- `swift-tutorial/scripts/verify_projects.sh`
- `swift-tutorial/scripts/verify_appendix.sh`
- `swift-tutorial/scripts/verify_focuslist_starter.sh`
- `swift-tutorial/scripts/verify_focuscore_focusctl.sh`

Part 1:

- `swift-tutorial/part1-app-first-foundations/01-create-your-first-cross-platform-swiftui-app.md`
- `swift-tutorial/part1-app-first-foundations/02-understand-view-state-and-rendering.md`
- `swift-tutorial/part1-app-first-foundations/03-build-lists-forms-and-navigation.md`
- `swift-tutorial/part1-app-first-foundations/04-shape-focuslist-v1.md`

Part 2:

- `swift-tutorial/part2-feature-growth-and-ui-organization/05-design-task-groups-and-tags.md`
- `swift-tutorial/part2-feature-growth-and-ui-organization/06-build-editing-flows-and-reusable-components.md`
- `swift-tutorial/part2-feature-growth-and-ui-organization/07-add-filtering-search-and-screen-organization.md`
- `swift-tutorial/part2-feature-growth-and-ui-organization/08-grow-focuslist-into-a-real-product.md`

Part 3:

- `swift-tutorial/part3-data-modeling-persistence-and-shared-core/09-model-tasks-projects-and-plans.md`
- `swift-tutorial/part3-data-modeling-persistence-and-shared-core/10-persist-state-with-swiftdata.md`
- `swift-tutorial/part3-data-modeling-persistence-and-shared-core/11-design-queries-storage-boundaries-and-failure-paths.md`
- `swift-tutorial/part3-data-modeling-persistence-and-shared-core/12-extract-focuscore-from-the-app.md`

Part 4:

- `swift-tutorial/part4-engineering-testing-and-modularization/13-organize-a-swift-package-workspace.md`
- `swift-tutorial/part4-engineering-testing-and-modularization/14-test-behavior-with-swift-testing.md`
- `swift-tutorial/part4-engineering-testing-and-modularization/15-design-feature-boundaries-and-dependencies.md`
- `swift-tutorial/part4-engineering-testing-and-modularization/16-build-focusctl-on-top-of-focuscore.md`

Part 5:

- `swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish/17-refresh-search-and-background-work.md`
- `swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish/18-handle-cancellation-errors-and-user-feedback.md`
- `swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish/19-manage-batch-operations-performance-and-observation-costs.md`
- `swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish/20-polish-ios-and-macos-experiences.md`

Part 6:

- `swift-tutorial/part6-capstone-and-shipping-readiness/21-refactor-the-feature-graph.md`
- `swift-tutorial/part6-capstone-and-shipping-readiness/22-harden-tests-previews-and-accessibility.md`
- `swift-tutorial/part6-capstone-and-shipping-readiness/23-prepare-focuslist-for-release.md`
- `swift-tutorial/part6-capstone-and-shipping-readiness/24-graduation-review-and-next-steps.md`

Project line:

- `swift-tutorial/projects/focuslist/README.md`
- `swift-tutorial/projects/focuslist/starter/README.md`
- `swift-tutorial/projects/focuslist/starter/Package.swift`
- `swift-tutorial/projects/focuslist/starter/Sources/FocusCore/FocusTask.swift`
- `swift-tutorial/projects/focuslist/starter/Sources/FocusCore/FocusProject.swift`
- `swift-tutorial/projects/focuslist/starter/Sources/FocusCore/FocusStore.swift`
- `swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/FocusListApp.swift`
- `swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Root/FocusListRootView.swift`
- `swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Features/Inbox/InboxView.swift`
- `swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Features/Projects/ProjectsView.swift`
- `swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Features/Settings/SettingsView.swift`
- `swift-tutorial/projects/focuslist/starter/Sources/focusctl/main.swift`
- `swift-tutorial/projects/focuslist/starter/Tests/FocusCoreTests/FocusCoreTests.swift`
- `swift-tutorial/projects/focuslist/checkpoints/README.md`
- `swift-tutorial/projects/focuslist/checkpoints/part1-focuslist-v1/README.md`
- `swift-tutorial/projects/focuslist/checkpoints/part2-product-shape/README.md`
- `swift-tutorial/projects/focuslist/checkpoints/part3-focuscore-split/README.md`
- `swift-tutorial/projects/focuslist/checkpoints/part4-engineering-v1/README.md`
- `swift-tutorial/projects/focuslist/checkpoints/part5-polish/README.md`
- `swift-tutorial/projects/focuslist/final/README.md`

Labs:

- `swift-tutorial/labs/part1-app-first-foundations.md`
- `swift-tutorial/labs/part2-feature-growth-and-ui-organization.md`
- `swift-tutorial/labs/part3-data-modeling-persistence-and-shared-core.md`
- `swift-tutorial/labs/part4-engineering-testing-and-modularization.md`
- `swift-tutorial/labs/part5-concurrency-reliability-and-cross-platform-polish.md`
- `swift-tutorial/labs/part6-capstone-and-shipping-readiness.md`

### Modify

- `docs/superpowers/plans/2026-04-22-modern-swift-app-first-tutorial.md`
  - This implementation plan.

### Do Not Modify

- `deepagents/`
- `langchain/`
- `langgraph/`
- `vllm/`
- `docs/superpowers/specs/2026-04-22-modern-swift-app-first-tutorial-design.md`
  - Approved design spec; implementation follows it.

### Verification Surface

- `bash swift-tutorial/scripts/verify_layout.sh`
- `bash swift-tutorial/scripts/verify_parts.sh`
- `bash swift-tutorial/scripts/verify_projects.sh`
- `bash swift-tutorial/scripts/verify_appendix.sh`
- `bash swift-tutorial/scripts/verify_focuslist_starter.sh`
- `bash swift-tutorial/scripts/verify_focuscore_focusctl.sh`
- `git diff --check -- swift-tutorial docs/superpowers`

Note: `verify_parts.sh`, `verify_projects.sh`, and `verify_appendix.sh` are full-product verifiers. Until the full tree exists, use targeted `rg`, `test -f`, and build commands for the files created in each task.

---

### Task 1: Replace the Old Tutorial Tree with the New Product Skeleton

**Files:**
- Delete: all tracked files under `swift-tutorial/`
- Create: `swift-tutorial/README.md`
- Create: `swift-tutorial/00-preface.md`
- Create: `swift-tutorial/projects/README.md`
- Create: `swift-tutorial/labs/README.md`
- Create: `swift-tutorial/appendix/environment-setup.md`
- Create: `swift-tutorial/appendix/glossary.md`
- Create: `swift-tutorial/appendix/swiftui-cheatsheet.md`
- Create: `swift-tutorial/appendix/swift-testing-cheatsheet.md`
- Create: `swift-tutorial/appendix/faq.md`
- Create: `swift-tutorial/appendix/answers.md`
- Create: `swift-tutorial/scripts/verify_layout.sh`

- [ ] **Step 1: Remove the existing tracked tutorial tree**

Run:

```bash
git rm -r swift-tutorial
mkdir -p \
  swift-tutorial/part1-app-first-foundations \
  swift-tutorial/part2-feature-growth-and-ui-organization \
  swift-tutorial/part3-data-modeling-persistence-and-shared-core \
  swift-tutorial/part4-engineering-testing-and-modularization \
  swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish \
  swift-tutorial/part6-capstone-and-shipping-readiness \
  swift-tutorial/projects/focuslist/checkpoints/part1-focuslist-v1 \
  swift-tutorial/projects/focuslist/checkpoints/part2-product-shape \
  swift-tutorial/projects/focuslist/checkpoints/part3-focuscore-split \
  swift-tutorial/projects/focuslist/checkpoints/part4-engineering-v1 \
  swift-tutorial/projects/focuslist/checkpoints/part5-polish \
  swift-tutorial/projects/focuslist/starter/Sources/FocusCore \
  swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Root \
  swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Features/Inbox \
  swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Features/Projects \
  swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Features/Settings \
  swift-tutorial/projects/focuslist/starter/Sources/focusctl \
  swift-tutorial/projects/focuslist/starter/Tests/FocusCoreTests \
  swift-tutorial/projects/focuslist/final \
  swift-tutorial/labs \
  swift-tutorial/appendix \
  swift-tutorial/scripts
```

Expected:

```text
rm 'swift-tutorial/...'
```

- [ ] **Step 2: Write the layout verifier first**

Create `swift-tutorial/scripts/verify_layout.sh` with:

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
  rg -q "$pattern" "$ROOT_DIR/$rel" || fail "missing-heading: $rel :: $pattern"
}

require_dir "swift-tutorial/part1-app-first-foundations"
require_dir "swift-tutorial/part2-feature-growth-and-ui-organization"
require_dir "swift-tutorial/part3-data-modeling-persistence-and-shared-core"
require_dir "swift-tutorial/part4-engineering-testing-and-modularization"
require_dir "swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish"
require_dir "swift-tutorial/part6-capstone-and-shipping-readiness"
require_dir "swift-tutorial/projects/focuslist"
require_dir "swift-tutorial/labs"
require_dir "swift-tutorial/appendix"
require_dir "swift-tutorial/scripts"

require_file "swift-tutorial/README.md"
require_file "swift-tutorial/00-preface.md"
require_file "swift-tutorial/projects/README.md"
require_file "swift-tutorial/labs/README.md"
require_file "swift-tutorial/appendix/environment-setup.md"
require_file "swift-tutorial/appendix/glossary.md"
require_file "swift-tutorial/appendix/swiftui-cheatsheet.md"
require_file "swift-tutorial/appendix/swift-testing-cheatsheet.md"
require_file "swift-tutorial/appendix/faq.md"
require_file "swift-tutorial/appendix/answers.md"

require_heading "swift-tutorial/README.md" '^# FocusList：从零到高阶的现代 Swift 教程$'
require_heading "swift-tutorial/00-preface.md" '^# 前言：如何使用这套 FocusList Swift 教程$'
require_heading "swift-tutorial/projects/README.md" '^# Projects：FocusList 项目主线$'
require_heading "swift-tutorial/labs/README.md" '^# Labs：分部综合实验$'

printf 'layout-ok\n'
```

- [ ] **Step 3: Run the layout verifier before files exist**

Run:

```bash
bash swift-tutorial/scripts/verify_layout.sh
```

Expected:

```text
missing-file: swift-tutorial/README.md
```

- [ ] **Step 4: Write the new root docs and appendix stubs**

Create these exact H1s and section groups:

```markdown
# FocusList：从零到高阶的现代 Swift 教程

## 教程定位
## 适合谁
## 六部分能力地图
## FocusList 项目主线
## 如何学习
## 教程特色
```

```markdown
# 前言：如何使用这套 FocusList Swift 教程

## 这套教程解决什么问题
## 默认读者画像
## 为什么采用 App-first
## 技术栈约定
## 学习建议
## 本教程不覆盖什么
```

```markdown
# Projects：FocusList 项目主线

## FocusList 是什么
## starter / checkpoints / final 如何配合
## FocusCore 与 focusctl 何时出现
```

```markdown
# Labs：分部综合实验

## labs 的作用
## 和章节内练习的区别
## 每一部做什么类型的实验
```

```markdown
# 环境准备
## 开发环境
## Swift 6 与 Xcode 要求
## 命令行工具
```

```markdown
# 术语表
## Swift 语言术语
## SwiftUI 术语
## 工程化术语
```

```markdown
# SwiftUI 速查
## View 与状态
## 容器与导航
## 数据流
```

```markdown
# Swift Testing 速查
## 基础断言
## 参数化测试
## 异步测试
```

```markdown
# 常见问题
## 为什么不是语言先行
## 为什么保留 FocusCore
## 为什么还有 focusctl
```

```markdown
# 练习与综合实验答案
## 使用方式
## Part 1-6 索引
```

- [ ] **Step 5: Run layout verification**

Run:

```bash
bash swift-tutorial/scripts/verify_layout.sh
```

Expected:

```text
layout-ok
```

- [ ] **Step 6: Commit the skeleton replacement**

Run:

```bash
git add swift-tutorial
git commit -m "feat: replace swift-tutorial root skeleton with focuslist product"
```

Expected:

```text
[<branch> <sha>] feat: replace swift-tutorial root skeleton with focuslist product
```

### Task 2: Build the FocusList Starter Package and Project-Line Readmes

**Files:**
- Create: `swift-tutorial/projects/focuslist/README.md`
- Create: `swift-tutorial/projects/focuslist/starter/README.md`
- Create: `swift-tutorial/projects/focuslist/starter/Package.swift`
- Create: `swift-tutorial/projects/focuslist/starter/Sources/FocusCore/FocusTask.swift`
- Create: `swift-tutorial/projects/focuslist/starter/Sources/FocusCore/FocusProject.swift`
- Create: `swift-tutorial/projects/focuslist/starter/Sources/FocusCore/FocusStore.swift`
- Create: `swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/FocusListApp.swift`
- Create: `swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Root/FocusListRootView.swift`
- Create: `swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Features/Inbox/InboxView.swift`
- Create: `swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Features/Projects/ProjectsView.swift`
- Create: `swift-tutorial/projects/focuslist/starter/Sources/FocusListApp/Features/Settings/SettingsView.swift`
- Create: `swift-tutorial/projects/focuslist/starter/Sources/focusctl/main.swift`
- Create: `swift-tutorial/projects/focuslist/starter/Tests/FocusCoreTests/FocusCoreTests.swift`
- Create: `swift-tutorial/projects/focuslist/checkpoints/README.md`
- Create: `swift-tutorial/projects/focuslist/checkpoints/part1-focuslist-v1/README.md`
- Create: `swift-tutorial/projects/focuslist/checkpoints/part2-product-shape/README.md`
- Create: `swift-tutorial/projects/focuslist/checkpoints/part3-focuscore-split/README.md`
- Create: `swift-tutorial/projects/focuslist/checkpoints/part4-engineering-v1/README.md`
- Create: `swift-tutorial/projects/focuslist/checkpoints/part5-polish/README.md`
- Create: `swift-tutorial/projects/focuslist/final/README.md`
- Create: `swift-tutorial/scripts/verify_projects.sh`
- Create: `swift-tutorial/scripts/verify_focuslist_starter.sh`
- Test: `swift-tutorial/projects/focuslist/starter/Tests/FocusCoreTests/FocusCoreTests.swift`

- [ ] **Step 1: Write the project and checkpoint readmes**

Each file must use these H1s:

```markdown
# FocusList：项目总览
```

```markdown
# FocusList Starter：起始工程
```

```markdown
# FocusList Checkpoints：阶段检查点
```

```markdown
# Part 1 Checkpoint：FocusList v1
```

```markdown
# Part 2 Checkpoint：产品化界面
```

```markdown
# Part 3 Checkpoint：抽出 FocusCore
```

```markdown
# Part 4 Checkpoint：工程化 v1
```

```markdown
# Part 5 Checkpoint：并发与跨平台打磨
```

```markdown
# FocusList Final：成品状态
```

Every checkpoint README must include:

- 当前阶段目标
- 该阶段新增能力
- 与前一阶段相比的结构变化

- [ ] **Step 2: Create the starter `Package.swift` and failing tests**

Write `swift-tutorial/projects/focuslist/starter/Package.swift`:

```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "FocusListStarter",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "FocusCore", targets: ["FocusCore"]),
        .executable(name: "FocusListApp", targets: ["FocusListApp"]),
        .executable(name: "focusctl", targets: ["focusctl"])
    ],
    targets: [
        .target(name: "FocusCore"),
        .executableTarget(
            name: "FocusListApp",
            dependencies: ["FocusCore"]
        ),
        .executableTarget(
            name: "focusctl",
            dependencies: ["FocusCore"]
        ),
        .testTarget(
            name: "FocusCoreTests",
            dependencies: ["FocusCore"]
        )
    ]
)
```

Write `swift-tutorial/projects/focuslist/starter/Tests/FocusCoreTests/FocusCoreTests.swift`:

```swift
import Testing
@testable import FocusCore

@Test func addTaskStoresItInInbox() {
    let store = FocusStore.sample()
    store.addTask(title: "Write first SwiftUI screen")
    #expect(store.inboxTasks.count == 1)
    #expect(store.inboxTasks[0].title == "Write first SwiftUI screen")
}

@Test func completingTaskMarksItDone() {
    let store = FocusStore.sample()
    store.addTask(title: "Review Part 1 draft")
    let task = try! #require(store.inboxTasks.first)
    store.toggleCompletion(task.id)
    #expect(store.inboxTasks[0].isDone)
}
```

- [ ] **Step 3: Run tests and confirm they fail before implementation**

Run:

```bash
cd swift-tutorial/projects/focuslist/starter
swift test
```

Expected:

```text
error: no such module 'FocusCore'
```

- [ ] **Step 4: Write the minimal `FocusCore`, app shell, and CLI**

Write `FocusTask.swift`:

```swift
import Foundation

public struct FocusTask: Identifiable, Equatable, Sendable {
    public let id: UUID
    public var title: String
    public var isDone: Bool

    public init(id: UUID = UUID(), title: String, isDone: Bool = false) {
        self.id = id
        self.title = title
        self.isDone = isDone
    }
}
```

Write `FocusProject.swift`:

```swift
import Foundation

public struct FocusProject: Identifiable, Equatable, Sendable {
    public let id: UUID
    public var name: String

    public init(id: UUID = UUID(), name: String) {
        self.id = id
        self.name = name
    }
}
```

Write `FocusStore.swift`:

```swift
import Foundation
import Observation

@Observable
public final class FocusStore {
    public private(set) var inboxTasks: [FocusTask]
    public private(set) var projects: [FocusProject]

    public init(inboxTasks: [FocusTask] = [], projects: [FocusProject] = []) {
        self.inboxTasks = inboxTasks
        self.projects = projects
    }

    public static func sample() -> FocusStore {
        FocusStore(
            inboxTasks: [
                FocusTask(title: "Sketch FocusList information architecture"),
                FocusTask(title: "Draft the Part 1 lesson map")
            ],
            projects: [
                FocusProject(name: "Tutorial Build"),
                FocusProject(name: "Product Polish")
            ]
        )
    }

    public func addTask(title: String) {
        inboxTasks.append(FocusTask(title: title))
    }

    public func toggleCompletion(_ id: UUID) {
        guard let index = inboxTasks.firstIndex(where: { $0.id == id }) else { return }
        inboxTasks[index].isDone.toggle()
    }
}
```

Write `FocusListApp.swift`:

```swift
import SwiftUI
import FocusCore

@main
struct FocusListApp: App {
    @State private var store = FocusStore.sample()

    var body: some Scene {
        WindowGroup {
            FocusListRootView(store: store)
        }
    }
}
```

Write `FocusListRootView.swift`:

```swift
import SwiftUI
import FocusCore

struct FocusListRootView: View {
    @Bindable var store: FocusStore

    var body: some View {
        NavigationSplitView {
            List {
                NavigationLink("Inbox") { InboxView(store: store) }
                NavigationLink("Projects") { ProjectsView(store: store) }
                NavigationLink("Settings") { SettingsView() }
            }
            .navigationTitle("FocusList")
        } detail: {
            InboxView(store: store)
        }
    }
}
```

Write `InboxView.swift`:

```swift
import SwiftUI
import FocusCore

struct InboxView: View {
    @Bindable var store: FocusStore

    var body: some View {
        List(store.inboxTasks) { task in
            HStack {
                Image(systemName: task.isDone ? "checkmark.circle.fill" : "circle")
                Text(task.title)
            }
        }
        .navigationTitle("Inbox")
    }
}
```

Write `ProjectsView.swift`:

```swift
import SwiftUI
import FocusCore

struct ProjectsView: View {
    @Bindable var store: FocusStore

    var body: some View {
        List(store.projects) { project in
            Text(project.name)
        }
        .navigationTitle("Projects")
    }
}
```

Write `SettingsView.swift`:

```swift
import SwiftUI

struct SettingsView: View {
    var body: some View {
        Form {
            Text("Focus mode settings land here later.")
        }
        .navigationTitle("Settings")
    }
}
```

Write `main.swift`:

```swift
import FocusCore

let store = FocusStore.sample()
for task in store.inboxTasks {
    print("- \(task.title)")
}
```

- [ ] **Step 5: Run tests and build verification**

Run:

```bash
cd swift-tutorial/projects/focuslist/starter
swift test
swift build --product focusctl
```

Expected:

```text
Build complete!
Test Suite 'All tests' passed
```

- [ ] **Step 6: Write the project verifiers**

`verify_projects.sh` must:

- require every project README listed above
- require the starter package files
- require the strings `FocusList`, `FocusCore`, and `focusctl` in `projects/focuslist/README.md`
- print `projects-ok`

`verify_focuslist_starter.sh` must run:

```bash
cd "$ROOT_DIR/swift-tutorial/projects/focuslist/starter"
swift test
swift build --product focusctl
```

and print `focuslist-starter-ok` on success.

- [ ] **Step 7: Run project verifiers**

Run:

```bash
bash swift-tutorial/scripts/verify_projects.sh
bash swift-tutorial/scripts/verify_focuslist_starter.sh
```

Expected:

```text
projects-ok
focuslist-starter-ok
```

- [ ] **Step 8: Commit the starter slice**

Run:

```bash
git add swift-tutorial
git commit -m "feat: add focuslist starter package and project line"
```

### Task 3: Author Part 1 and the Part 1 Lab

**Files:**
- Create: `swift-tutorial/part1-app-first-foundations/01-create-your-first-cross-platform-swiftui-app.md`
- Create: `swift-tutorial/part1-app-first-foundations/02-understand-view-state-and-rendering.md`
- Create: `swift-tutorial/part1-app-first-foundations/03-build-lists-forms-and-navigation.md`
- Create: `swift-tutorial/part1-app-first-foundations/04-shape-focuslist-v1.md`
- Create: `swift-tutorial/labs/part1-app-first-foundations.md`

- [ ] **Step 1: Write chapter 1 with this exact shell**

```markdown
# 第 1 章：创建第一个跨平台 SwiftUI 应用

## 当前问题
## Swift 与 SwiftUI 在这里各负责什么
## 创建 FocusList 最小应用壳
## `App`、`Scene`、`WindowGroup` 的作用
## 本章对项目造成了什么变化
## 常见误区
## 本章小结
```

The prose must explicitly connect:

- why the tutorial starts with an app
- why the reader does not need a full language detour first
- how the starter package relates to the app shell

- [ ] **Step 2: Write chapters 2-4 with exact titles and section groups**

Use these H1s:

```markdown
# 第 2 章：理解 View、状态与重新渲染
# 第 3 章：搭建列表、表单与基础导航
# 第 4 章：做出第一个可用的 FocusList
```

Each file must include:

- `## 当前问题`
- `## 核心机制`
- `## 在 FocusList 里的落点`
- `## 常见误区`
- `## 本章小结`

Chapter-specific coverage:

- Chapter 2: `View`, `body`, `@State`, `@Bindable`, rerender mental model
- Chapter 3: `List`, `Form`, `TextField`, `NavigationSplitView`
- Chapter 4: wire the starter into a coherent v1 screen flow

- [ ] **Step 3: Write the Part 1 lab**

`swift-tutorial/labs/part1-app-first-foundations.md` must start with:

```markdown
# Part 1 Lab：完成你的第一个 FocusList 冲刺
```

Include:

- a broken-state debugging exercise
- a small screen-extension exercise
- a reflection checklist tying App-first learning back to the part goals

- [ ] **Step 4: Run targeted content verification**

Run:

```bash
rg -n '^# 第 ' swift-tutorial/part1-app-first-foundations
rg -n '^# Part 1 Lab：完成你的第一个 FocusList 冲刺$' swift-tutorial/labs/part1-app-first-foundations.md
```

Expected:

```text
swift-tutorial/part1-app-first-foundations/01-create-your-first-cross-platform-swiftui-app.md:1:# 第 1 章：创建第一个跨平台 SwiftUI 应用
...
swift-tutorial/labs/part1-app-first-foundations.md:1:# Part 1 Lab：完成你的第一个 FocusList 冲刺
```

- [ ] **Step 5: Commit Part 1**

Run:

```bash
git add swift-tutorial
git commit -m "feat: author part 1 of the focuslist tutorial"
```

### Task 4: Author Part 2 and Product-Shaped UI Growth

**Files:**
- Create: `swift-tutorial/part2-feature-growth-and-ui-organization/05-design-task-groups-and-tags.md`
- Create: `swift-tutorial/part2-feature-growth-and-ui-organization/06-build-editing-flows-and-reusable-components.md`
- Create: `swift-tutorial/part2-feature-growth-and-ui-organization/07-add-filtering-search-and-screen-organization.md`
- Create: `swift-tutorial/part2-feature-growth-and-ui-organization/08-grow-focuslist-into-a-real-product.md`
- Create: `swift-tutorial/labs/part2-feature-growth-and-ui-organization.md`

- [ ] **Step 1: Write the four Part 2 chapters with exact titles**

```markdown
# 第 5 章：设计任务分组与标签
# 第 6 章：构建编辑流与可复用组件
# 第 7 章：加入筛选、搜索与界面组织
# 第 8 章：把 FocusList 推进成真正的产品界面
```

Every chapter must include:

- a concrete feature pressure
- the SwiftUI/UI-organization mechanism that answers it
- the app change it causes
- a section called `## 常见误区`

- [ ] **Step 2: Extend the checkpoint docs**

Write `part2-product-shape/README.md` so it names:

- sidebar organization
- tags and groups
- editing sheets/forms
- search/filter entry points

- [ ] **Step 3: Write the Part 2 lab**

`swift-tutorial/labs/part2-feature-growth-and-ui-organization.md` must include:

- one reusable-component extraction exercise
- one search/filter debugging exercise
- one UI coherence review checklist

- [ ] **Step 4: Verify Part 2 headings**

Run:

```bash
rg -n '^# 第 [5-8] 章：' swift-tutorial/part2-feature-growth-and-ui-organization
```

Expected:

```text
swift-tutorial/part2-feature-growth-and-ui-organization/05-design-task-groups-and-tags.md:1:# 第 5 章：设计任务分组与标签
...
```

- [ ] **Step 5: Commit Part 2**

Run:

```bash
git add swift-tutorial
git commit -m "feat: author part 2 of the focuslist tutorial"
```

### Task 5: Author Part 3 and Introduce FocusCore

**Files:**
- Create: `swift-tutorial/part3-data-modeling-persistence-and-shared-core/09-model-tasks-projects-and-plans.md`
- Create: `swift-tutorial/part3-data-modeling-persistence-and-shared-core/10-persist-state-with-swiftdata.md`
- Create: `swift-tutorial/part3-data-modeling-persistence-and-shared-core/11-design-queries-storage-boundaries-and-failure-paths.md`
- Create: `swift-tutorial/part3-data-modeling-persistence-and-shared-core/12-extract-focuscore-from-the-app.md`
- Create: `swift-tutorial/labs/part3-data-modeling-persistence-and-shared-core.md`
- Modify: `swift-tutorial/projects/focuslist/checkpoints/part3-focuscore-split/README.md`

- [ ] **Step 1: Write Part 3 chapter headings**

```markdown
# 第 9 章：建模任务、项目与计划
# 第 10 章：使用 SwiftData 持久化状态
# 第 11 章：设计查询、存储边界与失败路径
# 第 12 章：从应用里抽出 FocusCore
```

- [ ] **Step 2: Ensure the content explicitly covers**

- why `FocusCore` appears only now
- which logic stays in UI
- which logic moves into shared domain code
- how SwiftData changes error ownership and state recovery

Use these section headings in each file:

```markdown
## 当前问题
## 核心机制
## 在 FocusList / FocusCore 里的落点
## 常见误区
## 本章小结
```

- [ ] **Step 3: Write the Part 3 lab and checkpoint**

The lab must contain:

- one model-normalization exercise
- one persistence-failure reasoning exercise
- one boundary review exercise

The checkpoint README must summarize:

- app target responsibilities
- `FocusCore` responsibilities
- storage boundary responsibilities

- [ ] **Step 4: Verify Part 3 headings**

Run:

```bash
rg -n '^# 第 (9|10|11|12) 章：' swift-tutorial/part3-data-modeling-persistence-and-shared-core
```

- [ ] **Step 5: Commit Part 3**

Run:

```bash
git add swift-tutorial
git commit -m "feat: author part 3 and focuscore extraction guidance"
```

### Task 6: Author Part 4 and Define `focusctl`

**Files:**
- Create: `swift-tutorial/part4-engineering-testing-and-modularization/13-organize-a-swift-package-workspace.md`
- Create: `swift-tutorial/part4-engineering-testing-and-modularization/14-test-behavior-with-swift-testing.md`
- Create: `swift-tutorial/part4-engineering-testing-and-modularization/15-design-feature-boundaries-and-dependencies.md`
- Create: `swift-tutorial/part4-engineering-testing-and-modularization/16-build-focusctl-on-top-of-focuscore.md`
- Create: `swift-tutorial/labs/part4-engineering-testing-and-modularization.md`
- Create: `swift-tutorial/scripts/verify_focuscore_focusctl.sh`

- [ ] **Step 1: Write Part 4 chapter headings**

```markdown
# 第 13 章：组织 Swift Package 工作区
# 第 14 章：用 Swift Testing 锁定行为
# 第 15 章：设计功能边界与依赖关系
# 第 16 章：在 FocusCore 之上构建 focusctl
```

- [ ] **Step 2: Extend the starter package tests if needed**

Add at least one test that proves `focusctl`-relevant behavior already lives in `FocusCore`, for example:

```swift
@Test func togglingUnknownTaskDoesNothing() {
    let store = FocusStore.sample()
    let before = store.inboxTasks
    store.toggleCompletion(UUID())
    #expect(store.inboxTasks == before)
}
```

- [ ] **Step 3: Write the `focusctl` verifier**

`swift-tutorial/scripts/verify_focuscore_focusctl.sh` must run:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR/swift-tutorial/projects/focuslist/starter"

swift test
swift run focusctl

printf 'focuscore-focusctl-ok\n'
```

- [ ] **Step 4: Write the Part 4 lab**

Include:

- one test-first exercise
- one module-boundary review exercise
- one CLI output extension exercise

- [ ] **Step 5: Run the verifier**

Run:

```bash
bash swift-tutorial/scripts/verify_focuscore_focusctl.sh
```

Expected:

```text
focuscore-focusctl-ok
```

- [ ] **Step 6: Commit Part 4**

Run:

```bash
git add swift-tutorial
git commit -m "feat: author part 4 and focusctl engineering slice"
```

### Task 7: Author Part 5 and Cross-Platform Reliability Content

**Files:**
- Create: `swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish/17-refresh-search-and-background-work.md`
- Create: `swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish/18-handle-cancellation-errors-and-user-feedback.md`
- Create: `swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish/19-manage-batch-operations-performance-and-observation-costs.md`
- Create: `swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish/20-polish-ios-and-macos-experiences.md`
- Create: `swift-tutorial/labs/part5-concurrency-reliability-and-cross-platform-polish.md`
- Modify: `swift-tutorial/projects/focuslist/checkpoints/part5-polish/README.md`

- [ ] **Step 1: Write Part 5 chapter headings**

```markdown
# 第 17 章：刷新、搜索与后台工作
# 第 18 章：处理取消、错误与用户反馈
# 第 19 章：管理批量操作、性能与 Observation 成本
# 第 20 章：打磨 iOS 与 macOS 体验
```

- [ ] **Step 2: Ensure these content obligations are met**

- explain async ownership instead of only API usage
- distinguish user-cancelled work from real failures
- cover batch actions and UI feedback loops
- call out concrete `iOS` vs `macOS` product differences

- [ ] **Step 3: Write the Part 5 lab and checkpoint**

The lab must include:

- one async failure-path exercise
- one batch-action UX exercise
- one platform-polish comparison checklist

The checkpoint README must name:

- concurrency additions
- reliability improvements
- cross-platform polish points

- [ ] **Step 4: Verify Part 5 headings**

Run:

```bash
rg -n '^# 第 (17|18|19|20) 章：' swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish
```

- [ ] **Step 5: Commit Part 5**

Run:

```bash
git add swift-tutorial
git commit -m "feat: author part 5 of the focuslist tutorial"
```

### Task 8: Author Part 6, Final Guidance, and Graduation Surface

**Files:**
- Create: `swift-tutorial/part6-capstone-and-shipping-readiness/21-refactor-the-feature-graph.md`
- Create: `swift-tutorial/part6-capstone-and-shipping-readiness/22-harden-tests-previews-and-accessibility.md`
- Create: `swift-tutorial/part6-capstone-and-shipping-readiness/23-prepare-focuslist-for-release.md`
- Create: `swift-tutorial/part6-capstone-and-shipping-readiness/24-graduation-review-and-next-steps.md`
- Create: `swift-tutorial/labs/part6-capstone-and-shipping-readiness.md`
- Modify: `swift-tutorial/projects/focuslist/final/README.md`

- [ ] **Step 1: Write Part 6 chapter headings**

```markdown
# 第 21 章：重构功能图谱
# 第 22 章：加固测试、预览与无障碍
# 第 23 章：为发布准备 FocusList
# 第 24 章：毕业复盘与下一步路线
```

- [ ] **Step 2: Ensure the capstone content covers**

- architecture review and deliberate simplification
- test and preview hardening
- accessibility as engineering work
- release-readiness checklist
- realistic next-step learning roadmap

- [ ] **Step 3: Write the Part 6 lab and final README**

The lab must include:

- one architecture cleanup exercise
- one release-readiness audit
- one retrospective prompt on data flow, testing, and product judgment

The final README must summarize:

- the finished product shape
- what the reader should now understand
- how `FocusList`, `FocusCore`, and `focusctl` fit together

- [ ] **Step 4: Verify Part 6 headings**

Run:

```bash
rg -n '^# 第 (21|22|23|24) 章：' swift-tutorial/part6-capstone-and-shipping-readiness
```

- [ ] **Step 5: Commit Part 6**

Run:

```bash
git add swift-tutorial
git commit -m "feat: author part 6 and graduation surface"
```

### Task 9: Finish the Appendix and Full Product Verifiers

**Files:**
- Modify: `swift-tutorial/scripts/verify_parts.sh`
- Modify: `swift-tutorial/scripts/verify_appendix.sh`
- Modify: all appendix files
- Modify: `swift-tutorial/labs/README.md`

- [ ] **Step 1: Write the parts verifier**

`verify_parts.sh` must:

- require all 24 chapter files
- require every chapter to start with `# 第 `
- require the root README to mention all six parts
- print `parts-ok`

Use this exact shell:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

for file in \
  swift-tutorial/part1-app-first-foundations/*.md \
  swift-tutorial/part2-feature-growth-and-ui-organization/*.md \
  swift-tutorial/part3-data-modeling-persistence-and-shared-core/*.md \
  swift-tutorial/part4-engineering-testing-and-modularization/*.md \
  swift-tutorial/part5-concurrency-reliability-and-cross-platform-polish/*.md \
  swift-tutorial/part6-capstone-and-shipping-readiness/*.md
do
  [[ -f "$ROOT_DIR/$file" ]] || fail "missing-file: $file"
  head -n 1 "$ROOT_DIR/$file" | rg -q '^# 第 ' || fail "bad-heading: $file"
done

rg -q 'Part 1' "$ROOT_DIR/swift-tutorial/README.md" || fail "missing-part-map"
rg -q 'Part 6' "$ROOT_DIR/swift-tutorial/README.md" || fail "missing-part-map"

printf 'parts-ok\n'
```

- [ ] **Step 2: Write the appendix verifier**

`verify_appendix.sh` must require:

- all appendix files
- all six part labs
- these headings:
  - `# 环境准备`
  - `# 术语表`
  - `# SwiftUI 速查`
  - `# Swift Testing 速查`
  - `# 常见问题`
  - `# 练习与综合实验答案`

and print `appendix-ok`.

- [ ] **Step 3: Fill appendix content**

Each appendix file must move beyond a stub and include at least:

- one comparison table or checklist
- one task-oriented subsection
- cross-links back to the relevant part

- [ ] **Step 4: Run appendix and parts verification**

Run:

```bash
bash swift-tutorial/scripts/verify_parts.sh
bash swift-tutorial/scripts/verify_appendix.sh
```

Expected:

```text
parts-ok
appendix-ok
```

- [ ] **Step 5: Commit appendix and verifiers**

Run:

```bash
git add swift-tutorial
git commit -m "feat: finish appendix and full product verification"
```

### Task 10: Run the Final Verification Sweep

**Files:**
- Test: `swift-tutorial/scripts/verify_layout.sh`
- Test: `swift-tutorial/scripts/verify_parts.sh`
- Test: `swift-tutorial/scripts/verify_projects.sh`
- Test: `swift-tutorial/scripts/verify_appendix.sh`
- Test: `swift-tutorial/scripts/verify_focuslist_starter.sh`
- Test: `swift-tutorial/scripts/verify_focuscore_focusctl.sh`

- [ ] **Step 1: Run the full verifier set**

Run:

```bash
bash swift-tutorial/scripts/verify_layout.sh
bash swift-tutorial/scripts/verify_parts.sh
bash swift-tutorial/scripts/verify_projects.sh
bash swift-tutorial/scripts/verify_appendix.sh
bash swift-tutorial/scripts/verify_focuslist_starter.sh
bash swift-tutorial/scripts/verify_focuscore_focusctl.sh
```

Expected:

```text
layout-ok
parts-ok
projects-ok
appendix-ok
focuslist-starter-ok
focuscore-focusctl-ok
```

- [ ] **Step 2: Run diff hygiene checks**

Run:

```bash
git diff --check -- swift-tutorial docs/superpowers
git status --short
```

Expected:

```text
[no output from git diff --check]
 M swift-tutorial/...
```

- [ ] **Step 3: Commit the verified product**

Run:

```bash
git add swift-tutorial docs/superpowers/plans/2026-04-22-modern-swift-app-first-tutorial.md
git commit -m "feat: replace swift-tutorial with the focuslist course"
```

---

## Self-Review

### Spec Coverage

- Product replacement with no continuity: Task 1 deletes the old tree and creates the new root contract.
- New `FocusList` project identity: Tasks 2-8 build the project line, parts, labs, and final guidance around `FocusList`.
- Six-part architecture: Tasks 3-8 map one task per part group.
- `FocusCore` and `focusctl` introduced later: Tasks 5 and 6 explicitly add them after the app-first surface exists.
- Pure Chinese prose with English code/API names: enforced in the chapter and root-doc content requirements throughout Tasks 1 and 3-9.
- Verification-first product quality: Tasks 1, 2, 6, 9, and 10 create and run shell verifiers and package tests.

### Placeholder Scan

- No `TODO`, `TBD`, or deferred placeholders remain.
- All chapter files, appendix files, project files, and scripts are named explicitly.
- Commands and expected outputs are listed for each verification step.

### Type Consistency

- Shared domain types use `FocusTask`, `FocusProject`, and `FocusStore` throughout.
- The CLI target is consistently named `focusctl`.
- The shared library is consistently named `FocusCore`.

