# 术语表

这份术语表不是为了把英文全部翻成中文，而是为了让你在阅读本教程时形成稳定双语映射。中文负责解释概念和判断；英文保留类型名、API 名、协议名、命令名与社区常用说法。真正重要的不是“背翻译”，而是知道一个词在 Swift 语境里到底承担什么语义。

## Core Swift

| English | 中文 | 含义 | 在本教程里的位置 |
| --- | --- | --- | --- |
| value semantics | 值语义 | 复制一个值时，各副本彼此独立 | Part 2 判断 `struct` 是否合适的基础 |
| reference semantics | 引用语义 | 多个名字可能共享同一实例 | 用来判断何时真的需要 `class` |
| mutability | 可变性 | 一个值是否允许被修改 | 从 `let` / `var` 开始建立第一套心智模型 |
| type inference | 类型推断 | 编译器根据上下文推断类型 | 让代码简洁，但不能替代清楚建模 |
| explicit annotation | 显式标注 | 明确写出类型 | 当推断会让意图变糊时使用 |
| `Optional` | 可选值 | 表达“可能有，也可能没有” | Part 1 处理查找与安全解析的核心 |
| `enum` | 枚举 | 一组有限且显式的状态或分支 | 命令、状态、错误建模都大量依赖它 |
| `struct` | 结构体 | 以值语义为默认的数据类型 | 教程里的任务模型默认先从这里开始 |
| `class` | 类 | 引用类型，可支持共享可变状态和继承 | 只在确有身份共享需求时使用 |
| initializer | 初始化器 | 建立一个合法值的入口 | Part 2 强调它是“建值”，不是“补洞” |
| stored property | 存储属性 | 真正存储在实例里的属性 | 表达状态本身 |
| computed property | 计算属性 | 根据已有状态计算出的属性 | 表达衍生语义，不额外存储 |
| method | 方法 | 绑定在类型上的行为 | 用来让模型拥有自己的语义动作 |
| protocol | 协议 | 对能力或约束的抽象描述 | 只在变化点明确时引入 |
| protocol extension | 协议扩展 | 给协议添加默认实现或辅助能力 | 不能把它当“默认实现垃圾桶” |
| generic | 泛型 | 让同一逻辑作用于一类类型 | Part 2 和 Part 7 都会用到 |
| associated type | 关联类型 | 协议内部保留的类型关系 | 让协议不丢掉关键类型信息 |
| existential / `any` | 存在类型 / `any` | 表示“某个满足协议的值” | 适合隐藏具体类型，不适合保留复杂关系 |
| opaque type / `some` | 不透明类型 / `some` | 返回“某个具体但不公开的类型” | 在保留静态类型信息时很有价值 |
| type erasure | 类型擦除 | 用包装类型隐藏具体泛型实现 | Part 7 里只在边界确实需要时使用 |
| `throws` | 抛错 | 通过调用链传播失败 | 适合当前调用者就要处理的失败 |
| `Result` | 结果类型 | 把成功或失败都当值继续传递 | 适合批量、异步或组合场景 |
| API surface | API 表面 | 一个类型或模块对外暴露的入口总和 | Part 7 重点学习“收紧表面” |

## Swift Package Manager

| English | 中文 | 含义 | 在本教程里的位置 |
| --- | --- | --- | --- |
| Swift Package Manager / SPM | Swift 包管理器 | Swift 官方的 package、依赖与构建工具 | Part 3 的工程主线 |
| package | 包 | 一个 Swift 工程单位，通常由 `Package.swift` 描述 | `TaskCore + TaskCLI` 的外层容器 |
| manifest | 清单文件 | 通常指 `Package.swift` | 声明 products、targets、dependencies |
| tools version | 工具链版本声明 | `// swift-tools-version:` 指定 manifest 使用的工具 API 版本 | 影响包能使用哪些描述能力 |
| product | 产物 | 对外可消费的 library 或 executable | 决定包向外暴露什么 |
| target | 目标 | 一组一起编译的源码单元 | `TaskCore`、`TaskCLI`、test target 都是 target |
| library target | 库目标 | 编译成可复用模块的 target | 承担共享核心能力 |
| executable target | 可执行目标 | 编译成命令行程序的 target | 承担 CLI 入口 |
| test target | 测试目标 | 用于组织单元测试或集成测试的 target | 锁定核心行为 |
| dependency | 依赖 | 当前 package 需要的外部 package 或 target | Part 3 强调谨慎引入依赖 |
| semantic versioning | 语义化版本 | 用版本号表达破坏性变化和兼容变化 | 管理 package 依赖时的基本规则 |
| resource | 资源文件 | 随 target 一起打包的非源码文件 | 在 SwiftUI 或测试中常见 |
| plugin | 插件 | 参与 build 或 code generation 的扩展能力 | 本教程不作为早期重点 |
| module boundary | 模块边界 | 某个模块负责什么、不负责什么 | Part 3 的核心判断之一 |

## Testing And Reliability

| English | 中文 | 含义 | 在本教程里的位置 |
| --- | --- | --- | --- |
| XCTest | XCTest 测试框架 | Apple 传统测试框架 | 本教程的测试主线 |
| test case | 测试用例类 | 一组相关测试方法的容器 | 用来组织行为测试 |
| assertion | 断言 | 检查一个预期是否成立 | 测试“证明什么”靠它表达 |
| fixture | 测试夹具 / 测试数据场景 | 测试运行前的预置状态 | 帮你把输入状态写清楚 |
| test double | 测试替身 | 替代真实依赖的假对象、stub、fake 等 | Part 3 以后越来越重要 |
| regression test | 回归测试 | 防止旧 bug 再次出现的测试 | 重构前后都应优先补 |
| performance test | 性能测试 | 用来监控性能退化的测试 | Part 4 会开始建立意识 |
| failure surface | 失败面 | 系统中失败可能出现并向外暴露的表面 | Part 2、Part 4、Part 8 持续强调 |
| contract | 契约 | 一个 API 或模块承诺的行为 | 测试本质上在保护 contract |
| smoke test | 冒烟测试 | 快速确认系统基本可运行的验证 | CLI 与 package 常用 |

## Concurrency

| English | 中文 | 含义 | 在本教程里的位置 |
| --- | --- | --- | --- |
| concurrency | 并发 | 多个任务在时间上交错推进 | Part 4 核心主题 |
| async / await | 异步 / 等待 | 表达挂起与继续执行的语言机制 | 用来收紧异步调用链 |
| `Task` | 任务 | 一段异步工作单元 | 不是任意开新线程的同义词 |
| structured concurrency | 结构化并发 | 让异步任务保持父子关系和生命周期结构 | 避免失控后台任务 |
| actor | Actor | 隔离可变状态的并发类型 | 用来约束共享可变数据 |
| isolation | 隔离 | 限制谁能在什么上下文读写状态 | 真正的安全感来自这里 |
| `Sendable` | 可安全跨并发边界传递 | 表示值能安全穿过并发上下文 | Swift 6 时代尤其重要 |
| cancellation | 取消 | 提前结束异步任务 | 可靠性设计不可缺少的一环 |
| task group | 任务组 | 一组动态子任务的组织方式 | 用于更复杂并发场景 |
| race condition | 竞态条件 | 结果依赖不稳定执行顺序的问题 | Part 4 的典型风险 |
| snapshot | 快照 | 某一时刻的稳定状态视图 | 在 CLI / UI 间统一语义很重要 |
| mutation | 变更动作 | 对系统状态的显式修改 | 教程后期会把它当 contract 看待 |

## SwiftUI And App Architecture

| English | 中文 | 含义 | 在本教程里的位置 |
| --- | --- | --- | --- |
| SwiftUI | SwiftUI | Apple 的声明式 UI 框架 | Part 5 与 Part 6 主线 |
| declarative UI | 声明式 UI | 描述结果状态，而不是命令式操作控件 | SwiftUI 的第一心智模型 |
| `View` | 视图 | 描述 UI 的值类型 | 重点是描述，不是持有业务规则 |
| `body` | 视图主体 | 由状态推导出的 UI 描述 | 每次状态变化都可能重新计算 |
| identity | 身份 | SwiftUI 判断“是不是同一个视图/数据”的依据 | 列表刷新和 diff 常见问题根源 |
| `@State` | 本地状态 | View 自己拥有的短期可变状态 | 表单输入、切换开关等常用 |
| `@Binding` | 绑定 | 让子视图读写父层拥有的状态 | 表达“能改，但不拥有” |
| `@Observable` / `ObservableObject` | 可观察模型 | 能把状态变化发布给 View 的模型对象 | Part 5、Part 6 会讨论拥有权 |
| single source of truth | 单一真源 | 某份状态只有一个权威来源 | 防止多个页面各自维护一份事实 |
| app state | 应用状态 | 跨多个 feature 共享或驱动整个应用的状态 | Part 6 重点 |
| feature state | 功能状态 | 只属于某个功能模块的局部状态 | 不应轻易升级成全局状态 |
| preview | 预览 | 在 Xcode 中快速检查 View 与状态 | 不是平行实现，而是结构检查器 |
| environment | 环境 | 从上层向下传递共享上下文的机制 | 适合少量、稳定、跨层依赖 |
| navigation | 导航 | 界面间的路径与选择状态 | 也应围绕状态和语义组织 |
| data flow | 数据流 | 数据如何进入、变化、回到 UI | Part 6 的中心主题 |
