# 从零到高阶的TypeScript教程（AI应用实战版）

## 项目简介

本教程旨在为学习者提供一套系统、完整的TypeScript学习资源，从最基础的语法出发，循序渐进地覆盖类型系统、面向对象编程、高级特性、工程化实践等核心主题，最终深入到AI应用开发的实战场景。

**本教程的独特之处**：每章都包含「AI应用实战」部分，展示该章TypeScript知识在大语言模型（LLM）应用开发中的实际应用，包括API调用、SSE流式通信、Agent工具调用等现代AI开发场景。

---

## 目标受众

- JavaScript开发者，希望系统学习TypeScript
- 前端/全栈工程师，希望构建AI驱动的应用
- 对LLM应用开发感兴趣的开发者
- 希望提升代码质量和类型安全的团队
- 准备开发ChatGPT类应用的创业者和独立开发者

---

## 章节导航目录

### 开始之前

- [前言：如何使用本教程](./00-preface.md)

### 第一部分：TypeScript基础

| 章节 | 标题 | 主要内容 | AI应用实战 |
|------|------|----------|------------|
| 第1章 | [环境搭建与基本语法](./part1-basics/01-environment-syntax.md) | 安装配置、变量声明、基本类型 | 类型安全的配置管理 |
| 第2章 | [数据类型详解](./part1-basics/02-data-types.md) | 原始类型、数组、元组、枚举 | API响应类型定义 |
| 第3章 | [控制流与函数](./part1-basics/03-control-flow-functions.md) | 条件、循环、函数、箭头函数 | 重试逻辑与错误处理 |

### 第二部分：类型系统进阶

| 章节 | 标题 | 主要内容 | AI应用实战 |
|------|------|----------|------------|
| 第4章 | [联合类型与交叉类型](./part2-types/04-union-intersection.md) | Union、Intersection、字面量类型 | 多模型API统一接口 |
| 第5章 | [类型守卫与类型收窄](./part2-types/05-type-guards.md) | typeof、instanceof、自定义守卫 | 流式响应类型判断 |
| 第6章 | [泛型编程](./part2-types/06-generics.md) | 泛型函数、泛型类、泛型约束 | 通用API客户端封装 |

### 第三部分：面向对象编程

| 章节 | 标题 | 主要内容 | AI应用实战 |
|------|------|----------|------------|
| 第7章 | [类与接口](./part3-oop/07-classes-interfaces.md) | class、interface、继承、实现 | LLM客户端抽象设计 |
| 第8章 | [抽象类与装饰器](./part3-oop/08-abstract-decorators.md) | abstract、decorators、元编程 | 请求日志装饰器 |
| 第9章 | [模块系统](./part3-oop/09-modules.md) | ESM、CommonJS、动态导入 | SDK模块化组织 |

### 第四部分：高级类型特性

| 章节 | 标题 | 主要内容 | AI应用实战 |
|------|------|----------|------------|
| 第10章 | [映射类型与条件类型](./part4-advanced/10-mapped-conditional.md) | Mapped Types、Conditional Types、infer | API参数类型推导 |
| 第11章 | [实用工具类型](./part4-advanced/11-utility-types.md) | Partial、Required、Pick、Omit等 | 可选配置与必选参数 |
| 第12章 | [声明文件与类型定义](./part4-advanced/12-declaration-files.md) | .d.ts、@types、类型声明 | 第三方AI SDK类型补充 |

### 第五部分：工程化实践

| 章节 | 标题 | 主要内容 | AI应用实战 |
|------|------|----------|------------|
| 第13章 | [tsconfig配置详解](./part5-tooling/13-tsconfig.md) | 编译选项、路径映射、项目引用 | 多环境配置管理 |
| 第14章 | [构建工具集成](./part5-tooling/14-build-tools.md) | Vite、esbuild、tsx、打包优化 | AI应用构建流程 |
| 第15章 | [测试与调试](./part5-tooling/15-testing-debugging.md) | Vitest、类型测试、调试技巧 | API Mock与集成测试 |

### 第六部分：异步编程与网络通信

| 章节 | 标题 | 主要内容 | AI应用实战 |
|------|------|----------|------------|
| 第16章 | [Promise与async/await](./part6-async/16-promises-async.md) | 异步模式、错误处理、并发控制 | 并发API请求管理 |
| 第17章 | [HTTP客户端与API调用](./part6-async/17-http-clients.md) | fetch、axios、请求拦截、响应处理 | OpenAI API封装 |
| 第18章 | [Server-Sent Events](./part6-async/18-sse-streaming.md) | SSE协议、EventSource、流式处理 | LLM流式响应实现 |

### 第七部分：AI应用开发实战

| 章节 | 标题 | 主要内容 | AI应用实战 |
|------|------|----------|------------|
| 第19章 | [LLM API集成](./part7-ai-apps/19-llm-api-integration.md) | OpenAI SDK、Anthropic SDK、错误处理 | 多模型统一调用 |
| 第20章 | [流式对话实现](./part7-ai-apps/20-streaming-chat.md) | 流式解析、打字机效果、中断处理 | 实时聊天界面 |
| 第21章 | [Function Calling与Agent](./part7-ai-apps/21-function-calling.md) | 工具定义、调用解析、多轮对话 | AI Agent实现 |

### 第八部分：全栈AI应用

| 章节 | 标题 | 主要内容 | AI应用实战 |
|------|------|----------|------------|
| 第22章 | [Node.js后端开发](./part8-fullstack/22-nodejs-backend.md) | Express/Fastify、中间件、API设计 | AI网关服务 |
| 第23章 | [React前端集成](./part8-fullstack/23-react-frontend.md) | React+TypeScript、状态管理、Hooks | 聊天界面组件 |
| 第24章 | [完整项目实战](./part8-fullstack/24-complete-project.md) | 架构设计、部署、监控 | AI聊天应用全流程 |

### 附录

| 附录 | 标题 | 内容说明 |
|------|------|----------|
| 附录A | [TypeScript速查表](./appendix/typescript-cheatsheet.md) | 语法、类型、工具类型速查 |
| 附录B | [AI API参考](./appendix/ai-api-reference.md) | OpenAI、Anthropic常用API |
| 附录C | [练习答案汇总](./appendix/answers.md) | 各章练习题答案索引 |

---

## 学习路径建议

### 路径一：TypeScript入门（约 3-4 周）

适合有JavaScript基础、首次学习TypeScript的开发者：

1. 系统学习第1-3章（TypeScript基础）
2. 学习第4-6章（类型系统）
3. 学习第7-9章（面向对象）
4. 完成每章的基础练习题

### 路径二：进阶提升（约 2-3 周）

适合已有TypeScript基础、希望深入的开发者：

1. 快速复习第1-6章
2. 重点学习第10-12章（高级类型）
3. 深入学习第13-15章（工程化）
4. 完成中级和提高题

### 路径三：AI应用开发（约 3-4 周）

适合希望开发AI应用的开发者：

1. 快速浏览第1-9章，建立基础
2. 学习第16-18章（异步与网络）
3. 重点学习第19-21章（AI应用开发）
4. 实战第22-24章（全栈项目）

---

## 前置要求

学习本教程需要以下基础：

- **必需**：JavaScript基础知识（变量、函数、对象、数组）
- **必需**：基本的命令行操作能力
- **推荐**：了解Node.js和npm的基本使用
- **推荐**：HTML/CSS基础（学习前端章节时）

---

## 环境配置

本教程的代码示例使用以下环境：

```bash
# 推荐环境
Node.js >= 18.0
TypeScript >= 5.0
pnpm >= 8.0 (或 npm/yarn)

# 编辑器
VS Code + TypeScript相关插件
```

快速开始：
```bash
# 安装TypeScript
npm install -g typescript

# 验证安装
tsc --version

# 创建项目
mkdir my-ts-project && cd my-ts-project
npm init -y
npm install typescript @types/node -D
npx tsc --init
```

---

## 教程特色

- **24章完整内容**：从环境搭建到全栈AI应用，系统覆盖TypeScript
- **AI应用导向**：每章配有LLM应用开发的实战代码
- **120道练习题**：每章5道精选习题，含详细解答
- **现代工具链**：使用最新的构建工具和最佳实践
- **中文编写**：专为中文学习者设计，术语准确，表达清晰

---

## 许可证

本项目采用 MIT 许可证开源。

---

*如有建议或发现错误，欢迎提交 Issue 或 Pull Request。*
