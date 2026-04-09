# 第1章：环境搭建与基本语法

> **系列定位**：本教程面向希望将 TypeScript 应用于 AI 工程实践的开发者，从零开始构建类型安全的 AI 应用。

---

## 学习目标

完成本章学习后，你将能够：

1. 理解 TypeScript 相对于 JavaScript 的核心优势，以及它在 AI 应用开发中的价值
2. 在本地环境中安装并配置 Node.js、包管理器（npm/pnpm）和 VS Code
3. 编写、编译并运行第一个 TypeScript 程序
4. 正确区分 `let`、`const`、`var` 的作用域与使用场景
5. 使用基本类型注解（`string`、`number`、`boolean`）提升代码可读性与安全性

---

## 1.1 TypeScript 简介与发展历史

### 1.1.1 什么是 TypeScript？

TypeScript 是由微软在 2012 年发布的开源编程语言，是 JavaScript 的**超集**——所有合法的 JavaScript 代码都是合法的 TypeScript 代码。TypeScript 在此基础上添加了：

- **静态类型系统**：在编译期捕获类型错误，而不是等到运行时
- **现代语言特性**：装饰器、枚举、接口、泛型等
- **工具链支持**：更智能的自动补全、重构和代码导航

```
JavaScript (运行时错误) → TypeScript (编译期错误)
"快速失败，早期发现" 是工程质量的核心原则
```

### 1.1.2 发展历史时间线

| 年份 | 里程碑 |
|------|--------|
| 2012 | 微软发布 TypeScript 0.8，Anders Hejlsberg（C# 之父）主导设计 |
| 2014 | TypeScript 1.0 正式发布，Angular 团队开始采用 |
| 2016 | TypeScript 2.0 引入非空类型、标记联合等重要特性 |
| 2018 | TypeScript 3.0 发布，项目引用、元组改进 |
| 2020 | TypeScript 4.0，可变元组、标签模板字符串类型 |
| 2022 | TypeScript 4.9 引入 `satisfies` 运算符 |
| 2023 | TypeScript 5.0，支持装饰器标准、常量类型参数 |
| 2024 | TypeScript 5.x 持续演进，性能大幅提升 |

### 1.1.3 TypeScript 在 AI 应用开发中的优势

现代 AI 应用（调用 LLM API、处理 Embedding、构建 Agent 流程）面临以下挑战：

**挑战 1：复杂的数据结构**

AI API 返回的响应结构复杂且嵌套深，没有类型约束极易出错：

```typescript
// 没有类型注解 —— 开发者不知道 response 里有什么
const response = await openai.chat.completions.create(params);
console.log(response.choices[0].message.content); // 这行会报错吗？不确定

// 有类型注解 —— IDE 会自动提示所有可用字段
const response: ChatCompletion = await openai.chat.completions.create(params);
//  IDE 提示：response.choices[0].message.content -> string | null
```

**挑战 2：配置管理**

AI 应用需要管理模型名称、温度参数、Token 限制等配置，类型系统防止传错参数类型。

**挑战 3：团队协作**

类型注解即文档，新团队成员可以直接从类型定义理解数据契约，无需阅读大量注释。

---

## 1.2 开发环境搭建

### 1.2.1 安装 Node.js

TypeScript 代码最终编译为 JavaScript 并在 Node.js 环境中运行。

**方法一：使用 nvm（推荐）**

nvm（Node Version Manager）允许在同一机器上安装和切换多个 Node.js 版本。

```bash
# macOS / Linux 安装 nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# 重启终端后，安装 Node.js LTS 版本
nvm install --lts
nvm use --lts

# 验证安装
node --version   # 应输出 v20.x.x 或更高
npm --version    # 应输出 10.x.x 或更高
```

**方法二：直接下载安装包**

访问 [https://nodejs.org](https://nodejs.org) 下载 LTS 版本安装包，按提示安装。

**Windows 用户注意**：建议使用 [nvm-windows](https://github.com/coreybutler/nvm-windows) 或 WSL2（Windows Subsystem for Linux）。

```bash
# Windows PowerShell 验证
node --version
npm --version
```

### 1.2.2 配置包管理器

**npm**（随 Node.js 自动安装）是默认选项。**pnpm** 是性能更好的替代方案，推荐用于 AI 项目（依赖包通常体积较大）。

```bash
# 安装 pnpm
npm install -g pnpm

# 验证
pnpm --version  # 应输出 8.x.x 或更高

# 常用命令对比
# npm install          →  pnpm install
# npm install lodash   →  pnpm add lodash
# npm run dev          →  pnpm dev
```

pnpm 的优势：
- 硬链接共享依赖，节省 50-70% 磁盘空间
- 安装速度比 npm 快 2-3 倍
- 严格的依赖隔离，防止幽灵依赖问题

### 1.2.3 安装 TypeScript 编译器

```bash
# 全局安装 TypeScript（方便命令行使用）
npm install -g typescript

# 验证
tsc --version  # 应输出 Version 5.x.x

# 也可以在项目中安装（推荐用于生产项目）
pnpm add -D typescript
```

### 1.2.4 配置 VS Code

VS Code 是 TypeScript 开发的首选编辑器（本身也由 TypeScript 编写）。

**安装 VS Code**：访问 [https://code.visualstudio.com](https://code.visualstudio.com) 下载。

**推荐插件**：

| 插件名称 | 功能 |
|----------|------|
| **TypeScript Vue Plugin** (Volar) | Vue + TypeScript 支持 |
| **ESLint** | 代码风格检查 |
| **Prettier** | 代码格式化 |
| **Error Lens** | 内联显示错误信息 |
| **Thunder Client** | 内置 HTTP 客户端（测试 AI API） |

**推荐 VS Code 设置**（`.vscode/settings.json`）：

```json
{
  "typescript.preferences.quoteStyle": "double",
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "typescript.updateImportsOnFileMove.enabled": "always",
  "editor.inlayHints.enabled": "on"
}
```

开启 `inlayHints` 后，VS Code 会在变量旁边显示推断出的类型，对学习 TypeScript 类型推断非常有帮助。

### 1.2.5 创建第一个 TypeScript 项目

```bash
# 创建项目目录
mkdir my-first-ts-project
cd my-first-ts-project

# 初始化 Node.js 项目
pnpm init

# 安装 TypeScript（项目级）
pnpm add -D typescript

# 生成 TypeScript 配置文件
npx tsc --init
```

`tsconfig.json` 是 TypeScript 项目的核心配置文件，`--init` 生成的默认配置包含详细注释。对于初学者，重点关注以下选项：

```json
{
  "compilerOptions": {
    "target": "ES2020",          // 编译目标：生成 ES2020 语法的 JavaScript
    "module": "commonjs",        // 模块系统：CommonJS（Node.js 默认）
    "strict": true,              // 开启所有严格类型检查（强烈推荐）
    "outDir": "./dist",          // 编译输出目录
    "rootDir": "./src",          // 源码根目录
    "esModuleInterop": true,     // 允许 import defaultExport from 'module' 语法
    "skipLibCheck": true         // 跳过声明文件的类型检查（加快编译速度）
  },
  "include": ["src/**/*"],       // 包含的文件
  "exclude": ["node_modules"]    // 排除的文件
}
```

---

## 1.3 第一个 TypeScript 程序

### 1.3.1 项目结构

```
my-first-ts-project/
├── src/
│   └── index.ts        ← TypeScript 源文件
├── dist/               ← 编译后的 JavaScript（自动生成）
├── tsconfig.json       ← TypeScript 配置
└── package.json        ← 项目配置
```

### 1.3.2 编写 Hello World

创建 `src/index.ts`：

```typescript
// src/index.ts

// TypeScript 中的变量声明（带类型注解）
const greeting: string = "Hello, TypeScript!";
const year: number = 2024;
const isAIReady: boolean = true;

// 带类型的函数
function greet(name: string, language: string): string {
  return `你好，${name}！欢迎学习 ${language}。`;
}

// 调用函数
const message = greet("开发者", "TypeScript");

console.log(greeting);
console.log(`当前年份：${year}`);
console.log(`AI 就绪：${isAIReady}`);
console.log(message);
```

### 1.3.3 编译与运行

**方式一：手动编译**

```bash
# 编译 TypeScript -> JavaScript
npx tsc

# 运行编译后的 JavaScript
node dist/index.js
```

预期输出：
```
Hello, TypeScript!
当前年份：2024
AI 就绪：true
你好，开发者！欢迎学习 TypeScript。
```

**方式二：使用 ts-node（开发时推荐）**

`ts-node` 可以直接运行 TypeScript 文件，无需手动编译：

```bash
# 安装 ts-node
pnpm add -D ts-node

# 直接运行
npx ts-node src/index.ts
```

**方式三：使用 tsx（最快，推荐）**

`tsx` 是基于 esbuild 的极速 TypeScript 运行器：

```bash
# 安装 tsx
pnpm add -D tsx

# 运行
npx tsx src/index.ts

# 监听模式（文件修改后自动重新运行）
npx tsx watch src/index.ts
```

### 1.3.4 理解编译产物

查看 `dist/index.js`，观察 TypeScript 编译后的结果：

```javascript
// dist/index.js（自动生成，TypeScript 类型信息被擦除）
"use strict";
const greeting = "Hello, TypeScript!";
const year = 2024;
const isAIReady = true;

function greet(name, language) {
  return `你好，${name}！欢迎学习 ${language}。`;
}

const message = greet("开发者", "TypeScript");
console.log(greeting);
console.log(`当前年份：${year}`);
console.log(`AI 就绪：${isAIReady}`);
console.log(message);
```

**关键观察**：类型注解（`: string`、`: number`、`: boolean`）在编译后完全消失。TypeScript 的类型系统是**纯编译期**的工具，不影响运行时行为。这意味着：

- 运行时性能与原生 JavaScript 完全相同
- 类型错误在编译期就会被发现
- 最终部署的是标准 JavaScript

### 1.3.5 体验类型错误

修改 `src/index.ts`，故意引入类型错误：

```typescript
// 故意传入错误类型
const result = greet(42, "TypeScript"); // 错误：42 不是 string 类型
```

运行 `npx tsc` 会立即报错：

```
src/index.ts:XX:XX - error TS2345: Argument of type 'number' is not assignable
to parameter of type 'string'.
```

在 VS Code 中，不需要等到运行编译命令，错误会实时显示为红色波浪线。这就是 TypeScript 的核心价值：**在代码写错的那一刻就告诉你**，而不是等到用户在生产环境触发 bug。

---

## 1.4 变量声明：let、const 与 var

### 1.4.1 三者的核心区别

| 特性 | `var` | `let` | `const` |
|------|-------|-------|---------|
| 作用域 | 函数作用域 | 块作用域 | 块作用域 |
| 提升（Hoisting） | 是（值为 undefined） | 是（但不可访问，暂时性死区） | 是（但不可访问，暂时性死区） |
| 重复声明 | 允许 | 不允许 | 不允许 |
| 重新赋值 | 允许 | 允许 | 不允许（基本类型）|
| 全局对象属性 | 成为 `window` 属性 | 不会 | 不会 |

### 1.4.2 var 的问题（避免使用）

```typescript
// 问题 1：函数作用域而非块作用域
function varExample() {
  for (var i = 0; i < 3; i++) {
    // i 的作用域是整个函数，不是 for 块
  }
  console.log(i); // 输出 3！var 泄漏出了 for 块
}

// 问题 2：变量提升导致的诡异行为
console.log(x); // 输出 undefined（不报错！）
var x = 10;

// 问题 3：重复声明不报错
var name = "Alice";
var name = "Bob"; // 不报错，悄悄覆盖了
```

**TypeScript 中的表现**：即使在 TypeScript 中，`var` 也能编译通过。这是因为 TypeScript 保持了 JavaScript 的向后兼容性。但 `strict` 模式下 ESLint 通常会警告。

### 1.4.3 let：可重新赋值的块级变量

```typescript
// 基本用法
let count: number = 0;
count = count + 1; // 允许重新赋值
count = "hello";   // 错误：不能将 string 赋值给 number

// 块作用域
function letExample() {
  let message = "外部";

  if (true) {
    let message = "内部"; // 这是一个全新的变量，与外部无关
    console.log(message); // "内部"
  }

  console.log(message); // "外部"
}

// AI 应用场景：循环处理消息
const messages = ["你好", "请问", "谢谢"];
let totalLength: number = 0;

for (let i = 0; i < messages.length; i++) {
  let currentMessage: string = messages[i]; // 每次迭代的独立变量
  totalLength += currentMessage.length;
}

console.log(`总字符数：${totalLength}`);
```

### 1.4.4 const：不可重新赋值的块级变量

```typescript
// 基本用法：基本类型的 const 完全不可变
const MAX_TOKENS: number = 4096;
MAX_TOKENS = 8192; // 错误：无法分配到 "MAX_TOKENS" ，因为它是常数

// const 对象：引用不可变，但内容可变
const config = {
  model: "gpt-4",
  temperature: 0.7,
};

config.temperature = 0.9; // 允许：修改对象属性
config.model = "gpt-3.5"; // 允许：修改对象属性
config = { model: "gpt-4", temperature: 0.5 }; // 错误：不能重新赋值

// const 数组：同理
const history: string[] = [];
history.push("用户：你好"); // 允许：修改数组内容
history = ["重置"];         // 错误：不能重新赋值
```

**最佳实践**：默认使用 `const`，只有当你确定需要重新赋值时才用 `let`，永远不用 `var`。

```typescript
// 推荐的风格
const API_KEY = process.env.API_KEY ?? "";      // 配置不会变
const MODEL_NAME = "gpt-4o";                    // 模型名不会变

let retryCount = 0;                             // 重试次数需要递增
let lastResponse: string | null = null;         // 响应会被更新

for (let attempt = 0; attempt < 3; attempt++) {
  // attempt 需要递增，用 let
  retryCount++;
}
```

### 1.4.5 TypeScript 的类型推断

TypeScript 非常聪明，很多时候你不需要显式写类型注解：

```typescript
// TypeScript 自动推断类型
const name = "Alice";           // 推断为 string
const age = 30;                 // 推断为 number
const isActive = true;          // 推断为 boolean
const pi = 3.14159;             // 推断为 number

// 鼠标悬停在变量上，VS Code 会显示推断出的类型
// 推断失败时再手动添加注解

// 需要显式注解的情况：初始值为空或类型不确定
let userId: number;             // 没有初始值，必须注解
let result: string | null = null; // 联合类型，建议显式注解
```

---

## 1.5 基本类型注解

### 1.5.1 string 类型

```typescript
// 字面量赋值
const firstName: string = "张三";
const lastName: string = '李四';      // 单引号也可以
const fullName: string = `${firstName}${lastName}`; // 模板字符串

// 函数参数和返回值
function formatPrompt(userInput: string, systemPrompt: string): string {
  return `系统：${systemPrompt}\n用户：${userInput}`;
}

// 字符串方法（TypeScript 会提示所有可用方法）
const prompt = "请帮我写一首诗";
console.log(prompt.length);           // 9
console.log(prompt.includes("写"));   // true
console.log(prompt.toUpperCase());    // 英文字母转大写
console.log(prompt.trim());           // 去除首尾空白

// AI 场景：处理模型输出
function extractContent(rawResponse: string): string {
  return rawResponse.trim().replace(/\n{3,}/g, "\n\n");
}
```

### 1.5.2 number 类型

TypeScript（和 JavaScript）只有一种数字类型，整数和浮点数都是 `number`：

```typescript
// 整数
const maxRetries: number = 3;
const tokenLimit: number = 4096;

// 浮点数
const temperature: number = 0.7;
const topP: number = 0.95;

// 特殊数值
const infinity = Infinity;           // 正无穷
const negInfinity = -Infinity;       // 负无穷
const notANumber = NaN;              // 非数字（注意：typeof NaN === 'number'）

// 数字字面量
const hex = 0xFF;                    // 十六进制：255
const binary = 0b1010;               // 二进制：10
const octal = 0o17;                  // 八进制：15
const bigNum = 1_000_000;            // 数字分隔符（提高可读性）

// 常用数学操作
const inputTokens: number = 1024;
const outputTokens: number = 512;
const totalCost: number = (inputTokens * 0.001 + outputTokens * 0.002) / 1000;

console.log(`总费用：$${totalCost.toFixed(6)}`);

// 类型守卫：检查是否为有效数字
function isValidTemperature(value: number): boolean {
  return value >= 0 && value <= 2 && !isNaN(value);
}
```

### 1.5.3 boolean 类型

```typescript
// 基本用法
const isStreaming: boolean = true;
const hasSystemPrompt: boolean = false;

// 函数返回 boolean
function isTokenLimitExceeded(tokens: number, limit: number): boolean {
  return tokens > limit;
}

// 条件判断
const currentTokens = 3500;
if (isTokenLimitExceeded(currentTokens, MAX_TOKENS)) {
  console.log("警告：Token 数量超出限制");
}

// 注意：TypeScript 中 boolean 只有 true 和 false
// 不要与 truthy/falsy 混淆
const emptyString = "";
const zero = 0;
// if (emptyString) { ... }  // 这是 JavaScript 的 truthy 判断，TypeScript 允许但要小心

// 推荐：显式比较
if (emptyString !== "") {
  console.log("有内容");
}
```

### 1.5.4 其他基本类型简介

虽然本章重点是 `string`、`number`、`boolean`，但以下类型也很常见：

```typescript
// null 和 undefined（strict 模式下要注意区分）
let apiKey: string | null = null;      // 可能为 null
let userId: number | undefined;        // 可能未定义

// any（尽量避免使用）
let unknownData: any = "可以是任何值";
unknownData = 42;       // 不报错
unknownData = true;     // 不报错
// any 会绕过所有类型检查，失去 TypeScript 的优势

// unknown（比 any 更安全）
let userInput: unknown = getUserInput();
// 使用前必须进行类型检查
if (typeof userInput === "string") {
  console.log(userInput.toUpperCase()); // 现在安全了
}

// void（函数没有返回值）
function logMessage(msg: string): void {
  console.log(msg);
  // 没有 return 语句
}

// never（函数永远不会正常结束）
function throwError(message: string): never {
  throw new Error(message);
}
```

### 1.5.5 类型注解的时机

```typescript
// 情况 1：有初始值 —— 让 TypeScript 推断（省略注解）
const model = "gpt-4o";            // 推断为 string，无需注解
const temperature = 0.7;           // 推断为 number，无需注解

// 情况 2：没有初始值 —— 必须注解
let userMessage: string;
let tokenCount: number;

// 情况 3：初始值类型不够精确 —— 需要注解
let status: "pending" | "success" | "error" = "pending"; // 字面量联合类型
let items: string[] = [];           // 空数组无法推断元素类型

// 情况 4：函数参数 —— 总是注解（TypeScript 不会推断参数类型）
function callAPI(endpoint: string, timeout: number): void {
  // ...
}
```

---

## 本章小结

| 概念 | 要点 | AI 应用意义 |
|------|------|-------------|
| TypeScript 是什么 | JavaScript 超集，添加静态类型系统 | 复杂 AI API 响应的类型安全 |
| 开发环境 | Node.js + pnpm + VS Code + tsx | 高效的 AI 应用开发工具链 |
| 编译流程 | `.ts` → `tsc` → `.js` → Node.js 运行 | 类型错误在编译期暴露，不进入生产 |
| `var` | 函数作用域，避免使用 | 遗留代码识别 |
| `let` | 块作用域，可重新赋值 | 计数器、状态变量 |
| `const` | 块作用域，不可重新赋值（推荐默认） | 配置项、常量、模型名称 |
| `string` | 文本类型，支持模板字符串 | Prompt 构建、用户输入处理 |
| `number` | 整数和浮点数统一类型 | Token 计数、温度参数、费用计算 |
| `boolean` | 真/假值 | 开关配置、流式传输标志 |
| 类型推断 | TypeScript 自动推断大多数类型 | 减少冗余代码，保持可读性 |

---

## AI 应用实战：类型安全的配置管理

实际的 AI 应用需要管理大量配置参数。没有类型约束时，配置对象容易出现字段名拼错、参数类型错误等问题。本节展示如何用 TypeScript 构建类型安全的 AI 应用配置系统。

### 项目目标

构建一个 `AIConfig` 配置模块，支持：

1. 定义 OpenAI / Anthropic 等模型的配置结构
2. 提供默认值和验证函数
3. 从环境变量加载配置

### 完整实现

创建 `src/ai-config.ts`：

```typescript
// src/ai-config.ts
// AI 应用类型安全配置管理示例

// ===== 类型定义 =====

// 支持的 AI 提供商
const PROVIDERS = ["openai", "anthropic", "gemini", "local"] as const;
type AIProvider = typeof PROVIDERS[number]; // "openai" | "anthropic" | "gemini" | "local"

// 单个模型的配置
const MODEL_NAME_GPT4O: string = "gpt-4o";
const MODEL_NAME_CLAUDE: string = "claude-3-5-sonnet-20241022";
const MODEL_NAME_GEMINI: string = "gemini-1.5-pro";

// AI 请求参数配置
const DEFAULT_TEMPERATURE: number = 0.7;
const DEFAULT_MAX_TOKENS: number = 2048;
const DEFAULT_TOP_P: number = 1.0;
const DEFAULT_TIMEOUT_MS: number = 30000;

// 配置对象结构
const defaultOpenAIConfig = {
  provider: "openai" as AIProvider,
  apiKey: "",
  model: MODEL_NAME_GPT4O,
  temperature: DEFAULT_TEMPERATURE,
  maxTokens: DEFAULT_MAX_TOKENS,
  topP: DEFAULT_TOP_P,
  timeoutMs: DEFAULT_TIMEOUT_MS,
  isStreaming: false,
  debug: false,
};

// ===== 配置加载函数 =====

/**
 * 从环境变量加载 AI 配置
 * 在真实项目中，process.env 来自 .env 文件（通过 dotenv 加载）
 */
function loadConfigFromEnv(): typeof defaultOpenAIConfig {
  const apiKey: string = process.env.OPENAI_API_KEY ?? "";
  const model: string = process.env.AI_MODEL ?? MODEL_NAME_GPT4O;

  const rawTemperature: string = process.env.AI_TEMPERATURE ?? String(DEFAULT_TEMPERATURE);
  const temperature: number = parseFloat(rawTemperature);

  const rawMaxTokens: string = process.env.AI_MAX_TOKENS ?? String(DEFAULT_MAX_TOKENS);
  const maxTokens: number = parseInt(rawMaxTokens, 10);

  const isStreaming: boolean = process.env.AI_STREAMING === "true";
  const debug: boolean = process.env.DEBUG === "true";

  return {
    provider: "openai",
    apiKey,
    model,
    temperature: isNaN(temperature) ? DEFAULT_TEMPERATURE : temperature,
    maxTokens: isNaN(maxTokens) ? DEFAULT_MAX_TOKENS : maxTokens,
    topP: DEFAULT_TOP_P,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    isStreaming,
    debug,
  };
}

// ===== 配置验证函数 =====

/**
 * 验证温度参数是否在有效范围内
 * OpenAI: 0.0 - 2.0
 */
function isValidTemperature(temperature: number): boolean {
  return !isNaN(temperature) && temperature >= 0.0 && temperature <= 2.0;
}

/**
 * 验证 maxTokens 是否为正整数
 */
function isValidMaxTokens(maxTokens: number): boolean {
  return Number.isInteger(maxTokens) && maxTokens > 0 && maxTokens <= 128000;
}

/**
 * 验证整个配置对象，返回错误列表
 */
function validateConfig(config: typeof defaultOpenAIConfig): string[] {
  const errors: string[] = [];

  if (!config.apiKey) {
    errors.push("apiKey 不能为空，请设置环境变量 OPENAI_API_KEY");
  }

  if (!config.model) {
    errors.push("model 不能为空");
  }

  if (!isValidTemperature(config.temperature)) {
    errors.push(`temperature 必须在 0.0 到 2.0 之间，当前值：${config.temperature}`);
  }

  if (!isValidMaxTokens(config.maxTokens)) {
    errors.push(`maxTokens 必须为 1 到 128000 之间的整数，当前值：${config.maxTokens}`);
  }

  if (config.timeoutMs <= 0) {
    errors.push(`timeoutMs 必须大于 0，当前值：${config.timeoutMs}`);
  }

  return errors;
}

// ===== 配置格式化输出 =====

/**
 * 将配置格式化为可读字符串（隐藏 API Key 敏感信息）
 */
function formatConfigSummary(config: typeof defaultOpenAIConfig): string {
  const maskedKey: string =
    config.apiKey.length > 8
      ? `${config.apiKey.slice(0, 4)}...${config.apiKey.slice(-4)}`
      : "（未设置）";

  return [
    "===== AI 配置摘要 =====",
    `提供商：${config.provider}`,
    `模型：  ${config.model}`,
    `API Key：${maskedKey}`,
    `温度：  ${config.temperature}`,
    `最大 Token：${config.maxTokens}`,
    `超时：  ${config.timeoutMs}ms`,
    `流式传输：${config.isStreaming ? "开启" : "关闭"}`,
    `调试模式：${config.debug ? "开启" : "关闭"}`,
    "=======================",
  ].join("\n");
}

// ===== 主程序 =====

function main(): void {
  console.log("正在加载 AI 配置...\n");

  // 加载配置
  const config = loadConfigFromEnv();

  // 显示配置摘要
  console.log(formatConfigSummary(config));
  console.log();

  // 验证配置
  const errors: string[] = validateConfig(config);

  if (errors.length > 0) {
    console.log("配置验证失败，发现以下问题：");
    errors.forEach((error: string, index: number) => {
      console.log(`  ${index + 1}. ${error}`);
    });
  } else {
    console.log("配置验证通过，可以开始 AI 请求！");
  }

  // 演示类型安全的好处
  console.log("\n--- 类型安全演示 ---");

  // 这些操作都有类型提示和检查
  const updatedConfig = {
    ...config,
    temperature: 0.5,       // 修改温度（number 类型）
    isStreaming: true,       // 开启流式传输（boolean 类型）
    model: MODEL_NAME_CLAUDE, // 切换模型（string 类型）
  };

  console.log(`更新后的模型：${updatedConfig.model}`);
  console.log(`更新后的温度：${updatedConfig.temperature}`);
  console.log(`流式传输：${updatedConfig.isStreaming}`);
}

main();
```

### 运行示例

```bash
# 设置环境变量后运行
OPENAI_API_KEY="sk-test1234abcd5678" npx tsx src/ai-config.ts
```

输出：

```
正在加载 AI 配置...

===== AI 配置摘要 =====
提供商：openai
模型：  gpt-4o
API Key：sk-t...8abcd5678
温度：  0.7
最大 Token：2048
超时：  30000ms
流式传输：关闭
调试模式：关闭
=======================

配置验证通过，可以开始 AI 请求！

--- 类型安全演示 ---
更新后的模型：claude-3-5-sonnet-20241022
更新后的温度：0.5
流式传输：true
```

### 关键收益分析

```typescript
// 没有 TypeScript 类型检查时，这种错误在运行时才会暴露：
const badConfig = {
  temperature: "零点七",    // 字符串传给了数字字段
  maxTokens: -100,          // 负数的 Token 限制
  isStreaming: "yes",       // 字符串传给了布尔字段
};

// 有 TypeScript 类型检查时，上面的代码在编写时就会报错：
// Type 'string' is not assignable to type 'number'
// Type 'string' is not assignable to type 'boolean'
```

---

## 练习题

### 基础题

**练习 1.1**：环境验证

完成以下代码，使其能正确编译和运行。在注释处填入正确的类型注解：

```typescript
// 填入类型注解，使代码通过 TypeScript 编译
const modelName: ___ = "gpt-4o-mini";
const tokenBudget: ___ = 1000;
const enableCache: ___ = false;

function calculateCost(inputTokens: ___, outputTokens: ___): ___ {
  const inputCost = inputTokens * 0.00015;  // $0.15 / 1M tokens
  const outputCost = outputTokens * 0.0006; // $0.60 / 1M tokens
  return (inputCost + outputCost) / 1000;
}

const cost = calculateCost(500, 200);
console.log(`预估费用：$${cost.toFixed(8)}`);
```

**练习 1.2**：变量作用域

分析以下代码，指出每个 `console.log` 的输出结果，并解释原因：

```typescript
let x: number = 10;
const y: number = 20;

function scopeTest(): void {
  let x: number = 30;  // 注意：与外部 x 同名

  if (true) {
    let x: number = 40;
    const z: string = "内部";
    console.log("(A)", x); // 输出什么？
  }

  console.log("(B)", x); // 输出什么？
  // console.log("(C)", z); // 这行会发生什么？
}

scopeTest();
console.log("(D)", x); // 输出什么？
// console.log("(E)", y); // 输出什么？
```

---

### 进阶题

**练习 1.3**：AI 参数验证器

实现一个函数 `validateAIParams`，验证 AI 请求参数是否合法：

```typescript
// 要求：
// - temperature 必须在 0.0 到 2.0 之间
// - maxTokens 必须是 1 到 128000 之间的正整数
// - model 不能是空字符串
// - 返回 { valid: boolean, errors: string[] }

function validateAIParams(
  model: string,
  temperature: number,
  maxTokens: number
): { valid: boolean; errors: string[] } {
  // 在这里实现验证逻辑
}

// 测试你的实现
console.log(validateAIParams("gpt-4o", 0.7, 2048));
// 期望：{ valid: true, errors: [] }

console.log(validateAIParams("", 3.0, -100));
// 期望：{ valid: false, errors: [...3个错误信息] }
```

**练习 1.4**：Token 费用计算器

实现一个 Token 费用计算系统，支持多个模型的定价：

```typescript
// 模型定价（每 1000 个 token 的美元费用）
const GPT4O_INPUT_PRICE: number = 0.0025;   // $2.50 / 1M
const GPT4O_OUTPUT_PRICE: number = 0.01;    // $10 / 1M
const GPT4O_MINI_INPUT_PRICE: number = 0.00015;
const GPT4O_MINI_OUTPUT_PRICE: number = 0.0006;

// 实现以下函数：
// 1. calculateTokenCost(model, inputTokens, outputTokens) -> number（美元费用）
// 2. formatCostReport(model, inputTokens, outputTokens) -> string（格式化报告）

// 测试：
// calculateTokenCost("gpt-4o", 1000, 500) 应返回约 0.007500
// formatCostReport("gpt-4o-mini", 2000, 1000) 应输出格式化的费用报告
```

---

### 挑战题

**练习 1.5**：环境配置加载器

实现一个完整的配置加载器，要求：

1. 定义一个包含以下字段的配置结构（用 `const` 对象表示）：
   - `apiKey: string` - 必填，不能为空
   - `model: string` - 默认值 "gpt-4o"
   - `temperature: number` - 默认值 0.7，范围 [0, 2]
   - `maxTokens: number` - 默认值 2048
   - `isStreaming: boolean` - 默认值 false
   - `logLevel: string` - 只能是 "debug"、"info"、"warn"、"error" 之一

2. 实现 `loadConfig()` 函数，从以下来源加载配置（优先级从高到低）：
   - 环境变量（`process.env`）
   - 默认值

3. 实现 `validateConfig()` 函数，返回所有验证错误

4. 实现 `getConfigOrThrow()` 函数：如果配置验证失败则抛出错误，否则返回配置

5. 编写测试代码，覆盖有效配置和无效配置两种场景

---

## 练习答案

### 答案 1.1

```typescript
const modelName: string = "gpt-4o-mini";
const tokenBudget: number = 1000;
const enableCache: boolean = false;

function calculateCost(inputTokens: number, outputTokens: number): number {
  const inputCost = inputTokens * 0.00015;
  const outputCost = outputTokens * 0.0006;
  return (inputCost + outputCost) / 1000;
}

const cost = calculateCost(500, 200);
console.log(`预估费用：$${cost.toFixed(8)}`);
// 输出：预估费用：$0.00000019 （约 0.019 美分）
```

### 答案 1.2

```typescript
let x: number = 10;
const y: number = 20;

function scopeTest(): void {
  let x: number = 30;  // 函数内的新变量，遮蔽了外部 x

  if (true) {
    let x: number = 40; // if 块内的新变量，遮蔽了函数内 x
    const z: string = "内部";
    console.log("(A)", x); // 输出：(A) 40  ← if 块内的 x
  }

  console.log("(B)", x); // 输出：(B) 30  ← 函数内的 x，if 块的 x 已超出作用域
  // console.log("(C)", z); // 编译错误！z 的作用域仅在 if 块内
}

scopeTest();
console.log("(D)", x); // 输出：(D) 10  ← 全局 x，函数内的 x 已超出作用域
// console.log("(E)", y); // 输出：(E) 20  ← y 是全局 const
```

**解析**：
- `let` 有块作用域，每个 `{}` 块都可以有同名的独立变量
- 内部作用域的同名变量会"遮蔽"外部变量（shadowing）
- 块结束后，块内声明的变量不可访问

### 答案 1.3

```typescript
function validateAIParams(
  model: string,
  temperature: number,
  maxTokens: number
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // 验证 model
  if (!model || model.trim() === "") {
    errors.push("model 不能为空字符串");
  }

  // 验证 temperature
  if (isNaN(temperature) || temperature < 0.0 || temperature > 2.0) {
    errors.push(`temperature 必须在 0.0 到 2.0 之间，当前值：${temperature}`);
  }

  // 验证 maxTokens
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 128000) {
    errors.push(`maxTokens 必须是 1 到 128000 之间的整数，当前值：${maxTokens}`);
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

// 测试
console.log(validateAIParams("gpt-4o", 0.7, 2048));
// { valid: true, errors: [] }

console.log(validateAIParams("", 3.0, -100));
// { valid: false, errors: [
//   "model 不能为空字符串",
//   "temperature 必须在 0.0 到 2.0 之间，当前值：3",
//   "maxTokens 必须是 1 到 128000 之间的整数，当前值：-100"
// ]}
```

### 答案 1.4

```typescript
const GPT4O_INPUT_PRICE: number = 0.0025;
const GPT4O_OUTPUT_PRICE: number = 0.01;
const GPT4O_MINI_INPUT_PRICE: number = 0.00015;
const GPT4O_MINI_OUTPUT_PRICE: number = 0.0006;

function calculateTokenCost(
  model: string,
  inputTokens: number,
  outputTokens: number
): number {
  let inputPrice: number;
  let outputPrice: number;

  if (model === "gpt-4o") {
    inputPrice = GPT4O_INPUT_PRICE;
    outputPrice = GPT4O_OUTPUT_PRICE;
  } else if (model === "gpt-4o-mini") {
    inputPrice = GPT4O_MINI_INPUT_PRICE;
    outputPrice = GPT4O_MINI_OUTPUT_PRICE;
  } else {
    // 未知模型，使用 gpt-4o-mini 定价
    inputPrice = GPT4O_MINI_INPUT_PRICE;
    outputPrice = GPT4O_MINI_OUTPUT_PRICE;
  }

  // 价格单位是每 1000 token，转换为每 token 价格
  const inputCost: number = (inputTokens / 1000) * inputPrice;
  const outputCost: number = (outputTokens / 1000) * outputPrice;

  return inputCost + outputCost;
}

function formatCostReport(
  model: string,
  inputTokens: number,
  outputTokens: number
): string {
  const totalCost: number = calculateTokenCost(model, inputTokens, outputTokens);
  const totalTokens: number = inputTokens + outputTokens;

  return [
    `===== Token 费用报告 =====`,
    `模型：        ${model}`,
    `输入 Token：  ${inputTokens.toLocaleString()}`,
    `输出 Token：  ${outputTokens.toLocaleString()}`,
    `总 Token：    ${totalTokens.toLocaleString()}`,
    `预估费用：    $${totalCost.toFixed(6)}`,
    `=========================`,
  ].join("\n");
}

// 测试
console.log(calculateTokenCost("gpt-4o", 1000, 500).toFixed(6));
// 0.007500（1000 * 0.0025/1000 + 500 * 0.01/1000 = 0.0025 + 0.005 = 0.0075）

console.log(formatCostReport("gpt-4o-mini", 2000, 1000));
```

### 答案 1.5

```typescript
// src/config-loader.ts

// 合法的日志级别
const VALID_LOG_LEVELS = ["debug", "info", "warn", "error"] as const;
type LogLevel = typeof VALID_LOG_LEVELS[number];

// 默认配置
const CONFIG_DEFAULTS = {
  model: "gpt-4o",
  temperature: 0.7,
  maxTokens: 2048,
  isStreaming: false,
  logLevel: "info" as LogLevel,
};

// 配置结构（通过 typeof 引用）
type AppConfig = typeof CONFIG_DEFAULTS & { apiKey: string };

// 从环境变量加载配置
function loadConfig(): AppConfig {
  const apiKey: string = process.env.OPENAI_API_KEY ?? "";
  const model: string = process.env.AI_MODEL ?? CONFIG_DEFAULTS.model;

  const rawTemp: string = process.env.AI_TEMPERATURE ?? String(CONFIG_DEFAULTS.temperature);
  const temperature: number = parseFloat(rawTemp);

  const rawTokens: string = process.env.AI_MAX_TOKENS ?? String(CONFIG_DEFAULTS.maxTokens);
  const maxTokens: number = parseInt(rawTokens, 10);

  const isStreaming: boolean = process.env.AI_STREAMING === "true";

  const rawLogLevel: string = process.env.LOG_LEVEL ?? CONFIG_DEFAULTS.logLevel;
  const logLevel: LogLevel = (VALID_LOG_LEVELS as readonly string[]).includes(rawLogLevel)
    ? (rawLogLevel as LogLevel)
    : CONFIG_DEFAULTS.logLevel;

  return {
    apiKey,
    model,
    temperature: isNaN(temperature) ? CONFIG_DEFAULTS.temperature : temperature,
    maxTokens: isNaN(maxTokens) ? CONFIG_DEFAULTS.maxTokens : maxTokens,
    isStreaming,
    logLevel,
  };
}

// 验证配置
function validateConfig(config: AppConfig): string[] {
  const errors: string[] = [];

  if (!config.apiKey || config.apiKey.trim() === "") {
    errors.push("apiKey 不能为空（请设置 OPENAI_API_KEY 环境变量）");
  }

  if (!config.model || config.model.trim() === "") {
    errors.push("model 不能为空");
  }

  if (isNaN(config.temperature) || config.temperature < 0 || config.temperature > 2) {
    errors.push(`temperature 必须在 0 到 2 之间，当前：${config.temperature}`);
  }

  if (!Number.isInteger(config.maxTokens) || config.maxTokens < 1 || config.maxTokens > 128000) {
    errors.push(`maxTokens 必须是 1-128000 的整数，当前：${config.maxTokens}`);
  }

  if (!(VALID_LOG_LEVELS as readonly string[]).includes(config.logLevel)) {
    errors.push(`logLevel 必须是 ${VALID_LOG_LEVELS.join("/")} 之一，当前：${config.logLevel}`);
  }

  return errors;
}

// 获取配置，验证失败则抛出错误
function getConfigOrThrow(): AppConfig {
  const config = loadConfig();
  const errors = validateConfig(config);

  if (errors.length > 0) {
    const errorMessage = [
      "配置验证失败：",
      ...errors.map((e, i) => `  ${i + 1}. ${e}`),
    ].join("\n");

    throw new Error(errorMessage);
  }

  return config;
}

// ===== 测试代码 =====

// 场景 1：有效配置（设置了 API Key）
process.env.OPENAI_API_KEY = "sk-test-valid-key-12345";
process.env.AI_MODEL = "gpt-4o-mini";
process.env.AI_TEMPERATURE = "0.5";

try {
  const config = getConfigOrThrow();
  console.log("有效配置加载成功：", config.model, config.temperature);
} catch (error) {
  console.error("不应该失败：", error);
}

// 场景 2：无效配置（缺少 API Key）
delete process.env.OPENAI_API_KEY;
process.env.AI_TEMPERATURE = "5.0"; // 超出范围

try {
  const config = getConfigOrThrow();
  console.log("不应该到达这里");
} catch (error) {
  if (error instanceof Error) {
    console.log("正确捕获到配置错误：\n", error.message);
  }
}
```

---

## 下一章预告

第 2 章将深入 TypeScript 的**高级类型系统**：

- **接口（Interface）与类型别名（Type Alias）**：定义复杂数据结构
- **联合类型与交叉类型**：灵活组合类型
- **泛型（Generics）**：编写可复用的类型安全代码
- **AI 应用实战**：用类型系统建模 OpenAI Chat API 的完整请求/响应结构

---

*本章代码已上传至 GitHub，可以克隆运行：*

```bash
git clone https://github.com/example/typescript-ai-tutorial
cd typescript-ai-tutorial/chapter-01
pnpm install
npx tsx src/ai-config.ts
```
