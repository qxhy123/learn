# 第9章：模块系统

> **系列定位**：本章深入讲解 TypeScript 的模块系统，包括 ES 模块、命名空间、动态导入等核心概念，并通过组织一个完整的 AI SDK 模块结构展示最佳实践。

---

## 学习目标

完成本章学习后，你将能够：

1. 熟练使用 ES 模块的 `import` / `export` 语法，理解模块的边界与隔离机制
2. 区分默认导出与命名导出的适用场景，并在实际项目中正确选择
3. 理解 TypeScript 的模块解析策略（`Node` 与 `Classic`），能够排查模块找不到的常见错误
4. 使用 `namespace` 组织大型代码库，理解其与模块的本质差异
5. 掌握动态导入（`import()`）实现按需加载，将其应用于 AI SDK 的延迟初始化场景

---

## 9.1 ES 模块基础（import、export）

### 9.1.1 什么是模块

在 TypeScript（以及现代 JavaScript）中，**每个文件就是一个模块**。模块拥有自己独立的作用域——模块内定义的变量、函数、类，默认对外部不可见，必须通过 `export` 显式导出，其他文件通过 `import` 显式引入。

这与"脚本"（script）模式截然不同。判断一个 `.ts` 文件是模块还是脚本的规则很简单：**只要文件中出现了顶层 `import` 或 `export`，它就是模块**；否则它是脚本，其顶层变量属于全局作用域。

```typescript
// global-script.ts（脚本模式，没有 import/export）
var globalVar = "我是全局变量"; // 污染全局作用域 ⚠️

// module.ts（模块模式，有 export）
export const moduleVar = "我只在模块内可见"; // 安全隔离 ✅
```

如果一个文件没有任何导出但你想让它成为模块（避免全局污染），可以添加一行空导出：

```typescript
// 强制成为模块，避免全局作用域污染
export {};
```

### 9.1.2 命名导出（Named Export）

命名导出允许从一个模块导出多个绑定，每个绑定都有明确的名字：

```typescript
// math-utils.ts

// 导出方式一：声明时直接导出
export const PI = 3.14159265358979;

export function add(a: number, b: number): number {
  return a + b;
}

export function multiply(a: number, b: number): number {
  return a * b;
}

export interface Vector2D {
  x: number;
  y: number;
}

export type Operation = "add" | "subtract" | "multiply" | "divide";

// 导出方式二：统一在底部导出（推荐，便于一眼看到所有导出）
const E = 2.71828182845905;
class Matrix {
  constructor(public rows: number, public cols: number) {}
}
export { E, Matrix };
```

对应的导入语法：

```typescript
// main.ts

// 按需导入（推荐：tree-shaking 友好）
import { add, multiply, PI } from "./math-utils";

// 重命名导入（避免命名冲突）
import { add as mathAdd, Vector2D as Vec } from "./math-utils";

// 导入全部并挂载到命名空间对象
import * as MathUtils from "./math-utils";

console.log(MathUtils.PI);        // 3.14159265358979
console.log(MathUtils.add(1, 2)); // 3

// 仅执行模块的副作用，不导入任何绑定
import "./side-effects";
```

### 9.1.3 导出重定向（Re-export）

模块可以将从其他模块导入的内容再次导出，这是构建"桶文件"（barrel file）的核心机制：

```typescript
// utils/index.ts —— 桶文件，聚合所有工具函数

// 重新导出所有命名导出
export { add, multiply, PI } from "./math-utils";
export { formatDate, parseDate } from "./date-utils";

// 重新导出时重命名
export { fetchCompletion as callLLM } from "./api-utils";

// 重新导出全部（谨慎使用，可能导致命名冲突）
export * from "./string-utils";

// 重新导出默认导出并赋予名字
export { default as HttpClient } from "./http-client";
```

使用桶文件后，调用方只需记住一个路径：

```typescript
// 不使用桶文件 —— 需要记住每个文件的路径
import { add } from "./utils/math-utils";
import { formatDate } from "./utils/date-utils";

// 使用桶文件 —— 统一入口
import { add, formatDate, callLLM } from "./utils";
```

---

## 9.2 默认导出与命名导出

### 9.2.1 默认导出（Default Export）

每个模块只能有一个默认导出，使用 `export default` 语法：

```typescript
// ai-client.ts

class AIClient {
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async chat(prompt: string): Promise<string> {
    // 调用 AI API...
    return "AI 响应";
  }
}

// 方式一：声明时导出
export default AIClient;

// 方式二：匿名默认导出（不推荐，调试时难以识别）
// export default class { ... }
// export default function() { ... }
```

导入默认导出时，**不需要花括号**，且可以任意命名：

```typescript
// 导入时可以自定义名称（不受原始名称约束）
import AIClient from "./ai-client";
import MyAIClient from "./ai-client"; // 同一模块，不同别名，都合法

const client = new AIClient("sk-...");
```

### 9.2.2 混合导出

一个模块可以同时拥有默认导出和命名导出：

```typescript
// openai-adapter.ts

// 命名导出：辅助类型和工具函数
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface CompletionOptions {
  model: string;
  temperature?: number;
  maxTokens?: number;
}

export function buildSystemPrompt(instructions: string): ChatMessage {
  return { role: "system", content: instructions };
}

// 默认导出：核心类
export default class OpenAIAdapter {
  constructor(
    private apiKey: string,
    private baseUrl: string = "https://api.openai.com/v1"
  ) {}

  async complete(
    messages: ChatMessage[],
    options: CompletionOptions
  ): Promise<string> {
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: options.model,
        messages,
        temperature: options.temperature ?? 0.7,
        max_tokens: options.maxTokens,
      }),
    });
    const data = await response.json();
    return data.choices[0].message.content;
  }
}
```

```typescript
// 同时导入默认导出和命名导出
import OpenAIAdapter, { ChatMessage, buildSystemPrompt } from "./openai-adapter";

const adapter = new OpenAIAdapter(process.env.OPENAI_API_KEY!);
const messages: ChatMessage[] = [
  buildSystemPrompt("你是一个有帮助的助手"),
  { role: "user", content: "你好！" },
];
```

### 9.2.3 如何选择：默认导出 vs 命名导出

这是 TypeScript/JavaScript 社区长期争论的话题。以下是实践中被广泛认可的准则：

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| 模块的核心类/函数（一个文件一个概念） | 默认导出 | 语义清晰，导入简洁 |
| 工具函数集合 | 命名导出 | 按需导入，tree-shaking 友好 |
| 类型 / 接口 | 命名导出 | 类型通常是辅助概念，不应是默认 |
| React 组件（单文件） | 默认导出 | 社区约定，与文件名对应 |
| 常量配置 | 命名导出 | 多个常量并列，命名导出更直观 |
| 库的公开 API（index.ts） | 命名导出 | IDE 自动补全更友好，重命名风险低 |

> **经验法则**：如果一个模块只做"一件事"，用默认导出；如果它提供多个独立工具，用命名导出。避免在同一个模块里有多个默认导出的冲动（编译器也不允许）。

```typescript
// ❌ 反模式：用默认导出导出一个对象来绕过限制
export default {
  add,
  multiply,
  PI,
};
// 导入时：import MathUtils from "./math-utils"
// 问题：无法 tree-shaking，IDE 提示变差

// ✅ 正确做法：直接用命名导出
export { add, multiply, PI };
// 导入时：import { add } from "./math-utils"
```

---

## 9.3 模块解析策略（Node vs Classic）

### 9.3.1 什么是模块解析

当你写 `import { foo } from "./bar"` 时，TypeScript 编译器需要找到 `./bar` 对应的实际文件。**模块解析**（Module Resolution）就是这个查找过程的规则集合。

TypeScript 通过 `tsconfig.json` 中的 `moduleResolution` 字段控制解析策略。

### 9.3.2 Classic 策略（已过时）

Classic 是 TypeScript 早期的默认策略，查找规则简单：

```
import { foo } from "./bar"
→ ./bar.ts
→ ./bar.tsx
→ ./bar.d.ts

import { foo } from "lodash"（非相对路径）
→ ../lodash.ts
→ ../../lodash.ts
→ ... 逐级向上查找
```

Classic 策略不查找 `node_modules`，已不适合现代项目，**不推荐使用**。

### 9.3.3 Node 策略（主流选择）

Node 策略模仿 Node.js 的模块解析行为：

**相对路径导入**（`./`、`../`）：

```
import { foo } from "./utils"
查找顺序：
1. ./utils.ts
2. ./utils.tsx
3. ./utils.d.ts
4. ./utils/package.json（读取 "types" 字段）
5. ./utils/index.ts
6. ./utils/index.tsx
7. ./utils/index.d.ts
```

**非相对路径导入**（包名）：

```
import { foo } from "ai-sdk"
查找顺序：
1. ./node_modules/ai-sdk.ts
2. ./node_modules/ai-sdk.tsx
3. ./node_modules/ai-sdk.d.ts
4. ./node_modules/ai-sdk/package.json（读取 "types" 字段）
5. ./node_modules/ai-sdk/index.ts
6. ../node_modules/ai-sdk/...（逐级向上）
```

### 9.3.4 NodeNext / Bundler 策略（现代项目推荐）

TypeScript 4.7+ 引入了 `node16`/`nodenext` 策略，TypeScript 5.0 引入了 `bundler` 策略：

```json
// tsconfig.json
{
  "compilerOptions": {
    "module": "NodeNext",
    "moduleResolution": "NodeNext"
  }
}
```

`NodeNext` 要求导入语句**必须包含扩展名**（与 Node.js ESM 规范一致）：

```typescript
// NodeNext 模式下
import { foo } from "./utils.js"; // ✅ 必须写 .js（即使源文件是 .ts）
import { bar } from "./types.js"; // ✅
import { baz } from "./utils";    // ❌ 报错：缺少扩展名
```

`bundler` 策略适合使用 Webpack/Vite/esbuild 等打包工具的项目，不要求扩展名：

```json
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "bundler"
  }
}
```

### 9.3.5 路径别名（Path Aliases）

通过 `paths` 配置，可以为常用路径设置别名，避免 `../../../` 地狱：

```json
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@ai/*": ["src/ai/*"],
      "@utils/*": ["src/utils/*"]
    }
  }
}
```

```typescript
// 使用路径别名
import { OpenAIClient } from "@ai/clients/openai";
import { formatPrompt } from "@utils/prompt";
import { AppConfig } from "@/config";

// 等价于（不使用别名时的冗长写法）
import { OpenAIClient } from "../../src/ai/clients/openai";
import { formatPrompt } from "../../src/utils/prompt";
import { AppConfig } from "../../src/config";
```

> **注意**：`tsconfig.json` 中的 `paths` 只影响 TypeScript 的类型检查，不影响运行时。运行时路径解析需要额外配置（如 `tsconfig-paths`、Webpack 的 `resolve.alias`、Vite 的 `resolve.alias`）。

### 9.3.6 常见模块解析错误排查

```typescript
// 错误 1：找不到模块
// TS2307: Cannot find module './utils' or its corresponding type declarations

// 原因 A：文件不存在 → 检查文件路径和文件名大小写
// 原因 B：缺少类型声明 → 安装 @types/xxx 或添加 declare module

// 错误 2：找不到第三方包的类型
// TS7016: Could not find a declaration file for module 'some-package'
declare module "some-package" {
  export function doSomething(input: string): void;
}

// 错误 3：循环依赖导致的 undefined
// 症状：导入的值在运行时是 undefined，但编译通过
// 解决：重构代码，将共享依赖提取到单独模块
```

---

## 9.4 命名空间（namespace）

### 9.4.1 namespace 的概念

`namespace` 是 TypeScript 特有的语法（非 ES 标准），用于将相关代码组织在一个具名的作用域下，主要用于：

1. 在单文件脚本（非模块）中避免全局命名冲突
2. 为第三方 JavaScript 库编写类型声明（`.d.ts` 文件）
3. 组织大型类型定义（尽管现代项目更倾向用模块）

```typescript
// 基本语法
namespace Validation {
  // 接口（模块内可见）
  export interface StringValidator {
    isAcceptable(s: string): boolean;
  }

  // 内部辅助（不导出，外部不可见）
  const lettersRegexp = /^[A-Za-z]+$/;
  const numberRegexp = /^[0-9]+$/;

  // 导出的类
  export class LettersOnlyValidator implements StringValidator {
    isAcceptable(s: string): boolean {
      return lettersRegexp.test(s);
    }
  }

  export class ZipCodeValidator implements StringValidator {
    isAcceptable(s: string): boolean {
      return s.length === 5 && numberRegexp.test(s);
    }
  }
}

// 使用命名空间
const validator: Validation.StringValidator = new Validation.LettersOnlyValidator();
console.log(validator.isAcceptable("Hello")); // true
```

### 9.4.2 嵌套命名空间

命名空间可以嵌套，模拟层级结构：

```typescript
namespace AI {
  export namespace Models {
    export interface ModelConfig {
      name: string;
      maxTokens: number;
      temperature: number;
    }

    export const GPT4: ModelConfig = {
      name: "gpt-4",
      maxTokens: 8192,
      temperature: 0.7,
    };

    export const Claude3: ModelConfig = {
      name: "claude-3-opus-20240229",
      maxTokens: 200000,
      temperature: 0.7,
    };
  }

  export namespace Prompts {
    export function system(content: string): string {
      return `[SYSTEM]\n${content}\n[/SYSTEM]`;
    }

    export function user(content: string): string {
      return `[USER]\n${content}\n[/USER]`;
    }
  }
}

// 使用
const config = AI.Models.GPT4;
const prompt = AI.Prompts.system("你是一个 AI 助手");
```

使用别名简化深层嵌套：

```typescript
import ModelConfig = AI.Models.ModelConfig; // 类型别名
import GPT4 = AI.Models.GPT4;               // 值别名

const myConfig: ModelConfig = GPT4;
```

### 9.4.3 声明合并（Declaration Merging）

同名 namespace 会自动合并，常用于扩展第三方模块的类型：

```typescript
// 原始声明（来自第三方库）
namespace Express {
  interface Request {
    body: unknown;
  }
}

// 扩展声明（在你的项目中）
namespace Express {
  interface Request {
    user?: { id: string; role: string }; // 添加自定义字段
    aiContext?: { model: string; sessionId: string };
  }
}

// 合并结果：Request 同时拥有 body、user、aiContext
```

### 9.4.4 namespace vs 模块：如何选择

| 维度 | namespace | ES 模块 |
|------|-----------|---------|
| 标准化 | TypeScript 特有 | ES 标准，JavaScript 通用 |
| 文件边界 | 可跨多个文件 | 一文件一模块 |
| 依赖管理 | 需手动管理顺序 | `import/export` 自动管理 |
| Tree-shaking | 不支持 | 支持 |
| 适用场景 | 类型声明文件、遗留代码 | 现代应用开发 |

> **现代项目建议**：优先使用 ES 模块。`namespace` 主要保留在以下两种场景：(1) 编写 `.d.ts` 声明文件；(2) 需要扩展第三方库的类型（声明合并）。

---

## 9.5 动态导入与代码分割

### 9.5.1 静态导入的局限性

ES 静态导入（`import ... from ...`）有一个限制：**必须写在模块顶层**，无法在条件语句或函数内部使用。这意味着所有依赖在模块初始化时就被加载，对于大型应用或 AI SDK（初始化开销大）来说并不理想。

```typescript
// ❌ 非法：import 不能在块内部
if (process.env.USE_OPENAI) {
  import { OpenAI } from "openai"; // SyntaxError
}

// ❌ 非法：import 不能在函数内部
function loadOpenAI() {
  import { OpenAI } from "openai"; // SyntaxError
}
```

### 9.5.2 动态 import()

`import()` 是一个**返回 Promise 的函数**，可以在任何位置使用，实现真正的按需加载：

```typescript
// 基本语法
async function loadOpenAI() {
  // import() 返回 Promise<module>
  const { OpenAI } = await import("openai");
  return new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
}

// 条件加载
async function createAIClient(provider: "openai" | "anthropic" | "gemini") {
  switch (provider) {
    case "openai": {
      const { OpenAI } = await import("openai");
      return new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    }
    case "anthropic": {
      const { Anthropic } = await import("@anthropic-ai/sdk");
      return new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    }
    case "gemini": {
      const { GoogleGenerativeAI } = await import("@google/generative-ai");
      return new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!);
    }
  }
}
```

### 9.5.3 动态导入的类型安全

TypeScript 能正确推断动态导入的类型：

```typescript
// TypeScript 自动推断类型
const openaiModule = await import("openai");
// openaiModule 的类型是 typeof import("openai")
// 包含该模块导出的所有类型信息

// 使用 typeof import() 提取类型
type OpenAIModule = typeof import("openai");
type OpenAIInstance = InstanceType<OpenAIModule["OpenAI"]>;

// 获取默认导出的类型
type ConfigModule = typeof import("./config");
type AppConfig = ConfigModule["default"];
```

### 9.5.4 懒加载模式（Lazy Loading Pattern）

对于初始化开销大的模块（如 AI SDK、数据库连接），懒加载是常见优化手段：

```typescript
// lazy-loader.ts

type Loader<T> = () => Promise<T>;

/**
 * 创建一个懒加载实例：首次调用时初始化，后续调用复用缓存
 */
function createLazy<T>(loader: Loader<T>): Loader<T> {
  let instance: T | null = null;
  let loading: Promise<T> | null = null;

  return async () => {
    if (instance !== null) return instance;
    if (loading !== null) return loading;

    loading = loader().then((result) => {
      instance = result;
      loading = null;
      return result;
    });

    return loading;
  };
}

// 使用懒加载
const getOpenAIClient = createLazy(async () => {
  const { OpenAI } = await import("openai");
  return new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
});

// 第一次调用：实际加载并初始化
const client1 = await getOpenAIClient();
// 第二次调用：直接返回缓存实例
const client2 = await getOpenAIClient();
// client1 === client2 → true
```

### 9.5.5 代码分割（Code Splitting）

在前端应用中，动态导入与打包工具结合，实现代码分割（将大 bundle 拆分为按需加载的 chunk）：

```typescript
// Next.js / React 中的动态导入
import dynamic from "next/dynamic";

// AI 聊天组件（较大，只在需要时加载）
const AIChatPanel = dynamic(() => import("./components/AIChatPanel"), {
  loading: () => <div>加载 AI 助手中...</div>,
  ssr: false, // 禁用服务端渲染，因为 AI SDK 依赖浏览器 API
});

// 按路由分割
const routes = [
  {
    path: "/chat",
    component: () => import("./pages/ChatPage"),
  },
  {
    path: "/analyze",
    component: () => import("./pages/AnalyzePage"),
  },
];
```

### 9.5.6 import.meta

`import.meta` 提供当前模块的元信息，在 ES 模块中可用：

```typescript
// 获取当前模块的 URL（ESM 环境）
console.log(import.meta.url); // file:///Users/.../src/utils.ts

// Vite 项目中常用：环境变量
const apiUrl = import.meta.env.VITE_API_URL;
const isDev = import.meta.env.DEV;

// 热模块替换（HMR）
if (import.meta.hot) {
  import.meta.hot.accept((newModule) => {
    // 模块更新时的处理逻辑
  });
}
```

---

## 本章小结

| 概念 | 核心要点 | 适用场景 |
|------|----------|----------|
| **命名导出** | `export { name }`，按需导入，支持 tree-shaking | 工具函数、类型、常量 |
| **默认导出** | `export default`，每模块唯一，导入时可自定义名称 | 核心类、组件 |
| **桶文件** | `index.ts` 聚合多个模块的导出 | 提供统一 API 入口 |
| **模块解析** | `Node` / `NodeNext` / `bundler` 三种策略，影响文件查找规则 | 项目 tsconfig 配置 |
| **路径别名** | `paths` 配置简化深层相对路径 | 大型项目结构 |
| **namespace** | TypeScript 特有，声明合并、类型扩展 | `.d.ts` 文件、扩展第三方类型 |
| **动态导入** | `import()` 返回 Promise，按需加载模块 | 懒加载、条件加载、代码分割 |
| **import.meta** | 当前模块元信息（URL、环境变量） | Vite/ESM 项目 |

---

## AI 应用实战：SDK 模块化组织

本节展示如何将一个 AI SDK 组织成清晰、可维护的模块结构。这是一个综合运用本章所有知识点的完整示例。

### 项目结构

```
src/
├── index.ts                    # 公开 API 入口（桶文件）
├── core/
│   ├── client.ts               # 核心客户端类（默认导出）
│   ├── config.ts               # 配置类型与默认值（命名导出）
│   └── errors.ts               # 错误类层级（命名导出）
├── providers/
│   ├── index.ts                # 提供商桶文件
│   ├── openai.ts               # OpenAI 适配器
│   ├── anthropic.ts            # Anthropic 适配器
│   └── base.ts                 # 抽象基类
├── features/
│   ├── chat.ts                 # 聊天功能
│   ├── embeddings.ts           # Embedding 功能
│   └── streaming.ts            # 流式响应处理
└── utils/
    ├── index.ts                # 工具函数桶文件
    ├── retry.ts                # 重试逻辑
    └── token-counter.ts        # Token 计数
```

### 核心类型与错误定义

```typescript
// src/core/config.ts

export interface AIClientConfig {
  provider: "openai" | "anthropic" | "gemini";
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  maxRetries?: number;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatOptions {
  model: string;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
}

export interface ChatResponse {
  content: string;
  model: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

// 默认配置
export const DEFAULT_CONFIG: Partial<AIClientConfig> = {
  timeout: 30_000,
  maxRetries: 3,
};
```

```typescript
// src/core/errors.ts

// 错误基类
export class AIError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode?: number
  ) {
    super(message);
    this.name = "AIError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// 具体错误类型
export class AuthenticationError extends AIError {
  constructor(message = "API 密钥无效或已过期") {
    super(message, "AUTHENTICATION_ERROR", 401);
    this.name = "AuthenticationError";
  }
}

export class RateLimitError extends AIError {
  constructor(
    message = "请求频率超限",
    public readonly retryAfter?: number
  ) {
    super(message, "RATE_LIMIT_ERROR", 429);
    this.name = "RateLimitError";
  }
}

export class ModelNotFoundError extends AIError {
  constructor(model: string) {
    super(`模型 "${model}" 不存在或无权访问`, "MODEL_NOT_FOUND", 404);
    this.name = "ModelNotFoundError";
  }
}

// 错误类型守卫
export function isAIError(error: unknown): error is AIError {
  return error instanceof AIError;
}

export function isRateLimitError(error: unknown): error is RateLimitError {
  return error instanceof RateLimitError;
}
```

### 抽象提供商基类

```typescript
// src/providers/base.ts

import type { ChatMessage, ChatOptions, ChatResponse } from "../core/config";

// 抽象基类定义提供商契约
export abstract class BaseProvider {
  abstract readonly name: string;

  abstract chat(
    messages: ChatMessage[],
    options: ChatOptions
  ): Promise<ChatResponse>;

  abstract stream(
    messages: ChatMessage[],
    options: ChatOptions
  ): AsyncIterable<string>;

  // 通用工具方法（子类共享）
  protected buildHeaders(apiKey: string): Record<string, string> {
    return {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    };
  }
}
```

```typescript
// src/providers/openai.ts

import { BaseProvider } from "./base";
import { AuthenticationError, RateLimitError, ModelNotFoundError } from "../core/errors";
import type { ChatMessage, ChatOptions, ChatResponse } from "../core/config";

export class OpenAIProvider extends BaseProvider {
  readonly name = "openai";

  constructor(
    private apiKey: string,
    private baseUrl: string = "https://api.openai.com/v1"
  ) {
    super();
  }

  async chat(messages: ChatMessage[], options: ChatOptions): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: this.buildHeaders(this.apiKey),
      body: JSON.stringify({
        model: options.model,
        messages,
        temperature: options.temperature ?? 0.7,
        max_tokens: options.maxTokens,
      }),
    });

    if (!response.ok) {
      await this.handleError(response);
    }

    const data = await response.json();
    return {
      content: data.choices[0].message.content,
      model: data.model,
      usage: {
        promptTokens: data.usage.prompt_tokens,
        completionTokens: data.usage.completion_tokens,
        totalTokens: data.usage.total_tokens,
      },
    };
  }

  async *stream(messages: ChatMessage[], options: ChatOptions): AsyncIterable<string> {
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: this.buildHeaders(this.apiKey),
      body: JSON.stringify({
        model: options.model,
        messages,
        temperature: options.temperature ?? 0.7,
        stream: true,
      }),
    });

    if (!response.ok) await this.handleError(response);
    if (!response.body) return;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n").filter((l) => l.startsWith("data: "));

      for (const line of lines) {
        const data = line.slice(6);
        if (data === "[DONE]") return;
        try {
          const json = JSON.parse(data);
          const content = json.choices[0]?.delta?.content;
          if (content) yield content;
        } catch {
          // 忽略解析错误的行
        }
      }
    }
  }

  private async handleError(response: Response): Promise<never> {
    switch (response.status) {
      case 401: throw new AuthenticationError();
      case 429: throw new RateLimitError();
      case 404: throw new ModelNotFoundError("unknown");
      default:  throw new Error(`HTTP 错误: ${response.status}`);
    }
  }
}
```

```typescript
// src/providers/index.ts —— 提供商桶文件

export { BaseProvider } from "./base";
export { OpenAIProvider } from "./openai";

// 动态加载：只在需要时加载具体提供商，避免不必要的依赖
export async function loadProvider(name: "openai" | "anthropic" | "gemini") {
  switch (name) {
    case "openai": {
      const { OpenAIProvider } = await import("./openai");
      return OpenAIProvider;
    }
    case "anthropic": {
      const { AnthropicProvider } = await import("./anthropic");
      return AnthropicProvider;
    }
    case "gemini": {
      const { GeminiProvider } = await import("./gemini");
      return GeminiProvider;
    }
  }
}
```

### 核心客户端（整合所有模块）

```typescript
// src/core/client.ts

import { DEFAULT_CONFIG } from "./config";
import { isRateLimitError } from "./errors";
import { loadProvider } from "../providers";
import type { AIClientConfig, ChatMessage, ChatOptions, ChatResponse } from "./config";
import type { BaseProvider } from "../providers/base";

class AIClient {
  private config: Required<AIClientConfig>;
  private provider: BaseProvider | null = null;

  constructor(config: AIClientConfig) {
    this.config = { ...DEFAULT_CONFIG, ...config } as Required<AIClientConfig>;
  }

  // 懒初始化提供商
  private async getProvider(): Promise<BaseProvider> {
    if (this.provider) return this.provider;

    const ProviderClass = await loadProvider(this.config.provider);
    this.provider = new ProviderClass(
      this.config.apiKey,
      this.config.baseUrl
    );
    return this.provider;
  }

  // 带重试的聊天方法
  async chat(messages: ChatMessage[], options: ChatOptions): Promise<ChatResponse> {
    const provider = await this.getProvider();
    let lastError: unknown;

    for (let attempt = 0; attempt < this.config.maxRetries; attempt++) {
      try {
        return await provider.chat(messages, options);
      } catch (error) {
        lastError = error;

        if (isRateLimitError(error) && error.retryAfter) {
          await sleep(error.retryAfter * 1000);
          continue;
        }

        // 非限流错误，直接抛出
        throw error;
      }
    }

    throw lastError;
  }

  // 流式聊天
  async *stream(
    messages: ChatMessage[],
    options: ChatOptions
  ): AsyncIterable<string> {
    const provider = await this.getProvider();
    yield* provider.stream(messages, options);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// 默认导出核心类
export default AIClient;
```

### 公开 API 入口

```typescript
// src/index.ts —— SDK 的对外入口，只暴露公开 API

// 核心类：将默认导出重新以命名形式导出，方便使用方解构导入
export { default as AIClient } from "./core/client";

// 公开类型
export type {
  AIClientConfig,
  ChatMessage,
  ChatOptions,
  ChatResponse,
} from "./core/config";

// 公开错误类
export {
  AIError,
  AuthenticationError,
  RateLimitError,
  ModelNotFoundError,
  isAIError,
  isRateLimitError,
} from "./core/errors";

// 工具函数（按需导出）
export { retry, withTimeout } from "./utils";
```

> **说明**：桶文件使用 `export { default as AIClient }` 将内部的默认导出转换为具名导出，这是库设计的常见技巧——内部用默认导出保持单一职责，对外用命名导出方便 tree-shaking 和 IDE 补全。

### 使用方式

```typescript
// 用户代码
import { AIClient, isRateLimitError } from "my-ai-sdk";
import type { ChatMessage, ChatOptions } from "my-ai-sdk";

const client = new AIClient({
  provider: "openai",
  apiKey: process.env.OPENAI_API_KEY!,
});

const messages: ChatMessage[] = [
  { role: "user", content: "用一句话解释什么是大语言模型" },
];

const options: ChatOptions = { model: "gpt-4o", temperature: 0.5 };

try {
  // 普通聊天
  const response = await client.chat(messages, options);
  console.log(response.content);
  console.log(`Token 用量：${response.usage.totalTokens}`);

  // 流式响应
  for await (const chunk of client.stream(messages, options)) {
    process.stdout.write(chunk);
  }
} catch (error) {
  if (isRateLimitError(error)) {
    console.error("触发限流，请稍后重试");
  } else {
    throw error;
  }
}
```

---

## 练习题

### 基础题

**练习 1**：编写一个 `string-utils.ts` 模块，包含以下命名导出：
- `capitalize(str: string): string` — 将字符串首字母大写
- `truncate(str: string, maxLength: number, suffix?: string): string` — 截断字符串，超出部分用 `suffix`（默认 `"..."`）替代
- `countWords(str: string): number` — 统计字符串中的单词数（按空格分割）

同时创建一个 `index.ts` 将上述函数重新导出，要求 `capitalize` 重命名为 `toTitleCase` 导出。

---

**练习 2**：有如下代码存在模块解析问题，请识别并修复所有错误：

```typescript
// tsconfig.json 配置了 "moduleResolution": "node"
// 目录结构：
// src/
//   utils/
//     index.ts    （导出 formatDate）
//     math.ts     （导出 add, subtract）
//   main.ts

// main.ts
import formatDate from "./utils";           // 问题 A
import { add, subtract } from "./utils/math.ts"; // 问题 B
import * from "./utils/index";              // 问题 C
```

---

### 进阶题

**练习 3**：实现一个 `PluginLoader` 类，使用动态导入实现插件系统：

```typescript
interface Plugin {
  name: string;
  version: string;
  execute(input: string): Promise<string>;
}

class PluginLoader {
  // 实现以下方法：
  // 1. load(path: string): Promise<Plugin> — 动态加载插件
  // 2. loadAll(paths: string[]): Promise<Plugin[]> — 并行加载多个插件
  // 3. get(name: string): Plugin | undefined — 从缓存获取已加载的插件
}
```

要求：缓存已加载的插件，避免重复加载；`loadAll` 使用 `Promise.all` 并行加载。

---

**练习 4**：为以下第三方 JavaScript 库编写 TypeScript 声明（`.d.ts`），使用 `namespace` 组织：

```javascript
// 假设这是一个叫 "simple-ai" 的纯 JS 库（无类型声明）
// 使用方式：
const ai = SimpleAI.create({ apiKey: "..." });
const result = await ai.ask("问题");
const embeddings = SimpleAI.Embeddings.encode(["文本1", "文本2"]);
const distance = SimpleAI.Embeddings.cosineSimilarity(vec1, vec2);
```

---

### 挑战题

**练习 5**：设计并实现一个支持多环境的配置模块系统：

- `config/base.ts`：基础配置（所有环境共享）
- `config/development.ts`：开发环境配置（覆盖 base）
- `config/production.ts`：生产环境配置（覆盖 base）
- `config/index.ts`：根据 `process.env.NODE_ENV` 动态导入对应配置，合并后导出

要求：
1. 使用动态导入实现按环境加载，生产环境不打包开发配置
2. 使用 TypeScript 类型确保所有配置项均已定义（避免 `undefined` 配置项）
3. 提供一个 `getConfig()` 函数，首次调用加载并缓存，后续调用直接返回缓存

---

## 练习答案

### 答案 1

```typescript
// src/string-utils.ts

export function capitalize(str: string): string {
  if (str.length === 0) return str;
  return str[0].toUpperCase() + str.slice(1).toLowerCase();
}

export function truncate(
  str: string,
  maxLength: number,
  suffix: string = "..."
): string {
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength - suffix.length) + suffix;
}

export function countWords(str: string): number {
  const trimmed = str.trim();
  if (trimmed.length === 0) return 0;
  return trimmed.split(/\s+/).length;
}
```

```typescript
// src/index.ts

export {
  capitalize as toTitleCase, // 重命名导出
  truncate,
  countWords,
} from "./string-utils";

// 也可以保留原名导出，根据需求决定
// export { capitalize, truncate, countWords } from "./string-utils";
```

---

### 答案 2

```typescript
// 问题 A：formatDate 是命名导出，不是默认导出，需要花括号
// 修复：
import { formatDate } from "./utils";
// 或者如果 utils/index.ts 有默认导出：
// import formatDate from "./utils/index";

// 问题 B：模块路径不应包含文件扩展名（Node 解析策略）
// 修复：
import { add, subtract } from "./utils/math";

// 问题 C：import * 语法错误，必须有别名（as）
// 修复：
import * as Utils from "./utils/index";
// 或者按需导入：
import { formatDate } from "./utils/index";
```

---

### 答案 3

```typescript
// plugin-loader.ts

interface Plugin {
  name: string;
  version: string;
  execute(input: string): Promise<string>;
}

class PluginLoader {
  private cache = new Map<string, Plugin>();

  async load(path: string): Promise<Plugin> {
    // 检查缓存
    const cached = Array.from(this.cache.values()).find(
      (p) => p.name === path
    );
    if (cached) return cached;

    // 动态加载
    const module = await import(path);
    const plugin: Plugin = module.default ?? module;

    // 验证插件结构
    if (
      typeof plugin.name !== "string" ||
      typeof plugin.version !== "string" ||
      typeof plugin.execute !== "function"
    ) {
      throw new Error(`无效的插件格式：${path}`);
    }

    this.cache.set(plugin.name, plugin);
    return plugin;
  }

  async loadAll(paths: string[]): Promise<Plugin[]> {
    // 并行加载所有插件
    return Promise.all(paths.map((path) => this.load(path)));
  }

  get(name: string): Plugin | undefined {
    return this.cache.get(name);
  }
}

export default PluginLoader;
```

---

### 答案 4

```typescript
// simple-ai.d.ts

declare namespace SimpleAI {
  interface CreateOptions {
    apiKey: string;
    baseUrl?: string;
    timeout?: number;
  }

  interface AIInstance {
    ask(question: string): Promise<string>;
    askWithContext(question: string, context: string[]): Promise<string>;
  }

  function create(options: CreateOptions): AIInstance;

  namespace Embeddings {
    type Vector = number[];

    function encode(texts: string[]): Promise<Vector[]>;
    function cosineSimilarity(vec1: Vector, vec2: Vector): number;
    function euclideanDistance(vec1: Vector, vec2: Vector): number;
  }
}

export = SimpleAI;
```

---

### 答案 5

```typescript
// config/base.ts
export interface AppConfig {
  apiUrl: string;
  logLevel: "debug" | "info" | "warn" | "error";
  maxRetries: number;
  ai: {
    provider: "openai" | "anthropic";
    defaultModel: string;
    temperature: number;
  };
}

export const baseConfig: AppConfig = {
  apiUrl: "https://api.example.com",
  logLevel: "info",
  maxRetries: 3,
  ai: {
    provider: "openai",
    defaultModel: "gpt-4o",
    temperature: 0.7,
  },
};
```

```typescript
// config/development.ts
import type { AppConfig } from "./base";

export const devConfig: Partial<AppConfig> = {
  apiUrl: "http://localhost:3000",
  logLevel: "debug",
  maxRetries: 1,
};
```

```typescript
// config/production.ts
import type { AppConfig } from "./base";

export const prodConfig: Partial<AppConfig> = {
  logLevel: "warn",
  maxRetries: 5,
};
```

```typescript
// config/index.ts
import { baseConfig } from "./base";
import type { AppConfig } from "./base";

type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};

function mergeConfig(
  base: AppConfig,
  override: DeepPartial<AppConfig>
): AppConfig {
  return {
    ...base,
    ...override,
    ai: { ...base.ai, ...(override.ai ?? {}) },
  };
}

// 单例缓存
let configCache: AppConfig | null = null;
let loadingPromise: Promise<AppConfig> | null = null;

export async function getConfig(): Promise<AppConfig> {
  // 返回缓存
  if (configCache !== null) return configCache;
  // 防止并发重复加载
  if (loadingPromise !== null) return loadingPromise;

  loadingPromise = (async () => {
    const env = process.env.NODE_ENV ?? "development";

    let envConfig: DeepPartial<AppConfig> = {};

    if (env === "production") {
      const { prodConfig } = await import("./production");
      envConfig = prodConfig;
    } else {
      // 开发/测试环境
      const { devConfig } = await import("./development");
      envConfig = devConfig;
    }

    configCache = mergeConfig(baseConfig, envConfig);
    loadingPromise = null;
    return configCache;
  })();

  return loadingPromise;
}

// 同步获取（仅在已初始化后使用）
export function getConfigSync(): AppConfig {
  if (configCache === null) {
    throw new Error("配置未初始化，请先调用 await getConfig()");
  }
  return configCache;
}
```

```typescript
// 使用示例
import { getConfig } from "./config";

// 应用启动时预热配置
const config = await getConfig();
console.log(`连接到: ${config.apiUrl}`);
console.log(`AI 提供商: ${config.ai.provider}`);

// 后续代码可以同步获取（已缓存）
import { getConfigSync } from "./config";
const { ai } = getConfigSync();
```
