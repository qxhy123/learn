# 第12章：声明文件与类型定义

> **适用读者**：已完成第11章工具类型学习，希望深入理解 TypeScript 类型系统的"幕后机制"，并能为第三方库编写完整类型定义的开发者。

---

## 学习目标

完成本章学习后，你将能够：

1. 理解声明文件（`.d.ts`）的作用与工作原理，区分"类型声明"与"实现代码"
2. 掌握 `declare` 关键字的各种用法，为全局变量、函数、类和模块补充类型信息
3. 熟练使用 `@types` 包和 DefinitelyTyped 生态，管理第三方库的类型依赖
4. 区分模块声明（`declare module`）与全局声明（`declare global`）的适用场景
5. 为没有官方类型定义的 AI SDK 编写自定义声明文件，消除 `any` 类型的使用

---

## 12.1 声明文件基础（.d.ts 文件）

### 12.1.1 什么是声明文件

TypeScript 的类型系统分为两个层面：**运行时代码**（`.ts` 文件，会被编译为 JavaScript）和**纯类型信息**（`.d.ts` 文件，只用于类型检查，不产生任何 JavaScript 输出）。

声明文件（Declaration File）以 `.d.ts` 为扩展名，专门用于描述"某个 JavaScript 值的类型形状"，而不包含任何实际实现。它就像一份"类型合同"：告诉 TypeScript 编译器某个变量、函数、类或模块具有怎样的结构，但不提供具体逻辑。

```typescript
// math-utils.js（JavaScript 实现，运行时存在）
function add(a, b) {
  return a + b;
}
module.exports = { add };

// math-utils.d.ts（类型声明，仅供 TypeScript 使用）
export declare function add(a: number, b: number): number;
```

当 TypeScript 项目中 `import { add } from './math-utils'` 时，编译器会查找同名的 `.d.ts` 文件获取类型信息。

### 12.1.2 声明文件的三种来源

TypeScript 获取类型信息的方式有三种，优先级从高到低：

**1. 源码自带（内联声明）**

直接编写 TypeScript 源文件，编译时自动生成 `.d.ts`：

```typescript
// ai-client.ts（源文件）
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export async function chat(messages: ChatMessage[]): Promise<string> {
  // 实现...
  return "";
}
```

执行 `tsc --declaration` 后自动生成：

```typescript
// ai-client.d.ts（自动生成，无需手写）
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}
export declare function chat(messages: ChatMessage[]): Promise<string>;
```

**2. `@types` 包（社区维护）**

安装 `@types/xxx` 包提供第三方 JS 库的类型，详见 12.3 节。

**3. 手写声明文件**

为没有类型定义的库手动编写 `.d.ts`，详见 12.5 节。

### 12.1.3 声明文件的自动生成

在 `tsconfig.json` 中开启 `declaration` 选项，TypeScript 编译时会自动为每个 `.ts` 文件生成对应的 `.d.ts`：

```json
{
  "compilerOptions": {
    "declaration": true,
    "declarationDir": "./dist/types",
    "declarationMap": true
  }
}
```

```typescript
// src/token-counter.ts（源文件）
const TOKENS_PER_CHAR = 0.25;

export function estimateTokens(text: string): number {
  return Math.ceil(text.length * TOKENS_PER_CHAR);
}

export function estimateTokensBatch(texts: string[]): number[] {
  return texts.map(estimateTokens);
}

export type TokenEstimate = {
  text: string;
  tokens: number;
  chars: number;
};
```

编译后生成：

```typescript
// dist/types/token-counter.d.ts（自动生成）
declare const TOKENS_PER_CHAR = 0.25;
export declare function estimateTokens(text: string): number;
export declare function estimateTokensBatch(texts: string[]): number[];
export type TokenEstimate = {
  text: string;
  tokens: number;
  chars: number;
};
export {};
```

注意：`TOKENS_PER_CHAR` 虽然被声明，但因为没有 `export`，在 `.d.ts` 中不会被外部访问。

### 12.1.4 声明文件的查找规则

TypeScript 按照以下顺序查找类型声明：

```
import { foo } from "some-library"

查找顺序：
1. node_modules/some-library/package.json 的 "types" 或 "typings" 字段
2. node_modules/some-library/index.d.ts
3. node_modules/@types/some-library/index.d.ts
4. tsconfig.json 中 typeRoots 配置的路径
```

在 `package.json` 中指定声明文件位置：

```json
{
  "name": "my-ai-sdk",
  "main": "./dist/index.js",
  "types": "./dist/types/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/esm/index.js",
      "require": "./dist/cjs/index.js",
      "types": "./dist/types/index.d.ts"
    }
  }
}
```

---

## 12.2 declare 关键字

### 12.2.1 declare 的核心作用

`declare` 关键字告诉 TypeScript："这个值**已经存在**于运行时环境中，我只是在为它补充类型信息，不需要生成实际代码。"

它是声明文件的核心语法，也可以在普通 `.ts` 文件中用于描述外部环境。

```typescript
// 声明一个全局变量（由 HTML <script> 标签或外部环境注入）
declare const __API_BASE_URL__: string;
declare let __DEBUG_MODE__: boolean;

// 使用时，TypeScript 知道这些变量存在且有正确类型
console.log(__API_BASE_URL__);  // 合法，不会报 "变量未定义"
```

### 12.2.2 declare 声明变量、函数与类

```typescript
// 声明全局变量
declare const VERSION: string;
declare var globalConfig: { debug: boolean; logLevel: string };

// 声明全局函数
declare function fetchAIResponse(prompt: string): Promise<string>;
declare function initAISDK(apiKey: string, options?: { model: string }): void;

// 声明全局类
declare class EventEmitter {
  on(event: string, listener: (...args: unknown[]) => void): this;
  off(event: string, listener: (...args: unknown[]) => void): this;
  emit(event: string, ...args: unknown[]): boolean;
}

// 声明枚举（通常来自老式 JS 库的全局对象）
declare enum ModelStatus {
  Loading,
  Ready,
  Error,
}
```

### 12.2.3 declare 声明接口与类型

`interface` 和 `type` 本身就是纯类型声明，在 `.d.ts` 文件中不需要加 `declare`：

```typescript
// 在 .d.ts 文件中，interface 和 type 不需要 declare
interface AIConfig {
  apiKey: string;
  model: string;
  temperature?: number;
}

type Provider = "openai" | "anthropic" | "google";

// 但在普通 .ts 文件中声明"外部"类型时，可以加 declare（可选）
declare interface WindowAIExtension {
  askAI(prompt: string): Promise<string>;
}
```

### 12.2.4 declare namespace

`declare namespace` 用于描述老式 UMD 或 IIFE 风格的 JavaScript 库，这类库将所有 API 挂载到一个全局对象上：

```typescript
// 描述类似 jQuery 的库：window.$ 或 window.jQuery
declare namespace $ {
  function ajax(url: string, settings?: AjaxSettings): JQueryXHR;
  function get(url: string): JQueryXHR;

  interface AjaxSettings {
    method?: "GET" | "POST" | "PUT" | "DELETE";
    data?: Record<string, unknown>;
    success?: (data: unknown) => void;
    error?: (xhr: JQueryXHR, status: string) => void;
  }

  interface JQueryXHR {
    done(callback: (data: unknown) => void): this;
    fail(callback: (error: Error) => void): this;
    abort(): void;
  }
}

// 描述全局 AI SDK（通过 <script> 加载）
declare namespace SimpleAI {
  interface Options {
    apiKey: string;
    model?: string;
    maxTokens?: number;
  }

  interface Response {
    content: string;
    usage: { inputTokens: number; outputTokens: number };
  }

  function create(options: Options): AIInstance;

  interface AIInstance {
    ask(prompt: string): Promise<Response>;
    stream(prompt: string): AsyncIterable<string>;
  }

  namespace Embeddings {
    type Vector = number[];
    function encode(text: string): Promise<Vector>;
    function similarity(a: Vector, b: Vector): number;
  }
}
```

### 12.2.5 declare module（模块增强）

`declare module` 有两种用途：

**用途一：为整个 npm 包声明类型**（详见 12.4 节）

**用途二：模块增强（Module Augmentation）** — 为现有模块添加新的导出或扩展已有类型：

```typescript
// 扩展 Express 的 Request 类型，添加 AI 上下文字段
// express-augmentation.d.ts

import "express";  // 必须先导入，才能进行模块增强

declare module "express" {
  interface Request {
    aiContext?: {
      sessionId: string;
      model: string;
      conversationHistory: Array<{ role: string; content: string }>;
    };
    user?: {
      id: string;
      apiQuota: number;
    };
  }
}

// 扩展 String 原型（谨慎使用）
declare global {
  interface String {
    toPrompt(): string;
    countTokens(): number;
  }
}
```

### 12.2.6 declare global

`declare global` 用于在**模块文件**（含有 `import`/`export` 的文件）中向全局作用域添加声明：

```typescript
// globals.d.ts（含有 import，所以是模块文件）
import type { AIConfig } from "./types";

declare global {
  // 扩展 Window 接口
  interface Window {
    aiSDK: {
      config: AIConfig;
      version: string;
      isInitialized: boolean;
    };
  }

  // 添加全局函数
  function getAIConfig(): AIConfig;

  // 添加全局常量
  const BUILD_TIME: string;
  const APP_VERSION: string;
}

// 必须有这个导出，将文件标记为模块（否则 declare global 会报错）
export {};
```

```typescript
// 使用全局声明
window.aiSDK.config;           // 有类型提示
const config = getAIConfig();  // 有类型提示
console.log(BUILD_TIME);       // 有类型提示
```

---

## 12.3 @types 与 DefinitelyTyped

### 12.3.1 DefinitelyTyped 是什么

[DefinitelyTyped](https://github.com/DefinitelyTyped/DefinitelyTyped) 是世界上最大的 TypeScript 类型声明仓库，由社区维护，包含数千个流行 JavaScript 库的类型定义。通过 `@types/` 命名空间发布到 npm。

```bash
# 安装常见库的类型定义
npm install --save-dev @types/node          # Node.js 内置模块类型
npm install --save-dev @types/express       # Express 框架类型
npm install --save-dev @types/lodash        # Lodash 工具库类型
npm install --save-dev @types/jest          # Jest 测试框架类型
```

### 12.3.2 @types 包的工作原理

安装 `@types/xxx` 后，TypeScript 自动识别并应用这些类型，无需任何额外配置：

```typescript
// 安装 @types/node 后，Node.js 内置模块有了完整类型
import * as fs from "fs";
import * as path from "path";

// fs.readFile 有完整的类型签名和重载
fs.readFile("./config.json", "utf-8", (err, data) => {
  if (err) throw err;
  const config = JSON.parse(data);  // data: string，因为指定了 encoding
});

// path.join 的参数被正确检查
const fullPath = path.join(__dirname, "prompts", "system.txt");
```

### 12.3.3 tsconfig.json 中的类型控制

通过 `tsconfig.json` 精确控制哪些 `@types` 包被全局引入：

```json
{
  "compilerOptions": {
    "typeRoots": [
      "./node_modules/@types",
      "./src/types"
    ],
    "types": ["node", "jest"]
  }
}
```

- `typeRoots`：指定查找 `@types` 的根目录（默认 `node_modules/@types`）
- `types`：白名单——只有列出的包才会被自动全局引入；未列出的包仍可通过 `import` 手动引入

```typescript
// tsconfig.json 中 "types": ["node", "jest"]
// 则以下全局类型可直接使用（无需 import）：
process.env.NODE_ENV;   // Node.js 全局（来自 @types/node）
describe("test", () => {  // Jest 全局（来自 @types/jest）
  it("should pass", () => {
    expect(1 + 1).toBe(2);
  });
});

// @types/express 虽然已安装，但不在 types 白名单中，
// 需要通过 import 显式引入才能使用
import express from "express";  // 手动引入，正常工作
```

### 12.3.4 类型版本匹配

`@types` 包的版本应与对应库的主版本保持一致：

```json
{
  "dependencies": {
    "express": "^4.18.0"
  },
  "devDependencies": {
    "@types/express": "^4.17.0"
  }
}
```

许多现代库直接在包内附带类型（无需单独安装 `@types`）：

```bash
# 这些库内置了 TypeScript 类型，直接安装即可
npm install openai              # 内置类型，无需 @types/openai
npm install @anthropic-ai/sdk   # 内置类型
npm install zod                 # 内置类型
npm install axios               # 内置类型
npm install typescript          # 自然包含自身类型
```

判断方法：查看 `node_modules/xxx/package.json` 是否有 `"types"` 或 `"typings"` 字段。

### 12.3.5 类型缺失时的快速方案

当某个包既无内置类型、也无 `@types` 包时，最快的临时方案：

```typescript
// 方案一：在文件顶部用 @ts-ignore 忽略（不推荐，影响范围太广）
// @ts-ignore
import weirdLib from "weird-lib";

// 方案二：声明模块为 any（项目根目录下新建 declarations.d.ts）
declare module "weird-lib";
// 导入后，weirdLib 的类型为 any

// 方案三（推荐）：给出基本形状声明，逐步完善
declare module "weird-lib" {
  export function doSomething(input: string): Promise<string>;
  export const version: string;
}
```

---

## 12.4 模块声明与全局声明

### 12.4.1 模块声明（declare module）

当一个 npm 包没有类型定义时，可以在 `.d.ts` 文件中用 `declare module` 为其提供完整声明：

```typescript
// types/legacy-ai-sdk.d.ts
// 为假设中的 "legacy-ai-sdk" 包编写声明

declare module "legacy-ai-sdk" {
  // 导出接口
  export interface ChatOptions {
    model: string;
    prompt: string;
    temperature?: number;
    maxLength?: number;
    stopSequences?: string[];
  }

  export interface ChatResult {
    text: string;
    finishReason: "stop" | "length" | "content_filter";
    tokenUsage: {
      prompt: number;
      completion: number;
      total: number;
    };
  }

  // 导出函数
  export function complete(options: ChatOptions): Promise<ChatResult>;
  export function completeStream(
    options: ChatOptions,
    onChunk: (chunk: string) => void
  ): Promise<ChatResult>;

  // 导出类
  export class AIClient {
    constructor(apiKey: string, baseUrl?: string);
    complete(options: ChatOptions): Promise<ChatResult>;
    setDefaultModel(model: string): void;
  }

  // 导出常量
  export const VERSION: string;
  export const SUPPORTED_MODELS: readonly string[];
}
```

### 12.4.2 通配符模块声明

当项目中需要导入特殊文件（如 `.svg`、`.css`、`.mdx`）时，通配符声明非常有用：

```typescript
// types/assets.d.ts

// SVG 文件导入类型
declare module "*.svg" {
  import type { FC, SVGProps } from "react";
  const ReactComponent: FC<SVGProps<SVGSVGElement>>;
  export default ReactComponent;
  export const ReactComponent: FC<SVGProps<SVGSVGElement>>;
}

// CSS 模块
declare module "*.module.css" {
  const styles: Record<string, string>;
  export default styles;
}

// 图片文件
declare module "*.png" {
  const url: string;
  export default url;
}
declare module "*.jpg" {
  const url: string;
  export default url;
}

// MDX 文件（文档 AI 应用常用）
declare module "*.mdx" {
  import type { ComponentType } from "react";
  const Component: ComponentType<{ components?: Record<string, ComponentType> }>;
  export default Component;
}

// JSON 文件（直接导入时）
declare module "*.json" {
  const value: unknown;
  export default value;
}
```

### 12.4.3 全局声明与模块声明的区别

这是最容易混淆的概念之一，核心规则如下：

| 文件类型 | 判断标准 | `declare X` 的效果 |
|---------|---------|-------------------|
| **脚本文件** | 无顶层 `import`/`export` | 直接添加到全局作用域 |
| **模块文件** | 有顶层 `import`/`export` | 局部声明，不影响全局 |
| **模块文件** | 使用 `declare global {}` | 显式添加到全局作用域 |

```typescript
// ============================================================
// globals-script.d.ts（脚本文件：无 import/export）
// ============================================================
// 这里的所有声明自动成为全局声明

declare const AI_VERSION: string;          // 全局 ✓
declare function createClient(): void;      // 全局 ✓

interface GlobalAIConfig {                 // 全局 ✓
  apiKey: string;
}
```

```typescript
// ============================================================
// globals-module.d.ts（模块文件：有 export {}）
// ============================================================
// 有 export 使其成为模块文件
// 必须用 declare global {} 才能添加全局声明

export {};  // 标记为模块文件

declare global {
  const AI_VERSION: string;               // 全局 ✓
  function createClient(): void;           // 全局 ✓

  interface GlobalAIConfig {             // 全局 ✓
    apiKey: string;
  }

  // 扩展 Window 接口
  interface Window {
    aiHelper: {
      ask: (prompt: string) => Promise<string>;
    };
  }
}

// 模块级别的声明（不是全局的）
export interface LocalType {             // 仅模块内 ✓
  value: string;
}
```

### 12.4.4 环境模块与路径映射

当项目中存在非标准模块路径（如虚拟路径、代码生成路径）时，可以用声明配合路径映射：

```typescript
// types/virtual-modules.d.ts

// 声明构建工具生成的虚拟模块
declare module "virtual:ai-prompts" {
  export interface PromptTemplate {
    id: string;
    template: string;
    variables: string[];
  }

  export function getPrompt(id: string): PromptTemplate | undefined;
  export const prompts: PromptTemplate[];
}

// Vite 插件生成的虚拟模块
declare module "virtual:ai-config" {
  export const models: string[];
  export const defaultModel: string;
  export const apiBaseUrl: string;
}
```

```json
// tsconfig.json 配合路径映射
{
  "compilerOptions": {
    "paths": {
      "virtual:*": ["./types/virtual-modules.d.ts"]
    }
  }
}
```

---

## 12.5 编写自定义声明文件

### 12.5.1 声明文件的最佳实践

编写高质量声明文件的要点：

1. **精确而非宽泛**：避免 `any`，尽可能使用具体类型
2. **完整覆盖公开 API**：至少覆盖库的所有 `export`
3. **使用重载（Overloads）** 表达复杂的多态行为
4. **JSDoc 注释**：复制原库的文档注释，IDE 能显示

```typescript
// 精确类型 vs 宽泛类型的对比

// 不推荐：过于宽泛
declare function embed(input: any): any;

// 推荐：精确类型
declare function embed(input: string): Promise<number[]>;
declare function embed(input: string[]): Promise<number[][]>;
// （函数重载：单个字符串 → 单个向量；字符串数组 → 向量数组）
```

### 12.5.2 为复杂 JS 库编写声明文件

下面是一个完整示例：为假设的 `ai-toolkit` 库编写声明文件。

```typescript
// types/ai-toolkit.d.ts
// 为 "ai-toolkit" 库编写完整声明文件

declare module "ai-toolkit" {
  // ============================================================
  // 基础类型定义
  // ============================================================

  export type Role = "system" | "user" | "assistant" | "function";

  export interface Message {
    role: Role;
    content: string;
    name?: string;          // function 角色时使用
    functionCall?: {        // 模型请求调用函数时
      name: string;
      arguments: string;    // JSON 字符串
    };
  }

  export interface Usage {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  }

  export interface CompletionChoice {
    index: number;
    message: Message;
    finishReason: "stop" | "length" | "function_call" | "content_filter" | null;
  }

  // ============================================================
  // 请求参数
  // ============================================================

  export interface CompletionOptions {
    model: string;
    messages: Message[];
    temperature?: number;
    topP?: number;
    n?: number;                       // 生成 n 个结果
    stream?: boolean;
    stop?: string | string[];
    maxTokens?: number;
    presencePenalty?: number;
    frequencyPenalty?: number;
    logitBias?: Record<string, number>;
    user?: string;                    // 用于监控和限流
    functions?: FunctionDefinition[];
    functionCall?: "auto" | "none" | { name: string };
  }

  export interface FunctionDefinition {
    name: string;
    description?: string;
    parameters: {
      type: "object";
      properties: Record<string, {
        type: string;
        description?: string;
        enum?: string[];
        items?: { type: string };
      }>;
      required?: string[];
    };
  }

  // ============================================================
  // 响应类型
  // ============================================================

  export interface CompletionResponse {
    id: string;
    object: "chat.completion";
    created: number;
    model: string;
    choices: CompletionChoice[];
    usage: Usage;
  }

  // 流式响应的 chunk 类型
  export interface CompletionChunk {
    id: string;
    object: "chat.completion.chunk";
    created: number;
    model: string;
    choices: Array<{
      index: number;
      delta: Partial<Message>;
      finishReason: string | null;
    }>;
  }

  // ============================================================
  // 客户端类（函数重载处理流式与非流式）
  // ============================================================

  export class AIToolkit {
    constructor(options: {
      apiKey: string;
      baseUrl?: string;
      timeout?: number;
      maxRetries?: number;
      defaultModel?: string;
    });

    // 非流式调用
    complete(
      options: CompletionOptions & { stream?: false }
    ): Promise<CompletionResponse>;

    // 流式调用
    complete(
      options: CompletionOptions & { stream: true }
    ): Promise<AsyncIterable<CompletionChunk>>;

    // 通用重载（stream 未指定时）
    complete(
      options: CompletionOptions
    ): Promise<CompletionResponse | AsyncIterable<CompletionChunk>>;

    // 快捷方法
    ask(
      prompt: string,
      options?: Partial<Omit<CompletionOptions, "messages">>
    ): Promise<string>;

    // 配置管理
    setDefaultModel(model: string): void;
    getDefaultModel(): string;
  }

  // ============================================================
  // 独立工具函数
  // ============================================================

  export function countTokens(
    text: string,
    model?: string
  ): number;

  export function countTokensBatch(
    texts: string[],
    model?: string
  ): number[];

  export function truncateToTokenLimit(
    text: string,
    maxTokens: number,
    model?: string
  ): string;

  // ============================================================
  // 嵌入（Embedding）相关
  // ============================================================

  export namespace Embeddings {
    export interface EmbeddingOptions {
      model?: string;
      encodingFormat?: "float" | "base64";
    }

    export interface EmbeddingResponse {
      data: Array<{
        embedding: number[];
        index: number;
      }>;
      model: string;
      usage: { promptTokens: number; totalTokens: number };
    }

    export function encode(
      input: string,
      options?: EmbeddingOptions
    ): Promise<number[]>;

    export function encodeBatch(
      inputs: string[],
      options?: EmbeddingOptions
    ): Promise<EmbeddingResponse>;

    export function cosineSimilarity(
      vec1: number[],
      vec2: number[]
    ): number;

    export function findMostSimilar(
      query: number[],
      candidates: number[][],
      topK?: number
    ): Array<{ index: number; score: number }>;
  }

  // ============================================================
  // 错误类型
  // ============================================================

  export class AIToolkitError extends Error {
    code: string;
    statusCode?: number;
    constructor(message: string, code: string, statusCode?: number);
  }

  export class AuthError extends AIToolkitError {
    constructor(message?: string);
  }

  export class RateLimitError extends AIToolkitError {
    retryAfter?: number;
    constructor(message?: string, retryAfter?: number);
  }

  export class ModelError extends AIToolkitError {
    model: string;
    constructor(message: string, model: string);
  }

  // 类型守卫函数
  export function isAIToolkitError(error: unknown): error is AIToolkitError;
  export function isRateLimitError(error: unknown): error is RateLimitError;
}
```

### 12.5.3 使用 JSDoc 增强声明文件

在声明文件中添加 JSDoc 注释，IDE 可以在悬停时显示文档：

```typescript
declare module "ai-toolkit" {
  /**
   * 主客户端类，封装所有 AI API 交互。
   *
   * @example
   * ```typescript
   * const client = new AIToolkit({ apiKey: process.env.API_KEY });
   * const response = await client.ask("解释量子纠缠");
   * console.log(response);
   * ```
   */
  export class AIToolkit {
    /**
     * 创建 AI Toolkit 客户端实例。
     *
     * @param options - 客户端配置选项
     * @param options.apiKey - API 密钥（必填）
     * @param options.baseUrl - 自定义 API 基础 URL（可选，用于代理或自部署）
     * @param options.timeout - 请求超时时间（毫秒），默认 30000
     * @param options.maxRetries - 遇到限流时的最大重试次数，默认 3
     */
    constructor(options: {
      apiKey: string;
      baseUrl?: string;
      timeout?: number;
      maxRetries?: number;
    });

    /**
     * 发送聊天补全请求。
     *
     * 支持流式与非流式两种模式：
     * - 传入 `stream: false`（或不传）：返回完整响应
     * - 传入 `stream: true`：返回异步迭代器，逐 chunk 处理
     *
     * @param options - 请求选项
     * @returns 非流式时返回 CompletionResponse；流式时返回 AsyncIterable<CompletionChunk>
     *
     * @example
     * ```typescript
     * // 非流式
     * const result = await client.complete({
     *   model: "gpt-4",
     *   messages: [{ role: "user", content: "你好" }]
     * });
     *
     * // 流式
     * const stream = await client.complete({
     *   model: "gpt-4",
     *   messages: [{ role: "user", content: "你好" }],
     *   stream: true
     * });
     * for await (const chunk of stream) {
     *   process.stdout.write(chunk.choices[0].delta.content ?? "");
     * }
     * ```
     */
    complete(options: CompletionOptions & { stream: true }): Promise<AsyncIterable<CompletionChunk>>;
    complete(options: CompletionOptions & { stream?: false }): Promise<CompletionResponse>;
    complete(options: CompletionOptions): Promise<CompletionResponse | AsyncIterable<CompletionChunk>>;

    /**
     * 便捷方法：发送单条用户消息并返回文本响应。
     *
     * @param prompt - 用户输入的文本
     * @param options - 可选的补全参数（除 messages 外的所有参数）
     * @returns 模型生成的文本内容
     */
    ask(
      prompt: string,
      options?: Partial<Omit<CompletionOptions, "messages">>
    ): Promise<string>;
  }
}
```

---

## 本章小结

| 概念 | 核心要点 | 适用场景 |
|------|---------|---------|
| **`.d.ts` 文件** | 纯类型声明，不产生 JS 输出；三种来源：自动生成、`@types`、手写 | 为 JS 库提供类型，库开发时随源码发布 |
| **`declare var/let/const`** | 声明全局变量（已在运行时存在） | 构建注入变量、CDN 脚本全局变量 |
| **`declare function`** | 声明全局函数 | 全局工具函数、浏览器 API 扩展 |
| **`declare class`** | 声明全局类 | 老式 OOP 库的全局类 |
| **`declare namespace`** | 声明命名空间对象，支持嵌套 | UMD 库、将所有 API 挂载到一个对象上的库 |
| **`declare module "pkg"`** | 为整个 npm 包提供类型声明 | 无类型定义的第三方 JS 包 |
| **`declare module "*.ext"`** | 通配符模块声明 | SVG/CSS/图片等非 JS 资源的导入 |
| **`declare global {}`** | 在模块文件中扩展全局作用域 | 扩展 Window、全局接口增强 |
| **`@types/xxx`** | DefinitelyTyped 社区类型包 | 未内置类型的流行 JS 库（如 `@types/node`） |
| **模块增强** | `declare module "pkg"` + `import "pkg"` 扩展已有模块 | 给框架类型添加自定义字段（如 Express Request） |

**使用决策树**：

```
需要类型定义？
├── 自己写的 TypeScript 代码 → 开启 declaration: true，自动生成
├── 流行的 JS 库（如 lodash）→ 安装 @types/xxx
├── 现代 TS-first 库（如 openai）→ 直接安装，内置类型
└── 无类型的老旧 JS 库
    ├── 临时应付 → declare module "xxx";（类型为 any）
    └── 正式项目 → 手写 .d.ts 声明文件（参考本章 12.5 节）
```

---

## AI 应用实战：第三方 AI SDK 类型补充

在实际 AI 应用开发中，经常会遇到某些 SDK 缺少类型、类型不完整或类型不够精确的情况。本节展示如何为三个真实场景编写自定义声明文件。

### 场景一：为无类型的 Python-style SDK 编写声明

假设团队使用了一个内部 Python 服务封装的 HTTP SDK `@company/ai-gateway`，没有 TypeScript 类型：

```typescript
// types/@company/ai-gateway.d.ts

declare module "@company/ai-gateway" {
  // ============================================================
  // 基础数据类型
  // ============================================================

  export type ModelProvider = "openai" | "anthropic" | "gemini" | "local";
  export type MessageRole = "system" | "user" | "assistant";
  export type FinishReason = "stop" | "length" | "filtered" | "error";

  export interface GatewayMessage {
    role: MessageRole;
    content: string;
    /** 可选的消息 ID，用于追踪对话历史 */
    id?: string;
    /** 消息创建时间戳（Unix 毫秒） */
    timestamp?: number;
  }

  export interface GatewayUsage {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
    /** 缓存命中的 token 数（节省费用） */
    cachedTokens?: number;
    /** 本次请求的费用（美分） */
    costCents?: number;
  }

  // ============================================================
  // 请求/响应类型
  // ============================================================

  export interface ChatRequest {
    messages: GatewayMessage[];
    model?: string;
    provider?: ModelProvider;
    temperature?: number;
    maxTokens?: number;
    systemPrompt?: string;
    /** 启用响应缓存（相同输入返回缓存结果） */
    useCache?: boolean;
    /** 请求超时（秒） */
    timeoutSeconds?: number;
    /** 请求元数据（用于日志和计费） */
    metadata?: Record<string, string>;
  }

  export interface ChatResponse {
    content: string;
    model: string;
    provider: ModelProvider;
    finishReason: FinishReason;
    usage: GatewayUsage;
    /** 响应是否来自缓存 */
    fromCache: boolean;
    /** 请求唯一 ID，用于追踪和调试 */
    requestId: string;
    /** 服务端处理时间（毫秒） */
    latencyMs: number;
  }

  export interface StreamChunk {
    delta: string;
    requestId: string;
    done: boolean;
    /** 仅在 done=true 时存在 */
    finalUsage?: GatewayUsage;
  }

  // ============================================================
  // 嵌入（Embedding）
  // ============================================================

  export interface EmbedRequest {
    texts: string | string[];
    model?: string;
    /** 是否对向量进行 L2 归一化 */
    normalize?: boolean;
  }

  export interface EmbedResponse {
    embeddings: number[][];
    model: string;
    usage: { totalTokens: number };
  }

  // ============================================================
  // 主客户端类
  // ============================================================

  export interface GatewayClientOptions {
    /** 网关服务地址 */
    endpoint: string;
    /** 服务认证令牌 */
    authToken: string;
    /** 所属应用/服务名称（用于计费隔离） */
    appName: string;
    /** 默认超时（秒），默认 30 */
    defaultTimeout?: number;
    /** 连接池大小，默认 10 */
    poolSize?: number;
  }

  export class GatewayClient {
    constructor(options: GatewayClientOptions);

    /** 发送聊天请求，返回完整响应 */
    chat(request: ChatRequest): Promise<ChatResponse>;

    /** 发送聊天请求，返回流式迭代器 */
    chatStream(request: ChatRequest): AsyncIterable<StreamChunk>;

    /** 生成文本嵌入向量 */
    embed(request: EmbedRequest): Promise<EmbedResponse>;

    /** 检查服务健康状态 */
    healthCheck(): Promise<{ status: "ok" | "degraded" | "down"; latencyMs: number }>;

    /** 获取可用模型列表 */
    listModels(): Promise<Array<{
      id: string;
      provider: ModelProvider;
      maxTokens: number;
      supportsStream: boolean;
    }>>;
  }

  /** 创建 GatewayClient 的工厂函数（单例模式） */
  export function createClient(options: GatewayClientOptions): GatewayClient;

  /** 获取已创建的单例客户端（createClient 后可用） */
  export function getClient(): GatewayClient;

  // ============================================================
  // 错误类型
  // ============================================================

  export class GatewayError extends Error {
    code: string;
    httpStatus?: number;
    requestId?: string;
  }

  export class AuthError extends GatewayError {}
  export class QuotaExceededError extends GatewayError {
    /** 配额重置时间（Unix 时间戳） */
    resetAt: number;
  }
  export class ModelUnavailableError extends GatewayError {
    model: string;
  }

  export function isGatewayError(error: unknown): error is GatewayError;
}
```

### 场景二：扩展现有 SDK 的类型（模块增强）

Anthropic SDK 内置了类型，但假设我们的业务需要在消息上附加自定义元数据字段。通过模块增强，在不 fork SDK 的情况下扩展类型：

```typescript
// types/anthropic-augmentation.d.ts
// 扩展 @anthropic-ai/sdk 的类型，添加业务特有字段

import "@anthropic-ai/sdk";

declare module "@anthropic-ai/sdk" {
  // 扩展消息参数接口，添加业务追踪字段
  interface MessageCreateParamsNonStreaming {
    /** 业务请求追踪 ID，透传至响应 */
    _traceId?: string;
    /** 所属用户 ID，用于按用户计费统计 */
    _userId?: string;
    /** 功能标识符，用于 A/B 测试 */
    _feature?: string;
  }

  // 扩展响应类型，添加内部字段
  interface Message {
    /** 服务端分配的追踪 ID */
    _serverTraceId?: string;
    /** 此次请求的路由节点 */
    _routeNode?: string;
  }
}

// 使用示例（仅作说明，不产生 JS）
export {};
```

### 场景三：为流式响应工具函数库编写声明

一个内部的流处理工具 `stream-utils`，提供处理 AI 流式响应的工具函数：

```typescript
// types/stream-utils.d.ts

declare module "@internal/stream-utils" {
  // ============================================================
  // 核心类型
  // ============================================================

  export type StreamSource<T> = AsyncIterable<T> | ReadableStream<T>;

  export interface StreamOptions {
    /** 超时时间（毫秒） */
    timeout?: number;
    /** 遇到错误时是否继续（跳过错误 chunk） */
    skipErrors?: boolean;
    /** 缓冲区大小（chunk 数量） */
    bufferSize?: number;
  }

  export interface StreamStats {
    chunkCount: number;
    totalBytes: number;
    durationMs: number;
    errorCount: number;
  }

  // ============================================================
  // 转换函数（函数重载）
  // ============================================================

  /**
   * 将流的所有 chunk 合并为单个值。
   * - string chunk 合并为 string
   * - 其他类型合并为数组
   */
  export function collect(source: StreamSource<string>, options?: StreamOptions): Promise<string>;
  export function collect<T>(source: StreamSource<T>, options?: StreamOptions): Promise<T[]>;

  /**
   * 限制流的速率（用于演示或防止 UI 更新过快）
   */
  export function throttle<T>(
    source: StreamSource<T>,
    delayMs: number
  ): AsyncIterable<T>;

  /**
   * 将 ReadableStream（浏览器 API）转换为 AsyncIterable
   */
  export function fromReadableStream<T>(stream: ReadableStream<T>): AsyncIterable<T>;

  /**
   * 将文本流按行分割
   */
  export function splitLines(source: StreamSource<string>): AsyncIterable<string>;

  /**
   * 解析 Server-Sent Events（SSE）格式的流
   */
  export function parseSSE(source: StreamSource<string>): AsyncIterable<{
    event?: string;
    data: string;
    id?: string;
    retry?: number;
  }>;

  /**
   * 流式 JSON 解析（每个 chunk 是完整的 JSON 行）
   */
  export function parseJSONLines<T = unknown>(
    source: StreamSource<string>
  ): AsyncIterable<T>;

  /**
   * 收集流统计信息（不消耗流，返回增强版迭代器）
   */
  export function withStats<T>(
    source: StreamSource<T>
  ): AsyncIterable<T> & { getStats(): StreamStats };

  /**
   * 将流分叉为多个独立消费者（每个消费者都能完整消费）
   */
  export function tee<T>(
    source: StreamSource<T>,
    count: number
  ): AsyncIterable<T>[];

  // ============================================================
  // 错误处理
  // ============================================================

  export class StreamTimeoutError extends Error {
    timeoutMs: number;
  }

  export class StreamAbortError extends Error {}
}
```

### 完整使用示例

```typescript
// ai-service.ts
// 综合运用上述三个自定义声明文件

import { GatewayClient, createClient, isGatewayError, QuotaExceededError }
  from "@company/ai-gateway";
import { collect, parseSSE, withStats }
  from "@internal/stream-utils";

// ============================================================
// 初始化（类型完整，无任何 any）
// ============================================================

const gateway = createClient({
  endpoint: process.env.GATEWAY_ENDPOINT!,
  authToken: process.env.GATEWAY_TOKEN!,
  appName: "customer-support-bot",
  defaultTimeout: 60,
});

// ============================================================
// 非流式对话（类型安全的 ChatResponse）
// ============================================================

async function answerQuestion(
  question: string,
  userId: string
): Promise<{ answer: string; tokens: number; latency: number }> {
  try {
    const response = await gateway.chat({
      messages: [
        { role: "system", content: "你是客服助手，请简洁、准确地回答用户问题。" },
        { role: "user", content: question },
      ],
      provider: "anthropic",
      maxTokens: 512,
      metadata: { userId, feature: "support-chat" },
    });

    // response 类型完整：ChatResponse
    return {
      answer: response.content,
      tokens: response.usage.totalTokens,
      latency: response.latencyMs,
    };
  } catch (error) {
    if (isGatewayError(error)) {
      if (error instanceof QuotaExceededError) {
        const resetDate = new Date(error.resetAt * 1000);
        throw new Error(`配额已用尽，将于 ${resetDate.toLocaleString()} 重置`);
      }
      console.error(`网关错误 [${error.code}]: ${error.message}`);
    }
    throw error;
  }
}

// ============================================================
// 流式对话（类型安全的 StreamChunk）
// ============================================================

async function streamAnswer(
  question: string,
  onChunk: (text: string) => void
): Promise<{ totalTokens: number }> {
  const stream = gateway.chatStream({
    messages: [{ role: "user", content: question }],
    provider: "openai",
  });

  // 用 withStats 包装，收集统计信息
  const trackedStream = withStats(stream);

  for await (const chunk of trackedStream) {
    // chunk 类型：StreamChunk
    if (!chunk.done) {
      onChunk(chunk.delta);
    }

    if (chunk.done && chunk.finalUsage) {
      // finalUsage 类型：GatewayUsage
      const stats = trackedStream.getStats();
      console.log(`流统计: ${stats.chunkCount} chunks, ${stats.durationMs}ms`);
      return { totalTokens: chunk.finalUsage.totalTokens };
    }
  }

  return { totalTokens: 0 };
}

// ============================================================
// 嵌入与语义搜索
// ============================================================

async function semanticSearch(
  query: string,
  documents: string[]
): Promise<Array<{ document: string; score: number }>> {
  // embedResponse 类型：EmbedResponse
  const embedResponse = await gateway.embed({
    texts: [query, ...documents],
    normalize: true,
  });

  const [queryVec, ...docVecs] = embedResponse.embeddings;

  // 计算余弦相似度（向量已归一化，点积即余弦相似度）
  return documents
    .map((doc, i) => ({
      document: doc,
      score: queryVec.reduce((sum, val, j) => sum + val * docVecs[i][j], 0),
    }))
    .sort((a, b) => b.score - a.score);
}

export { answerQuestion, streamAnswer, semanticSearch };
```

### 类型验证

得益于完整的声明文件，TypeScript 能捕获以下错误：

```typescript
// 错误示例：类型系统将在编译时报告这些问题

await gateway.chat({
  messages: [{ role: "god", content: "..." }],
  //              ^^^^^ 错误：类型 "god" 不能赋值给 MessageRole
});

await gateway.chat({
  messages: [],
  provider: "azure",
  //        ^^^^^^^ 错误：类型 "azure" 不能赋值给 ModelProvider
});

const response = await gateway.chat({ messages: [] });
console.log(response.text);
//                   ^^^^ 错误：ChatResponse 上不存在属性 "text"（正确是 content）

const chunks = gateway.chatStream({ messages: [] });
for await (const chunk of chunks) {
  console.log(chunk.content);
  //                ^^^^^^^ 错误：StreamChunk 上不存在属性 "content"（正确是 delta）
}
```

---

## 练习题

### 基础题

**练习 12-1**：有以下 JavaScript 对象被通过 `<script>` 标签全局注入到页面，编写一个 `.d.ts` 声明文件，让 TypeScript 能够识别并正确提示其类型：

```javascript
// 由 CDN 脚本注入的全局对象
window.ChatWidget = {
  version: "2.1.0",
  isLoaded: false,

  init: function(config) {
    // config: { apiKey, position, theme, language }
  },

  open: function() {},
  close: function() {},

  sendMessage: function(text, options) {
    // options: { role?, metadata? }
    // 返回 Promise<{ messageId: string, timestamp: number }>
  },

  on: function(event, callback) {
    // event: "open" | "close" | "message" | "error"
  },

  destroy: function() {},
};
```

**练习 12-2**：`@types/node` 提供了 Node.js 的类型，但某些项目只需要其中的部分内容。在 `tsconfig.json` 中配置：(a) 只加载 `@types/node` 和 `@types/jest` 的全局声明；(b) 项目自定义的类型目录 `./src/types` 也应被扫描。同时解释为什么有时需要限制 `types` 白名单。

### 进阶题

**练习 12-3**：以下是一个假设的向量数据库 SDK `vecdb-js` 的使用示例（JavaScript），没有类型定义。请编写完整的 `declare module "vecdb-js"` 声明：

```javascript
// vecdb-js 的 JavaScript 使用示例
const { VecDB } = require("vecdb-js");

const db = new VecDB({
  url: "http://localhost:6333",
  apiKey: "optional-key",
});

// 创建集合
await db.collections.create("my-docs", {
  vectorSize: 1536,
  distance: "cosine",    // "cosine" | "euclidean" | "dot"
  onDiskPayload: true,
});

// 插入向量
await db.collections.insert("my-docs", [
  {
    id: "doc-1",
    vector: [0.1, 0.2, ...],
    payload: { title: "文档标题", text: "内容", tags: ["tag1"] }
  }
]);

// 搜索
const results = await db.collections.search("my-docs", {
  vector: [0.1, 0.2, ...],
  topK: 5,
  filter: { tags: { $contains: "tag1" } },
  withPayload: true,
});
// results: [{ id, score, payload? }]

// 删除
await db.collections.delete("my-docs", { ids: ["doc-1"] });
await db.collections.drop("my-docs");
```

**练习 12-4**：项目使用了 Express，需要在 `Request` 对象上添加三个自定义字段：
- `requestId: string`（由中间件注入）
- `aiSession: { sessionId: string; model: string; messageCount: number } | undefined`（由 AI 会话中间件注入）
- `user: { id: string; tier: "free" | "pro" | "enterprise"; monthlyTokensUsed: number } | undefined`（由认证中间件注入）

使用模块增强（不修改 Express 源码）实现这个扩展，并展示如何在路由处理函数中使用这些类型。

### 挑战题

**练习 12-5**：团队开发了一个内部的"提示词模板引擎" `prompt-engine`（纯 JavaScript），API 如下。请为其编写完整的声明文件，要求类型精确（避免 `any`），并使用泛型使模板变量类型安全：

```javascript
// prompt-engine 的 JavaScript API

// 创建模板（变量用 {{变量名}} 标记）
const template = PromptEngine.create(`
  你是 {{role}}。
  用户提问：{{question}}
  请用 {{language}} 回答，字数不超过 {{maxWords}} 字。
`);

// 渲染（传入变量值）
const rendered = template.render({
  role: "TypeScript 专家",
  question: "什么是声明文件？",
  language: "中文",
  maxWords: 200,
});
// rendered: string

// 验证变量完整性
const isValid = template.validate({
  role: "...",
  question: "...",
  // 缺少 language 和 maxWords → 返回 false
});
// isValid: false

// 获取模板信息
const info = template.info();
// info: { variables: ["role", "question", "language", "maxWords"], charCount: number }

// 组合多个模板
const combined = PromptEngine.combine([systemTemplate, userTemplate], "\n\n---\n\n");
// combined: Template

// 从文件加载（Node.js 环境）
const fromFile = await PromptEngine.fromFile("./prompts/system.txt");
```

**挑战要求**：使用模板字面量类型（Template Literal Types）和 `infer` 实现类型安全的变量提取，使 `render()` 方法的参数类型能精确反映模板中使用的变量名（而不是 `Record<string, unknown>`）。

---

## 练习答案

### 练习 12-1 答案

```typescript
// types/chat-widget.d.ts

// 使用 declare global 添加到全局 window 类型
export {};

declare global {
  interface ChatWidgetConfig {
    /** API 密钥（必填） */
    apiKey: string;
    /** 悬浮按钮位置 */
    position?: "bottom-right" | "bottom-left" | "top-right" | "top-left";
    /** 主题 */
    theme?: "light" | "dark" | "auto";
    /** 界面语言 */
    language?: string;
  }

  interface ChatWidgetMessage {
    role?: "user" | "assistant";
    metadata?: Record<string, unknown>;
  }

  interface ChatWidgetSendResult {
    messageId: string;
    timestamp: number;
  }

  type ChatWidgetEvent = "open" | "close" | "message" | "error";

  interface ChatWidgetInstance {
    /** SDK 版本号 */
    version: string;
    /** 是否已完成初始化 */
    isLoaded: boolean;

    /**
     * 初始化聊天组件
     * @param config 初始化配置
     */
    init(config: ChatWidgetConfig): void;

    /** 打开聊天窗口 */
    open(): void;

    /** 关闭聊天窗口 */
    close(): void;

    /**
     * 发送消息
     * @param text 消息文本
     * @param options 消息选项
     */
    sendMessage(
      text: string,
      options?: ChatWidgetMessage
    ): Promise<ChatWidgetSendResult>;

    /**
     * 监听聊天组件事件
     * @param event 事件名称
     * @param callback 事件回调
     */
    on(event: ChatWidgetEvent, callback: (...args: unknown[]) => void): void;

    /** 销毁聊天组件，释放资源 */
    destroy(): void;
  }

  interface Window {
    ChatWidget: ChatWidgetInstance;
  }
}
```

### 练习 12-2 答案

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "CommonJS",
    "typeRoots": [
      "./node_modules/@types",
      "./src/types"
    ],
    "types": ["node", "jest"]
  }
}
```

**为什么需要限制 `types` 白名单**：

默认情况下，TypeScript 会将 `typeRoots` 下所有 `@types/xxx` 包的**全局声明**自动注入到所有文件中。当项目同时安装了 `@types/node`（提供 `process`、`Buffer` 等 Node.js 全局）和 `@types/browser`（提供 `window`、`document`）时，可能发生命名冲突或不符合运行环境的类型声明（如在 Node.js 项目中出现 `window`）。

通过指定 `"types": ["node", "jest"]`：
- 只有 `@types/node` 和 `@types/jest` 的全局声明被自动引入
- 其他 `@types` 包（如 `@types/express`）仍可通过 `import express from 'express'` 正常使用，只是不会污染全局类型空间
- 减少了 IDE 在全局补全中显示无关的类型提示

### 练习 12-3 答案

```typescript
// types/vecdb-js.d.ts

declare module "vecdb-js" {
  export type DistanceMetric = "cosine" | "euclidean" | "dot";

  // ============================================================
  // 集合操作类型
  // ============================================================

  export interface CreateCollectionOptions {
    /** 向量维度 */
    vectorSize: number;
    /** 距离度量方式 */
    distance: DistanceMetric;
    /** 是否将 payload 存储到磁盘（节省内存） */
    onDiskPayload?: boolean;
    /** 索引配置 */
    indexConfig?: {
      m?: number;
      efConstruct?: number;
    };
  }

  export type VectorPayload = Record<string, unknown>;

  export interface VectorPoint {
    /** 向量唯一 ID */
    id: string;
    /** 向量数据 */
    vector: number[];
    /** 附加的业务数据 */
    payload?: VectorPayload;
  }

  // ============================================================
  // 查询过滤器（支持简单的 MongoDB-like 语法）
  // ============================================================

  type FilterValue = string | number | boolean;
  type FilterCondition = {
    $eq?: FilterValue;
    $ne?: FilterValue;
    $gt?: number;
    $gte?: number;
    $lt?: number;
    $lte?: number;
    $in?: FilterValue[];
    $contains?: FilterValue;
  };

  export type VectorFilter = Record<string, FilterValue | FilterCondition>;

  export interface SearchOptions {
    /** 查询向量 */
    vector: number[];
    /** 返回结果数量，默认 10 */
    topK?: number;
    /** 过滤条件 */
    filter?: VectorFilter;
    /** 是否在结果中包含 payload，默认 false */
    withPayload?: boolean;
    /** 相似度分数阈值（低于此值的结果被过滤掉） */
    scoreThreshold?: number;
  }

  export interface SearchResult {
    id: string;
    score: number;
    payload?: VectorPayload;
    vector?: number[];
  }

  export interface DeleteOptions {
    ids?: string[];
    filter?: VectorFilter;
  }

  // ============================================================
  // 集合操作接口
  // ============================================================

  export interface CollectionOperations {
    /**
     * 创建新集合
     */
    create(name: string, options: CreateCollectionOptions): Promise<void>;

    /**
     * 插入或更新向量（upsert 语义）
     */
    insert(name: string, points: VectorPoint[]): Promise<{ inserted: number }>;

    /**
     * 向量相似度搜索
     */
    search(name: string, options: SearchOptions): Promise<SearchResult[]>;

    /**
     * 删除向量
     */
    delete(name: string, options: DeleteOptions): Promise<{ deleted: number }>;

    /**
     * 删除整个集合
     */
    drop(name: string): Promise<void>;

    /**
     * 获取集合信息
     */
    info(name: string): Promise<{
      name: string;
      vectorSize: number;
      distance: DistanceMetric;
      pointsCount: number;
      status: "green" | "yellow" | "grey";
    }>;

    /**
     * 列出所有集合名称
     */
    list(): Promise<string[]>;
  }

  // ============================================================
  // 主客户端类
  // ============================================================

  export interface VecDBOptions {
    url: string;
    apiKey?: string;
    timeout?: number;
  }

  export class VecDB {
    constructor(options: VecDBOptions);
    collections: CollectionOperations;
    isConnected(): boolean;
    disconnect(): Promise<void>;
  }

  export function createClient(options: VecDBOptions): VecDB;
}
```

### 练习 12-4 答案

```typescript
// types/express-augmentation.d.ts

import "express";

declare module "express-serve-static-core" {
  // Express 内部使用 express-serve-static-core 定义核心类型
  // 在这里扩展更可靠

  interface Request {
    /**
     * 请求唯一 ID，由 request-id 中间件注入。
     * 所有请求都有此字段（中间件在应用最早阶段注入）。
     */
    requestId: string;

    /**
     * AI 会话信息，由 AI 会话中间件注入。
     * 仅在调用了 AI 功能的路由上存在。
     */
    aiSession?: {
      sessionId: string;
      model: string;
      messageCount: number;
    };

    /**
     * 已认证的用户信息，由认证中间件注入。
     * 未登录用户为 undefined。
     */
    user?: {
      id: string;
      tier: "free" | "pro" | "enterprise";
      monthlyTokensUsed: number;
    };
  }
}

export {};
```

```typescript
// routes/chat.ts —— 在路由中使用扩展后的类型

import { Router, Request, Response } from "express";

const router = Router();

router.post("/chat", async (req: Request, res: Response) => {
  // req.requestId 类型：string（始终存在）
  console.log(`处理请求 ${req.requestId}`);

  // req.user 类型：{ id: string; tier: "free" | "pro" | "enterprise"; monthlyTokensUsed: number } | undefined
  if (!req.user) {
    return res.status(401).json({ error: "请先登录" });
  }

  // 限制免费用户的 token 使用量
  if (req.user.tier === "free" && req.user.monthlyTokensUsed > 10000) {
    return res.status(429).json({ error: "本月免费额度已用尽，请升级到 Pro 套餐" });
  }

  // req.aiSession 类型：{ sessionId: string; model: string; messageCount: number } | undefined
  const sessionId = req.aiSession?.sessionId ?? "new-session";
  const model = req.aiSession?.model ?? "gpt-4o-mini";

  res.json({
    requestId: req.requestId,
    userId: req.user.id,
    sessionId,
    model,
  });
});

export default router;
```

### 练习 12-5 答案

```typescript
// types/prompt-engine.d.ts

declare module "prompt-engine" {
  // ============================================================
  // 核心类型：从模板字符串提取变量名
  // ============================================================

  /**
   * 从模板字符串中提取所有 {{变量名}} 的变量名组成联合类型。
   *
   * 示例：
   *   ExtractVariables<"你好 {{name}}，你是 {{role}}"> → "name" | "role"
   */
  type ExtractVariables<T extends string> =
    T extends `${string}{{${infer VarName}}}${infer Rest}`
      ? VarName | ExtractVariables<Rest>
      : never;

  /**
   * 根据变量名联合类型生成 Record 类型（所有变量为必填 string）。
   *
   * 示例：
   *   VariableRecord<"name" | "role"> → { name: string; role: string }
   */
  type VariableRecord<Vars extends string> = {
    [K in Vars]: string;
  };

  // ============================================================
  // 模板实例类型
  // ============================================================

  /**
   * 泛型模板类型，T 为模板字符串字面量类型。
   * 通过 T 推导出所有变量名，使 render/validate 参数类型安全。
   */
  export interface Template<T extends string> {
    /**
     * 渲染模板，用提供的变量替换所有 {{变量名}}。
     * @param variables 变量值映射（所有变量名均为必填）
     */
    render(variables: VariableRecord<ExtractVariables<T>>): string;

    /**
     * 验证变量是否完整提供。
     * @param variables 待验证的变量映射（使用 Partial 允许缺失）
     */
    validate(variables: Partial<VariableRecord<ExtractVariables<T>>>): boolean;

    /**
     * 获取模板元信息。
     */
    info(): {
      variables: ExtractVariables<T>[];
      charCount: number;
      /** 原始模板字符串 */
      source: T;
    };
  }

  // ============================================================
  // PromptEngine 命名空间
  // ============================================================

  export namespace PromptEngine {
    /**
     * 创建类型安全的模板。
     * TypeScript 会自动推断 T 为传入字符串的字面量类型。
     *
     * @example
     * const t = PromptEngine.create("你好 {{name}}");
     * t.render({ name: "World" });    // ✓
     * t.render({ wrong: "key" });     // 编译错误！
     */
    function create<T extends string>(template: T): Template<T>;

    /**
     * 组合多个模板，使用分隔符连接。
     * 返回组合后的通用模板（变量类型合并为 string Record）。
     */
    function combine(
      templates: Template<string>[],
      separator?: string
    ): Template<string>;

    /**
     * 从文件加载模板（Node.js 环境）。
     * 注意：文件内容在编译时未知，故返回通用 Template<string>。
     */
    function fromFile(filePath: string): Promise<Template<string>>;

    /**
     * 从字符串创建模板（create 的别名，语义更明确）。
     */
    function fromString<T extends string>(template: T): Template<T>;
  }

  export default PromptEngine;
}
```

使用示例，展示类型安全效果：

```typescript
import PromptEngine from "prompt-engine";

// TypeScript 自动推导变量类型
const template = PromptEngine.create(`
  你是 {{role}}。
  用户提问：{{question}}
  请用 {{language}} 回答，字数不超过 {{maxWords}} 字。
`);

// 正确使用：所有变量已提供
const rendered = template.render({
  role: "TypeScript 专家",
  question: "什么是声明文件？",
  language: "中文",
  maxWords: "200",
});

// 编译错误示例：
template.render({
  role: "专家",
  question: "问题",
  language: "中文",
  // maxWords: "200",   // 错误：缺少属性 maxWords
});

template.render({
  role: "专家",
  question: "问题",
  language: "中文",
  maxWords: "200",
  extra: "多余字段",   // 错误：对象字面量只能指定已知属性
});

// 获取变量列表（类型为 ("role" | "question" | "language" | "maxWords")[]）
const { variables } = template.info();
console.log(variables); // ["role", "question", "language", "maxWords"]
```

---

*下一章：[第13章：工程化配置与 tsconfig 详解](./13-tsconfig-engineering.md)*
