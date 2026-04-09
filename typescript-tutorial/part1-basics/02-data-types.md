# 第2章：数据类型详解

> **系列定位**：本章是 TypeScript 基础篇的第二章。在第一章了解 TypeScript 的基本概念之后，本章将系统讲解 TypeScript 的类型系统，为后续 AI 应用开发打下坚实基础。

---

## 学习目标

完成本章学习后，你将能够：

1. **掌握所有原始类型**：熟练使用 `string`、`number`、`boolean`、`null`、`undefined`、`symbol`、`bigint` 七种原始类型，并理解各自的适用场景
2. **灵活运用数组与元组**：区分 `Array<T>` 与 `T[]` 的写法，使用只读数组保护数据，用元组精确描述固定结构的数据
3. **正确使用枚举类型**：理解数字枚举、字符串枚举和 `const` 枚举的差异，在 AI 应用的状态管理中合理选择枚举
4. **驾驭特殊类型**：明确 `any`、`unknown`、`never`、`void`、`object` 各自的语义，避免滥用 `any` 降低类型安全性
5. **为 AI API 响应定义类型**：能够为 OpenAI / Anthropic 等 LLM API 的响应结构编写完整的 TypeScript 类型定义

---

## 2.1 原始类型

TypeScript 的原始类型（Primitive Types）直接映射到 JavaScript 的基础值类型，共有七种。

### 2.1.1 string

字符串类型，用于表示文本数据，是 AI 应用中最常用的类型（prompt、response 均为字符串）。

```typescript
// 基本声明
let modelName: string = "gpt-4o";
let systemPrompt: string = `你是一个专业的数学教师，请用简洁的语言解释概念。`;

// 模板字符串
const maxTokens: number = 4096;
const config: string = `模型：${modelName}，最大 token 数：${maxTokens}`;

// 字符串方法（类型推断自动生效）
const upperModel = modelName.toUpperCase(); // 类型推断为 string
const tokenCount = systemPrompt.length;     // 类型推断为 number

// 函数参数类型
function buildPrompt(userMessage: string, context: string): string {
  return `上下文：${context}\n\n用户问题：${userMessage}`;
}
```

### 2.1.2 number

数值类型，TypeScript 中整数和浮点数统一使用 `number`（对应 JavaScript 的 64 位浮点数）。

```typescript
// 整数与浮点数
let temperature: number = 0.7;    // LLM 采样温度，0-1 之间
let topP: number = 0.95;
let maxTokens: number = 2048;
let requestCount: number = 0;

// 各种数字字面量形式
const hexColor: number = 0xff0000;     // 十六进制
const binaryFlag: number = 0b1010;    // 二进制
const octalValue: number = 0o755;     // 八进制
const bigNumber: number = 1_000_000;  // 数字分隔符（可读性）

// NaN 和 Infinity 也是 number 类型
const notANumber: number = NaN;
const positiveInfinity: number = Infinity;

// 计算 token 费用（AI 应用常见场景）
function calculateCost(
  inputTokens: number,
  outputTokens: number,
  inputPricePerMillion: number,
  outputPricePerMillion: number
): number {
  return (inputTokens * inputPricePerMillion + outputTokens * outputPricePerMillion) / 1_000_000;
}

const cost = calculateCost(1000, 500, 2.5, 10.0);
console.log(`本次请求费用：$${cost.toFixed(6)}`);
```

### 2.1.3 boolean

布尔类型，只有 `true` 和 `false` 两个值。

```typescript
// AI 应用中的配置开关
let streamEnabled: boolean = true;
let cacheEnabled: boolean = false;
let debugMode: boolean = process.env.NODE_ENV === "development";

// 函数返回布尔值
function isValidTemperature(temp: number): boolean {
  return temp >= 0 && temp <= 2;
}

function shouldStream(contentLength: number): boolean {
  return contentLength > 500; // 内容较长时启用流式输出
}

// 注意：避免把其他值隐式转换为 boolean
// TypeScript 不允许将 number 直接赋值给 boolean
// let flag: boolean = 1; // 错误！
let flag: boolean = true; // 正确
```

### 2.1.4 null 与 undefined

`null` 表示"有意为之的空值"，`undefined` 表示"变量已声明但未赋值"。

```typescript
// undefined：未赋值的变量
let apiKey: string | undefined;
console.log(apiKey); // undefined

// null：明确表示无值
let lastResponse: string | null = null;

// 函数可选返回
function findCachedResponse(key: string): string | null {
  // 从缓存中查找，没有则返回 null
  const cache = new Map<string, string>();
  return cache.get(key) ?? null;
}

// 处理 null/undefined（可选链 + 空值合并）
const response = findCachedResponse("prompt-hash-123");
const displayText = response ?? "（缓存未命中，正在请求 API...）";

// strictNullChecks 开启时，null 和 undefined 不能赋给其他类型
// let name: string = null;       // 错误！（strictNullChecks 开启时）
let name: string | null = null;   // 正确
```

### 2.1.5 symbol

`symbol` 是唯一且不可变的值，常用于对象属性的唯一键。

```typescript
// 创建唯一 symbol
const REQUEST_ID = Symbol("requestId");
const SESSION_TOKEN = Symbol("sessionToken");

// 用作对象属性键（避免命名冲突）
const metadata = {
  [REQUEST_ID]: "req-abc-123",
  [SESSION_TOKEN]: "tok-xyz-789",
  model: "claude-3-5-sonnet"
};

// 访问 symbol 属性
console.log(metadata[REQUEST_ID]); // "req-abc-123"

// Symbol.for 创建全局共享 symbol
const GLOBAL_CONFIG = Symbol.for("app.config");
```

### 2.1.6 bigint

`bigint` 用于表示任意精度的大整数，在普通 `number` 精度不足时使用。

```typescript
// 声明 bigint（字面量后缀 n）
const maxSafeInt: number = Number.MAX_SAFE_INTEGER; // 9007199254740991
const bigInt: bigint = 9007199254740993n;           // 超出 number 精度范围

// AI 应用场景：统计大量 token 消耗（企业级）
let totalTokensConsumed: bigint = 0n;
totalTokensConsumed += 1_500_000n;
totalTokensConsumed += 2_300_000n;
console.log(`累计消耗：${totalTokensConsumed} tokens`);

// bigint 不能与 number 直接混算
// const result = bigInt + 1; // 错误！
const result = bigInt + 1n;   // 正确
```

### 原始类型速查表

| 类型 | 示例值 | 典型 AI 应用场景 |
|------|--------|-----------------|
| `string` | `"hello"` | prompt、response、model name |
| `number` | `0.7`, `2048` | temperature、max_tokens、cost |
| `boolean` | `true`, `false` | stream、cache、debug |
| `null` | `null` | 明确表示无缓存、无结果 |
| `undefined` | `undefined` | 可选配置未设置 |
| `symbol` | `Symbol("id")` | 唯一键、防冲突属性 |
| `bigint` | `9007199254740993n` | 大规模 token 统计 |

---

## 2.2 数组类型

TypeScript 提供了多种数组类型的写法，并支持只读数组。

### 2.2.1 两种声明语法

```typescript
// 语法一：T[]（更简洁，推荐日常使用）
let models: string[] = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"];
let temperatures: number[] = [0.0, 0.5, 0.7, 1.0];
let flags: boolean[] = [true, false, true];

// 语法二：Array<T>（泛型形式，复杂类型时更清晰）
let messages: Array<string> = ["你好", "请问有什么可以帮助你？"];
let scores: Array<number> = [0.95, 0.82, 0.71];

// 复杂类型时，Array<T> 更易读
type Message = { role: string; content: string };
let chatHistory: Array<Message> = [];
// 等价于：let chatHistory: Message[] = [];

// 多维数组
let embeddingMatrix: number[][] = [
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6]
];
```

### 2.2.2 数组操作与类型推断

```typescript
// TypeScript 自动推断 push 的参数类型
const supportedModels: string[] = ["claude-3-5-sonnet-20241022"];
supportedModels.push("claude-3-opus-20240229"); // 正确
// supportedModels.push(42);                    // 错误！不能 push number

// map / filter / reduce 的类型推断
const tokenCounts: number[] = [150, 320, 480, 220];
const totalTokens: number = tokenCounts.reduce((sum, n) => sum + n, 0);
const longRequests: number[] = tokenCounts.filter(n => n > 300);
const tokenStrings: string[] = tokenCounts.map(n => `${n} tokens`);

// 函数参数与返回值
function getModelNames(models: Array<{ id: string; name: string }>): string[] {
  return models.map(m => m.name);
}
```

### 2.2.3 只读数组（ReadonlyArray）

只读数组防止数组被意外修改，适用于配置数据和常量列表。

```typescript
// 方式一：readonly T[]
const ALLOWED_MODELS: readonly string[] = [
  "gpt-4o",
  "gpt-4-turbo",
  "claude-3-5-sonnet-20241022",
  "claude-3-opus-20240229"
];

// 方式二：ReadonlyArray<T>
const DEFAULT_STOP_SEQUENCES: ReadonlyArray<string> = ["\n\nHuman:", "\n\nAssistant:"];

// 只读数组禁止修改操作
// ALLOWED_MODELS.push("new-model");    // 错误！
// ALLOWED_MODELS[0] = "other-model";  // 错误！

// 只读数组可以赋值给普通数组（需要类型断言）
const mutableModels: string[] = [...ALLOWED_MODELS]; // 展开后可修改

// 函数参数使用只读数组，表明不会修改传入的数组
function isModelAllowed(model: string, allowedModels: readonly string[]): boolean {
  return allowedModels.includes(model);
}

// 实际使用
console.log(isModelAllowed("gpt-4o", ALLOWED_MODELS)); // true
```

### 2.2.4 数组类型的常见模式

```typescript
// AI 消息历史记录
type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

// 消息数组
const conversation: ChatMessage[] = [
  { role: "system", content: "你是一个有用的助手。" },
  { role: "user", content: "TypeScript 是什么？" },
  { role: "assistant", content: "TypeScript 是 JavaScript 的超集..." }
];

// 追加消息
function addMessage(
  history: ChatMessage[],
  role: ChatMessage["role"],
  content: string
): ChatMessage[] {
  return [...history, { role, content }];
}
```

---

## 2.3 元组类型

元组（Tuple）是长度固定、每个位置类型确定的数组，适合表示结构化的小数据集。

### 2.3.1 基本元组

```typescript
// 基本元组：[类型1, 类型2, ...]
let point: [number, number] = [10, 20];
let nameAndAge: [string, number] = ["Alice", 30];

// AI 应用：模型名称 + 最大 token 数的配对
type ModelConfig = [string, number];
const gpt4Config: ModelConfig = ["gpt-4o", 128000];
const claudeConfig: ModelConfig = ["claude-3-5-sonnet-20241022", 200000];

// 解构元组
const [modelName, contextWindow] = gpt4Config;
console.log(`${modelName} 支持 ${contextWindow} tokens 的上下文窗口`);

// 函数返回元组（代替返回对象，适用于简单场景）
function parseApiResponse(raw: string): [boolean, string] {
  try {
    JSON.parse(raw);
    return [true, raw];
  } catch {
    return [false, "解析失败"];
  }
}

const [success, data] = parseApiResponse('{"model": "gpt-4o"}');
if (success) {
  console.log("解析成功：", data);
}
```

### 2.3.2 命名元组（TypeScript 4.0+）

命名元组为每个位置赋予语义名称，大幅提升可读性。

```typescript
// 命名元组
type LLMResponse = [success: boolean, content: string, tokensUsed: number];

function callLLM(prompt: string): LLMResponse {
  // 模拟 API 调用
  return [true, "这是模型的回复内容", 256];
}

const [success, content, tokensUsed] = callLLM("解释递归的概念");
console.log(`成功：${success}，用了 ${tokensUsed} 个 token`);
console.log(`回复：${content}`);

// 命名元组中参数名会出现在 IDE 提示中
type EmbeddingResult = [
  modelId: string,
  dimensions: number,
  vector: number[]
];

const embedding: EmbeddingResult = [
  "text-embedding-3-small",
  1536,
  [0.012, -0.045, 0.089 /*, ...更多维度 */]
];
```

### 2.3.3 可选元素与剩余元素

```typescript
// 可选元素（? 后缀）
type RequestOptions = [
  prompt: string,
  temperature?: number,
  maxTokens?: number
];

// 以下都是合法的
const req1: RequestOptions = ["解释量子计算"];
const req2: RequestOptions = ["解释量子计算", 0.7];
const req3: RequestOptions = ["解释量子计算", 0.7, 2048];

// 剩余元素（...T[]）
type LogEntry = [timestamp: number, level: string, ...messages: string[]];

const log1: LogEntry = [Date.now(), "INFO", "请求开始"];
const log2: LogEntry = [Date.now(), "ERROR", "API 调用失败", "超时", "重试中"];

// 元组与数组混合：前几个元素固定类型，后续元素任意数量
type ApiCallRecord = [requestId: string, model: string, ...tokenCounts: number[]];
const record: ApiCallRecord = ["req-001", "gpt-4o", 150, 320, 480];
```

---

## 2.4 枚举类型

枚举（Enum）为一组相关常量提供有意义的名称，在 AI 应用的状态管理中非常实用。

### 2.4.1 数字枚举

默认情况下，枚举成员从 0 开始自动递增。

```typescript
// 基本数字枚举
enum RequestStatus {
  Pending,    // 0
  InProgress, // 1
  Completed,  // 2
  Failed,     // 3
  Cancelled   // 4
}

// 自定义起始值
enum HttpStatus {
  OK = 200,
  Created = 201,
  BadRequest = 400,
  Unauthorized = 401,
  TooManyRequests = 429,
  InternalServerError = 500
}

// 使用枚举
let status: RequestStatus = RequestStatus.Pending;
status = RequestStatus.InProgress;

function handleResponse(statusCode: HttpStatus): string {
  switch (statusCode) {
    case HttpStatus.OK:
      return "请求成功";
    case HttpStatus.TooManyRequests:
      return "请求频率超限，请稍后重试";
    case HttpStatus.Unauthorized:
      return "API 密钥无效或已过期";
    default:
      return `未知状态：${statusCode}`;
  }
}

// 数字枚举支持反向映射
console.log(RequestStatus[0]);             // "Pending"
console.log(RequestStatus[RequestStatus.Completed]); // "Completed"
```

### 2.4.2 字符串枚举

字符串枚举的值更具可读性，调试时直接看到有意义的字符串，**推荐在 AI 应用中使用**。

```typescript
// 字符串枚举（无反向映射，但值可读）
enum AIProvider {
  OpenAI = "openai",
  Anthropic = "anthropic",
  Google = "google",
  Cohere = "cohere",
  Mistral = "mistral"
}

enum MessageRole {
  System = "system",
  User = "user",
  Assistant = "assistant",
  Tool = "tool"
}

enum FinishReason {
  Stop = "stop",           // 正常结束
  Length = "length",       // 达到 max_tokens
  ToolCall = "tool_calls", // 触发工具调用
  ContentFilter = "content_filter" // 内容过滤
}

// 实际使用场景
interface LLMRequest {
  provider: AIProvider;
  model: string;
  messages: Array<{
    role: MessageRole;
    content: string;
  }>;
}

const request: LLMRequest = {
  provider: AIProvider.Anthropic,
  model: "claude-3-5-sonnet-20241022",
  messages: [
    { role: MessageRole.System, content: "你是一个有帮助的助手。" },
    { role: MessageRole.User, content: "什么是 TypeScript？" }
  ]
};

// 字符串枚举不支持反向映射
// console.log(AIProvider["openai"]); // undefined（不支持）
```

### 2.4.3 const 枚举

`const enum` 在编译时被内联，不生成额外的 JavaScript 对象，性能更优。

```typescript
// const 枚举：编译时完全内联
const enum ModelTier {
  Basic = 1,
  Standard = 2,
  Advanced = 3,
  Premium = 4
}

const enum TokenLimit {
  Small = 4096,
  Medium = 32768,
  Large = 128000,
  XLarge = 200000
}

// 使用 const 枚举
function getTokenLimit(tier: ModelTier): TokenLimit {
  switch (tier) {
    case ModelTier.Basic:
      return TokenLimit.Small;
    case ModelTier.Standard:
      return TokenLimit.Medium;
    case ModelTier.Advanced:
      return TokenLimit.Large;
    case ModelTier.Premium:
      return TokenLimit.XLarge;
  }
}

// 编译后，ModelTier.Advanced 直接变为数字 3，无运行时开销
const limit = getTokenLimit(ModelTier.Advanced);
console.log(`高级模型支持 ${limit} tokens`);
```

### 2.4.4 枚举 vs 联合类型

在现代 TypeScript 中，字符串字面量联合类型有时比枚举更简洁：

```typescript
// 枚举方式
enum Role {
  Admin = "admin",
  User = "user",
  Guest = "guest"
}

// 联合类型方式（更简洁，近年更流行）
type RoleUnion = "admin" | "user" | "guest";

// 什么时候用枚举？
// - 需要反向映射（数字枚举）
// - 成员数量多（超过 5 个）
// - 需要在运行时遍历所有值
// - 枚举值需要文档注释

// 什么时候用联合类型？
// - 成员数量少
// - 值已经是有意义的字符串
// - 不需要命名空间隔离
```

---

## 2.5 特殊类型

TypeScript 提供了几种特殊类型，正确理解它们对编写健壮代码至关重要。

### 2.5.1 any

`any` 完全跳过类型检查，相当于告诉 TypeScript "我知道我在做什么，别管我"。

```typescript
// any 的使用（尽量避免）
let rawData: any = fetchRawApiResponse();
rawData.someMethod();      // 不报错（but 运行时可能崩溃）
rawData = 42;              // 不报错
rawData = "string";        // 不报错

// any 的合理使用场景
// 1. 处理真正动态的第三方数据
// 2. 渐进式迁移 JavaScript 到 TypeScript
// 3. 临时原型阶段

// 从 any 收窄类型（正确做法）
function processApiData(data: any): string {
  if (typeof data === "string") {
    return data.toUpperCase();
  }
  if (typeof data === "object" && data !== null && "content" in data) {
    return String(data.content);
  }
  return String(data);
}

// 反例：不要这样做
function badProcess(data: any): string {
  return data.content.text; // 没有类型检查，运行时可能崩溃
}
```

### 2.5.2 unknown

`unknown` 是 `any` 的类型安全替代品：同样表示"类型未知"，但使用前必须进行类型检查。

```typescript
// unknown：类型安全的 any
async function fetchFromAPI(url: string): Promise<unknown> {
  const response = await fetch(url);
  return response.json(); // 返回值类型不确定，用 unknown
}

// 使用 unknown 时必须先做类型检查
async function processResponse(url: string): Promise<string> {
  const data: unknown = await fetchFromAPI(url);

  // 方式一：typeof 检查
  if (typeof data === "string") {
    return data;
  }

  // 方式二：类型谓词（Type Guard）
  if (isLLMResponse(data)) {
    return data.choices[0].message.content;
  }

  throw new Error("未知的响应格式");
}

// 类型谓词函数
interface LLMChoiceMessage {
  role: string;
  content: string;
}
interface LLMChoice {
  message: LLMChoiceMessage;
  finish_reason: string;
}
interface LLMResponse {
  id: string;
  choices: LLMChoice[];
  usage: { prompt_tokens: number; completion_tokens: number };
}

function isLLMResponse(value: unknown): value is LLMResponse {
  return (
    typeof value === "object" &&
    value !== null &&
    "id" in value &&
    "choices" in value &&
    Array.isArray((value as LLMResponse).choices)
  );
}
```

### 2.5.3 never

`never` 表示永远不会出现的类型，常见于：穷尽性检查、永远不返回的函数。

```typescript
// 永远不返回的函数（抛出异常或无限循环）
function throwError(message: string): never {
  throw new Error(message);
}

function infiniteLoop(): never {
  while (true) {
    // 处理消息队列...
  }
}

// 穷尽性检查（exhaustive check）—— never 最重要的用途
type LLMProvider = "openai" | "anthropic" | "google";

function getBaseURL(provider: LLMProvider): string {
  switch (provider) {
    case "openai":
      return "https://api.openai.com/v1";
    case "anthropic":
      return "https://api.anthropic.com/v1";
    case "google":
      return "https://generativelanguage.googleapis.com/v1beta";
    default:
      // 如果所有 case 都处理了，这里 provider 的类型就是 never
      // 如果漏掉了某个 case，TypeScript 会在这里报错！
      const _exhaustiveCheck: never = provider;
      throw new Error(`未处理的 provider：${_exhaustiveCheck}`);
  }
}

// 当你后来添加了新的 provider，比如 "cohere"
// TypeScript 会立刻在 switch 的 default 处报错，提醒你补充处理逻辑
```

### 2.5.4 void

`void` 表示函数不返回有意义的值（返回 `undefined`）。

```typescript
// void 函数
function logRequest(requestId: string, model: string): void {
  console.log(`[${new Date().toISOString()}] ${requestId}: 调用 ${model}`);
  // 不需要 return 语句，或 return; （无值）
}

// void vs undefined 的细微差别
function returnsVoid(): void {
  return undefined; // 允许
}

function returnsUndefined(): undefined {
  return undefined; // 必须显式返回 undefined
}

// 回调函数中的 void
const models: string[] = ["gpt-4o", "claude-3-5-sonnet-20241022"];

// forEach 的回调返回类型是 void，不关心返回值
models.forEach((model): void => {
  console.log(model);
  // 即使这里写了 return "something"，forEach 也不会使用这个值
});
```

### 2.5.5 object

`object` 类型表示所有非原始值（即对象、数组、函数等）。

```typescript
// object 类型（很宽泛，通常不推荐直接使用）
let config: object = { model: "gpt-4o", temperature: 0.7 };
// config.model; // 错误！object 类型不能访问属性

// 更好的做法：使用具体的接口或类型别名
interface ModelConfig {
  model: string;
  temperature: number;
  maxTokens?: number;
}

let betterConfig: ModelConfig = { model: "gpt-4o", temperature: 0.7 };
betterConfig.model;      // 正确，有类型提示
betterConfig.temperature; // 正确

// {} 类型：表示任何非 null/undefined 的值（不推荐）
let anything: {} = 42;      // 可以是数字
anything = "string";         // 可以是字符串
anything = { key: "value" }; // 可以是对象

// Record<K, V>：更实用的对象类型
type ModelPricing = Record<string, { input: number; output: number }>;
const pricing: ModelPricing = {
  "gpt-4o": { input: 2.5, output: 10.0 },
  "gpt-4o-mini": { input: 0.15, output: 0.6 },
  "claude-3-5-sonnet-20241022": { input: 3.0, output: 15.0 }
};
```

### 特殊类型对比速查

| 类型 | 含义 | 何时使用 | 安全性 |
|------|------|----------|--------|
| `any` | 任意类型，跳过检查 | 迁移期、第三方无类型库 | 最低 |
| `unknown` | 任意类型，使用前必须检查 | 外部 API 响应、动态数据 | 高 |
| `never` | 不可能出现的类型 | 穷尽检查、不返回的函数 | - |
| `void` | 无返回值 | 纯副作用函数 | - |
| `object` | 非原始对象 | 极少直接使用 | 低 |

---

## 本章小结

```
TypeScript 类型系统概览
├── 原始类型（7种）
│   ├── string     - 文本（prompt、response）
│   ├── number     - 数值（temperature、tokens）
│   ├── boolean    - 布尔（stream、cache）
│   ├── null       - 显式空值
│   ├── undefined  - 未定义
│   ├── symbol     - 唯一键
│   └── bigint     - 大整数
├── 数组类型
│   ├── T[] / Array<T>    - 普通数组
│   └── readonly T[]      - 只读数组
├── 元组类型
│   ├── [T1, T2]          - 基本元组
│   ├── [name: T1, ...]   - 命名元组（4.0+）
│   └── [T1, T2?, ...T[]] - 可选/剩余元素
├── 枚举类型
│   ├── enum              - 数字枚举（支持反向映射）
│   ├── string enum       - 字符串枚举（可读性强）
│   └── const enum        - 内联枚举（零运行时开销）
└── 特殊类型（5种）
    ├── any     - 跳过检查（尽量避免）
    ├── unknown - 安全的未知类型（推荐）
    ├── never   - 不可能类型（穷尽检查）
    ├── void    - 无返回值
    └── object  - 非原始对象
```

**核心原则**：
- 优先使用具体类型，而非 `any`
- 处理外部数据时用 `unknown` 代替 `any`
- 用 `const enum` 减少运行时开销
- 字符串枚举比数字枚举更易调试

---

## AI 应用实战：API 响应类型定义

本节展示如何为真实的 LLM API 响应定义完整的 TypeScript 类型，这是 AI 应用开发中的基础工作。

### 实战场景

我们将为以下两个主流 API 定义类型：
1. **OpenAI Chat Completions API**（`/v1/chat/completions`）
2. **Anthropic Messages API**（`/v1/messages`）

### OpenAI 类型定义

```typescript
// ============================================
// OpenAI Chat Completions API 类型定义
// 参考：https://platform.openai.com/docs/api-reference/chat
// ============================================

// --- 请求类型 ---

enum OpenAIModel {
  GPT4O = "gpt-4o",
  GPT4OMin = "gpt-4o-mini",
  GPT4Turbo = "gpt-4-turbo",
  O1Preview = "o1-preview"
}

enum OpenAIRole {
  System = "system",
  User = "user",
  Assistant = "assistant",
  Tool = "tool"
}

interface OpenAIMessage {
  role: OpenAIRole;
  content: string | null;
  name?: string;
  tool_call_id?: string;
}

interface OpenAIToolFunction {
  name: string;
  description?: string;
  parameters: Record<string, unknown>; // JSON Schema 对象
}

interface OpenAITool {
  type: "function";
  function: OpenAIToolFunction;
}

interface OpenAIChatRequest {
  model: OpenAIModel | string;
  messages: OpenAIMessage[];
  temperature?: number;        // 0-2，默认 1
  top_p?: number;              // 0-1，默认 1
  max_tokens?: number;
  stream?: boolean;
  stop?: string | string[];
  tools?: OpenAITool[];
  tool_choice?: "none" | "auto" | "required";
  response_format?: { type: "text" | "json_object" };
  seed?: number;
  user?: string;               // 用于滥用检测
}

// --- 响应类型 ---

interface OpenAIToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;         // JSON 字符串
  };
}

interface OpenAIResponseMessage {
  role: "assistant";
  content: string | null;      // tool_calls 时为 null
  tool_calls?: OpenAIToolCall[];
}

type OpenAIFinishReason =
  | "stop"
  | "length"
  | "tool_calls"
  | "content_filter"
  | "function_call";

interface OpenAIChoice {
  index: number;
  message: OpenAIResponseMessage;
  finish_reason: OpenAIFinishReason;
  logprobs: null | object;
}

interface OpenAIUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: {
    cached_tokens: number;
    audio_tokens: number;
  };
  completion_tokens_details?: {
    reasoning_tokens: number;
    audio_tokens: number;
    accepted_prediction_tokens: number;
    rejected_prediction_tokens: number;
  };
}

interface OpenAIChatResponse {
  id: string;
  object: "chat.completion";
  created: number;             // Unix 时间戳
  model: string;
  choices: OpenAIChoice[];
  usage: OpenAIUsage;
  system_fingerprint: string | null;
}

// --- 错误类型 ---

interface OpenAIError {
  message: string;
  type: string;
  param: string | null;
  code: string | null;
}

interface OpenAIErrorResponse {
  error: OpenAIError;
}
```

### Anthropic 类型定义

```typescript
// ============================================
// Anthropic Messages API 类型定义
// 参考：https://docs.anthropic.com/en/api/messages
// ============================================

// --- 请求类型 ---

enum AnthropicModel {
  Claude35Sonnet = "claude-3-5-sonnet-20241022",
  Claude35Haiku = "claude-3-5-haiku-20241022",
  Claude3Opus = "claude-3-opus-20240229",
  Claude3Haiku = "claude-3-haiku-20240307"
}

// Anthropic 使用 content block 结构
type AnthropicContentBlock =
  | { type: "text"; text: string }
  | { type: "image"; source: AnthropicImageSource }
  | { type: "tool_use"; id: string; name: string; input: unknown }
  | { type: "tool_result"; tool_use_id: string; content: string };

interface AnthropicImageSource {
  type: "base64" | "url";
  media_type?: "image/jpeg" | "image/png" | "image/gif" | "image/webp";
  data?: string;
  url?: string;
}

interface AnthropicMessage {
  role: "user" | "assistant";
  content: string | AnthropicContentBlock[];
}

interface AnthropicSystemMessage {
  type: "text";
  text: string;
}

interface AnthropicTool {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>; // JSON Schema
}

interface AnthropicMessagesRequest {
  model: AnthropicModel | string;
  max_tokens: number;          // 必填！无默认值
  messages: AnthropicMessage[];
  system?: string | AnthropicSystemMessage[];
  temperature?: number;        // 0-1
  top_p?: number;
  top_k?: number;
  stream?: boolean;
  stop_sequences?: string[];
  tools?: AnthropicTool[];
  tool_choice?: { type: "auto" | "any" | "tool"; name?: string };
}

// --- 响应类型 ---

type AnthropicResponseContentBlock =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> };

type AnthropicStopReason =
  | "end_turn"      // 正常结束
  | "max_tokens"    // 达到 max_tokens
  | "stop_sequence" // 触发停止序列
  | "tool_use";     // 触发工具调用

interface AnthropicUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

interface AnthropicMessagesResponse {
  id: string;
  type: "message";
  role: "assistant";
  content: AnthropicResponseContentBlock[];
  model: string;
  stop_reason: AnthropicStopReason | null;
  stop_sequence: string | null;
  usage: AnthropicUsage;
}

// --- 错误类型 ---

type AnthropicErrorType =
  | "invalid_request_error"
  | "authentication_error"
  | "permission_error"
  | "not_found_error"
  | "rate_limit_error"
  | "api_error"
  | "overloaded_error";

interface AnthropicErrorResponse {
  type: "error";
  error: {
    type: AnthropicErrorType;
    message: string;
  };
}
```

### 统一封装：多 Provider 客户端

```typescript
// ============================================
// 统一的 LLM 客户端类型封装
// ============================================

// 统一的输入类型
interface UnifiedMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface UnifiedLLMRequest {
  model: string;
  messages: UnifiedMessage[];
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
}

// 统一的输出类型
interface UnifiedLLMResponse {
  id: string;
  provider: "openai" | "anthropic";
  model: string;
  content: string;
  finishReason: string;
  usage: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
  rawResponse: OpenAIChatResponse | AnthropicMessagesResponse;
}

// 类型守卫：判断响应来自哪个 Provider
function isOpenAIResponse(
  response: unknown
): response is OpenAIChatResponse {
  return (
    typeof response === "object" &&
    response !== null &&
    (response as OpenAIChatResponse).object === "chat.completion"
  );
}

function isAnthropicResponse(
  response: unknown
): response is AnthropicMessagesResponse {
  return (
    typeof response === "object" &&
    response !== null &&
    (response as AnthropicMessagesResponse).type === "message"
  );
}

// 转换函数
function normalizeResponse(
  raw: unknown,
  provider: "openai" | "anthropic"
): UnifiedLLMResponse {
  if (provider === "openai" && isOpenAIResponse(raw)) {
    const choice = raw.choices[0];
    return {
      id: raw.id,
      provider: "openai",
      model: raw.model,
      content: choice.message.content ?? "",
      finishReason: choice.finish_reason,
      usage: {
        inputTokens: raw.usage.prompt_tokens,
        outputTokens: raw.usage.completion_tokens,
        totalTokens: raw.usage.total_tokens
      },
      rawResponse: raw
    };
  }

  if (provider === "anthropic" && isAnthropicResponse(raw)) {
    const textBlock = raw.content.find(b => b.type === "text");
    const content = textBlock?.type === "text" ? textBlock.text : "";
    return {
      id: raw.id,
      provider: "anthropic",
      model: raw.model,
      content,
      finishReason: raw.stop_reason ?? "unknown",
      usage: {
        inputTokens: raw.usage.input_tokens,
        outputTokens: raw.usage.output_tokens,
        totalTokens: raw.usage.input_tokens + raw.usage.output_tokens
      },
      rawResponse: raw
    };
  }

  throw new Error("无法识别的 API 响应格式");
}

// 使用示例
async function callLLM(
  request: UnifiedLLMRequest,
  provider: "openai" | "anthropic"
): Promise<UnifiedLLMResponse> {
  // 实际项目中这里会调用真实的 API
  const rawResponse: unknown = await mockAPICall(provider, request);
  return normalizeResponse(rawResponse, provider);
}

// 模拟 API 调用（实际项目替换为真实实现）
async function mockAPICall(
  provider: "openai" | "anthropic",
  request: UnifiedLLMRequest
): Promise<unknown> {
  // 返回模拟数据
  if (provider === "openai") {
    const mockResponse: OpenAIChatResponse = {
      id: "chatcmpl-mock-001",
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: request.model,
      choices: [{
        index: 0,
        message: { role: "assistant", content: "这是一个模拟的 OpenAI 响应。" },
        finish_reason: "stop",
        logprobs: null
      }],
      usage: { prompt_tokens: 50, completion_tokens: 20, total_tokens: 70 },
      system_fingerprint: null
    };
    return mockResponse;
  }

  const mockResponse: AnthropicMessagesResponse = {
    id: "msg-mock-001",
    type: "message",
    role: "assistant",
    content: [{ type: "text", text: "这是一个模拟的 Anthropic 响应。" }],
    model: request.model,
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: 50, output_tokens: 20 }
  };
  return mockResponse;
}
```

---

## 练习题

### 基础题

**练习 1**：类型注解补全

下面的代码缺少类型注解，请为所有变量和函数参数/返回值添加正确的 TypeScript 类型：

```typescript
// 待补全类型注解
let apiKey = "";
let maxRetries = 3;
let isStreaming = false;
let models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"];
let lastError = null;

function calculateCost(inputTokens, outputTokens, pricePerMillion) {
  return (inputTokens + outputTokens) * pricePerMillion / 1_000_000;
}

function getModelById(id, models) {
  return models.find(m => m.id === id) ?? null;
}
```

---

**练习 2**：枚举设计

为一个 AI 聊天应用设计枚举类型，包含：
- 聊天会话状态（创建中、活跃、已暂停、已结束、出错）
- 消息状态（发送中、已发送、已读、失败）
- 支持的输出格式（纯文本、Markdown、JSON、代码）

要求：使用字符串枚举，值要有意义。

---

### 中级题

**练习 3**：元组应用

使用命名元组定义以下数据结构，并实现相应的处理函数：

1. API 调用记录：包含请求 ID（字符串）、时间戳（数字）、耗时毫秒数（数字）、是否成功（布尔）、token 数量（可选数字）
2. 实现 `formatRecord(record: ApiCallRecord): string` 函数，将记录格式化为日志字符串
3. 实现 `calculateSuccessRate(records: ApiCallRecord[]): number` 函数，计算成功率

---

**练习 4**：类型收窄与 unknown

实现一个 `parseWebhookEvent` 函数，接收 `unknown` 类型的 webhook 数据，并安全地解析为以下联合类型之一：

```typescript
type WebhookEvent =
  | { type: "message.created"; data: { id: string; content: string; role: string } }
  | { type: "message.delta"; data: { id: string; delta: string } }
  | { type: "message.completed"; data: { id: string; totalTokens: number } }
  | { type: "error"; data: { code: string; message: string } };
```

要求：使用类型守卫，对非法数据抛出有意义的错误信息。

---

### 高级题

**练习 5**：综合类型系统设计

设计一个 `LLMRequestBuilder` 类型系统，实现建造者模式的类型支持。要求：

1. 定义所有必要的接口和类型
2. 用元组表示消息历史中的"用户-助手"对话轮次：`[userMessage: string, assistantMessage: string]`
3. 用 `const enum` 定义所有枚举参数（role、provider、finish reason 等）
4. 定义一个泛型函数 `buildRequest<T extends LLMProvider>(...)`，根据不同的 provider 返回对应格式的请求体（OpenAI 格式或 Anthropic 格式）
5. 使用 `never` 实现穷尽性检查，确保未来添加新 provider 时编译器会提示开发者补充处理逻辑

---

## 练习答案

### 答案 1

```typescript
let apiKey: string = "";
let maxRetries: number = 3;
let isStreaming: boolean = false;
let models: string[] = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"];
let lastError: string | null = null;

interface ModelInfo {
  id: string;
  name: string;
  [key: string]: unknown;
}

function calculateCost(
  inputTokens: number,
  outputTokens: number,
  pricePerMillion: number
): number {
  return (inputTokens + outputTokens) * pricePerMillion / 1_000_000;
}

function getModelById(id: string, models: ModelInfo[]): ModelInfo | null {
  return models.find(m => m.id === id) ?? null;
}
```

---

### 答案 2

```typescript
enum SessionStatus {
  Creating = "creating",
  Active = "active",
  Paused = "paused",
  Ended = "ended",
  Error = "error"
}

enum MessageStatus {
  Sending = "sending",
  Sent = "sent",
  Read = "read",
  Failed = "failed"
}

enum OutputFormat {
  PlainText = "plain_text",
  Markdown = "markdown",
  JSON = "json",
  Code = "code"
}

// 使用示例
interface ChatSession {
  id: string;
  status: SessionStatus;
  outputFormat: OutputFormat;
  createdAt: number;
}

interface ChatMessage {
  id: string;
  sessionId: string;
  content: string;
  status: MessageStatus;
  role: "user" | "assistant";
}
```

---

### 答案 3

```typescript
// 命名元组定义
type ApiCallRecord = [
  requestId: string,
  timestamp: number,
  durationMs: number,
  success: boolean,
  tokenCount?: number
];

// 格式化记录
function formatRecord(record: ApiCallRecord): string {
  const [requestId, timestamp, durationMs, success, tokenCount] = record;
  const date = new Date(timestamp).toISOString();
  const status = success ? "成功" : "失败";
  const tokenInfo = tokenCount !== undefined ? `，${tokenCount} tokens` : "";
  return `[${date}] ${requestId}: ${status}，耗时 ${durationMs}ms${tokenInfo}`;
}

// 计算成功率
function calculateSuccessRate(records: ApiCallRecord[]): number {
  if (records.length === 0) return 0;
  const successCount = records.filter(([, , , success]) => success).length;
  return successCount / records.length;
}

// 测试
const records: ApiCallRecord[] = [
  ["req-001", Date.now(), 350, true, 256],
  ["req-002", Date.now(), 520, true, 512],
  ["req-003", Date.now(), 100, false],
  ["req-004", Date.now(), 480, true, 128]
];

records.forEach(r => console.log(formatRecord(r)));
console.log(`成功率：${(calculateSuccessRate(records) * 100).toFixed(1)}%`);
// 成功率：75.0%
```

---

### 答案 4

```typescript
type WebhookEvent =
  | { type: "message.created"; data: { id: string; content: string; role: string } }
  | { type: "message.delta"; data: { id: string; delta: string } }
  | { type: "message.completed"; data: { id: string; totalTokens: number } }
  | { type: "error"; data: { code: string; message: string } };

// 辅助类型守卫
function hasProperty<K extends string>(
  obj: unknown,
  key: K
): obj is Record<K, unknown> {
  return typeof obj === "object" && obj !== null && key in obj;
}

function parseWebhookEvent(raw: unknown): WebhookEvent {
  if (!hasProperty(raw, "type") || !hasProperty(raw, "data")) {
    throw new Error("无效的 webhook 事件：缺少 type 或 data 字段");
  }

  const { type, data } = raw;

  if (typeof type !== "string") {
    throw new Error(`无效的 webhook 事件：type 必须是字符串，收到 ${typeof type}`);
  }

  if (typeof data !== "object" || data === null) {
    throw new Error("无效的 webhook 事件：data 必须是对象");
  }

  switch (type) {
    case "message.created":
      if (
        hasProperty(data, "id") && typeof (data as { id: unknown }).id === "string" &&
        hasProperty(data, "content") && typeof (data as { content: unknown }).content === "string" &&
        hasProperty(data, "role") && typeof (data as { role: unknown }).role === "string"
      ) {
        return {
          type: "message.created",
          data: data as { id: string; content: string; role: string }
        };
      }
      throw new Error("message.created 事件数据格式错误");

    case "message.delta":
      if (
        hasProperty(data, "id") && typeof (data as { id: unknown }).id === "string" &&
        hasProperty(data, "delta") && typeof (data as { delta: unknown }).delta === "string"
      ) {
        return {
          type: "message.delta",
          data: data as { id: string; delta: string }
        };
      }
      throw new Error("message.delta 事件数据格式错误");

    case "message.completed":
      if (
        hasProperty(data, "id") && typeof (data as { id: unknown }).id === "string" &&
        hasProperty(data, "totalTokens") && typeof (data as { totalTokens: unknown }).totalTokens === "number"
      ) {
        return {
          type: "message.completed",
          data: data as { id: string; totalTokens: number }
        };
      }
      throw new Error("message.completed 事件数据格式错误");

    case "error":
      if (
        hasProperty(data, "code") && typeof (data as { code: unknown }).code === "string" &&
        hasProperty(data, "message") && typeof (data as { message: unknown }).message === "string"
      ) {
        return {
          type: "error",
          data: data as { code: string; message: string }
        };
      }
      throw new Error("error 事件数据格式错误");

    default:
      throw new Error(`未知的 webhook 事件类型：${type}`);
  }
}

// 测试
try {
  const event = parseWebhookEvent({
    type: "message.created",
    data: { id: "msg-001", content: "你好！", role: "user" }
  });
  console.log("解析成功：", event.type, event.data);
} catch (e) {
  console.error("解析失败：", e);
}
```

---

### 答案 5

```typescript
// 使用 const enum 定义所有枚举参数
const enum LLMProvider {
  OpenAI = "openai",
  Anthropic = "anthropic"
}

const enum MessageRole {
  System = "system",
  User = "user",
  Assistant = "assistant"
}

const enum FinishReason {
  Stop = "stop",
  Length = "length",
  ToolCall = "tool_calls",
  EndTurn = "end_turn",
  MaxTokens = "max_tokens"
}

// 对话轮次元组
type ConversationTurn = [userMessage: string, assistantMessage: string];

// 基础接口
interface BaseMessage {
  role: MessageRole;
  content: string;
}

// OpenAI 格式请求
interface OpenAIFormatRequest {
  model: string;
  messages: BaseMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
}

// Anthropic 格式请求
interface AnthropicFormatRequest {
  model: string;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
  system?: string;
  max_tokens: number;
  temperature?: number;
  stream?: boolean;
}

// Provider 到请求格式的映射
type ProviderRequestMap = {
  [LLMProvider.OpenAI]: OpenAIFormatRequest;
  [LLMProvider.Anthropic]: AnthropicFormatRequest;
};

// 构建器配置
interface BuilderConfig {
  provider: LLMProvider;
  model: string;
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
  conversationHistory?: ConversationTurn[];
  additionalMessages?: BaseMessage[];
}

// 泛型构建函数，根据 provider 返回对应格式
function buildRequest<T extends LLMProvider>(
  config: BuilderConfig & { provider: T }
): ProviderRequestMap[T] {
  // 将对话历史转换为消息数组
  const historyMessages: BaseMessage[] = (config.conversationHistory ?? []).flatMap(
    ([userMsg, assistantMsg]): BaseMessage[] => [
      { role: MessageRole.User, content: userMsg },
      { role: MessageRole.Assistant, content: assistantMsg }
    ]
  );

  const allMessages: BaseMessage[] = [
    ...(config.systemPrompt && config.provider === LLMProvider.OpenAI
      ? [{ role: MessageRole.System, content: config.systemPrompt }]
      : []),
    ...historyMessages,
    ...(config.additionalMessages ?? [])
  ];

  switch (config.provider) {
    case LLMProvider.OpenAI: {
      const request: OpenAIFormatRequest = {
        model: config.model,
        messages: allMessages,
        temperature: config.temperature,
        max_tokens: config.maxTokens,
        stream: config.stream
      };
      return request as ProviderRequestMap[T];
    }

    case LLMProvider.Anthropic: {
      // Anthropic 的 system 是顶层字段，messages 不含 system role
      const userAssistantMessages = allMessages.filter(
        m => m.role !== MessageRole.System
      ) as Array<{ role: "user" | "assistant"; content: string }>;

      const request: AnthropicFormatRequest = {
        model: config.model,
        messages: userAssistantMessages,
        system: config.systemPrompt,
        max_tokens: config.maxTokens ?? 1024,
        temperature: config.temperature,
        stream: config.stream
      };
      return request as ProviderRequestMap[T];
    }

    default: {
      // 穷尽性检查：如果未来添加新 Provider 而未处理，编译器会在此报错
      const _exhaustive: never = config.provider;
      throw new Error(`未实现的 LLM Provider：${_exhaustive}`);
    }
  }
}

// 使用示例
const openAIRequest = buildRequest({
  provider: LLMProvider.OpenAI,
  model: "gpt-4o",
  systemPrompt: "你是一个数学教师。",
  temperature: 0.7,
  maxTokens: 2048,
  conversationHistory: [
    ["什么是导数？", "导数是函数在某点处的瞬时变化率..."],
    ["能举个例子吗？", "当然，f(x) = x² 的导数是 f'(x) = 2x..."]
  ],
  additionalMessages: [
    { role: MessageRole.User, content: "那积分又是什么？" }
  ]
});

const anthropicRequest = buildRequest({
  provider: LLMProvider.Anthropic,
  model: "claude-3-5-sonnet-20241022",
  systemPrompt: "你是一个数学教师。",
  temperature: 0.7,
  maxTokens: 2048,
  conversationHistory: [
    ["什么是导数？", "导数是函数在某点处的瞬时变化率..."]
  ],
  additionalMessages: [
    { role: MessageRole.User, content: "积分和导数有什么关系？" }
  ]
});

console.log("OpenAI 请求格式：", JSON.stringify(openAIRequest, null, 2));
console.log("Anthropic 请求格式：", JSON.stringify(anthropicRequest, null, 2));
```

---

> **下一章预告**：第3章将介绍 TypeScript 的接口（Interface）与类型别名（Type Alias），学习如何用它们构建复杂的对象类型，并探讨两者的异同与最佳实践。
