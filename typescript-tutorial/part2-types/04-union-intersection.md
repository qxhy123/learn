# 第4章：联合类型与交叉类型

> **适用读者**：已完成第3章基础类型学习，希望深入掌握 TypeScript 高级类型组合能力的开发者。

---

## 学习目标

完成本章学习后，你将能够：

1. 理解联合类型（Union Types）的概念，并在函数参数、返回值中灵活运用
2. 掌握字面量类型与联合类型的组合，实现精确的值约束
3. 使用可辨识联合（Discriminated Unions）设计类型安全的状态机和消息系统
4. 理解交叉类型（Intersection Types）的语义，用于合并多个类型的属性
5. 综合运用联合类型与交叉类型，为多模型 AI API 设计统一、类型安全的接口层

---

## 4.1 联合类型（Union Types）基础

### 4.1.1 什么是联合类型

联合类型表示一个值可以是**多种类型之一**，用管道符 `|` 分隔各成员类型。

```typescript
// 最基本的联合类型：字符串或数字
type StringOrNumber = string | number;

let value: StringOrNumber;
value = "hello";  // 合法
value = 42;       // 合法
value = true;     // 错误：boolean 不在联合类型中
```

联合类型常用于函数参数，描述函数能接受多种输入形式：

```typescript
// AI 接口中的 token 数量可以是数字或 null（表示未知）
function formatTokenCount(count: number | null): string {
  if (count === null) {
    return "未知";
  }
  return count.toLocaleString();
}

console.log(formatTokenCount(1024));  // "1,024"
console.log(formatTokenCount(null));  // "未知"
```

### 4.1.2 类型缩窄（Type Narrowing）

当使用联合类型时，TypeScript 要求在访问特定类型的属性前先进行**类型缩窄**——即通过条件判断缩小变量的可能类型范围。

```typescript
type ApiResponse = string | { content: string; model: string };

function extractContent(response: ApiResponse): string {
  // 直接访问 .content 会报错，因为 string 类型没有该属性
  // return response.content; // 错误！

  // 需要先缩窄类型
  if (typeof response === "string") {
    return response;
  } else {
    return response.content;
  }
}
```

TypeScript 支持多种缩窄方式：

```typescript
// 1. typeof 缩窄（适用于原始类型）
function processInput(input: string | number | boolean) {
  if (typeof input === "string") {
    console.log(input.toUpperCase());  // 此处 input 是 string
  } else if (typeof input === "number") {
    console.log(input.toFixed(2));     // 此处 input 是 number
  } else {
    console.log(input ? "true" : "false"); // 此处 input 是 boolean
  }
}

// 2. instanceof 缩窄（适用于类实例）
class OpenAIClient {
  model = "gpt-4";
}
class AnthropicClient {
  model = "claude-opus-4-5-20251101";
}

type AIClient = OpenAIClient | AnthropicClient;

function getModelName(client: AIClient): string {
  if (client instanceof OpenAIClient) {
    return `OpenAI: ${client.model}`;
  } else {
    return `Anthropic: ${client.model}`;
  }
}

// 3. in 缩窄（检查属性是否存在）
type StreamResponse = { stream: true; chunks: string[] };
type CompleteResponse = { stream: false; content: string };
type Response = StreamResponse | CompleteResponse;

function handleResponse(res: Response) {
  if ("chunks" in res) {
    // TypeScript 推断 res 为 StreamResponse
    res.chunks.forEach(chunk => process.stdout.write(chunk));
  } else {
    console.log(res.content);
  }
}
```

### 4.1.3 联合类型的公共属性

对于联合类型，无需缩窄即可访问**所有成员类型共有**的属性：

```typescript
type Cat = { name: string; meow(): void };
type Dog = { name: string; bark(): void };
type Pet = Cat | Dog;

function greetPet(pet: Pet) {
  console.log(`你好，${pet.name}！`);  // 合法：name 是公共属性
  // pet.meow();  // 错误：Dog 没有 meow 方法
  // pet.bark();  // 错误：Cat 没有 bark 方法
}
```

---

## 4.2 字面量类型与联合类型

### 4.2.1 字面量类型

TypeScript 允许将具体的值作为类型，称为**字面量类型**（Literal Types）。

```typescript
// 字符串字面量类型
type Direction = "north" | "south" | "east" | "west";

// 数字字面量类型
type DiceValue = 1 | 2 | 3 | 4 | 5 | 6;

// 布尔字面量类型（较少单独使用）
type AlwaysTrue = true;
```

字面量类型与联合类型组合，是 TypeScript 中定义枚举值的惯用方式，比 `enum` 更轻量：

```typescript
// AI 模型提供商
type ModelProvider = "openai" | "anthropic" | "google" | "mistral";

// AI 请求的角色
type MessageRole = "system" | "user" | "assistant";

// 使用示例
interface ChatMessage {
  role: MessageRole;
  content: string;
}

function createMessage(role: MessageRole, content: string): ChatMessage {
  return { role, content };
}

createMessage("user", "你好");      // 合法
createMessage("system", "你是助手"); // 合法
createMessage("admin", "管理员");   // 错误：'admin' 不是合法的 MessageRole
```

### 4.2.2 模板字面量类型

TypeScript 4.1+ 支持模板字面量类型，可以组合字符串字面量生成新类型：

```typescript
type Provider = "openai" | "anthropic";
type Version = "v1" | "v2";

// 自动生成所有组合：'openai-v1' | 'openai-v2' | 'anthropic-v1' | 'anthropic-v2'
type ApiEndpoint = `${Provider}-${Version}`;

// AI 模型名称模式
type GPTModel = `gpt-${3 | 4}-${string}`;
type ClaudeModel = `claude-${"opus" | "sonnet" | "haiku"}-${number}`;
```

### 4.2.3 const 断言与字面量推断

TypeScript 默认会将字面量值**拓宽**为更宽泛的类型：

```typescript
// 没有 const 断言时，TypeScript 推断 config.provider 为 string 类型
const config = {
  provider: "openai",
  model: "gpt-4",
};
// config.provider 的类型是 string，不是 "openai"

// 使用 as const，TypeScript 推断为精确的字面量类型
const configConst = {
  provider: "openai",
  model: "gpt-4",
} as const;
// configConst.provider 的类型是 "openai"，configConst.model 的类型是 "gpt-4"

type ConfigProvider = typeof configConst.provider; // "openai"
```

---

## 4.3 可辨识联合（Discriminated Unions）

### 4.3.1 概念与结构

可辨识联合（也称**标签联合**或**代数数据类型**）是 TypeScript 最强大的模式之一。它要求联合的每个成员都包含一个**公共字面量属性**（判别式），用于区分不同成员。

结构要素：
1. 多个接口，每个接口包含一个**相同名称但不同字面量值**的属性（判别式）
2. 将这些接口组成联合类型
3. 通过判别式属性进行类型缩窄

```typescript
// AI 请求的三种状态
interface PendingState {
  status: "pending";        // 判别式
  startedAt: number;
}

interface SuccessState {
  status: "success";        // 判别式
  content: string;
  tokenUsage: number;
  completedAt: number;
}

interface ErrorState {
  status: "error";          // 判别式
  error: string;
  code: number;
  retryable: boolean;
}

// 联合类型
type RequestState = PendingState | SuccessState | ErrorState;

// 基于判别式缩窄
function renderState(state: RequestState): string {
  switch (state.status) {
    case "pending":
      const elapsed = Date.now() - state.startedAt;
      return `处理中... (已等待 ${elapsed}ms)`;

    case "success":
      return `完成：${state.content.slice(0, 50)}... (${state.tokenUsage} tokens)`;

    case "error":
      const retryHint = state.retryable ? "（可重试）" : "（不可重试）";
      return `错误 ${state.code}：${state.error} ${retryHint}`;
  }
}
```

### 4.3.2 穷举检查（Exhaustiveness Checking）

可辨识联合配合 `never` 类型，可实现编译期的**穷举检查**——如果新增了联合成员但忘记处理，TypeScript 会报错。

```typescript
// 使用 never 实现穷举断言
function assertNever(value: never): never {
  throw new Error(`未处理的值：${JSON.stringify(value)}`);
}

type ModelProvider = "openai" | "anthropic" | "google";

interface BaseConfig {
  provider: ModelProvider;
  maxTokens: number;
}

interface OpenAIConfig extends BaseConfig {
  provider: "openai";
  apiKey: string;
  organization?: string;
}

interface AnthropicConfig extends BaseConfig {
  provider: "anthropic";
  apiKey: string;
  version: string;
}

interface GoogleConfig extends BaseConfig {
  provider: "google";
  projectId: string;
  location: string;
}

type ProviderConfig = OpenAIConfig | AnthropicConfig | GoogleConfig;

function getBaseUrl(config: ProviderConfig): string {
  switch (config.provider) {
    case "openai":
      return "https://api.openai.com/v1";
    case "anthropic":
      return "https://api.anthropic.com";
    case "google":
      return `https://${config.location}-aiplatform.googleapis.com/v1`;
    default:
      // 如果新增了提供商但忘记在 switch 中处理，
      // 此处会产生编译错误，因为 config 的类型变为 never
      return assertNever(config);
  }
}
```

### 4.3.3 实际案例：流式响应事件系统

可辨识联合非常适合建模**事件流**，这在 AI 流式输出中极为常见：

```typescript
// 流式响应事件
interface StreamStartEvent {
  type: "stream_start";
  model: string;
  requestId: string;
}

interface ContentDeltaEvent {
  type: "content_delta";
  index: number;
  delta: string;
}

interface TokenCountEvent {
  type: "token_count";
  inputTokens: number;
  outputTokens: number;
}

interface StreamEndEvent {
  type: "stream_end";
  stopReason: "end_turn" | "max_tokens" | "stop_sequence";
}

interface StreamErrorEvent {
  type: "stream_error";
  error: string;
  recoverable: boolean;
}

type StreamEvent =
  | StreamStartEvent
  | ContentDeltaEvent
  | TokenCountEvent
  | StreamEndEvent
  | StreamErrorEvent;

// 事件处理器
function handleStreamEvent(event: StreamEvent): void {
  switch (event.type) {
    case "stream_start":
      console.log(`开始流式传输：模型=${event.model}, 请求ID=${event.requestId}`);
      break;
    case "content_delta":
      process.stdout.write(event.delta);
      break;
    case "token_count":
      console.log(`\nTokens：输入=${event.inputTokens}, 输出=${event.outputTokens}`);
      break;
    case "stream_end":
      console.log(`\n流式传输结束，原因：${event.stopReason}`);
      break;
    case "stream_error":
      console.error(`流式传输错误：${event.error}`);
      if (event.recoverable) {
        console.log("尝试重新连接...");
      }
      break;
    default:
      assertNever(event);
  }
}
```

---

## 4.4 交叉类型（Intersection Types）

### 4.4.1 什么是交叉类型

交叉类型用 `&` 符号将多个类型**合并**为一个类型，结果类型同时拥有所有成员类型的属性。

```typescript
type A = { name: string; age: number };
type B = { email: string; phone: string };

// 交叉类型：同时拥有 A 和 B 的所有属性
type C = A & B;

const person: C = {
  name: "Alice",
  age: 30,
  email: "alice@example.com",
  phone: "123-456-7890",
  // 缺少任意属性都会报错
};
```

### 4.4.2 接口合并 vs 交叉类型

交叉类型与接口继承效果相似，但语义不同：

```typescript
// 接口继承：通过 extends 扩展
interface BaseRequest {
  model: string;
  temperature: number;
}

interface StreamableRequest extends BaseRequest {
  stream: boolean;
  onChunk: (chunk: string) => void;
}

// 交叉类型：通过 & 组合（更适合组合已有类型）
type WithTimestamp<T> = T & { createdAt: number; updatedAt: number };
type WithMetadata<T> = T & { requestId: string; userId: string };

// 叠加应用
type TrackedRequest = WithTimestamp<WithMetadata<BaseRequest>>;

const request: TrackedRequest = {
  model: "gpt-4",
  temperature: 0.7,
  requestId: "req-001",
  userId: "user-123",
  createdAt: Date.now(),
  updatedAt: Date.now(),
};
```

### 4.4.3 Mixin 模式

交叉类型常用于实现 **Mixin**（混入）模式，为类型添加可复用的能力集：

```typescript
// 能力类型
type Serializable = {
  serialize(): string;
  deserialize(data: string): void;
};

type Cacheable = {
  cacheKey: string;
  cacheTtl: number;
  invalidateCache(): void;
};

type Retryable = {
  maxRetries: number;
  retryDelay: number;
  shouldRetry(error: Error): boolean;
};

// AI 请求类型：具备序列化、缓存和重试能力
type RobustAIRequest = {
  prompt: string;
  model: string;
} & Serializable & Cacheable & Retryable;
```

### 4.4.4 交叉类型的冲突处理

当交叉类型的成员包含**同名但类型不兼容**的属性时，结果为 `never`：

```typescript
type X = { id: string };
type Y = { id: number };

// id 的类型是 string & number，即 never
type Z = X & Y;

// 实际上无法创建满足 Z 的对象
// const z: Z = { id: "hello" };  // 错误：string 不能赋给 never
// const z: Z = { id: 42 };       // 错误：number 不能赋给 never

// 正确做法：避免同名不兼容属性
type BetterX = { stringId: string };
type BetterY = { numberId: number };
type BetterZ = BetterX & BetterY; // 没有冲突
```

### 4.4.5 函数类型的交叉

函数类型的交叉会产生**函数重载**效果：

```typescript
type StringFormatter = (input: string) => string;
type NumberFormatter = (input: number) => string;

// 交叉后，函数可以接受 string 或 number
type Formatter = StringFormatter & NumberFormatter;

// 实际上等价于函数重载
declare const format: Formatter;
const s = format("hello"); // 返回 string
const n = format(42);      // 返回 string
```

---

## 4.5 联合类型与交叉类型的组合使用

### 4.5.1 分配律

联合类型和交叉类型满足分配律，理解这一点有助于化简复杂类型：

```typescript
// A & (B | C) 等价于 (A & B) | (A & C)
type Base = { id: string };
type WithName = { name: string };
type WithEmail = { email: string };

// 两种写法等价
type T1 = Base & (WithName | WithEmail);
type T2 = (Base & WithName) | (Base & WithEmail);

// T1 和 T2 都表示：要么同时有 id 和 name，要么同时有 id 和 email
```

### 4.5.2 泛型与联合/交叉类型

泛型和联合/交叉类型结合，可以构建高度灵活的类型工具：

```typescript
// 使某些属性变为可选
type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

interface FullRequest {
  model: string;
  prompt: string;
  temperature: number;
  maxTokens: number;
  stream: boolean;
}

// temperature、maxTokens、stream 变为可选
type MinimalRequest = PartialBy<FullRequest, "temperature" | "maxTokens" | "stream">;

const req: MinimalRequest = {
  model: "claude-sonnet-4-5-20251101",
  prompt: "你好",
  // temperature、maxTokens、stream 可以省略
};

// 为类型添加必填的鉴权信息
type Authenticated<T> = T & { apiKey: string; timestamp: number };

type AuthenticatedRequest = Authenticated<MinimalRequest>;
```

### 4.5.3 条件类型中的联合分发

在条件类型中，联合类型会被**分发**（distribute）处理：

```typescript
// 从联合类型中过滤出可赋值给 T 的类型
type Filter<U, T> = U extends T ? U : never;

type Providers = "openai" | "anthropic" | "google" | "local";
type CloudProviders = Filter<Providers, "openai" | "anthropic" | "google">;
// 结果："openai" | "anthropic" | "google"（去掉了 "local"）

// 提取对象类型中值类型满足条件的键
type KeysOfType<T, V> = {
  [K in keyof T]: T[K] extends V ? K : never;
}[keyof T];

interface Config {
  provider: string;
  maxTokens: number;
  stream: boolean;
  temperature: number;
  tags: string[];
}

type StringKeys = KeysOfType<Config, string>;    // "provider"
type NumberKeys = KeysOfType<Config, number>;    // "maxTokens" | "temperature"
type BooleanKeys = KeysOfType<Config, boolean>;  // "stream"
```

---

## 本章小结

| 概念 | 语法 | 语义 | 典型用途 |
|------|------|------|---------|
| 联合类型 | `A \| B` | 值是 A **或** B | 函数参数多态、可选值 |
| 字面量类型 | `"a" \| "b"` | 值必须是列出的字面量之一 | 枚举值、状态标记 |
| 可辨识联合 | 含公共判别式属性的联合 | 通过判别式安全缩窄 | 状态机、事件系统 |
| 交叉类型 | `A & B` | 值同时满足 A **和** B | 类型合并、Mixin |
| 组合使用 | `(A & C) \| (B & C)` | 分配律展开 | 复杂业务类型建模 |

**选择指南**：
- 需要表达"多选一"→ 用联合类型
- 需要约束为具体值→ 用字面量类型联合
- 需要类型安全的 switch/case → 用可辨识联合 + 穷举检查
- 需要合并多个接口的属性 → 用交叉类型
- 需要动态添加属性 → 用泛型交叉 `T & { extra: ... }`

---

## AI应用实战：多模型API统一接口

在实际 AI 应用开发中，往往需要同时支持多个模型提供商（如 OpenAI、Anthropic），同时保持调用代码的统一性。本节展示如何综合运用联合类型与交叉类型实现这一目标。

### 需求分析

- 支持 OpenAI（GPT 系列）和 Anthropic（Claude 系列）
- 统一的请求/响应接口，调用方无需关心底层差异
- 类型安全：每个提供商的特有配置有独立类型约束
- 支持流式和非流式两种响应模式

### 完整实现

```typescript
// ==========================================
// 1. 基础类型定义
// ==========================================

// 消息角色
type MessageRole = "system" | "user" | "assistant";

// 聊天消息
interface ChatMessage {
  role: MessageRole;
  content: string;
}

// 停止原因（不同提供商的原因统一映射）
type StopReason = "end_turn" | "max_tokens" | "stop_sequence" | "tool_use";

// Token 使用统计
interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

// ==========================================
// 2. 提供商特有的请求参数（交叉类型组合）
// ==========================================

// 所有提供商共有的基础参数
interface BaseRequestParams {
  messages: ChatMessage[];
  maxTokens?: number;
  temperature?: number;
  stream?: boolean;
}

// OpenAI 特有参数
interface OpenAISpecificParams {
  provider: "openai";
  model: "gpt-4" | "gpt-4-turbo" | "gpt-3.5-turbo" | `gpt-${string}`;
  presencePenalty?: number;
  frequencyPenalty?: number;
  logitBias?: Record<string, number>;
}

// Anthropic 特有参数
interface AnthropicSpecificParams {
  provider: "anthropic";
  model:
    | "claude-opus-4-5-20251101"
    | "claude-sonnet-4-5-20251101"
    | "claude-haiku-3-20240307"
    | `claude-${string}`;
  systemPrompt?: string;      // Anthropic 将 system 提示单独处理
  topK?: number;
  topP?: number;
}

// 各提供商的完整请求类型（使用交叉类型）
type OpenAIRequest = BaseRequestParams & OpenAISpecificParams;
type AnthropicRequest = BaseRequestParams & AnthropicSpecificParams;

// 统一请求类型（使用联合类型）
type UnifiedRequest = OpenAIRequest | AnthropicRequest;

// ==========================================
// 3. 统一响应类型（可辨识联合）
// ==========================================

// 非流式响应
interface CompleteResponse {
  type: "complete";
  provider: "openai" | "anthropic";
  model: string;
  content: string;
  stopReason: StopReason;
  usage: TokenUsage;
  rawResponse?: unknown; // 原始响应（用于调试）
}

// 流式响应块
interface StreamChunkResponse {
  type: "chunk";
  provider: "openai" | "anthropic";
  delta: string;
  index: number;
}

// 流式响应完成信号
interface StreamDoneResponse {
  type: "stream_done";
  provider: "openai" | "anthropic";
  stopReason: StopReason;
  usage: TokenUsage;
}

// 错误响应
interface ErrorResponse {
  type: "error";
  provider: "openai" | "anthropic" | "unknown";
  code: number;
  message: string;
  retryable: boolean;
}

// 统一响应联合类型
type UnifiedResponse =
  | CompleteResponse
  | StreamChunkResponse
  | StreamDoneResponse
  | ErrorResponse;

// ==========================================
// 4. 类型守卫（Type Guards）
// ==========================================

function isOpenAIRequest(req: UnifiedRequest): req is OpenAIRequest {
  return req.provider === "openai";
}

function isAnthropicRequest(req: UnifiedRequest): req is AnthropicRequest {
  return req.provider === "anthropic";
}

function isErrorResponse(res: UnifiedResponse): res is ErrorResponse {
  return res.type === "error";
}

function isCompleteResponse(res: UnifiedResponse): res is CompleteResponse {
  return res.type === "complete";
}

// ==========================================
// 5. 适配器实现（模拟）
// ==========================================

// 将 OpenAI 响应格式映射为统一格式
function adaptOpenAIResponse(raw: {
  choices: Array<{ message: { content: string }; finish_reason: string }>;
  model: string;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}): CompleteResponse {
  const stopReasonMap: Record<string, StopReason> = {
    stop: "end_turn",
    length: "max_tokens",
    content_filter: "stop_sequence",
    tool_calls: "tool_use",
  };

  return {
    type: "complete",
    provider: "openai",
    model: raw.model,
    content: raw.choices[0].message.content,
    stopReason: stopReasonMap[raw.choices[0].finish_reason] ?? "end_turn",
    usage: {
      inputTokens: raw.usage.prompt_tokens,
      outputTokens: raw.usage.completion_tokens,
      totalTokens: raw.usage.total_tokens,
    },
  };
}

// 将 Anthropic 响应格式映射为统一格式
function adaptAnthropicResponse(raw: {
  content: Array<{ text: string }>;
  model: string;
  stop_reason: string;
  usage: { input_tokens: number; output_tokens: number };
}): CompleteResponse {
  const stopReasonMap: Record<string, StopReason> = {
    end_turn: "end_turn",
    max_tokens: "max_tokens",
    stop_sequence: "stop_sequence",
    tool_use: "tool_use",
  };

  return {
    type: "complete",
    provider: "anthropic",
    model: raw.model,
    content: raw.content.map(c => c.text).join(""),
    stopReason: stopReasonMap[raw.stop_reason] ?? "end_turn",
    usage: {
      inputTokens: raw.usage.input_tokens,
      outputTokens: raw.usage.output_tokens,
      totalTokens: raw.usage.input_tokens + raw.usage.output_tokens,
    },
  };
}

// ==========================================
// 6. 统一客户端接口
// ==========================================

// 客户端配置（使用可辨识联合 + 交叉类型）
type WithApiKey = { apiKey: string };

type OpenAIClientConfig = {
  provider: "openai";
  organization?: string;
  baseUrl?: string;
} & WithApiKey;

type AnthropicClientConfig = {
  provider: "anthropic";
  version?: string;
  baseUrl?: string;
} & WithApiKey;

type ClientConfig = OpenAIClientConfig | AnthropicClientConfig;

// 统一客户端（示意性实现）
class UnifiedAIClient {
  private config: ClientConfig;

  constructor(config: ClientConfig) {
    this.config = config;
  }

  // 构建实际请求 URL
  private getBaseUrl(): string {
    if (this.config.provider === "openai") {
      return this.config.baseUrl ?? "https://api.openai.com/v1";
    } else {
      return this.config.baseUrl ?? "https://api.anthropic.com";
    }
  }

  // 构建请求头
  private buildHeaders(): Record<string, string> {
    const baseHeaders: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (this.config.provider === "openai") {
      baseHeaders["Authorization"] = `Bearer ${this.config.apiKey}`;
      if (this.config.organization) {
        baseHeaders["OpenAI-Organization"] = this.config.organization;
      }
    } else {
      // anthropic
      baseHeaders["x-api-key"] = this.config.apiKey;
      baseHeaders["anthropic-version"] = this.config.version ?? "2023-06-01";
    }

    return baseHeaders;
  }

  // 构建请求体（处理提供商差异）
  private buildRequestBody(request: UnifiedRequest): Record<string, unknown> {
    if (isOpenAIRequest(request)) {
      return {
        model: request.model,
        messages: request.messages,
        max_tokens: request.maxTokens,
        temperature: request.temperature,
        stream: request.stream,
        presence_penalty: request.presencePenalty,
        frequency_penalty: request.frequencyPenalty,
      };
    } else {
      // 将 system 消息从 messages 数组中提取出来
      const systemMessages = request.messages
        .filter(m => m.role === "system")
        .map(m => m.content)
        .join("\n");

      const nonSystemMessages = request.messages.filter(m => m.role !== "system");

      return {
        model: request.model,
        messages: nonSystemMessages,
        system: request.systemPrompt ?? systemMessages,
        max_tokens: request.maxTokens ?? 1024,
        temperature: request.temperature,
        stream: request.stream,
        top_k: request.topK,
        top_p: request.topP,
      };
    }
  }

  // 统一的 chat 方法
  async chat(request: UnifiedRequest): Promise<UnifiedResponse> {
    try {
      const url = this.getBaseUrl() + (isOpenAIRequest(request) ? "/chat/completions" : "/v1/messages");
      const headers = this.buildHeaders();
      const body = this.buildRequestBody(request);

      // 实际项目中这里会发起真实的 HTTP 请求
      // const response = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
      // const data = await response.json();

      // 模拟响应（示意）
      console.log(`发起请求：${url}`);
      console.log(`请求体：${JSON.stringify(body, null, 2)}`);

      // 模拟成功响应
      const mockResponse: CompleteResponse = {
        type: "complete",
        provider: request.provider,
        model: request.model,
        content: "这是模拟的AI响应内容。",
        stopReason: "end_turn",
        usage: {
          inputTokens: 50,
          outputTokens: 20,
          totalTokens: 70,
        },
      };

      return mockResponse;
    } catch (err) {
      const errorResponse: ErrorResponse = {
        type: "error",
        provider: request.provider,
        code: 500,
        message: err instanceof Error ? err.message : "未知错误",
        retryable: true,
      };
      return errorResponse;
    }
  }
}

// ==========================================
// 7. 使用示例
// ==========================================

// 创建 OpenAI 客户端
const openaiClient = new UnifiedAIClient({
  provider: "openai",
  apiKey: "sk-xxx",
  organization: "org-xxx",
});

// 创建 Anthropic 客户端
const anthropicClient = new UnifiedAIClient({
  provider: "anthropic",
  apiKey: "sk-ant-xxx",
  version: "2023-06-01",
});

// 统一的请求写法
async function runDemo() {
  const openaiRequest: OpenAIRequest = {
    provider: "openai",
    model: "gpt-4",
    messages: [
      { role: "system", content: "你是一个有帮助的AI助手。" },
      { role: "user", content: "请简要介绍TypeScript的优点。" },
    ],
    maxTokens: 500,
    temperature: 0.7,
  };

  const anthropicRequest: AnthropicRequest = {
    provider: "anthropic",
    model: "claude-sonnet-4-5-20251101",
    messages: [
      { role: "user", content: "请简要介绍TypeScript的优点。" },
    ],
    systemPrompt: "你是一个有帮助的AI助手。",
    maxTokens: 500,
    temperature: 0.7,
  };

  // 处理统一响应
  function processResponse(response: UnifiedResponse): void {
    switch (response.type) {
      case "complete":
        console.log(`[${response.provider}] ${response.model}`);
        console.log(`内容：${response.content}`);
        console.log(`Token使用：${response.usage.totalTokens}`);
        break;
      case "chunk":
        process.stdout.write(response.delta);
        break;
      case "stream_done":
        console.log(`\n流式传输完成，停止原因：${response.stopReason}`);
        break;
      case "error":
        console.error(`错误 ${response.code}：${response.message}`);
        if (response.retryable) {
          console.log("可以重试");
        }
        break;
      default:
        assertNever(response);
    }
  }

  const res1 = await openaiClient.chat(openaiRequest);
  processResponse(res1);

  const res2 = await anthropicClient.chat(anthropicRequest);
  processResponse(res2);
}
```

### 设计要点总结

1. **可辨识联合** 用于区分不同提供商的请求/响应（`provider` 和 `type` 作为判别式）
2. **交叉类型** 用于将公共参数与提供商特有参数合并（`BaseRequestParams & OpenAISpecificParams`）
3. **字面量类型** 约束模型名称、停止原因等枚举值
4. **类型守卫** 在运行时安全地缩窄联合类型
5. **穷举检查** 通过 `assertNever` 确保所有响应类型都被处理

---

## 练习题

### 基础题

**练习 4-1**：定义一个 `Shape` 联合类型，包含 `Circle`、`Rectangle` 和 `Triangle` 三种形状，每种形状都有 `type` 判别式和计算面积所需的尺寸属性。编写一个 `calculateArea(shape: Shape): number` 函数，使用 `switch` 语句计算并返回面积。

**练习 4-2**：定义一个 `HttpMethod` 字面量类型联合（包含 `"GET" | "POST" | "PUT" | "PATCH" | "DELETE"`），以及一个 `ApiEndpoint` 接口，包含 `path: string`、`method: HttpMethod` 和可选的 `body: unknown` 属性。编写一个函数 `buildFetchOptions`，根据 `HttpMethod` 是否为 `"GET"` 或 `"DELETE"` 决定是否在 fetch 选项中包含 `body`。

### 进阶题

**练习 4-3**：为一个简单的任务队列系统设计类型。任务有三种状态：`queued`（包含 `priority: number`）、`running`（包含 `startedAt: number` 和 `progress: number`）、`done`（包含 `result: string` 和 `duration: number`）。使用可辨识联合定义 `Task` 类型，并实现：
- `advanceTask(task: Task): Task`：将任务推进到下一状态（queued → running → done）
- `getTaskSummary(task: Task): string`：返回任务状态描述字符串

**练习 4-4**：使用交叉类型和泛型，实现以下工具类型：
- `Required<T>`：使 T 的所有属性变为必填（TypeScript 内置，尝试自己实现）
- `WithId<T>`：为类型 T 添加 `id: string` 属性
- `Timestamped<T>`：为类型 T 添加 `createdAt: number` 和 `updatedAt: number` 属性
- `AuditedRecord<T>`：结合 `WithId<T>` 和 `Timestamped<T>`，创建一个带有 id 和时间戳的记录类型

### 挑战题

**练习 4-5**：设计一个类型安全的**命令模式（Command Pattern）**系统，用于管理 AI 对话历史的操作。

需要支持以下命令：
- `AppendMessage`：追加一条消息（包含消息内容）
- `DeleteMessage`：删除指定索引的消息
- `UpdateMessage`：更新指定索引消息的内容
- `ClearHistory`：清空全部历史
- `Undo`：撤销上一条命令

要求：
1. 使用可辨识联合定义 `HistoryCommand` 类型
2. 定义 `ConversationHistory` 状态类型
3. 实现 `executeCommand(history: ConversationHistory, command: HistoryCommand): ConversationHistory` 函数（返回新状态，不修改原状态）
4. 使用穷举检查确保所有命令都被处理
5. 对 `Undo` 命令，需要维护命令历史栈（在 `ConversationHistory` 中包含 `commandStack`）

---

## 练习答案

### 练习 4-1 答案

```typescript
interface Circle {
  type: "circle";
  radius: number;
}

interface Rectangle {
  type: "rectangle";
  width: number;
  height: number;
}

interface Triangle {
  type: "triangle";
  base: number;
  height: number;
}

type Shape = Circle | Rectangle | Triangle;

function calculateArea(shape: Shape): number {
  switch (shape.type) {
    case "circle":
      return Math.PI * shape.radius ** 2;
    case "rectangle":
      return shape.width * shape.height;
    case "triangle":
      return (shape.base * shape.height) / 2;
    default:
      assertNever(shape);
  }
}

// 测试
console.log(calculateArea({ type: "circle", radius: 5 }));           // ~78.54
console.log(calculateArea({ type: "rectangle", width: 4, height: 6 })); // 24
console.log(calculateArea({ type: "triangle", base: 3, height: 8 }));  // 12
```

### 练习 4-2 答案

```typescript
type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

interface ApiEndpoint {
  path: string;
  method: HttpMethod;
  body?: unknown;
}

function buildFetchOptions(endpoint: ApiEndpoint): RequestInit {
  const options: RequestInit = {
    method: endpoint.method,
    headers: { "Content-Type": "application/json" },
  };

  if (endpoint.method !== "GET" && endpoint.method !== "DELETE" && endpoint.body !== undefined) {
    options.body = JSON.stringify(endpoint.body);
  }

  return options;
}

// 测试
console.log(buildFetchOptions({ path: "/users", method: "GET" }));
// { method: "GET", headers: {...} }  — 没有 body

console.log(buildFetchOptions({
  path: "/messages",
  method: "POST",
  body: { content: "你好" },
}));
// { method: "POST", headers: {...}, body: '{"content":"你好"}' }
```

### 练习 4-3 答案

```typescript
interface QueuedTask {
  id: string;
  status: "queued";
  priority: number;
}

interface RunningTask {
  id: string;
  status: "running";
  startedAt: number;
  progress: number; // 0-100
}

interface DoneTask {
  id: string;
  status: "done";
  result: string;
  duration: number;
}

type Task = QueuedTask | RunningTask | DoneTask;

function advanceTask(task: Task): Task {
  switch (task.status) {
    case "queued":
      const running: RunningTask = {
        id: task.id,
        status: "running",
        startedAt: Date.now(),
        progress: 0,
      };
      return running;
    case "running":
      const done: DoneTask = {
        id: task.id,
        status: "done",
        result: "任务执行成功",
        duration: Date.now() - task.startedAt,
      };
      return done;
    case "done":
      // 已完成的任务无法继续推进，返回自身
      return task;
    default:
      assertNever(task);
  }
}

function getTaskSummary(task: Task): string {
  switch (task.status) {
    case "queued":
      return `任务 ${task.id} 排队中，优先级：${task.priority}`;
    case "running":
      return `任务 ${task.id} 运行中，进度：${task.progress}%`;
    case "done":
      return `任务 ${task.id} 已完成，耗时：${task.duration}ms，结果：${task.result}`;
    default:
      assertNever(task);
  }
}
```

### 练习 4-4 答案

```typescript
// 自实现 Required
type MyRequired<T> = {
  [K in keyof T]-?: T[K];  // -? 移除可选修饰符
};

// WithId
type WithId<T> = T & { id: string };

// Timestamped
type Timestamped<T> = T & {
  createdAt: number;
  updatedAt: number;
};

// AuditedRecord
type AuditedRecord<T> = WithId<Timestamped<T>>;

// 测试
interface UserProfile {
  name: string;
  email: string;
  bio?: string;
}

type AuditedUser = AuditedRecord<UserProfile>;

const user: AuditedUser = {
  id: "user-001",
  name: "Alice",
  email: "alice@example.com",
  createdAt: Date.now(),
  updatedAt: Date.now(),
};

// MyRequired 测试
type FullProfile = MyRequired<UserProfile>;
const full: FullProfile = {
  name: "Bob",
  email: "bob@example.com",
  bio: "工程师",  // 现在是必填的
};
```

### 练习 4-5 答案

```typescript
// 消息类型
interface Message {
  role: MessageRole;
  content: string;
}

// 命令类型（可辨识联合）
interface AppendMessageCommand {
  type: "AppendMessage";
  message: Message;
}

interface DeleteMessageCommand {
  type: "DeleteMessage";
  index: number;
}

interface UpdateMessageCommand {
  type: "UpdateMessage";
  index: number;
  newContent: string;
}

interface ClearHistoryCommand {
  type: "ClearHistory";
}

interface UndoCommand {
  type: "Undo";
}

type HistoryCommand =
  | AppendMessageCommand
  | DeleteMessageCommand
  | UpdateMessageCommand
  | ClearHistoryCommand
  | UndoCommand;

// 对话历史状态
interface ConversationHistory {
  messages: Message[];
  commandStack: Exclude<HistoryCommand, UndoCommand>[]; // Undo 本身不入栈
}

// 工具函数：穷举断言
function assertNever(value: never): never {
  throw new Error(`未处理的命令类型：${JSON.stringify(value)}`);
}

function executeCommand(
  history: ConversationHistory,
  command: HistoryCommand
): ConversationHistory {
  switch (command.type) {
    case "AppendMessage": {
      return {
        messages: [...history.messages, command.message],
        commandStack: [...history.commandStack, command],
      };
    }

    case "DeleteMessage": {
      if (command.index < 0 || command.index >= history.messages.length) {
        console.warn(`索引 ${command.index} 越界，忽略 DeleteMessage 命令`);
        return history;
      }
      return {
        messages: history.messages.filter((_, i) => i !== command.index),
        commandStack: [...history.commandStack, command],
      };
    }

    case "UpdateMessage": {
      if (command.index < 0 || command.index >= history.messages.length) {
        console.warn(`索引 ${command.index} 越界，忽略 UpdateMessage 命令`);
        return history;
      }
      const updatedMessages = history.messages.map((msg, i) =>
        i === command.index ? { ...msg, content: command.newContent } : msg
      );
      return {
        messages: updatedMessages,
        commandStack: [...history.commandStack, command],
      };
    }

    case "ClearHistory": {
      return {
        messages: [],
        commandStack: [...history.commandStack, command],
      };
    }

    case "Undo": {
      if (history.commandStack.length === 0) {
        console.warn("没有可撤销的操作");
        return history;
      }
      // 移除最后一条命令，然后重放剩余命令
      const remainingCommands = history.commandStack.slice(0, -1);
      const initialHistory: ConversationHistory = {
        messages: [],
        commandStack: [],
      };
      // 重放所有剩余命令（不包括 Undo 命令自身）
      return remainingCommands.reduce(
        (acc, cmd) => executeCommand(acc, cmd),
        initialHistory
      );
    }

    default:
      return assertNever(command);
  }
}

// 测试
let history: ConversationHistory = { messages: [], commandStack: [] };

history = executeCommand(history, {
  type: "AppendMessage",
  message: { role: "user", content: "你好" },
});
history = executeCommand(history, {
  type: "AppendMessage",
  message: { role: "assistant", content: "你好！有什么我可以帮助你的？" },
});
history = executeCommand(history, {
  type: "AppendMessage",
  message: { role: "user", content: "介绍一下TypeScript" },
});

console.log("当前消息数：", history.messages.length); // 3

history = executeCommand(history, { type: "Undo" });
console.log("撤销后消息数：", history.messages.length); // 2
console.log("最后一条消息：", history.messages[history.messages.length - 1].content);
// "你好！有什么我可以帮助你的？"

history = executeCommand(history, {
  type: "UpdateMessage",
  index: 0,
  newContent: "嗨！",
});
console.log("更新后第一条：", history.messages[0].content); // "嗨！"
```

---

*下一章：[第5章：泛型（Generics）](./05-generics.md)*
