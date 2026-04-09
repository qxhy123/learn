# 第5章：类型守卫与类型收窄

## 学习目标

完成本章学习后，你将能够：

1. 理解类型收窄（Type Narrowing）的概念，以及TypeScript如何在控制流中自动推断更精确的类型
2. 掌握 `typeof` 和 `instanceof` 类型守卫，用于原始类型和类实例的判断
3. 使用 `in` 操作符和用户自定义类型守卫（`is` 关键字）安全地区分联合类型
4. 理解断言函数（`asserts` 关键字）的作用，实现运行时类型验证
5. 将类型守卫技术应用于SSE流式响应的事件类型判断，构建类型安全的AI流处理管道

---

## 5.1 类型收窄的概念

### 什么是类型收窄

在TypeScript中，**类型收窄**（Type Narrowing）是指在代码的特定分支中，TypeScript编译器能够将一个宽泛的类型自动推断为更精确的子类型。

考虑一个简单的例子：

```typescript
function printLength(value: string | number): void {
  // 此处 value 的类型是 string | number
  // value.length  // ❌ 错误：number 类型没有 length 属性

  if (typeof value === "string") {
    // 此处 TypeScript 知道 value 是 string
    console.log(value.length);   // ✅ 正确
    console.log(value.toUpperCase()); // ✅ 正确
  } else {
    // 此处 TypeScript 知道 value 是 number
    console.log(value.toFixed(2));    // ✅ 正确
  }
}
```

`if (typeof value === "string")` 这个条件就是一个**类型守卫**（Type Guard）——它让TypeScript在该分支内"收窄"了 `value` 的类型范围。

### 控制流分析

TypeScript会对代码进行**控制流分析**（Control Flow Analysis），追踪每个分支中变量的可能类型：

```typescript
type Status = "pending" | "success" | "error";

function handleStatus(status: Status): string {
  if (status === "pending") {
    // status: "pending"
    return "处理中...";
  }

  if (status === "success") {
    // status: "success"
    return "操作成功";
  }

  // 到达此处，TypeScript 推断 status 只能是 "error"
  // status: "error"
  return "操作失败";
}
```

### 穷举检查（Exhaustiveness Checking）

配合 `never` 类型，可以实现编译时穷举检查——确保联合类型的所有成员都被处理：

```typescript
type ModelProvider = "openai" | "anthropic" | "gemini";

function getBaseUrl(provider: ModelProvider): string {
  switch (provider) {
    case "openai":
      return "https://api.openai.com/v1";
    case "anthropic":
      return "https://api.anthropic.com/v1";
    case "gemini":
      return "https://generativelanguage.googleapis.com/v1";
    default:
      // 如果所有情况都处理了，provider 的类型是 never
      // 如果漏掉了某个成员，这里会产生编译错误
      const _exhaustive: never = provider;
      throw new Error(`未知的模型提供商: ${_exhaustive}`);
  }
}
```

> **提示**：`never` 类型表示"永远不会发生的值"。当一个变量被收窄到 `never`，说明代码在逻辑上不可达。利用这个特性，可以在编译时确保 `switch/if-else` 覆盖了所有可能的分支。

### 赋值收窄

TypeScript也会根据赋值语句收窄类型：

```typescript
let value: string | number;

value = "hello";
// 此处 value: string
console.log(value.toUpperCase()); // ✅

value = 42;
// 此处 value: number
console.log(value.toFixed()); // ✅
```

---

## 5.2 typeof 类型守卫

### 基本用法

`typeof` 是JavaScript的内置操作符，TypeScript为其赋予了类型收窄的能力。它可以识别以下类型字符串：

| `typeof` 返回值 | 对应TypeScript类型 |
|---|---|
| `"string"` | `string` |
| `"number"` | `number` |
| `"bigint"` | `bigint` |
| `"boolean"` | `boolean` |
| `"symbol"` | `symbol` |
| `"undefined"` | `undefined` |
| `"object"` | `object \| null`（注意null的陷阱） |
| `"function"` | `Function` |

```typescript
function formatValue(value: string | number | boolean | null): string {
  if (typeof value === "string") {
    return `"${value}"`;
  }

  if (typeof value === "number") {
    return value.toLocaleString();
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  // value: null
  return "null";
}

console.log(formatValue("hello"));  // 输出: "hello"
console.log(formatValue(1234567));  // 输出: 1,234,567
console.log(formatValue(true));     // 输出: true
console.log(formatValue(null));     // 输出: null
```

### null 的陷阱

`typeof null === "object"` 是JavaScript的历史遗留问题，TypeScript也继承了这个行为。处理 `null` 时需要额外小心：

```typescript
function processConfig(config: object | null): void {
  if (typeof config === "object") {
    // config: object | null  —— 仍然可能是 null！
    console.log(config);  // ❌ 不安全
  }
}

// 正确做法：显式检查 null
function processConfigSafe(config: object | null): void {
  if (config !== null && typeof config === "object") {
    // config: object  ✅
    console.log(config);
  }
}

// 或者使用严格相等检查
function processConfigSafe2(config: object | null): void {
  if (config === null) {
    console.log("配置为空");
    return;
  }
  // config: object  ✅
  console.log(config);
}
```

### 在AI应用中处理多种输入类型

```typescript
type LLMInput = string | string[] | { role: string; content: string }[];

function normalizeLLMInput(input: LLMInput): { role: string; content: string }[] {
  if (typeof input === "string") {
    // 字符串 -> 单条用户消息
    return [{ role: "user", content: input }];
  }

  if (Array.isArray(input)) {
    // 检查数组元素类型
    if (input.length === 0) {
      return [];
    }

    if (typeof input[0] === "string") {
      // string[] -> 多条用户消息
      return (input as string[]).map(content => ({ role: "user", content }));
    }

    // { role, content }[] -> 直接返回
    return input as { role: string; content: string }[];
  }

  // 理论上不会到达此处（TypeScript 会提示）
  throw new Error("不支持的输入格式");
}

// 测试
console.log(normalizeLLMInput("你好"));
// 输出: [{ role: 'user', content: '你好' }]

console.log(normalizeLLMInput(["消息1", "消息2"]));
// 输出: [{ role: 'user', content: '消息1' }, { role: 'user', content: '消息2' }]
```

> **注意**：`Array.isArray()` 也是一种类型守卫，TypeScript会在该分支内将类型收窄为数组类型。

---

## 5.3 instanceof 类型守卫

### 基本用法

`instanceof` 操作符用于检查对象是否是某个类的实例。TypeScript利用它来收窄联合类型中的类类型：

```typescript
class NetworkError extends Error {
  constructor(
    public statusCode: number,
    message: string
  ) {
    super(message);
    this.name = "NetworkError";
  }
}

class RateLimitError extends NetworkError {
  constructor(public retryAfter: number) {
    super(429, `请求过于频繁，请在 ${retryAfter} 秒后重试`);
    this.name = "RateLimitError";
  }
}

class AuthError extends NetworkError {
  constructor() {
    super(401, "认证失败，请检查API密钥");
    this.name = "AuthError";
  }
}

function handleApiError(error: unknown): string {
  if (error instanceof RateLimitError) {
    // error: RateLimitError
    return `限流错误，${error.retryAfter}秒后重试`;
  }

  if (error instanceof AuthError) {
    // error: AuthError
    return `认证错误：${error.message}`;
  }

  if (error instanceof NetworkError) {
    // error: NetworkError（排除了子类后的剩余情况）
    return `网络错误 ${error.statusCode}：${error.message}`;
  }

  if (error instanceof Error) {
    // error: Error
    return `未知错误：${error.message}`;
  }

  // error: unknown
  return `发生了意外错误：${String(error)}`;
}
```

### 继承层次中的类型收窄

`instanceof` 对继承链同样有效——子类实例也是父类的实例：

```typescript
class BaseModel {
  constructor(public modelId: string) {}

  abstract generate(prompt: string): Promise<string>;
}

class ChatModel extends BaseModel {
  constructor(
    modelId: string,
    public maxTokens: number
  ) {
    super(modelId);
  }

  async generate(prompt: string): Promise<string> {
    return `[ChatModel ${this.modelId}] ${prompt}`;
  }

  async generateStream(prompt: string): AsyncGenerator<string> {
    // 流式生成实现...
    yield `[Stream] ${prompt}`;
  }
}

class EmbeddingModel extends BaseModel {
  constructor(
    modelId: string,
    public dimensions: number
  ) {
    super(modelId);
  }

  async generate(prompt: string): Promise<string> {
    return `[EmbeddingModel] vector(${this.dimensions})`;
  }

  async embed(text: string): Promise<number[]> {
    // 返回嵌入向量...
    return new Array(this.dimensions).fill(0);
  }
}

function describeModel(model: BaseModel): string {
  if (model instanceof EmbeddingModel) {
    // model: EmbeddingModel
    return `嵌入模型 ${model.modelId}，维度：${model.dimensions}`;
  }

  if (model instanceof ChatModel) {
    // model: ChatModel
    return `对话模型 ${model.modelId}，最大Token：${model.maxTokens}`;
  }

  // model: BaseModel
  return `基础模型 ${model.modelId}`;
}
```

### 处理 unknown 类型的错误

在 `catch` 块中，错误类型是 `unknown`，`instanceof` 是最安全的处理方式：

```typescript
async function callOpenAI(prompt: string): Promise<string> {
  try {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: "gpt-4",
        messages: [{ role: "user", content: prompt }],
      }),
    });

    if (!response.ok) {
      throw new NetworkError(response.status, await response.text());
    }

    const data = await response.json();
    return data.choices[0].message.content;
  } catch (error) {
    // error: unknown
    if (error instanceof NetworkError) {
      console.error(`API错误 [${error.statusCode}]:`, error.message);
      throw error;
    }

    if (error instanceof TypeError) {
      // 网络连接失败（fetch本身抛出的错误）
      console.error("网络连接失败:", error.message);
      throw new NetworkError(0, "网络连接失败");
    }

    throw error;
  }
}
```

---

## 5.4 in 操作符与自定义类型守卫

### in 操作符

`in` 操作符检查对象是否拥有指定属性，TypeScript会据此收窄联合类型：

```typescript
interface TextChunk {
  type: "text";
  content: string;
}

interface ToolCallChunk {
  type: "tool_call";
  toolName: string;
  arguments: Record<string, unknown>;
}

interface ErrorChunk {
  type: "error";
  code: number;
  message: string;
}

type StreamChunk = TextChunk | ToolCallChunk | ErrorChunk;

function processChunk(chunk: StreamChunk): void {
  if ("toolName" in chunk) {
    // chunk: ToolCallChunk
    console.log(`调用工具: ${chunk.toolName}`, chunk.arguments);
    return;
  }

  if ("code" in chunk) {
    // chunk: ErrorChunk
    console.error(`流错误 [${chunk.code}]: ${chunk.message}`);
    return;
  }

  // chunk: TextChunk
  process.stdout.write(chunk.content);
}
```

> **提示**：当联合类型的各成员拥有不同的**判别属性**（discriminant property）时，使用 `type` 字段配合字面量类型（如 `type: "text"`）是更推荐的做法，见下文的判别联合模式。

### 判别联合（Discriminated Union）

判别联合是最常用的类型收窄模式——为联合类型的每个成员添加一个共同的字面量类型字段：

```typescript
// 使用 type 字段作为判别属性
interface TextDelta {
  type: "text_delta";
  text: string;
}

interface InputJsonDelta {
  type: "input_json_delta";
  partial_json: string;
}

interface MessageStartEvent {
  type: "message_start";
  message: {
    id: string;
    model: string;
    usage: { input_tokens: number };
  };
}

interface ContentBlockStopEvent {
  type: "content_block_stop";
  index: number;
}

interface MessageStopEvent {
  type: "message_stop";
}

type AnthropicStreamEvent =
  | TextDelta
  | InputJsonDelta
  | MessageStartEvent
  | ContentBlockStopEvent
  | MessageStopEvent;

// TypeScript 根据 type 字段精确收窄类型
function handleStreamEvent(event: AnthropicStreamEvent): void {
  switch (event.type) {
    case "text_delta":
      // event: TextDelta
      process.stdout.write(event.text);
      break;

    case "input_json_delta":
      // event: InputJsonDelta
      console.log("工具参数片段:", event.partial_json);
      break;

    case "message_start":
      // event: MessageStartEvent
      console.log(`\n消息开始 [${event.message.model}]`);
      console.log(`输入Token: ${event.message.usage.input_tokens}`);
      break;

    case "content_block_stop":
      // event: ContentBlockStopEvent
      console.log(`\n内容块 ${event.index} 结束`);
      break;

    case "message_stop":
      // event: MessageStopEvent
      console.log("\n消息结束");
      break;

    default:
      // 穷举检查
      const _exhaustive: never = event;
      console.warn("未知事件类型:", _exhaustive);
  }
}
```

### 自定义类型守卫（is 关键字）

当内置的类型守卫不够用时，可以编写**自定义类型守卫函数**。函数的返回类型使用 `参数名 is 类型` 语法：

```typescript
interface OpenAIResponse {
  id: string;
  choices: { message: { content: string } }[];
  usage: { total_tokens: number };
}

interface AnthropicResponse {
  id: string;
  content: { type: string; text: string }[];
  usage: { input_tokens: number; output_tokens: number };
}

// 自定义类型守卫：判断是否为 OpenAI 响应
function isOpenAIResponse(
  response: OpenAIResponse | AnthropicResponse
): response is OpenAIResponse {
  return "choices" in response;
}

// 自定义类型守卫：判断是否为 Anthropic 响应
function isAnthropicResponse(
  response: OpenAIResponse | AnthropicResponse
): response is AnthropicResponse {
  return "content" in response && Array.isArray(
    (response as AnthropicResponse).content
  );
}

function extractText(response: OpenAIResponse | AnthropicResponse): string {
  if (isOpenAIResponse(response)) {
    // response: OpenAIResponse
    return response.choices[0].message.content;
  }

  if (isAnthropicResponse(response)) {
    // response: AnthropicResponse
    return response.content
      .filter(block => block.type === "text")
      .map(block => block.text)
      .join("");
  }

  throw new Error("无法识别的响应格式");
}
```

### 类型守卫与泛型结合

自定义类型守卫可以与泛型结合，实现通用的类型检查：

```typescript
// 通用的非空检查守卫
function isNotNull<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}

// 使用示例：过滤掉数组中的 null/undefined
const responses: (string | null | undefined)[] = [
  "响应1",
  null,
  "响应2",
  undefined,
  "响应3",
];

const validResponses: string[] = responses.filter(isNotNull);
// validResponses: ["响应1", "响应2", "响应3"]

// 通用的类型检查守卫
function hasProperty<T extends object, K extends PropertyKey>(
  obj: T,
  key: K
): obj is T & Record<K, unknown> {
  return key in obj;
}

// 使用示例
function parseApiResponse(data: unknown): string {
  if (
    typeof data === "object" &&
    data !== null &&
    hasProperty(data, "choices") &&
    Array.isArray((data as any).choices) &&
    (data as any).choices.length > 0
  ) {
    // data 现在具有 choices 属性
    return String((data as any).choices[0]?.message?.content ?? "");
  }
  throw new Error("响应格式无效");
}
```

---

## 5.5 断言函数（asserts 关键字）

### 什么是断言函数

**断言函数**（Assertion Functions）是TypeScript 3.7引入的特性。与类型守卫返回 `boolean` 不同，断言函数在类型检查失败时会抛出错误，成功时则通过 `asserts` 关键字向TypeScript保证参数满足某种类型。

语法：
```typescript
function assert(condition: unknown): asserts condition;
function assertIsType(value: unknown): asserts value is SomeType;
```

### 基本断言函数

```typescript
// 通用断言函数：断言条件为真
function assert(
  condition: unknown,
  message: string = "断言失败"
): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

// 使用示例
function processApiKey(apiKey: string | undefined): void {
  assert(apiKey !== undefined, "API密钥未设置");
  // 此处 TypeScript 知道 apiKey 是 string
  console.log(`使用API密钥: ${apiKey.slice(0, 8)}...`);
}

processApiKey(process.env.OPENAI_API_KEY);
```

### 断言特定类型

```typescript
interface LLMConfig {
  apiKey: string;
  model: string;
  maxTokens: number;
}

// 断言值是 LLMConfig 类型
function assertIsLLMConfig(value: unknown): asserts value is LLMConfig {
  if (typeof value !== "object" || value === null) {
    throw new TypeError("配置必须是对象");
  }

  const config = value as Record<string, unknown>;

  if (typeof config.apiKey !== "string") {
    throw new TypeError("apiKey 必须是字符串");
  }

  if (typeof config.model !== "string") {
    throw new TypeError("model 必须是字符串");
  }

  if (typeof config.maxTokens !== "number") {
    throw new TypeError("maxTokens 必须是数字");
  }
}

// 加载配置并进行类型断言
function loadConfig(rawConfig: unknown): LLMConfig {
  assertIsLLMConfig(rawConfig);
  // rawConfig: LLMConfig ✅
  return rawConfig;
}

// 使用示例
try {
  const config = loadConfig({
    apiKey: "sk-xxx",
    model: "gpt-4",
    maxTokens: 2048,
  });
  console.log(`模型: ${config.model}`);  // 完整的类型提示
} catch (error) {
  console.error("配置无效:", error);
}
```

### 断言函数 vs 类型守卫的区别

```typescript
type UserId = string & { __brand: "UserId" };

// 方式一：类型守卫（返回 boolean，不抛出）
function isUserId(value: string): value is UserId {
  return /^user_[a-z0-9]{16}$/.test(value);
}

// 方式二：断言函数（失败时抛出，成功后类型自动收窄）
function assertIsUserId(value: string): asserts value is UserId {
  if (!/^user_[a-z0-9]{16}$/.test(value)) {
    throw new Error(`无效的用户ID格式: "${value}"`);
  }
}

// 类型守卫的使用：需要 if 分支
function processUserId_Guard(value: string): void {
  if (isUserId(value)) {
    // value: UserId
    console.log("有效的用户ID:", value);
  } else {
    console.error("无效的用户ID");
  }
}

// 断言函数的使用：直接收窄，简洁
function processUserId_Assert(value: string): void {
  assertIsUserId(value);
  // 之后 value: UserId，后续代码无需 if 分支
  console.log("有效的用户ID:", value);
}
```

> **何时选择断言函数？** 当你确定某个值"必须"满足某个类型，否则整个流程无法继续时，使用断言函数；当你需要根据类型执行不同逻辑时，使用类型守卫。

### 实际场景：解析SSE事件数据

```typescript
interface SSEEvent {
  event?: string;
  data: string;
  id?: string;
}

// 断言函数：验证原始数据是有效的 SSE 事件
function assertIsSSEEvent(value: unknown): asserts value is SSEEvent {
  if (typeof value !== "object" || value === null) {
    throw new TypeError("SSE事件必须是对象");
  }

  const obj = value as Record<string, unknown>;

  if (typeof obj.data !== "string") {
    throw new TypeError("SSE事件的 data 字段必须是字符串");
  }

  if (obj.event !== undefined && typeof obj.event !== "string") {
    throw new TypeError("SSE事件的 event 字段必须是字符串");
  }
}

function parseSSELine(rawLine: string): SSEEvent | null {
  try {
    const parsed = JSON.parse(rawLine);
    assertIsSSEEvent(parsed);
    // parsed: SSEEvent ✅
    return parsed;
  } catch {
    return null;
  }
}
```

---

## 本章小结

| 技术 | 语法 | 适用场景 | 示例 |
|---|---|---|---|
| **类型收窄** | 控制流自动推断 | 条件分支内自动生效 | `if (x !== null)` 后 x 不为 null |
| **typeof守卫** | `typeof x === "string"` | 原始类型判断 | 区分 string / number / boolean |
| **instanceof守卫** | `x instanceof ClassName` | 类实例判断 | 区分 Error 子类、自定义类 |
| **in操作符** | `"key" in obj` | 属性存在性判断 | 区分接口成员 |
| **判别联合** | `obj.type === "xxx"` | 带判别字段的联合类型 | SSE事件类型区分（推荐） |
| **自定义守卫** | `x is SomeType` 返回类型 | 复杂类型判断逻辑封装 | `isOpenAIResponse(resp)` |
| **断言函数** | `asserts x is SomeType` | 必须满足类型否则抛出 | 配置验证、环境变量检查 |

**选型建议**：
- 简单场景优先用**判别联合**（`type` 字段），语义清晰，IDE支持最好
- 需要复用判断逻辑时用**自定义类型守卫**
- 程序"无法继续"时用**断言函数**（相当于 `assert` + 类型收窄）
- 避免过度使用 `as` 类型断言替代类型守卫——`as` 会绕过类型检查

---

## AI应用实战：流式响应类型判断

### 场景描述

现代AI应用广泛使用**Server-Sent Events（SSE）**实现流式输出。以Anthropic Claude API为例，流式响应会依次发送多种事件类型：

- `message_start`：消息开始，包含模型信息
- `content_block_start`：内容块开始（文本块或工具调用块）
- `content_block_delta`：内容块增量（文本片段或工具参数片段）
- `content_block_stop`：内容块结束
- `message_delta`：消息级增量（stop_reason等）
- `message_stop`：消息结束

每种事件携带不同的数据结构，我们需要用类型守卫精确区分，才能安全地提取内容。

### 完整实现

```typescript
// ============================================================
// 类型定义：Anthropic 流式响应事件
// ============================================================

// --- 基础块类型 ---
interface TextBlock {
  type: "text";
  text: string;
}

interface ToolUseBlock {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
}

type ContentBlock = TextBlock | ToolUseBlock;

// --- 增量类型 ---
interface TextDelta {
  type: "text_delta";
  text: string;
}

interface InputJsonDelta {
  type: "input_json_delta";
  partial_json: string;
}

type Delta = TextDelta | InputJsonDelta;

// --- 事件类型（判别联合） ---
interface MessageStartEvent {
  type: "message_start";
  message: {
    id: string;
    type: "message";
    role: "assistant";
    model: string;
    usage: {
      input_tokens: number;
      output_tokens: number;
    };
  };
}

interface ContentBlockStartEvent {
  type: "content_block_start";
  index: number;
  content_block: ContentBlock;
}

interface ContentBlockDeltaEvent {
  type: "content_block_delta";
  index: number;
  delta: Delta;
}

interface ContentBlockStopEvent {
  type: "content_block_stop";
  index: number;
}

interface MessageDeltaEvent {
  type: "message_delta";
  delta: {
    stop_reason: "end_turn" | "tool_use" | "max_tokens" | "stop_sequence";
    stop_sequence: string | null;
  };
  usage: {
    output_tokens: number;
  };
}

interface MessageStopEvent {
  type: "message_stop";
}

// 所有可能的流式事件
type AnthropicStreamEvent =
  | MessageStartEvent
  | ContentBlockStartEvent
  | ContentBlockDeltaEvent
  | ContentBlockStopEvent
  | MessageDeltaEvent
  | MessageStopEvent;

// ============================================================
// 自定义类型守卫
// ============================================================

// 守卫：判断是否为文本增量事件
function isTextDeltaEvent(
  event: AnthropicStreamEvent
): event is ContentBlockDeltaEvent & { delta: TextDelta } {
  return (
    event.type === "content_block_delta" &&
    event.delta.type === "text_delta"
  );
}

// 守卫：判断是否为工具调用增量事件
function isToolCallDeltaEvent(
  event: AnthropicStreamEvent
): event is ContentBlockDeltaEvent & { delta: InputJsonDelta } {
  return (
    event.type === "content_block_delta" &&
    event.delta.type === "input_json_delta"
  );
}

// 守卫：判断是否为工具调用块开始事件
function isToolUseBlockStart(
  event: AnthropicStreamEvent
): event is ContentBlockStartEvent & { content_block: ToolUseBlock } {
  return (
    event.type === "content_block_start" &&
    event.content_block.type === "tool_use"
  );
}

// 守卫：判断事件是否为结束信号
function isTerminalEvent(
  event: AnthropicStreamEvent
): event is MessageStopEvent | MessageDeltaEvent {
  return event.type === "message_stop" || event.type === "message_delta";
}

// 断言函数：验证原始数据是合法的流式事件
function assertIsStreamEvent(data: unknown): asserts data is AnthropicStreamEvent {
  if (typeof data !== "object" || data === null) {
    throw new TypeError(`流事件必须是对象，收到: ${typeof data}`);
  }

  const obj = data as Record<string, unknown>;

  const validTypes = [
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
  ];

  if (!validTypes.includes(obj.type as string)) {
    throw new TypeError(`未知的流事件类型: "${obj.type}"`);
  }
}

// ============================================================
// 流式响应处理器
// ============================================================

interface StreamProcessorCallbacks {
  onText: (text: string) => void;
  onToolCallStart: (toolName: string, toolId: string) => void;
  onToolCallDelta: (partialJson: string) => void;
  onToolCallEnd: (toolName: string) => void;
  onComplete: (stopReason: string, outputTokens: number) => void;
  onError: (error: Error) => void;
}

class AnthropicStreamProcessor {
  private activeToolName: string | null = null;
  private activeToolId: string | null = null;
  private totalOutputTokens: number = 0;

  constructor(private callbacks: StreamProcessorCallbacks) {}

  // 解析 SSE 原始行，提取事件对象
  private parseSSELine(line: string): AnthropicStreamEvent | null {
    // SSE 格式: "data: {...json...}"
    if (!line.startsWith("data: ")) {
      return null;
    }

    const jsonStr = line.slice(6); // 去掉 "data: " 前缀

    if (jsonStr === "[DONE]") {
      return null; // OpenAI 兼容格式的结束标记
    }

    try {
      const parsed: unknown = JSON.parse(jsonStr);
      assertIsStreamEvent(parsed);
      return parsed;
    } catch (error) {
      if (error instanceof SyntaxError) {
        this.callbacks.onError(new Error(`JSON解析失败: ${jsonStr}`));
      } else if (error instanceof TypeError) {
        this.callbacks.onError(error);
      }
      return null;
    }
  }

  // 处理单个流式事件
  processEvent(event: AnthropicStreamEvent): void {
    switch (event.type) {
      case "message_start":
        // 记录模型信息（可选）
        console.log(`[流开始] 模型: ${event.message.model}`);
        break;

      case "content_block_start":
        if (isToolUseBlockStart(event)) {
          // event.content_block: ToolUseBlock ✅
          this.activeToolName = event.content_block.name;
          this.activeToolId = event.content_block.id;
          this.callbacks.onToolCallStart(
            event.content_block.name,
            event.content_block.id
          );
        }
        break;

      case "content_block_delta":
        if (isTextDeltaEvent(event)) {
          // event.delta: TextDelta ✅
          this.callbacks.onText(event.delta.text);
        } else if (isToolCallDeltaEvent(event)) {
          // event.delta: InputJsonDelta ✅
          this.callbacks.onToolCallDelta(event.delta.partial_json);
        }
        break;

      case "content_block_stop":
        if (this.activeToolName !== null) {
          this.callbacks.onToolCallEnd(this.activeToolName);
          this.activeToolName = null;
          this.activeToolId = null;
        }
        break;

      case "message_delta":
        this.totalOutputTokens = event.usage.output_tokens;
        this.callbacks.onComplete(
          event.delta.stop_reason,
          event.usage.output_tokens
        );
        break;

      case "message_stop":
        // 消息彻底结束
        break;

      default:
        // 穷举检查：如果新增了事件类型但忘记处理，编译时报错
        const _exhaustive: never = event;
        console.warn("未处理的事件类型:", _exhaustive);
    }
  }

  // 处理 SSE 原始行
  processLine(line: string): void {
    const event = this.parseSSELine(line);
    if (event !== null) {
      this.processEvent(event);
    }
  }

  // 处理完整的 SSE 流
  async processStream(response: Response): Promise<void> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("响应体不可读");
    }

    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        // 按换行符分割，处理完整的 SSE 行
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? ""; // 保留不完整的最后一行

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed) {
            this.processLine(trimmed);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}

// ============================================================
// 使用示例
// ============================================================

async function streamingChat(
  prompt: string,
  apiKey: string
): Promise<void> {
  // 收集完整的响应文本
  let fullText = "";
  const toolCalls: Map<string, string> = new Map();
  let currentToolId: string | null = null;

  const processor = new AnthropicStreamProcessor({
    onText: (text) => {
      process.stdout.write(text);
      fullText += text;
    },

    onToolCallStart: (toolName, toolId) => {
      currentToolId = toolId;
      toolCalls.set(toolId, "");
      console.log(`\n[工具调用开始] ${toolName} (${toolId})`);
    },

    onToolCallDelta: (partialJson) => {
      if (currentToolId) {
        toolCalls.set(
          currentToolId,
          (toolCalls.get(currentToolId) ?? "") + partialJson
        );
      }
    },

    onToolCallEnd: (toolName) => {
      console.log(`\n[工具调用结束] ${toolName}`);
      if (currentToolId) {
        const args = toolCalls.get(currentToolId);
        console.log("参数:", args);
      }
    },

    onComplete: (stopReason, outputTokens) => {
      console.log(`\n\n[完成] 停止原因: ${stopReason}, 输出Token: ${outputTokens}`);
    },

    onError: (error) => {
      console.error("[流错误]", error.message);
    },
  });

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-beta": "messages-2023-12-15",
    },
    body: JSON.stringify({
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 1024,
      stream: true,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  if (!response.ok) {
    throw new NetworkError(response.status, await response.text());
  }

  await processor.processStream(response);
}

// 运行示例（需要真实 API 密钥）
// streamingChat("用一句话描述TypeScript的最大优点", process.env.ANTHROPIC_API_KEY ?? "")
//   .catch(console.error);
```

### 关键设计要点

1. **判别联合优先**：所有事件类型都有 `type` 字段，`switch (event.type)` 是最简洁的分发方式，且TypeScript能验证穷举性。

2. **嵌套收窄**：`isTextDeltaEvent` 和 `isToolCallDeltaEvent` 处理了两层收窄——先确认是 `content_block_delta`，再确认 `delta` 的子类型。

3. **断言函数用于边界验证**：`assertIsStreamEvent` 在数据进入系统时验证，之后代码可以安全地使用类型。

4. **穷举检查保护未来**：`default` 分支的 `never` 赋值确保新增事件类型时不会被遗漏。

---

## 练习题

### 基础题

**练习 5-1**：使用 `typeof` 守卫实现函数

实现一个函数 `formatTokenCount`，接受参数类型为 `number | string | null`：
- 如果是 `number`，返回 `"${value} tokens"`
- 如果是 `string`，先尝试转换为整数，再格式化；如果转换失败返回 `"无效数值"`
- 如果是 `null`，返回 `"未知"`

```typescript
function formatTokenCount(value: number | string | null): string {
  // 你的代码
}

// 期望输出：
console.log(formatTokenCount(1234));       // "1234 tokens"
console.log(formatTokenCount("567"));     // "567 tokens"
console.log(formatTokenCount("abc"));     // "无效数值"
console.log(formatTokenCount(null));      // "未知"
```

---

**练习 5-2**：使用 `instanceof` 区分错误类型

定义三个错误类：`ApiError`（含 `statusCode: number`）、`TimeoutError`（含 `timeoutMs: number`）、`ParseError`（含 `rawData: string`）。实现函数 `describeError` 接受 `unknown` 类型，返回中文错误描述字符串。

```typescript
class ApiError extends Error { /* ... */ }
class TimeoutError extends Error { /* ... */ }
class ParseError extends Error { /* ... */ }

function describeError(error: unknown): string {
  // 你的代码
}

// 期望输出：
console.log(describeError(new ApiError(404, "Not Found")));
// "API错误 [404]: Not Found"

console.log(describeError(new TimeoutError(30000)));
// "请求超时，超时时间: 30000ms"

console.log(describeError(new ParseError('{"broken"')));
// "数据解析失败，原始数据: {\"broken\""

console.log(describeError("unknown"));
// "未知错误: unknown"
```

---

### 中级题

**练习 5-3**：实现自定义类型守卫

定义以下类型，并实现所有类型守卫函数：

```typescript
interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface ToolMessage {
  role: "tool";
  tool_call_id: string;
  content: string;
}

interface SystemMessage {
  role: "system";
  content: string;
}

type Message = ChatMessage | ToolMessage | SystemMessage;

// 实现以下类型守卫：
function isChatMessage(msg: Message): msg is ChatMessage { /* ... */ }
function isToolMessage(msg: Message): msg is ToolMessage { /* ... */ }
function isSystemMessage(msg: Message): msg is SystemMessage { /* ... */ }

// 实现以下函数，统计各类型消息数量
function countMessageTypes(messages: Message[]): {
  chat: number;
  tool: number;
  system: number;
} {
  // 你的代码
}
```

---

**练习 5-4**：判别联合与穷举检查

实现一个OpenAI流式响应处理器。定义以下判别联合类型并处理所有分支：

```typescript
interface OpenAIChunk {
  id: string;
  object: "chat.completion.chunk";
  choices: Array<{
    index: number;
    delta: {
      role?: "assistant";
      content?: string;
      tool_calls?: Array<{
        index: number;
        id?: string;
        type?: "function";
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason: "stop" | "tool_calls" | "length" | null;
  }>;
}

// 定义三种语义明确的块类型（判别联合）
type ProcessedChunk =
  | { kind: "text"; content: string }
  | { kind: "tool_call"; name: string; argumentsDelta: string }
  | { kind: "done"; reason: "stop" | "tool_calls" | "length" };

// 实现转换函数（必须处理所有情况，加穷举检查）
function processOpenAIChunk(raw: OpenAIChunk): ProcessedChunk | null {
  // 你的代码
}
```

---

### 提高题

**练习 5-5**：实现通用的流式事件路由器

实现一个类型安全的事件路由器 `StreamEventRouter`，能够：
1. 注册事件处理器：`on(eventType, handler)`
2. 处理事件：`emit(event)`
3. 支持通配符处理器：`onAny(handler)`
4. 处理器类型必须与事件类型严格匹配（利用条件类型和重载实现）

```typescript
// 要求：以下代码应该类型安全且不报错
const router = new StreamEventRouter();

// 注册文本事件处理器（handler 参数应为 TextDelta 类型）
router.on("text_delta", (event: TextDelta) => {
  process.stdout.write(event.text);
});

// 注册消息开始处理器
router.on("message_start", (event: MessageStartEvent) => {
  console.log("模型:", event.message.model);
});

// 注册通配符处理器
router.onAny((event: AnthropicStreamEvent) => {
  console.log("事件类型:", event.type);
});

// 类型错误：handler 类型与 "text_delta" 事件不匹配
// router.on("text_delta", (event: MessageStartEvent) => {});  // ❌ 编译错误

// 分发事件
const event: AnthropicStreamEvent = { type: "text_delta", text: "Hello" };
router.emit(event);
```

---

## 练习答案

### 答案 5-1

```typescript
function formatTokenCount(value: number | string | null): string {
  if (value === null) {
    return "未知";
  }

  if (typeof value === "number") {
    return `${value} tokens`;
  }

  // typeof value === "string"
  const parsed = parseInt(value, 10);
  if (isNaN(parsed)) {
    return "无效数值";
  }

  return `${parsed} tokens`;
}

console.log(formatTokenCount(1234));   // "1234 tokens"
console.log(formatTokenCount("567")); // "567 tokens"
console.log(formatTokenCount("abc")); // "无效数值"
console.log(formatTokenCount(null));  // "未知"
```

---

### 答案 5-2

```typescript
class ApiError extends Error {
  constructor(
    public statusCode: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

class TimeoutError extends Error {
  constructor(public timeoutMs: number) {
    super(`请求在 ${timeoutMs}ms 后超时`);
    this.name = "TimeoutError";
  }
}

class ParseError extends Error {
  constructor(public rawData: string) {
    super("数据解析失败");
    this.name = "ParseError";
  }
}

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    return `API错误 [${error.statusCode}]: ${error.message}`;
  }

  if (error instanceof TimeoutError) {
    return `请求超时，超时时间: ${error.timeoutMs}ms`;
  }

  if (error instanceof ParseError) {
    return `数据解析失败，原始数据: ${error.rawData}`;
  }

  if (error instanceof Error) {
    return `系统错误: ${error.message}`;
  }

  return `未知错误: ${String(error)}`;
}

console.log(describeError(new ApiError(404, "Not Found")));
// "API错误 [404]: Not Found"

console.log(describeError(new TimeoutError(30000)));
// "请求超时，超时时间: 30000ms"

console.log(describeError(new ParseError('{"broken"')));
// "数据解析失败，原始数据: {\"broken\""

console.log(describeError("unknown"));
// "未知错误: unknown"
```

---

### 答案 5-3

```typescript
interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface ToolMessage {
  role: "tool";
  tool_call_id: string;
  content: string;
}

interface SystemMessage {
  role: "system";
  content: string;
}

type Message = ChatMessage | ToolMessage | SystemMessage;

function isChatMessage(msg: Message): msg is ChatMessage {
  return msg.role === "user" || msg.role === "assistant";
}

function isToolMessage(msg: Message): msg is ToolMessage {
  return msg.role === "tool";
}

function isSystemMessage(msg: Message): msg is SystemMessage {
  return msg.role === "system";
}

function countMessageTypes(messages: Message[]): {
  chat: number;
  tool: number;
  system: number;
} {
  return messages.reduce(
    (counts, msg) => {
      if (isChatMessage(msg)) {
        counts.chat += 1;
      } else if (isToolMessage(msg)) {
        counts.tool += 1;
      } else if (isSystemMessage(msg)) {
        counts.system += 1;
      }
      return counts;
    },
    { chat: 0, tool: 0, system: 0 }
  );
}

// 测试
const messages: Message[] = [
  { role: "system", content: "你是一个助手" },
  { role: "user", content: "你好" },
  { role: "assistant", content: "你好！" },
  { role: "tool", tool_call_id: "call_1", content: "42" },
  { role: "user", content: "谢谢" },
];

console.log(countMessageTypes(messages));
// { chat: 3, tool: 1, system: 1 }
```

---

### 答案 5-4

```typescript
interface OpenAIChunk {
  id: string;
  object: "chat.completion.chunk";
  choices: Array<{
    index: number;
    delta: {
      role?: "assistant";
      content?: string;
      tool_calls?: Array<{
        index: number;
        id?: string;
        type?: "function";
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason: "stop" | "tool_calls" | "length" | null;
  }>;
}

type ProcessedChunk =
  | { kind: "text"; content: string }
  | { kind: "tool_call"; name: string; argumentsDelta: string }
  | { kind: "done"; reason: "stop" | "tool_calls" | "length" };

function processOpenAIChunk(raw: OpenAIChunk): ProcessedChunk | null {
  const choice = raw.choices[0];
  if (!choice) {
    return null;
  }

  // 处理结束信号
  if (choice.finish_reason !== null) {
    return { kind: "done", reason: choice.finish_reason };
  }

  const { delta } = choice;

  // 处理工具调用
  if (delta.tool_calls && delta.tool_calls.length > 0) {
    const toolCall = delta.tool_calls[0];
    return {
      kind: "tool_call",
      name: toolCall.function?.name ?? "",
      argumentsDelta: toolCall.function?.arguments ?? "",
    };
  }

  // 处理文本内容
  if (delta.content !== undefined && delta.content !== null) {
    return { kind: "text", content: delta.content };
  }

  return null;
}

// 使用示例
function renderChunk(chunk: ProcessedChunk): void {
  switch (chunk.kind) {
    case "text":
      // chunk: { kind: "text"; content: string } ✅
      process.stdout.write(chunk.content);
      break;

    case "tool_call":
      // chunk: { kind: "tool_call"; ... } ✅
      console.log(`\n[工具: ${chunk.name}] ${chunk.argumentsDelta}`);
      break;

    case "done":
      // chunk: { kind: "done"; reason: ... } ✅
      console.log(`\n[完成，原因: ${chunk.reason}]`);
      break;

    default:
      // 穷举检查
      const _exhaustive: never = chunk;
      console.warn("未处理的块类型:", _exhaustive);
  }
}
```

---

### 答案 5-5

```typescript
// 利用条件类型从联合类型中提取特定 type 对应的事件类型
type ExtractEvent<T extends AnthropicStreamEvent, K extends string> =
  T extends { type: K } ? T : never;

// 类型安全的事件路由器
class StreamEventRouter {
  private handlers: Map<string, ((event: AnthropicStreamEvent) => void)[]> =
    new Map();
  private anyHandlers: ((event: AnthropicStreamEvent) => void)[] = [];

  // 重载：根据 eventType 推断 handler 的参数类型
  on<K extends AnthropicStreamEvent["type"]>(
    eventType: K,
    handler: (event: ExtractEvent<AnthropicStreamEvent, K>) => void
  ): this {
    const list = this.handlers.get(eventType) ?? [];
    // 类型转换：内部实现时使用宽松类型，对外接口类型安全
    list.push(handler as (event: AnthropicStreamEvent) => void);
    this.handlers.set(eventType, list);
    return this;
  }

  onAny(handler: (event: AnthropicStreamEvent) => void): this {
    this.anyHandlers.push(handler);
    return this;
  }

  emit(event: AnthropicStreamEvent): void {
    // 调用特定类型的处理器
    const handlers = this.handlers.get(event.type) ?? [];
    for (const handler of handlers) {
      handler(event);
    }

    // 调用通配符处理器
    for (const handler of this.anyHandlers) {
      handler(event);
    }
  }
}

// 测试
const router = new StreamEventRouter();

router.on("text_delta", (event) => {
  // event: TextDelta ✅
  process.stdout.write(event.text);
});

router.on("message_start", (event) => {
  // event: MessageStartEvent ✅
  console.log("模型:", event.message.model);
});

router.onAny((event) => {
  console.log("事件类型:", event.type);
});

// 分发测试事件
router.emit({ type: "text_delta", text: "Hello, TypeScript!" });
router.emit({
  type: "message_start",
  message: {
    id: "msg_01",
    type: "message",
    role: "assistant",
    model: "claude-3-5-sonnet-20241022",
    usage: { input_tokens: 10, output_tokens: 0 },
  },
});
```

---

[上一章：联合类型与交叉类型](./04-union-intersection.md) | [下一章：泛型编程](./06-generics.md)
