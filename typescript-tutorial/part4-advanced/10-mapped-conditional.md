# 第10章：映射类型与条件类型

> **适用读者**：已完成第9章模块系统学习，希望深入掌握 TypeScript 高级类型变换能力的开发者。

---

## 学习目标

完成本章学习后，你将能够：

1. 理解映射类型（Mapped Types）的 `in keyof` 语法，并用它批量转换对象类型的属性
2. 掌握映射修饰符 `readonly`、`?` 及其移除前缀 `-`，灵活控制属性的可变性与可选性
3. 使用条件类型（Conditional Types）`T extends U ? X : Y` 根据类型关系做出类型决策
4. 掌握 `infer` 关键字，从复杂类型中提取内部类型信息
5. 理解分布式条件类型（Distributive Conditional Types）的运作规则，并将其应用于 AI API 参数的自动类型推导

---

## 10.1 映射类型基础（in keyof）

### 10.1.1 什么是映射类型

映射类型是 TypeScript 中一种通过**遍历已有类型的键**来创建新类型的机制。它的核心语法是：

```typescript
type NewType = {
  [K in keyof SourceType]: NewPropertyType;
};
```

这里 `K in keyof SourceType` 意味着"遍历 `SourceType` 的每一个键 `K`"，并为每个键定义新的值类型。

最简单的例子：将类型的所有属性值改为 `string`：

```typescript
interface AIModel {
  name: string;
  version: number;
  contextWindow: number;
  supportsVision: boolean;
}

// 将所有属性值变为 string
type Stringified<T> = {
  [K in keyof T]: string;
};

type StringifiedModel = Stringified<AIModel>;
// 等价于：
// {
//   name: string;
//   version: string;
//   contextWindow: string;
//   supportsVision: string;
// }
```

### 10.1.2 同态映射（Homomorphic Mapping）

当映射类型的形式是 `[K in keyof T]: ...` 时，称为**同态映射**。它会自动保留原类型属性的 `readonly` 和 `?` 修饰符：

```typescript
interface RequestConfig {
  readonly model: string;
  temperature?: number;
  maxTokens: number;
}

// 同态映射：保留 readonly 和 ? 修饰符
type Cloned<T> = {
  [K in keyof T]: T[K];
};

type ClonedConfig = Cloned<RequestConfig>;
// {
//   readonly model: string;    ← 保留 readonly
//   temperature?: number;      ← 保留 ?
//   maxTokens: number;
// }
```

### 10.1.3 通过映射类型实现内置工具类型

TypeScript 内置的 `Readonly<T>` 和 `Partial<T>` 都是映射类型实现的：

```typescript
// 自己实现 Readonly<T>
type MyReadonly<T> = {
  readonly [K in keyof T]: T[K];
};

// 自己实现 Partial<T>
type MyPartial<T> = {
  [K in keyof T]?: T[K];
};

// 自己实现 Required<T>（移除可选修饰符）
type MyRequired<T> = {
  [K in keyof T]-?: T[K];
};

// 使用示例
interface ChatOptions {
  model: string;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
}

type ReadonlyOptions = MyReadonly<ChatOptions>;
// 所有属性变为 readonly

type RequiredOptions = MyRequired<ChatOptions>;
// 所有属性变为必填，包括原来的 temperature、maxTokens、stream
```

### 10.1.4 键重映射（Key Remapping）— TypeScript 4.1+

使用 `as` 子句可以在映射时**重命名键**：

```typescript
// 将所有键名加上 "get" 前缀，生成 getter 类型
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K];
};

interface ModelInfo {
  name: string;
  version: number;
  provider: string;
}

type ModelGetters = Getters<ModelInfo>;
// {
//   getName: () => string;
//   getVersion: () => number;
//   getProvider: () => string;
// }

// 结合条件类型过滤键（值为 never 的键会被移除）
type OnlyStringValues<T> = {
  [K in keyof T as T[K] extends string ? K : never]: T[K];
};

interface MixedConfig {
  apiKey: string;
  baseUrl: string;
  timeout: number;
  retries: number;
  debug: boolean;
}

type StringConfig = OnlyStringValues<MixedConfig>;
// { apiKey: string; baseUrl: string }
```

---

## 10.2 映射修饰符（readonly、?、-）

### 10.2.1 添加修饰符

在映射类型中，可以为每个属性添加 `readonly`（只读）或 `?`（可选）修饰符：

```typescript
// 添加 readonly
type Immutable<T> = {
  readonly [K in keyof T]: T[K];
};

// 添加 ?（可选）
type Optional<T> = {
  [K in keyof T]?: T[K];
};

// 同时添加两者
type ImmutableOptional<T> = {
  readonly [K in keyof T]?: T[K];
};

// 实际应用：AI 请求的不可变快照
interface ChatRequest {
  model: string;
  messages: Array<{ role: string; content: string }>;
  temperature: number;
}

type RequestSnapshot = Immutable<ChatRequest>;

const snapshot: RequestSnapshot = {
  model: "claude-sonnet-4-5-20251101",
  messages: [{ role: "user", content: "你好" }],
  temperature: 0.7,
};

// snapshot.model = "gpt-4";  // 错误：只读属性不可修改
```

### 10.2.2 移除修饰符（前缀 `-`）

在修饰符前加 `-` 表示**移除**该修饰符。这是 `Required<T>` 的实现原理：

```typescript
// 移除所有属性的 readonly
type Mutable<T> = {
  -readonly [K in keyof T]: T[K];
};

// 移除所有属性的 ?（变为必填）
type Complete<T> = {
  [K in keyof T]-?: T[K];
};

// 同时移除 readonly 和 ?
type MutableComplete<T> = {
  -readonly [K in keyof T]-?: T[K];
};

// 实际应用：从只读配置创建可修改副本
interface FrozenConfig {
  readonly apiKey: string;
  readonly model?: string;
  readonly maxTokens?: number;
}

type EditableConfig = Mutable<FrozenConfig>;
// { apiKey: string; model?: string; maxTokens?: number }  ← readonly 被移除

type FullConfig = MutableComplete<FrozenConfig>;
// { apiKey: string; model: string; maxTokens: number }  ← 移除 readonly 和 ?
```

### 10.2.3 深层映射（Recursive Mapped Types）

映射类型可以递归应用，处理嵌套对象：

```typescript
// 深度只读：递归使所有嵌套属性变为 readonly
type DeepReadonly<T> = {
  readonly [K in keyof T]: T[K] extends object ? DeepReadonly<T[K]> : T[K];
};

interface ProviderConfig {
  openai: {
    apiKey: string;
    models: {
      chat: string;
      embedding: string;
    };
  };
  anthropic: {
    apiKey: string;
    version: string;
  };
}

type FrozenProviderConfig = DeepReadonly<ProviderConfig>;

const config: FrozenProviderConfig = {
  openai: {
    apiKey: "sk-xxx",
    models: { chat: "gpt-4", embedding: "text-embedding-3-small" },
  },
  anthropic: {
    apiKey: "sk-ant-xxx",
    version: "2023-06-01",
  },
};

// config.openai.apiKey = "new-key";        // 错误：只读
// config.openai.models.chat = "gpt-4o";   // 错误：深层也是只读
```

### 10.2.4 映射类型与联合类型键

`keyof` 不仅适用于接口/类，还可与联合类型结合，生成动态的键集合：

```typescript
type Provider = "openai" | "anthropic" | "google";

// 为每个提供商生成配置键
type ProviderKeys = {
  [K in Provider]: `${K}ApiKey`;
}[Provider];
// 结果："openaiApiKey" | "anthropicApiKey" | "googleApiKey"

// 为每个提供商生成配置对象
type ProviderConfigs = {
  [K in Provider]: {
    apiKey: string;
    baseUrl: string;
    model: string;
  };
};

type AllProviderConfigs = ProviderConfigs;
// {
//   openai: { apiKey: string; baseUrl: string; model: string };
//   anthropic: { apiKey: string; baseUrl: string; model: string };
//   google: { apiKey: string; baseUrl: string; model: string };
// }
```

---

## 10.3 条件类型基础（extends ? :）

### 10.3.1 条件类型语法

条件类型的语法类似三元运算符，根据类型关系返回不同类型：

```typescript
type IsString<T> = T extends string ? true : false;

type A = IsString<string>;    // true
type B = IsString<number>;    // false
type C = IsString<"hello">;   // true（"hello" 可赋值给 string）
type D = IsString<string | number>; // boolean（分布式，见 10.5 节）
```

条件类型最常用的场景是**根据泛型参数的形状选择不同类型**：

```typescript
// 如果 T 是数组，返回元素类型；否则返回 T 本身
type Flatten<T> = T extends Array<infer Item> ? Item : T;

type F1 = Flatten<string[]>;        // string
type F2 = Flatten<number[][]>;      // number[]
type F3 = Flatten<string>;          // string（不是数组，返回自身）

// AI 实战：根据是否传入 stream 参数决定返回类型
type ChatResponse<TStream extends boolean> =
  TStream extends true
    ? AsyncIterable<string>  // 流式模式返回异步迭代器
    : string;                // 非流式模式返回字符串

declare function chat<TStream extends boolean = false>(
  prompt: string,
  stream?: TStream
): Promise<ChatResponse<TStream>>;

const text = await chat("你好");            // Promise<string>
const stream = await chat("你好", true);    // Promise<AsyncIterable<string>>
```

### 10.3.2 嵌套条件类型

条件类型可以嵌套，处理多个类型分支：

```typescript
// 获取 TypeScript 基础类型的名称字符串
type TypeName<T> =
  T extends string ? "string" :
  T extends number ? "number" :
  T extends boolean ? "boolean" :
  T extends undefined ? "undefined" :
  T extends null ? "null" :
  T extends Function ? "function" :
  T extends Array<unknown> ? "array" :
  "object";

type T1 = TypeName<string>;         // "string"
type T2 = TypeName<42>;             // "number"
type T3 = TypeName<string[]>;       // "array"
type T4 = TypeName<{ a: string }>;  // "object"

// AI 实战：根据参数类型决定如何格式化提示词
type FormatPrompt<T> =
  T extends string ? T :
  T extends number ? `${T}` :
  T extends string[] ? string :
  never;

function buildPrompt<T extends string | number | string[]>(input: T): FormatPrompt<T> {
  if (typeof input === "string") return input as FormatPrompt<T>;
  if (typeof input === "number") return String(input) as FormatPrompt<T>;
  return (input as string[]).join("\n") as FormatPrompt<T>;
}
```

### 10.3.3 条件类型的延迟求值

当条件类型依赖于**尚未确定的泛型参数**时，TypeScript 会延迟（defer）其求值，直到类型参数被具体化：

```typescript
// 在泛型函数中，T 未确定时条件类型保持"延迟"状态
function processValue<T>(value: T): T extends string ? string[] : never {
  // 此处需要类型断言，因为 TypeScript 无法在函数体内"看穿"条件类型
  if (typeof value === "string") {
    return value.split("") as T extends string ? string[] : never;
  }
  throw new Error("仅支持 string 类型");
}

const chars = processValue("hello");  // string[]
// processValue(42);  // 返回类型是 never，调用时类型系统会提示问题
```

### 10.3.4 非空类型提取

条件类型可以用来过滤掉 `null` 和 `undefined`：

```typescript
// 移除 null 和 undefined
type NonNullable<T> = T extends null | undefined ? never : T;

type MaybeString = string | null | undefined;
type DefinitelyString = NonNullable<MaybeString>; // string

// AI 实战：过滤掉可能为空的模型响应字段
interface RawModelResponse {
  content: string | null;
  model: string;
  stopReason: string | null | undefined;
  tokenCount: number | undefined;
}

type CleanResponse = {
  [K in keyof RawModelResponse]: NonNullable<RawModelResponse[K]>;
};
// {
//   content: string;
//   model: string;
//   stopReason: string;
//   tokenCount: number;
// }
```

---

## 10.4 infer关键字与类型推断

### 10.4.1 infer 基础语法

`infer` 关键字用于在条件类型的 `extends` 子句中**声明一个待推断的类型变量**，TypeScript 会从实际类型中提取并填充它：

```typescript
// 提取数组元素类型
type ElementType<T> = T extends Array<infer E> ? E : never;

type E1 = ElementType<string[]>;          // string
type E2 = ElementType<Array<number>>;     // number
type E3 = ElementType<[string, number]>;  // string | number（元组）
type E4 = ElementType<string>;            // never（不是数组）

// 提取 Promise 的解析值类型
type Awaited<T> = T extends Promise<infer R> ? R : T;

type A1 = Awaited<Promise<string>>;           // string
type A2 = Awaited<Promise<Promise<number>>>;  // Promise<number>（只解一层）
type A3 = Awaited<string>;                    // string（不是 Promise）
```

### 10.4.2 提取函数参数和返回值类型

`infer` 在处理函数类型时极为有用：

```typescript
// 提取函数参数类型（元组形式）
type Parameters<T extends (...args: any) => any> =
  T extends (...args: infer P) => any ? P : never;

// 提取函数返回值类型
type ReturnType<T extends (...args: any) => any> =
  T extends (...args: any) => infer R ? R : never;

// 提取函数第一个参数类型
type FirstParameter<T extends (...args: any) => any> =
  T extends (first: infer F, ...rest: any[]) => any ? F : never;

// 使用示例
type FetchFn = (url: string, options: RequestInit) => Promise<Response>;

type FetchParams = Parameters<FetchFn>;    // [url: string, options: RequestInit]
type FetchReturn = ReturnType<FetchFn>;    // Promise<Response>
type FetchFirst = FirstParameter<FetchFn>; // string

// AI 实战：提取 AI 处理函数的输入输出类型
type ChatHandler = (
  messages: Array<{ role: string; content: string }>,
  options: { model: string; temperature: number }
) => Promise<{ content: string; usage: { total: number } }>;

type ChatInput = Parameters<ChatHandler>;
// [messages: Array<{ role: string; content: string }>, options: { model: string; temperature: number }]

type ChatOutput = ReturnType<ChatHandler>;
// Promise<{ content: string; usage: { total: number } }>

type ChatResult = Awaited<ChatOutput>;
// { content: string; usage: { total: number } }
```

### 10.4.3 从对象类型中提取嵌套类型

`infer` 可以从更复杂的类型结构中提取嵌套类型：

```typescript
// 提取对象类型某个键的值类型
type PropType<T, K extends keyof T> = T extends { [key in K]: infer V } ? V : never;

interface APIConfig {
  endpoints: {
    chat: string;
    embedding: string;
  };
  auth: {
    type: "bearer" | "apikey";
    token: string;
  };
}

type EndpointsType = PropType<APIConfig, "endpoints">;
// { chat: string; embedding: string }

type AuthType = PropType<APIConfig, "auth">;
// { type: "bearer" | "apikey"; token: string }

// 提取构造函数的实例类型
type InstanceType<T extends new (...args: any) => any> =
  T extends new (...args: any) => infer I ? I : never;

class OpenAIClient {
  chat(prompt: string): Promise<string> {
    return Promise.resolve(`Response to: ${prompt}`);
  }
}

type ClientInstance = InstanceType<typeof OpenAIClient>;
// OpenAIClient（即该类的实例类型）
```

### 10.4.4 多处 infer 与协变逆变

在同一条件类型中可以多次使用 `infer`，也可以在协变（covariant）位置或逆变（contravariant）位置使用：

```typescript
// 同时推断多个类型
type FunctionShape<T> =
  T extends (arg: infer A) => infer R
    ? { argument: A; returnType: R }
    : never;

type Shape = FunctionShape<(x: number) => string>;
// { argument: number; returnType: string }

// 交换函数参数与返回值
type SwapFunction<T extends (arg: any) => any> =
  T extends (arg: infer A) => infer R ? (arg: R) => A : never;

type Original = (x: number) => string;
type Swapped = SwapFunction<Original>; // (arg: string) => number

// 在联合类型中，协变位置的多个 infer 会合并为联合
type UnionInfer<T> =
  T extends { a: infer U; b: infer U } ? U : never;

type UI = UnionInfer<{ a: string; b: number }>; // string | number
```

---

## 10.5 分布式条件类型

### 10.5.1 分布律（Distributive Law）

当条件类型的**被检查类型是裸类型参数**（naked type parameter）且传入的是联合类型时，条件类型会**对每个联合成员分别求值**，然后将结果合并为联合类型：

```typescript
// T 是裸类型参数（未被包裹）
type IsString<T> = T extends string ? "yes" : "no";

// 传入联合类型时，分布式处理
type Result = IsString<string | number | boolean>;
// 等价于：IsString<string> | IsString<number> | IsString<boolean>
// 等价于："yes" | "no" | "no"
// 最终：  "yes" | "no"
```

### 10.5.2 Exclude 与 Extract 的实现原理

TypeScript 内置的 `Exclude<T, U>` 和 `Extract<T, U>` 都依赖分布式条件类型：

```typescript
// Exclude：从 T 中排除可赋值给 U 的类型
type MyExclude<T, U> = T extends U ? never : T;

// Extract：从 T 中提取可赋值给 U 的类型
type MyExtract<T, U> = T extends U ? U : never;

// 实际应用
type AllProviders = "openai" | "anthropic" | "google" | "local" | "mock";
type CloudProviders = MyExtract<AllProviders, "openai" | "anthropic" | "google">;
// "openai" | "anthropic" | "google"

type NonCloudProviders = MyExclude<AllProviders, "openai" | "anthropic" | "google">;
// "local" | "mock"

// AI 实战：过滤掉调试用的模型名称
type ModelName =
  | "gpt-4"
  | "gpt-3.5-turbo"
  | "claude-opus-4-5-20251101"
  | "claude-sonnet-4-5-20251101"
  | "mock-model"
  | "test-model";

type ProductionModel = MyExclude<ModelName, "mock-model" | "test-model">;
// "gpt-4" | "gpt-3.5-turbo" | "claude-opus-4-5-20251101" | "claude-sonnet-4-5-20251101"
```

### 10.5.3 阻止分布性（Wrapping in Tuple）

如果不希望触发分布式行为，可以将类型参数**用元组包裹**：

```typescript
// 分布式版本
type IsNever<T> = T extends never ? true : false;
type D1 = IsNever<never>;    // never（分布式：对空联合求值，结果为 never）
type D2 = IsNever<string>;   // false

// 非分布式版本（用元组包裹）
type IsNeverStrict<T> = [T] extends [never] ? true : false;
type S1 = IsNeverStrict<never>;    // true（不触发分布，直接检查整体）
type S2 = IsNeverStrict<string>;   // false

// 非分布式条件类型
type ToArray<T> = T extends any ? T[] : never;        // 分布式
type ToArrayStrict<T> = [T] extends [any] ? T[] : never; // 非分布式

type DA = ToArray<string | number>;       // string[] | number[]（分布处理）
type SA = ToArrayStrict<string | number>; // (string | number)[]（整体处理）
```

### 10.5.4 分布式条件类型的链式组合

多个分布式条件类型可以链式组合，构建强大的类型过滤器：

```typescript
// 从类型中移除 null、undefined 以及指定类型
type StrictExclude<T, Excluded> = T extends null | undefined | Excluded ? never : T;

type Value = string | number | null | undefined | symbol;
type CleanValue = StrictExclude<Value, symbol>;
// string | number

// 获取对象类型中满足条件的键的联合
type FilteredKeys<T, Condition> = {
  [K in keyof T]: T[K] extends Condition ? K : never;
}[keyof T];

interface ModelCapabilities {
  chat: boolean;
  embedding: boolean;
  maxContextWindow: number;
  supportedLanguages: string[];
  finetunable: boolean;
  latencyMs: number;
}

// 提取值为 boolean 类型的键
type BooleanKeys = FilteredKeys<ModelCapabilities, boolean>;
// "chat" | "embedding" | "finetunable"

// 提取值为 number 类型的键
type NumberKeys = FilteredKeys<ModelCapabilities, number>;
// "maxContextWindow" | "latencyMs"
```

---

## 本章小结

| 特性 | 语法 | 用途 | 典型示例 |
|------|------|------|---------|
| 映射类型 | `[K in keyof T]: NewType` | 批量转换对象属性类型 | `Readonly<T>`、`Partial<T>` |
| 添加修饰符 | `readonly [K in keyof T]?:` | 批量添加只读或可选 | 不可变配置快照 |
| 移除修饰符 | `-readonly [K in keyof T]-?:` | 批量移除只读或可选 | `Required<T>`、`Mutable<T>` |
| 键重映射 | `[K in keyof T as NewKey]:` | 重命名键或过滤键 | 生成 getter 方法映射 |
| 条件类型 | `T extends U ? X : Y` | 根据类型关系分支 | `NonNullable<T>`、类型守卫 |
| infer | `T extends F<infer R> ? R : Y` | 从类型结构中提取内部类型 | `ReturnType<T>`、`Parameters<T>` |
| 分布式条件 | 裸泛型参数 + 联合类型输入 | 对联合类型逐个成员处理 | `Exclude<T,U>`、`Extract<T,U>` |
| 阻止分布 | `[T] extends [U]` | 对联合类型整体处理 | `IsNever<T>` |

**使用建议**：
- 需要批量变换类型属性 → 用映射类型
- 需要根据类型"形状"做分支决策 → 用条件类型
- 需要从函数/Promise/数组中"挖出"内部类型 → 用 `infer`
- 需要对联合类型每个成员单独处理 → 利用分布式条件类型
- 需要精确控制是否分布 → 用元组包裹 `[T] extends [U]`

---

## AI应用实战：API参数类型推导

在 AI 应用开发中，不同 API 端点接受不同的请求参数，返回不同形态的响应。本节展示如何综合运用映射类型与条件类型，**根据 API 端点自动推导请求和响应类型**，消除手动维护类型映射的负担。

### 需求场景

设想我们有一个统一的 AI 网关，支持以下端点：
- `chat/completions`：聊天对话
- `embeddings`：文本向量化
- `images/generate`：图像生成
- `audio/transcribe`：语音转文字

每个端点有各自独特的请求参数和响应格式。目标是实现一个 `apiCall<Endpoint>()` 函数，使 TypeScript 能够根据传入的端点字符串**自动推导**正确的请求和响应类型。

### 完整实现

```typescript
// ============================================================
// 1. 定义各端点的请求/响应类型
// ============================================================

// 聊天对话
interface ChatRequest {
  model: string;
  messages: Array<{ role: "system" | "user" | "assistant"; content: string }>;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
}

interface ChatResponse {
  id: string;
  model: string;
  content: string;
  stopReason: "end_turn" | "max_tokens" | "stop_sequence";
  usage: { inputTokens: number; outputTokens: number };
}

// 文本向量化
interface EmbeddingRequest {
  model: string;
  input: string | string[];
  encodingFormat?: "float" | "base64";
}

interface EmbeddingResponse {
  model: string;
  embeddings: number[][];
  usage: { totalTokens: number };
}

// 图像生成
interface ImageGenerationRequest {
  model: string;
  prompt: string;
  size?: "256x256" | "512x512" | "1024x1024" | "1792x1024" | "1024x1792";
  quality?: "standard" | "hd";
  n?: number;
}

interface ImageGenerationResponse {
  created: number;
  images: Array<{ url: string; revisedPrompt?: string }>;
}

// 语音转文字
interface AudioTranscribeRequest {
  model: string;
  audioData: Blob | ArrayBuffer;
  language?: string;
  prompt?: string;
}

interface AudioTranscribeResponse {
  text: string;
  language: string;
  duration: number;
  segments?: Array<{ start: number; end: number; text: string }>;
}

// ============================================================
// 2. 端点注册表：将端点路径映射到请求/响应类型对
// ============================================================

interface EndpointRegistry {
  "chat/completions": {
    request: ChatRequest;
    response: ChatResponse;
  };
  "embeddings": {
    request: EmbeddingRequest;
    response: EmbeddingResponse;
  };
  "images/generate": {
    request: ImageGenerationRequest;
    response: ImageGenerationResponse;
  };
  "audio/transcribe": {
    request: AudioTranscribeRequest;
    response: AudioTranscribeResponse;
  };
}

// 所有合法的端点名称
type EndpointName = keyof EndpointRegistry;

// ============================================================
// 3. 条件类型：根据端点提取请求/响应类型
// ============================================================

// 提取端点的请求类型
type RequestOf<E extends EndpointName> =
  EndpointRegistry[E]["request"];

// 提取端点的响应类型
type ResponseOf<E extends EndpointName> =
  EndpointRegistry[E]["response"];

// 使用示例
type ChatReq = RequestOf<"chat/completions">;
// ChatRequest

type EmbedRes = ResponseOf<"embeddings">;
// EmbeddingResponse

// ============================================================
// 4. 映射类型：生成所有端点的默认参数
// ============================================================

// 为所有端点生成"仅包含必填字段"的最小请求类型
type MinimalRequests = {
  [E in EndpointName]: Pick<
    RequestOf<E>,
    // 过滤出非可选键（值类型不含 undefined 的键）
    {
      [K in keyof RequestOf<E>]: undefined extends RequestOf<E>[K] ? never : K;
    }[keyof RequestOf<E>]
  >;
};

// ============================================================
// 5. 条件类型：根据 stream 参数决定响应类型
// ============================================================

// 对于支持流式传输的端点，返回类型随 stream 参数变化
type StreamableResponse<
  E extends EndpointName,
  TStream extends boolean = false
> = E extends "chat/completions"
  ? TStream extends true
    ? AsyncIterable<{ delta: string; done: boolean }>
    : ResponseOf<E>
  : ResponseOf<E>; // 非 chat 端点不支持流式，始终返回完整响应

// ============================================================
// 6. infer：提取响应中的核心内容字段
// ============================================================

// 提取响应中第一个 string 类型的字段值类型
type PrimaryContent<T> =
  T extends { content: infer C }
    ? C extends string ? C : never
    : T extends { text: infer X }
      ? X extends string ? X : never
      : T extends { images: Array<infer I> }
        ? I
        : never;

type ChatContent = PrimaryContent<ChatResponse>;
// string（来自 ChatResponse.content）

type AudioContent = PrimaryContent<AudioTranscribeResponse>;
// string（来自 AudioTranscribeResponse.text）

type ImageContent = PrimaryContent<ImageGenerationResponse>;
// { url: string; revisedPrompt?: string }（来自数组元素）

// ============================================================
// 7. 分布式条件类型：过滤支持特定能力的端点
// ============================================================

// 判断端点是否支持流式传输（仅 chat/completions 支持）
type SupportsStreaming<E extends EndpointName> =
  E extends "chat/completions" ? true : false;

// 获取所有支持流式传输的端点
type StreamingEndpoints = {
  [E in EndpointName]: SupportsStreaming<E> extends true ? E : never;
}[EndpointName];
// "chat/completions"

// 获取所有处理多媒体输入的端点（有 audioData 或 images 字段）
type MultimediaEndpoints = {
  [E in EndpointName]: RequestOf<E> extends { audioData: any }
    ? E
    : RequestOf<E> extends { prompt: string; size?: any }
      ? E
      : never;
}[EndpointName];
// "images/generate" | "audio/transcribe"

// ============================================================
// 8. 统一 API 调用函数（完整类型推导）
// ============================================================

// HTTP 方法（部分端点用 POST，部分用 GET）
type EndpointMethod = {
  [E in EndpointName]: E extends "embeddings" ? "POST" : "POST";
}[EndpointName];

// API 调用选项
interface ApiCallOptions<E extends EndpointName> {
  endpoint: E;
  request: RequestOf<E>;
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
}

// 流式调用选项（仅限支持流式的端点）
interface StreamApiCallOptions<E extends EndpointName> extends ApiCallOptions<E> {
  stream: true;
  onChunk: (chunk: string) => void;
  onComplete: (response: ResponseOf<E>) => void;
}

// 统一调用函数签名（函数重载：流式 vs 非流式）
async function apiCall<E extends EndpointName>(
  options: StreamApiCallOptions<E>
): Promise<void>;
async function apiCall<E extends EndpointName>(
  options: ApiCallOptions<E>
): Promise<ResponseOf<E>>;
async function apiCall<E extends EndpointName>(
  options: ApiCallOptions<E> | StreamApiCallOptions<E>
): Promise<ResponseOf<E> | void> {
  const { endpoint, request, apiKey, baseUrl = "https://api.example.com" } = options;

  const url = `${baseUrl}/v1/${endpoint}`;

  // 构建请求头
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${apiKey}`,
  };

  // 构建请求体
  const body = JSON.stringify(request);

  console.log(`调用端点：${endpoint}`);
  console.log(`请求体：${body}`);

  // 模拟 API 响应（实际项目中替换为真实 fetch 调用）
  const mockResponses: { [E in EndpointName]: ResponseOf<E> } = {
    "chat/completions": {
      id: "msg-001",
      model: "claude-sonnet-4-5-20251101",
      content: "这是模拟的聊天回复内容。",
      stopReason: "end_turn",
      usage: { inputTokens: 20, outputTokens: 15 },
    },
    "embeddings": {
      model: "text-embedding-3-small",
      embeddings: [[0.1, 0.2, 0.3, 0.4, 0.5]],
      usage: { totalTokens: 8 },
    },
    "images/generate": {
      created: Date.now(),
      images: [{ url: "https://example.com/generated-image.png" }],
    },
    "audio/transcribe": {
      text: "这是模拟的语音转文字结果。",
      language: "zh",
      duration: 3.5,
    },
  };

  if ("stream" in options && options.stream) {
    // 流式模式：模拟逐块输出
    const fullContent = (mockResponses[endpoint] as ChatResponse).content ?? "";
    const chunks = fullContent.split("");
    for (const chunk of chunks) {
      await new Promise(resolve => setTimeout(resolve, 10));
      (options as StreamApiCallOptions<E>).onChunk(chunk);
    }
    (options as StreamApiCallOptions<E>).onComplete(mockResponses[endpoint]);
    return;
  }

  return mockResponses[endpoint];
}

// ============================================================
// 9. 使用示例：TypeScript 自动推导类型
// ============================================================

async function demo() {
  // 聊天对话：TypeScript 自动推导 response 类型为 ChatResponse
  const chatResponse = await apiCall({
    endpoint: "chat/completions",
    request: {
      model: "claude-sonnet-4-5-20251101",
      messages: [
        { role: "system", content: "你是一个有帮助的助手。" },
        { role: "user", content: "TypeScript 有什么优势？" },
      ],
      temperature: 0.7,
      maxTokens: 500,
    },
    apiKey: "sk-ant-xxx",
  });
  // chatResponse.content 类型为 string ✓
  // chatResponse.usage.inputTokens 类型为 number ✓
  console.log(`聊天回复：${chatResponse.content}`);
  console.log(`Token 使用：${chatResponse.usage.inputTokens + chatResponse.usage.outputTokens}`);

  // 文本向量化：TypeScript 自动推导 response 类型为 EmbeddingResponse
  const embedResponse = await apiCall({
    endpoint: "embeddings",
    request: {
      model: "text-embedding-3-small",
      input: ["TypeScript 教程", "AI 应用开发"],
    },
    apiKey: "sk-xxx",
  });
  // embedResponse.embeddings 类型为 number[][] ✓
  console.log(`向量维度：${embedResponse.embeddings[0].length}`);

  // 图像生成：TypeScript 自动推导 response 类型为 ImageGenerationResponse
  const imageResponse = await apiCall({
    endpoint: "images/generate",
    request: {
      model: "dall-e-3",
      prompt: "一只在写代码的猫咪，像素艺术风格",
      size: "1024x1024",
      quality: "hd",
    },
    apiKey: "sk-xxx",
  });
  // imageResponse.images[0].url 类型为 string ✓
  console.log(`图片 URL：${imageResponse.images[0].url}`);

  // 流式聊天（通过函数重载自动切换返回类型）
  await apiCall({
    endpoint: "chat/completions",
    request: {
      model: "claude-sonnet-4-5-20251101",
      messages: [{ role: "user", content: "讲一个关于 TypeScript 的笑话。" }],
    },
    apiKey: "sk-ant-xxx",
    stream: true,
    onChunk: (chunk) => process.stdout.write(chunk),
    onComplete: (response) => {
      // response 类型自动推导为 ChatResponse ✓
      console.log(`\n完成，共 ${response.usage.outputTokens} 个 token`);
    },
  });
}

// ============================================================
// 10. 高级：通用响应处理器，使用 infer 提取内容
// ============================================================

// 泛型响应处理器：自动提取响应中的主要内容
function extractContent<E extends EndpointName>(
  endpoint: E,
  response: ResponseOf<E>
): PrimaryContent<ResponseOf<E>> {
  // 运行时逻辑（编译期类型已经精确推导）
  if ("content" in response && typeof response.content === "string") {
    return response.content as PrimaryContent<ResponseOf<E>>;
  }
  if ("text" in response && typeof response.text === "string") {
    return response.text as PrimaryContent<ResponseOf<E>>;
  }
  if ("images" in response && Array.isArray(response.images)) {
    return response.images[0] as PrimaryContent<ResponseOf<E>>;
  }
  throw new Error(`无法从端点 ${endpoint} 的响应中提取内容`);
}
```

### 设计要点总结

| 技术 | 在本案例中的作用 |
|------|----------------|
| 映射类型 `[E in EndpointName]` | 为所有端点批量生成最小请求类型、能力映射 |
| 条件类型 `extends ? :` | 根据端点名称和 stream 参数决定响应类型 |
| `infer` | 从 `ChatResponse`、`AudioTranscribeResponse` 等类型中提取主要内容字段的类型 |
| 分布式条件类型 | 过滤支持流式传输或多媒体的端点子集 |
| 函数重载 | 根据是否传入 `stream: true` 切换返回类型（`Promise<Response>` vs `Promise<void>`） |

---

## 练习题

### 基础题

**练习 10-1**：实现一个 `Nullable<T>` 映射类型，将对象类型 `T` 的所有属性值类型变为"原类型或 null"（即 `T[K] | null`）。同时实现其逆操作 `NonNullableProperties<T>`，移除所有属性值中的 `null`。

验证示例：
```typescript
interface Config {
  apiKey: string;
  model: string | null;
  maxTokens: number;
}

type NullableConfig = Nullable<Config>;
// { apiKey: string | null; model: string | null; maxTokens: number | null }

type CleanConfig = NonNullableProperties<Config>;
// { apiKey: string; model: string; maxTokens: number }
```

**练习 10-2**：使用条件类型实现 `IsArray<T>` 和 `Unwrap<T>`：
- `IsArray<T>`：如果 T 是数组类型返回 `true`，否则返回 `false`
- `Unwrap<T>`：如果 T 是 `Promise<X>` 或 `Array<X>`，返回 `X`；否则返回 `T` 本身

### 进阶题

**练习 10-3**：实现一个 `PickByValue<T, V>` 工具类型，从对象类型 `T` 中选出所有值类型可赋值给 `V` 的属性。

验证示例：
```typescript
interface APIStats {
  endpoint: string;
  requestCount: number;
  successCount: number;
  averageLatency: number;
  isActive: boolean;
  lastCalledAt: Date;
}

type StringStats = PickByValue<APIStats, string>;
// { endpoint: string }

type NumberStats = PickByValue<APIStats, number>;
// { requestCount: number; successCount: number; averageLatency: number }
```

**练习 10-4**：使用 `infer` 实现 `UnpackPromiseArray<T>` 类型：接受一个 Promise 数组类型（如 `Promise<string>[]`），提取出其中所有 Promise 解析值的联合类型。

验证示例：
```typescript
type PA1 = UnpackPromiseArray<Promise<string>[]>;   // string
type PA2 = UnpackPromiseArray<Promise<number>[]>;   // number
type PA3 = UnpackPromiseArray<Array<Promise<string | number>>>; // string | number
type PA4 = UnpackPromiseArray<string[]>;             // never（不是 Promise 数组）
```

### 挑战题

**练习 10-5**：设计一个类型安全的**插件系统**类型框架，用于扩展 AI 聊天客户端的能力。

需求如下：

1. 定义三种插件类型（可辨识联合）：
   - `TransformPlugin`：转换消息内容，包含 `transform(message: string): string`
   - `FilterPlugin`：过滤不合规消息，包含 `filter(message: string): boolean`
   - `LogPlugin`：记录日志，包含 `log(event: string, data: unknown): void`

2. 使用映射类型，为插件注册表 `PluginRegistry` 生成类型，使得通过插件名称（如 `"transform"` | `"filter"` | `"log"`）可以取得对应的插件类型。

3. 使用条件类型实现 `PluginHandler<P>`：给定插件类型 `P`，返回调用该插件时需要的处理函数签名：
   - `TransformPlugin` → `(message: string) => string`
   - `FilterPlugin` → `(message: string) => boolean`
   - `LogPlugin` → `(event: string, data: unknown) => void`

4. 使用 `infer` 实现 `PluginReturnType<P>`，从插件的主方法中提取返回值类型。

5. 实现 `applyPlugin<P extends Plugin>(plugin: P, ...args: Parameters<PluginHandler<P>>): ReturnType<PluginHandler<P>>` 函数（签名即可，无需实现完整逻辑）。

---

## 练习答案

### 练习 10-1 答案

```typescript
// Nullable：所有属性值加上 | null
type Nullable<T> = {
  [K in keyof T]: T[K] | null;
};

// NonNullableProperties：移除所有属性值中的 null 和 undefined
type NonNullableProperties<T> = {
  [K in keyof T]: NonNullable<T[K]>;
};

// 验证
interface Config {
  apiKey: string;
  model: string | null;
  maxTokens: number;
}

type NullableConfig = Nullable<Config>;
// { apiKey: string | null; model: string | null; maxTokens: number | null }

type CleanConfig = NonNullableProperties<Config>;
// { apiKey: string; model: string; maxTokens: number }

// 测试
const nullableConf: NullableConfig = {
  apiKey: null,          // 合法
  model: "gpt-4",       // 合法
  maxTokens: null,       // 合法
};

const cleanConf: CleanConfig = {
  apiKey: "sk-xxx",
  model: "gpt-4",
  maxTokens: 1000,
  // model 不能为 null ✓
};
```

### 练习 10-2 答案

```typescript
// IsArray：检查是否为数组类型
type IsArray<T> = T extends Array<any> ? true : false;

// Unwrap：解包 Promise 或 Array 的内部类型
type Unwrap<T> =
  T extends Promise<infer R> ? R :
  T extends Array<infer E> ? E :
  T;

// 验证 IsArray
type IA1 = IsArray<string[]>;              // true
type IA2 = IsArray<Array<number>>;         // true
type IA3 = IsArray<string>;               // false
type IA4 = IsArray<{ length: number }>;   // false

// 验证 Unwrap
type U1 = Unwrap<Promise<string>>;        // string
type U2 = Unwrap<number[]>;              // number
type U3 = Unwrap<Promise<boolean>>;      // boolean
type U4 = Unwrap<string>;               // string（不是 Promise/Array，返回自身）
type U5 = Unwrap<Array<{ id: string }>>; // { id: string }
```

### 练习 10-3 答案

```typescript
// PickByValue：选出值类型可赋值给 V 的属性
type PickByValue<T, V> = {
  [K in keyof T as T[K] extends V ? K : never]: T[K];
};

// 验证
interface APIStats {
  endpoint: string;
  requestCount: number;
  successCount: number;
  averageLatency: number;
  isActive: boolean;
  lastCalledAt: Date;
}

type StringStats = PickByValue<APIStats, string>;
// { endpoint: string }

type NumberStats = PickByValue<APIStats, number>;
// { requestCount: number; successCount: number; averageLatency: number }

type BooleanStats = PickByValue<APIStats, boolean>;
// { isActive: boolean }

// 也可以用联合类型作为 V
type PrimitiveStats = PickByValue<APIStats, string | number | boolean>;
// { endpoint: string; requestCount: number; successCount: number; averageLatency: number; isActive: boolean }
```

### 练习 10-4 答案

```typescript
// UnpackPromiseArray：从 Promise 数组中提取解析值类型
type UnpackPromiseArray<T> =
  T extends Array<Promise<infer R>> ? R : never;

// 验证
type PA1 = UnpackPromiseArray<Promise<string>[]>;                   // string
type PA2 = UnpackPromiseArray<Promise<number>[]>;                   // number
type PA3 = UnpackPromiseArray<Array<Promise<string | number>>>;     // string | number
type PA4 = UnpackPromiseArray<string[]>;                            // never
type PA5 = UnpackPromiseArray<Array<Promise<{ id: string }>>>;      // { id: string }

// 更灵活的版本：递归解包嵌套 Promise
type DeepAwaited<T> =
  T extends Promise<infer R>
    ? DeepAwaited<R>
    : T extends Array<Promise<infer E>>
      ? DeepAwaited<E>
      : T;

type DA1 = DeepAwaited<Promise<Promise<string>>>;  // string（递归解包两层）
type DA2 = DeepAwaited<Array<Promise<number>>>;   // number
```

### 练习 10-5 答案

```typescript
// 1. 定义三种插件类型（可辨识联合）
interface TransformPlugin {
  type: "transform";
  name: string;
  transform(message: string): string;
}

interface FilterPlugin {
  type: "filter";
  name: string;
  filter(message: string): boolean;
}

interface LogPlugin {
  type: "log";
  name: string;
  log(event: string, data: unknown): void;
}

type Plugin = TransformPlugin | FilterPlugin | LogPlugin;

// 2. 插件注册表映射类型
type PluginTypeMap = {
  transform: TransformPlugin;
  filter: FilterPlugin;
  log: LogPlugin;
};

type PluginRegistry = {
  [K in keyof PluginTypeMap]: PluginTypeMap[K];
};

// 3. 条件类型：根据插件类型生成处理函数签名
type PluginHandler<P extends Plugin> =
  P extends TransformPlugin ? (message: string) => string :
  P extends FilterPlugin ? (message: string) => boolean :
  P extends LogPlugin ? (event: string, data: unknown) => void :
  never;

// 4. infer：从插件主方法中提取返回值类型
type PluginReturnType<P extends Plugin> =
  P extends TransformPlugin ? ReturnType<P["transform"]> :
  P extends FilterPlugin ? ReturnType<P["filter"]> :
  P extends LogPlugin ? ReturnType<P["log"]> :
  never;

// 验证类型
type TransformHandler = PluginHandler<TransformPlugin>; // (message: string) => string
type FilterHandler = PluginHandler<FilterPlugin>;       // (message: string) => boolean
type LogHandler = PluginHandler<LogPlugin>;             // (event: string, data: unknown) => void

type TransformReturn = PluginReturnType<TransformPlugin>; // string
type FilterReturn = PluginReturnType<FilterPlugin>;       // boolean
type LogReturn = PluginReturnType<LogPlugin>;             // void

// 5. applyPlugin 函数签名
function applyPlugin<P extends Plugin>(
  plugin: P,
  ...args: Parameters<PluginHandler<P>>
): ReturnType<PluginHandler<P>> {
  // 运行时分发（编译期类型已精确）
  switch (plugin.type) {
    case "transform":
      return plugin.transform(...(args as [string])) as ReturnType<PluginHandler<P>>;
    case "filter":
      return plugin.filter(...(args as [string])) as ReturnType<PluginHandler<P>>;
    case "log":
      plugin.log(...(args as [string, unknown]));
      return undefined as ReturnType<PluginHandler<P>>;
  }
}

// 使用示例
const uppercasePlugin: TransformPlugin = {
  type: "transform",
  name: "uppercase",
  transform: (msg) => msg.toUpperCase(),
};

const profanityFilter: FilterPlugin = {
  type: "filter",
  name: "profanity-filter",
  filter: (msg) => !msg.includes("bad-word"),
};

const consoleLogger: LogPlugin = {
  type: "log",
  name: "console-logger",
  log: (event, data) => console.log(`[${event}]`, data),
};

const transformed = applyPlugin(uppercasePlugin, "hello typescript");
// transformed 类型自动推导为 string ✓
// 值为："HELLO TYPESCRIPT"

const isAllowed = applyPlugin(profanityFilter, "clean message");
// isAllowed 类型自动推导为 boolean ✓
// 值为：true

applyPlugin(consoleLogger, "api_call", { model: "gpt-4", tokens: 100 });
// 返回类型自动推导为 void ✓
```

---

*下一章：[第11章：实用工具类型](./11-utility-types.md)*
