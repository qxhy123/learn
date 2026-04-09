# 第6章：泛型编程

## 学习目标

完成本章学习后，你将能够：

1. 理解泛型（Generics）的核心思想，掌握类型参数的声明与使用方式
2. 使用泛型接口和泛型类型别名设计灵活、可复用的类型结构
3. 定义泛型类，构建类型安全的数据结构（如栈、队列、缓存）
4. 使用 `extends` 关键字添加泛型约束，限制类型参数的可接受范围
5. 利用泛型默认值和多个类型参数设计通用API客户端，实现端到端的类型安全

---

## 6.1 泛型基础

### 什么是泛型

**泛型**（Generics）是指"参数化类型"——将类型本身作为参数传递，使同一段代码能处理多种类型，同时保持类型安全。

先看一个没有泛型时的问题：

```typescript
// ❌ 方案一：使用 any，丢失类型信息
function identity_any(value: any): any {
  return value;
}

const result1 = identity_any("hello"); // result1: any —— 丢失了 string 信息
result1.toFixed(2); // 编译时不报错，运行时崩溃！

// ❌ 方案二：函数重载，代码冗余
function identity_string(value: string): string {
  return value;
}
function identity_number(value: number): number {
  return value;
}
// 如果类型增多，需要写大量重复代码...
```

泛型的解决方案：

```typescript
// ✅ 方案三：泛型函数，类型安全且可复用
function identity<T>(value: T): T {
  return value;
}

const result2 = identity("hello"); // result2: string ✅ TypeScript自动推断
const result3 = identity(42);      // result3: number ✅
const result4 = identity(true);    // result4: boolean ✅

// 也可以显式指定类型参数
const result5 = identity<string>("world"); // result5: string ✅
```

`<T>` 就是**类型参数**（Type Parameter）。调用 `identity("hello")` 时，TypeScript推断 `T = string`，函数签名变成 `(value: string): string`。

### 泛型函数

泛型函数是最常见的泛型形式。类型参数写在函数名后的尖括号内：

```typescript
// 返回数组的第一个元素
function first<T>(arr: T[]): T | undefined {
  return arr[0];
}

const firstName = first(["Alice", "Bob", "Charlie"]); // firstName: string | undefined
const firstNum = first([1, 2, 3]);                    // firstNum: number | undefined
const firstEmpty = first([]);                          // firstEmpty: undefined

// 交换元组中的两个元素
function swap<A, B>(pair: [A, B]): [B, A] {
  return [pair[1], pair[0]];
}

const swapped = swap(["hello", 42]); // swapped: [number, string]
console.log(swapped); // [42, "hello"]

// 创建对象的浅拷贝并合并额外字段
function withDefaults<T extends object>(obj: T, defaults: Partial<T>): T {
  return { ...defaults, ...obj };
}

const config = withDefaults(
  { apiKey: "sk-xxx" },
  { model: "gpt-4", maxTokens: 1024 }
);
// config: { apiKey: string }（合并后包含所有字段）
```

### 多个类型参数

一个函数可以有多个类型参数，用逗号分隔：

```typescript
// 将值映射到键值对
function zipToObject<K extends string, V>(keys: K[], values: V[]): Record<K, V> {
  const result = {} as Record<K, V>;
  keys.forEach((key, index) => {
    result[key] = values[index];
  });
  return result;
}

const modelConfig = zipToObject(
  ["model", "temperature", "maxTokens"],
  ["gpt-4", 0.7, 2048]
);
// modelConfig: Record<"model" | "temperature" | "maxTokens", string | number>

// 将函数的返回类型包裹为 Promise
function promisify<TArgs extends unknown[], TReturn>(
  fn: (...args: TArgs) => TReturn
): (...args: TArgs) => Promise<TReturn> {
  return (...args: TArgs) => Promise.resolve(fn(...args));
}

const syncAdd = (a: number, b: number): number => a + b;
const asyncAdd = promisify(syncAdd);
// asyncAdd: (a: number, b: number) => Promise<number>
```

### 箭头函数中的泛型

箭头函数也支持泛型，但在 `.tsx` 文件中需要特殊处理（防止与JSX语法冲突）：

```typescript
// 普通 .ts 文件中的泛型箭头函数
const wrapInArray = <T>(value: T): T[] => [value];

// .tsx 文件中：在类型参数后加逗号，避免与 JSX 混淆
const wrapInArray2 = <T,>(value: T): T[] => [value];

// 或者使用 extends unknown（更明确）
const wrapInArray3 = <T extends unknown>(value: T): T[] => [value];
```

---

## 6.2 泛型接口与泛型类型别名

### 泛型接口

接口可以带类型参数，描述通用的数据结构：

```typescript
// 通用响应结构
interface ApiResponse<T> {
  data: T;
  status: number;
  message: string;
  timestamp: number;
}

// 分页数据结构
interface PaginatedResult<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

// 使用具体类型实例化
interface User {
  id: string;
  name: string;
  email: string;
}

interface Model {
  id: string;
  name: string;
  provider: "openai" | "anthropic" | "gemini";
  contextLength: number;
}

// 实例化泛型接口
type UserResponse = ApiResponse<User>;
type ModelListResponse = ApiResponse<PaginatedResult<Model>>;

// 使用示例
const userResp: UserResponse = {
  data: { id: "user_01", name: "张三", email: "zhang@example.com" },
  status: 200,
  message: "success",
  timestamp: Date.now(),
};

const modelListResp: ModelListResponse = {
  data: {
    items: [
      { id: "gpt-4", name: "GPT-4", provider: "openai", contextLength: 128000 },
      { id: "claude-3-5", name: "Claude 3.5", provider: "anthropic", contextLength: 200000 },
    ],
    total: 2,
    page: 1,
    pageSize: 10,
    hasMore: false,
  },
  status: 200,
  message: "success",
  timestamp: Date.now(),
};
```

### 泛型接口描述函数类型

泛型接口可以描述函数的形状：

```typescript
// 描述通用的转换函数
interface Transformer<TInput, TOutput> {
  transform(input: TInput): TOutput;
  inverse?(output: TOutput): TInput; // 可选的逆变换
}

// 描述通用的异步处理器
interface AsyncProcessor<TInput, TOutput> {
  process(input: TInput): Promise<TOutput>;
  onError?(error: Error): Promise<TOutput | null>;
}

// 实现示例：将AI响应转换为结构化数据
interface RawAIResponse {
  content: string;
  usage: { input_tokens: number; output_tokens: number };
}

interface ParsedAIResponse {
  text: string;
  tokenCount: number;
  costEstimate: number;
}

class AIResponseTransformer implements Transformer<RawAIResponse, ParsedAIResponse> {
  constructor(private costPerToken: number = 0.00001) {}

  transform(input: RawAIResponse): ParsedAIResponse {
    const tokenCount = input.usage.input_tokens + input.usage.output_tokens;
    return {
      text: input.content,
      tokenCount,
      costEstimate: tokenCount * this.costPerToken,
    };
  }
}
```

### 泛型类型别名

`type` 别名同样支持泛型参数，且比接口更灵活（可以表示联合类型、交叉类型等）：

```typescript
// 通用的 Result 类型（类似 Rust 的 Result<T, E>）
type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

// 通用的 Maybe 类型
type Maybe<T> = T | null | undefined;

// 通用的异步函数类型
type AsyncFn<TArgs extends unknown[], TReturn> = (
  ...args: TArgs
) => Promise<TReturn>;

// 使用示例
type FetchUserResult = Result<User>;
type FetchModelResult = Result<Model, { code: number; message: string }>;

async function fetchUser(id: string): Promise<FetchUserResult> {
  try {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) {
      return {
        success: false,
        error: new Error(`HTTP ${response.status}: ${response.statusText}`),
      };
    }
    const data = await response.json() as User;
    return { success: true, data };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error : new Error(String(error)),
    };
  }
}

// 使用 Result 类型处理响应
async function showUser(id: string): Promise<void> {
  const result = await fetchUser(id);

  if (result.success) {
    // result.data: User ✅ TypeScript 自动收窄
    console.log(`用户名: ${result.data.name}`);
    console.log(`邮箱: ${result.data.email}`);
  } else {
    // result.error: Error ✅
    console.error(`获取用户失败: ${result.error.message}`);
  }
}
```

### 递归泛型类型

泛型类型别名可以自引用，表达递归结构：

```typescript
// JSON 值的完整类型表示（递归泛型）
type JSONValue =
  | string
  | number
  | boolean
  | null
  | JSONValue[]
  | { [key: string]: JSONValue };

// 深度只读类型（递归泛型）
type DeepReadonly<T> = {
  readonly [K in keyof T]: T[K] extends object ? DeepReadonly<T[K]> : T[K];
};

// 使用示例：冻结AI配置，防止运行时修改
interface LLMConfig {
  provider: string;
  model: string;
  params: {
    temperature: number;
    maxTokens: number;
    topP: number;
  };
}

type FrozenConfig = DeepReadonly<LLMConfig>;

const config: FrozenConfig = {
  provider: "anthropic",
  model: "claude-3-5-sonnet-20241022",
  params: { temperature: 0.7, maxTokens: 2048, topP: 1.0 },
};

// config.model = "gpt-4";              // ❌ 编译错误：只读属性
// config.params.temperature = 0.5;     // ❌ 编译错误：深度只读
```

---

## 6.3 泛型类

### 基本泛型类

类也可以有类型参数，在类名后声明：

```typescript
// 泛型栈（Stack）数据结构
class Stack<T> {
  private items: T[] = [];

  push(item: T): void {
    this.items.push(item);
  }

  pop(): T | undefined {
    return this.items.pop();
  }

  peek(): T | undefined {
    return this.items[this.items.length - 1];
  }

  isEmpty(): boolean {
    return this.items.length === 0;
  }

  get size(): number {
    return this.items.length;
  }
}

// 类型安全地使用
const numberStack = new Stack<number>();
numberStack.push(1);
numberStack.push(2);
numberStack.push(3);
console.log(numberStack.pop()); // 3

const messageStack = new Stack<{ role: string; content: string }>();
messageStack.push({ role: "user", content: "你好" });
messageStack.push({ role: "assistant", content: "你好！有什么可以帮你？" });
// numberStack.push("hello"); // ❌ 编译错误：类型不匹配
```

### 泛型缓存类

```typescript
// LRU缓存（简化版）：用于缓存AI API响应，避免重复调用
class LRUCache<K, V> {
  private cache: Map<K, V> = new Map();

  constructor(private maxSize: number = 100) {}

  get(key: K): V | undefined {
    if (!this.cache.has(key)) {
      return undefined;
    }
    // 将访问的键移到最后（表示最近使用）
    const value = this.cache.get(key)!;
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }

  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      // 删除最久未使用的（Map 的第一个键）
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }

  has(key: K): boolean {
    return this.cache.has(key);
  }

  clear(): void {
    this.cache.clear();
  }

  get size(): number {
    return this.cache.size;
  }
}

// 使用示例：缓存AI嵌入向量（避免重复计算）
interface EmbeddingResult {
  vector: number[];
  model: string;
  tokenCount: number;
}

const embeddingCache = new LRUCache<string, EmbeddingResult>(500);

async function getEmbedding(text: string): Promise<EmbeddingResult> {
  const cacheKey = text.trim().toLowerCase();

  if (embeddingCache.has(cacheKey)) {
    console.log("缓存命中，跳过API调用");
    return embeddingCache.get(cacheKey)!;
  }

  // 实际API调用（示例）
  const result: EmbeddingResult = {
    vector: new Array(1536).fill(0).map(() => Math.random()),
    model: "text-embedding-3-small",
    tokenCount: text.split(" ").length,
  };

  embeddingCache.set(cacheKey, result);
  return result;
}
```

### 泛型类的类型推断

实例化泛型类时，TypeScript通常可以从构造函数参数推断类型参数：

```typescript
// 事件发射器（Event Emitter）
class TypedEventEmitter<TEvents extends Record<string, unknown>> {
  private listeners: {
    [K in keyof TEvents]?: Array<(data: TEvents[K]) => void>;
  } = {};

  on<K extends keyof TEvents>(
    event: K,
    listener: (data: TEvents[K]) => void
  ): this {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event]!.push(listener);
    return this;
  }

  off<K extends keyof TEvents>(
    event: K,
    listener: (data: TEvents[K]) => void
  ): this {
    const listeners = this.listeners[event];
    if (listeners) {
      this.listeners[event] = listeners.filter(l => l !== listener) as typeof listeners;
    }
    return this;
  }

  emit<K extends keyof TEvents>(event: K, data: TEvents[K]): void {
    const listeners = this.listeners[event] ?? [];
    for (const listener of listeners) {
      listener(data);
    }
  }
}

// 定义事件映射（事件名 -> 事件数据类型）
interface StreamEvents {
  text: { content: string; index: number };
  tool_call: { name: string; id: string; arguments: string };
  complete: { stopReason: string; totalTokens: number };
  error: { message: string; code: number };
}

// TypeScript 从类型参数推断出所有事件和数据类型
const emitter = new TypedEventEmitter<StreamEvents>();

emitter.on("text", (data) => {
  // data: { content: string; index: number } ✅
  process.stdout.write(data.content);
});

emitter.on("error", (data) => {
  // data: { message: string; code: number } ✅
  console.error(`[错误 ${data.code}]: ${data.message}`);
});

// emitter.on("unknown", () => {}); // ❌ 编译错误："unknown" 不是有效事件
// emitter.emit("text", { content: "hi" }); // ❌ 编译错误：缺少 index 字段
```

---

## 6.4 泛型约束（extends 关键字）

### 基本约束

不加约束的类型参数 `T` 可以是任意类型，有时这太宽泛了。使用 `extends` 可以限制 `T` 必须满足某个结构：

```typescript
// 没有约束：T 可以是任何类型
function getLength_unsafe<T>(value: T): number {
  return (value as any).length; // 不得不用 as any，不安全
}

// 有约束：T 必须有 length 属性
function getLength<T extends { length: number }>(value: T): number {
  return value.length; // ✅ 安全，TypeScript 知道 T 一定有 length
}

console.log(getLength("hello"));         // 5
console.log(getLength([1, 2, 3]));       // 3
console.log(getLength({ length: 10 })); // 10
// console.log(getLength(42));           // ❌ 编译错误：number 没有 length 属性
```

### 约束与键名访问

`keyof` 与泛型约束结合，可以安全地访问对象属性：

```typescript
// 安全地访问对象的任意属性
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

const model = {
  id: "claude-3-5-sonnet",
  name: "Claude 3.5 Sonnet",
  provider: "anthropic",
  contextLength: 200000,
  costPer1kTokens: 0.003,
};

const modelName = getProperty(model, "name");         // string ✅
const contextLen = getProperty(model, "contextLength"); // number ✅
// getProperty(model, "unknown");  // ❌ 编译错误："unknown" 不是 model 的键

// 安全地设置对象属性
function setProperty<T, K extends keyof T>(obj: T, key: K, value: T[K]): T {
  return { ...obj, [key]: value };
}

const updatedModel = setProperty(model, "contextLength", 300000); // ✅
// setProperty(model, "contextLength", "很长"); // ❌ 编译错误：类型不匹配
```

### 约束继承层次

`extends` 约束可以限制泛型为某个类或接口的子类型：

```typescript
// 所有 AI 模型的基础接口
interface BaseModel {
  id: string;
  name: string;
  invoke(prompt: string): Promise<string>;
}

interface ChatModel extends BaseModel {
  maxTokens: number;
  streamInvoke(prompt: string): AsyncGenerator<string>;
}

interface EmbeddingModel extends BaseModel {
  dimensions: number;
  embed(texts: string[]): Promise<number[][]>;
}

// 只接受 ChatModel 或其子类型
function createChatSession<T extends ChatModel>(
  model: T,
  systemPrompt: string
): { model: T; systemPrompt: string; history: string[] } {
  return {
    model,    // 类型保留为 T（而不是 ChatModel），避免类型信息丢失
    systemPrompt,
    history: [],
  };
}

// 工厂函数：只能处理 BaseModel 的子类型
function registerModel<T extends BaseModel>(
  registry: Map<string, BaseModel>,
  model: T
): T {
  registry.set(model.id, model);
  return model; // 返回 T 而非 BaseModel，保留具体类型
}
```

### 条件约束：infer 关键字初探

泛型约束与条件类型结合可以提取嵌套类型（第7章会详细介绍条件类型）：

```typescript
// 提取 Promise 包裹的类型
type Awaited<T> = T extends Promise<infer U> ? U : T;

type StringPromise = Awaited<Promise<string>>;  // string
type NumberDirect = Awaited<number>;             // number

// 提取函数返回类型
type ReturnType<T extends (...args: any[]) => any> =
  T extends (...args: any[]) => infer R ? R : never;

async function fetchModels(): Promise<Model[]> {
  return [];
}

type FetchModelsReturn = Awaited<ReturnType<typeof fetchModels>>;
// FetchModelsReturn: Model[]
```

### 多重约束

类型参数可以同时满足多个约束，使用交叉类型 `&`：

```typescript
interface Identifiable {
  id: string;
}

interface Timestamped {
  createdAt: Date;
  updatedAt: Date;
}

interface Serializable {
  toJSON(): Record<string, unknown>;
}

// T 必须同时满足三个约束
function persistRecord<T extends Identifiable & Timestamped & Serializable>(
  record: T,
  store: Map<string, Record<string, unknown>>
): void {
  const now = new Date();
  record.updatedAt = now;
  store.set(record.id, record.toJSON());
  console.log(`已持久化记录 ${record.id}`);
}
```

---

## 6.5 泛型默认值与多个类型参数

### 泛型默认值

类型参数可以有默认值，使用 `= DefaultType` 语法。当调用时不指定该类型参数，TypeScript 会使用默认值：

```typescript
// 带默认值的泛型接口
interface Repository<
  T,
  TId = string,       // ID类型默认为 string
  TError = Error      // 错误类型默认为 Error
> {
  findById(id: TId): Promise<T | null>;
  findAll(filter?: Partial<T>): Promise<T[]>;
  save(entity: T): Promise<Result<T, TError>>;
  delete(id: TId): Promise<Result<void, TError>>;
}

// 使用默认值：不指定 TId 和 TError
interface UserRepository extends Repository<User> {
  findByEmail(email: string): Promise<User | null>;
}

// 覆盖默认值：使用数字ID
interface ConversationRepository extends Repository<
  Conversation,
  number,  // TId = number
  { code: string; detail: string }  // TError = 自定义错误类型
> {
  findByUserId(userId: string): Promise<Conversation[]>;
}

interface Conversation {
  id: number;
  userId: string;
  title: string;
  messages: Array<{ role: string; content: string }>;
  createdAt: Date;
}
```

### 带默认值的泛型函数

函数类型参数也可以有默认值，常用于工厂函数和构建器模式：

```typescript
// 通用的请求选项，带默认错误类型
interface RequestOptions<
  TBody = unknown,
  TResponse = unknown,
  TError = { message: string; status: number }
> {
  body?: TBody;
  headers?: Record<string, string>;
  timeout?: number;
  onSuccess?: (data: TResponse) => void;
  onError?: (error: TError) => void;
}

// 通用的 fetch 包装函数
async function typedFetch<
  TResponse = unknown,
  TBody = unknown,
  TError = { message: string; status: number }
>(
  url: string,
  options: RequestOptions<TBody, TResponse, TError> = {}
): Promise<Result<TResponse, TError>> {
  try {
    const response = await fetch(url, {
      method: options.body ? "POST" : "GET",
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: options.timeout
        ? AbortSignal.timeout(options.timeout)
        : undefined,
    });

    if (!response.ok) {
      const errorData = await response.json() as TError;
      options.onError?.(errorData);
      return { success: false, error: errorData };
    }

    const data = await response.json() as TResponse;
    options.onSuccess?.(data);
    return { success: true, data };
  } catch (error) {
    const typedError = { message: String(error), status: 0 } as unknown as TError;
    options.onError?.(typedError);
    return { success: false, error: typedError };
  }
}

// 使用：指定响应类型，其他使用默认值
const result = await typedFetch<User[]>("/api/users");
if (result.success) {
  // result.data: User[] ✅
  result.data.forEach(user => console.log(user.name));
}
```

### 多个类型参数的设计原则

设计多类型参数的泛型时，有几个重要原则：

```typescript
// 原则一：相关的类型参数放在一起
// ❌ 不好：TInput 和 TOutput 语义上相关，却分散
interface Converter<T1, T2, TConfig, T3, T4> { /* ... */ }

// ✅ 好：按语义分组
interface Converter<
  TInput,
  TOutput,
  TConfig = Record<string, unknown>
> {
  convert(input: TInput, config?: TConfig): TOutput;
  convertBatch(inputs: TInput[], config?: TConfig): TOutput[];
}

// 原则二：有默认值的参数放在最后
// ❌ 错误：有默认值的参数在必需参数前
// interface Bad<T = string, U> {} // 语法错误

// ✅ 正确：默认参数排在最后
interface Good<T, U, V = string> {}

// 原则三：通过约束减少调用方的类型标注负担
// ❌ 需要调用方写很多类型
function process1<T, K, V>(obj: T, key: K, value: V): T { return obj; }

// ✅ 利用约束推断，调用方无需手动标注
function process2<T, K extends keyof T>(obj: T, key: K, value: T[K]): T {
  return { ...obj, [key]: value };
}
// 调用时 TypeScript 自动推断所有类型
const updated = process2(model, "name", "GPT-4 Turbo"); // ✅ 无需任何类型标注
```

### 类型参数的命名约定

TypeScript 社区对泛型参数命名有约定俗成的规范：

| 名称 | 含义 | 典型场景 |
|------|------|----------|
| `T` | Type（通用类型） | 单类型参数的首选 |
| `U`, `V`, `W` | 第二、三、四个类型参数 | 多参数场景 |
| `K` | Key（键类型） | `keyof` 相关操作 |
| `V` | Value（值类型） | 与 `K` 配合使用 |
| `E` | Element（元素类型） | 数组、集合元素 |
| `R` | Return（返回类型） | 函数返回值 |
| `TXxx` | 有意义的描述性名称 | 公共API，清晰优先 |

```typescript
// 约定的单字母命名（简洁场景）
function map<T, U>(arr: T[], fn: (item: T) => U): U[] {
  return arr.map(fn);
}

// 描述性命名（公共API，清晰场景）
interface HttpClient<TConfig, TResponse, TError> {
  request(config: TConfig): Promise<Result<TResponse, TError>>;
}
```

---

## 本章小结

| 特性 | 语法 | 用途 | 示例 |
|------|------|------|------|
| **泛型函数** | `function fn<T>(x: T): T` | 类型安全的通用函数 | `identity<T>`, `first<T>` |
| **泛型接口** | `interface Foo<T> { ... }` | 通用数据结构描述 | `ApiResponse<T>`, `Repository<T>` |
| **泛型类型别名** | `type Bar<T> = ...` | 联合/交叉等复杂泛型类型 | `Result<T,E>`, `Maybe<T>` |
| **泛型类** | `class Baz<T> { ... }` | 类型安全的数据结构 | `Stack<T>`, `LRUCache<K,V>` |
| **泛型约束** | `<T extends SomeType>` | 限制类型参数范围 | `<T extends { length: number }>` |
| **键约束** | `<K extends keyof T>` | 安全的属性访问 | `getProperty<T, K extends keyof T>` |
| **默认类型** | `<T = DefaultType>` | 简化调用方标注 | `<TError = Error>` |
| **多类型参数** | `<T, U, V>` | 描述多个相关类型 | `<TInput, TOutput, TConfig>` |

**核心设计原则**：
- 尽量让TypeScript**自动推断**类型参数，避免调用方手动标注
- 使用约束（`extends`）缩小类型参数范围，提高代码安全性和IDE提示质量
- 默认类型参数优先放在参数列表末尾
- 优先用**描述性名称**（`TInput`, `TResponse`）代替单字母（公共API场景）

---

## AI应用实战：通用API客户端封装

### 场景描述

在开发AI应用时，我们通常需要调用多个大模型API（OpenAI、Anthropic、本地模型等）。这些API的请求/响应结构各不相同，但底层的HTTP交互模式是一样的。本节将用泛型设计一个**通用、类型安全的API客户端**，实现：

1. 类型安全的请求/响应（编译时检查，而非运行时）
2. 统一的错误处理（重试、超时、限流）
3. 可扩展的中间件机制
4. 完整的TypeScript类型推断（调用方无需任何类型标注）

### 完整实现

```typescript
// ============================================================
// 基础类型定义
// ============================================================

// HTTP 方法
type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

// 通用 API 错误类型
interface ApiErrorBody {
  error: {
    message: string;
    type: string;
    code?: string | number;
  };
}

// 自定义 API 错误类
class ApiClientError extends Error {
  constructor(
    public readonly statusCode: number,
    public readonly errorBody: ApiErrorBody,
    public readonly url: string
  ) {
    super(errorBody.error.message);
    this.name = "ApiClientError";
  }

  get isRateLimit(): boolean {
    return this.statusCode === 429;
  }

  get isAuthError(): boolean {
    return this.statusCode === 401 || this.statusCode === 403;
  }

  get isServerError(): boolean {
    return this.statusCode >= 500;
  }
}

// 请求配置接口（泛型：TBody 为请求体类型）
interface RequestConfig<TBody = undefined> {
  method?: HttpMethod;
  body?: TBody;
  headers?: Record<string, string>;
  timeout?: number;        // 超时毫秒数
  retries?: number;        // 重试次数
  retryDelay?: number;     // 重试间隔毫秒数
}

// 中间件类型（用于请求拦截/响应处理）
type RequestMiddleware = (
  url: string,
  config: RequestConfig<unknown>
) => RequestConfig<unknown> | Promise<RequestConfig<unknown>>;

// ============================================================
// 核心：通用 API 客户端类
// ============================================================

class ApiClient {
  private middlewares: RequestMiddleware[] = [];

  constructor(
    private readonly baseUrl: string,
    private readonly defaultHeaders: Record<string, string> = {}
  ) {}

  // 注册请求中间件（用于添加认证、日志等）
  use(middleware: RequestMiddleware): this {
    this.middlewares.push(middleware);
    return this;
  }

  // 核心 request 方法（泛型：TResponse 为响应类型，TBody 为请求体类型）
  async request<TResponse, TBody = undefined>(
    endpoint: string,
    config: RequestConfig<TBody> = {}
  ): Promise<TResponse> {
    const url = `${this.baseUrl}${endpoint}`;
    const maxRetries = config.retries ?? 0;
    const retryDelay = config.retryDelay ?? 1000;
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      if (attempt > 0) {
        // 指数退避：每次重试等待时间翻倍
        const delay = retryDelay * Math.pow(2, attempt - 1);
        await sleep(delay);
        console.log(`第 ${attempt} 次重试...`);
      }

      try {
        const result = await this.executeRequest<TResponse, TBody>(
          url,
          config
        );
        return result;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        if (error instanceof ApiClientError) {
          // 仅对服务器错误（5xx）和限流（429）进行重试
          if (!error.isServerError && !error.isRateLimit) {
            throw error; // 4xx 错误（除429外）不重试
          }
        }

        if (attempt === maxRetries) {
          throw lastError;
        }
      }
    }

    throw lastError ?? new Error("请求失败");
  }

  // 便捷方法：GET 请求
  async get<TResponse>(
    endpoint: string,
    config: Omit<RequestConfig<undefined>, "method" | "body"> = {}
  ): Promise<TResponse> {
    return this.request<TResponse, undefined>(endpoint, {
      ...config,
      method: "GET",
    });
  }

  // 便捷方法：POST 请求
  async post<TResponse, TBody>(
    endpoint: string,
    body: TBody,
    config: Omit<RequestConfig<TBody>, "method" | "body"> = {}
  ): Promise<TResponse> {
    return this.request<TResponse, TBody>(endpoint, {
      ...config,
      method: "POST",
      body,
    });
  }

  // 流式请求（返回 ReadableStream 而非 JSON）
  async stream<TBody>(
    endpoint: string,
    body: TBody,
    config: Omit<RequestConfig<TBody>, "method" | "body"> = {}
  ): Promise<ReadableStream<Uint8Array>> {
    const url = `${this.baseUrl}${endpoint}`;
    const mergedConfig: RequestConfig<TBody> = {
      ...config,
      method: "POST",
      body,
    };

    const finalConfig = await this.applyMiddlewares(
      url,
      mergedConfig as RequestConfig<unknown>
    );

    const fetchOptions = this.buildFetchOptions(finalConfig);
    const response = await fetch(url, fetchOptions);

    if (!response.ok) {
      const errorBody = await response.json() as ApiErrorBody;
      throw new ApiClientError(response.status, errorBody, url);
    }

    if (!response.body) {
      throw new Error("响应体为空，无法进行流式读取");
    }

    return response.body;
  }

  // 内部：执行实际的 HTTP 请求
  private async executeRequest<TResponse, TBody>(
    url: string,
    config: RequestConfig<TBody>
  ): Promise<TResponse> {
    const finalConfig = await this.applyMiddlewares(
      url,
      config as RequestConfig<unknown>
    );

    const fetchOptions = this.buildFetchOptions(finalConfig);

    const response = await fetch(url, fetchOptions);

    if (!response.ok) {
      let errorBody: ApiErrorBody;
      try {
        errorBody = await response.json() as ApiErrorBody;
      } catch {
        errorBody = {
          error: {
            message: `HTTP ${response.status}: ${response.statusText}`,
            type: "http_error",
          },
        };
      }
      throw new ApiClientError(response.status, errorBody, url);
    }

    return response.json() as Promise<TResponse>;
  }

  // 内部：应用所有中间件
  private async applyMiddlewares(
    url: string,
    config: RequestConfig<unknown>
  ): Promise<RequestConfig<unknown>> {
    let currentConfig = config;
    for (const middleware of this.middlewares) {
      currentConfig = await middleware(url, currentConfig);
    }
    return currentConfig;
  }

  // 内部：将 RequestConfig 转换为 fetch 选项
  private buildFetchOptions(config: RequestConfig<unknown>): RequestInit {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...this.defaultHeaders,
      ...config.headers,
    };

    const init: RequestInit = {
      method: config.method ?? "GET",
      headers,
    };

    if (config.body !== undefined) {
      init.body = JSON.stringify(config.body);
    }

    if (config.timeout) {
      init.signal = AbortSignal.timeout(config.timeout);
    }

    return init;
  }
}

// 工具函数：等待指定毫秒数
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================
// 具体 AI API 客户端：基于 ApiClient 构建
// ============================================================

// --- OpenAI 类型定义 ---
interface OpenAIChatRequest {
  model: string;
  messages: Array<{ role: string; content: string }>;
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
  tools?: Array<{
    type: "function";
    function: { name: string; description: string; parameters: JSONValue };
  }>;
}

interface OpenAIChatResponse {
  id: string;
  object: "chat.completion";
  model: string;
  choices: Array<{
    index: number;
    message: { role: "assistant"; content: string | null };
    finish_reason: "stop" | "tool_calls" | "length" | "content_filter";
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface OpenAIEmbeddingRequest {
  model: string;
  input: string | string[];
}

interface OpenAIEmbeddingResponse {
  object: "list";
  model: string;
  data: Array<{
    object: "embedding";
    index: number;
    embedding: number[];
  }>;
  usage: { prompt_tokens: number; total_tokens: number };
}

type JSONValue = string | number | boolean | null | JSONValue[] | { [key: string]: JSONValue };

// --- OpenAI 客户端 ---
class OpenAIClient {
  private client: ApiClient;

  constructor(apiKey: string, baseUrl: string = "https://api.openai.com/v1") {
    this.client = new ApiClient(baseUrl, {
      Authorization: `Bearer ${apiKey}`,
    });

    // 添加日志中间件
    this.client.use((url, config) => {
      console.log(`[OpenAI] ${config.method ?? "GET"} ${url}`);
      return config;
    });
  }

  // 对话补全（完整的类型推断，调用方无需标注任何类型）
  async chat(request: OpenAIChatRequest): Promise<OpenAIChatResponse> {
    return this.client.post<OpenAIChatResponse, OpenAIChatRequest>(
      "/chat/completions",
      request,
      { retries: 2, timeout: 30000 }
    );
  }

  // 文本嵌入
  async embed(request: OpenAIEmbeddingRequest): Promise<OpenAIEmbeddingResponse> {
    return this.client.post<OpenAIEmbeddingResponse, OpenAIEmbeddingRequest>(
      "/embeddings",
      request
    );
  }

  // 流式对话
  async streamChat(
    request: Omit<OpenAIChatRequest, "stream">
  ): Promise<ReadableStream<Uint8Array>> {
    return this.client.stream<OpenAIChatRequest>(
      "/chat/completions",
      { ...request, stream: true }
    );
  }
}

// --- Anthropic 类型定义 ---
interface AnthropicMessageRequest {
  model: string;
  max_tokens: number;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
  system?: string;
  stream?: boolean;
}

interface AnthropicMessageResponse {
  id: string;
  type: "message";
  role: "assistant";
  model: string;
  content: Array<{ type: "text"; text: string }>;
  stop_reason: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use";
  usage: { input_tokens: number; output_tokens: number };
}

// --- Anthropic 客户端 ---
class AnthropicClient {
  private client: ApiClient;

  constructor(
    apiKey: string,
    baseUrl: string = "https://api.anthropic.com/v1"
  ) {
    this.client = new ApiClient(baseUrl, {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    });
  }

  async createMessage(
    request: AnthropicMessageRequest
  ): Promise<AnthropicMessageResponse> {
    return this.client.post<AnthropicMessageResponse, AnthropicMessageRequest>(
      "/messages",
      request,
      { retries: 2, timeout: 60000 }
    );
  }

  async streamMessage(
    request: Omit<AnthropicMessageRequest, "stream">
  ): Promise<ReadableStream<Uint8Array>> {
    return this.client.stream<AnthropicMessageRequest>(
      "/messages",
      { ...request, stream: true }
    );
  }
}

// ============================================================
// 统一的模型调用接口（Provider 无关）
// ============================================================

// 标准化的对话请求
interface StandardChatRequest {
  messages: Array<{ role: "user" | "assistant" | "system"; content: string }>;
  maxTokens?: number;
  temperature?: number;
}

// 标准化的对话响应
interface StandardChatResponse {
  content: string;
  model: string;
  provider: string;
  usage: { inputTokens: number; outputTokens: number };
}

// 抽象的模型提供商接口（使用泛型约束）
interface ModelProvider<
  TRequest extends StandardChatRequest = StandardChatRequest,
  TResponse extends StandardChatResponse = StandardChatResponse
> {
  name: string;
  chat(request: TRequest): Promise<TResponse>;
}

// OpenAI Provider 适配器
class OpenAIProvider implements ModelProvider {
  constructor(
    private openai: OpenAIClient,
    private model: string = "gpt-4o"
  ) {}

  get name(): string {
    return "openai";
  }

  async chat(request: StandardChatRequest): Promise<StandardChatResponse> {
    const response = await this.openai.chat({
      model: this.model,
      messages: request.messages,
      max_tokens: request.maxTokens,
      temperature: request.temperature,
    });

    return {
      content: response.choices[0].message.content ?? "",
      model: response.model,
      provider: this.name,
      usage: {
        inputTokens: response.usage.prompt_tokens,
        outputTokens: response.usage.completion_tokens,
      },
    };
  }
}

// Anthropic Provider 适配器
class AnthropicProvider implements ModelProvider {
  constructor(
    private anthropic: AnthropicClient,
    private model: string = "claude-3-5-sonnet-20241022"
  ) {}

  get name(): string {
    return "anthropic";
  }

  async chat(request: StandardChatRequest): Promise<StandardChatResponse> {
    const systemMsg = request.messages.find(m => m.role === "system");
    const userMsgs = request.messages
      .filter((m): m is { role: "user" | "assistant"; content: string } =>
        m.role !== "system"
      );

    const response = await this.anthropic.createMessage({
      model: this.model,
      max_tokens: request.maxTokens ?? 2048,
      messages: userMsgs,
      system: systemMsg?.content,
    });

    return {
      content: response.content.map(b => b.text).join(""),
      model: response.model,
      provider: this.name,
      usage: {
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
      },
    };
  }
}

// ============================================================
// 使用示例
// ============================================================

async function demonstrateApiClient(): Promise<void> {
  // 初始化客户端（实际使用时从环境变量读取密钥）
  const openai = new OpenAIClient(process.env.OPENAI_API_KEY ?? "demo");
  const anthropic = new AnthropicClient(process.env.ANTHROPIC_API_KEY ?? "demo");

  // 通过 Provider 适配器统一调用
  const providers: ModelProvider[] = [
    new OpenAIProvider(openai, "gpt-4o"),
    new AnthropicProvider(anthropic, "claude-3-5-sonnet-20241022"),
  ];

  const request: StandardChatRequest = {
    messages: [
      { role: "system", content: "你是一个专业的TypeScript助手。" },
      { role: "user", content: "请用一句话解释TypeScript泛型的作用。" },
    ],
    maxTokens: 256,
    temperature: 0.7,
  };

  // 并发调用所有 Provider（类型完全安全）
  console.log("正在调用所有模型...\n");

  const results = await Promise.allSettled(
    providers.map(provider => provider.chat(request))
  );

  results.forEach((result, index) => {
    const provider = providers[index];
    if (result.status === "fulfilled") {
      const { content, model, usage } = result.value;
      console.log(`[${provider.name}] 模型: ${model}`);
      console.log(`回答: ${content}`);
      console.log(`Token: 输入 ${usage.inputTokens}, 输出 ${usage.outputTokens}\n`);
    } else {
      console.error(`[${provider.name}] 调用失败:`, result.reason);
    }
  });
}

// demonstrateApiClient().catch(console.error);
```

### 关键设计要点

1. **泛型隔离请求与响应类型**：`request<TResponse, TBody>` 将请求体和响应体的类型分离，两者可以完全不同，调用方通过类型参数显式标注，TypeScript全程验证类型匹配。

2. **默认类型参数简化调用**：`TBody = undefined` 表示GET请求没有请求体，调用 `get<User[]>("/users")` 时无需标注 `TBody`，代码更简洁。

3. **泛型约束实现协议**：`ModelProvider<TRequest, TResponse>` 通过泛型约束确保所有Provider的 `chat` 方法接受标准化输入、返回标准化输出，实现统一调用。

4. **类型信息不丢失**：`registerModel<T extends BaseModel>(registry, model): T` 返回 `T` 而非 `BaseModel`，调用方得到的是具体类型，可以访问子类的特有方法。

5. **中间件保持类型透明**：中间件只修改请求配置（headers等），不改变请求/响应的数据类型，因此不破坏泛型的类型传递链。

---

## 练习题

### 基础题

**练习 6-1**：实现泛型工具函数

实现以下三个泛型工具函数：

```typescript
// (1) 从数组中去除重复项（使用 Set 去重）
function unique<T>(arr: T[]): T[] {
  // 你的代码
}

// (2) 将数组按指定键分组，返回 Map
function groupBy<T, K extends string>(
  arr: T[],
  keyFn: (item: T) => K
): Map<K, T[]> {
  // 你的代码
}

// (3) 安全地从嵌套对象读取属性（不存在时返回 undefined）
function safeGet<T, K1 extends keyof T>(obj: T, k1: K1): T[K1];
function safeGet<T, K1 extends keyof T, K2 extends keyof T[K1]>(
  obj: T,
  k1: K1,
  k2: K2
): T[K1][K2];
function safeGet(obj: any, ...keys: string[]): unknown {
  // 你的代码
}

// 期望行为：
const nums = unique([1, 2, 2, 3, 3, 3]); // [1, 2, 3]
const words = unique(["a", "b", "a", "c"]); // ["a", "b", "c"]

const models = [
  { provider: "openai", name: "gpt-4" },
  { provider: "anthropic", name: "claude-3" },
  { provider: "openai", name: "gpt-3.5" },
];
const grouped = groupBy(models, m => m.provider);
// Map { "openai" => [...], "anthropic" => [...] }

const config = { db: { host: "localhost", port: 5432 } };
const host = safeGet(config, "db", "host"); // "localhost"
```

---

**练习 6-2**：实现泛型 `Queue` 类

实现一个先进先出（FIFO）的泛型队列：

```typescript
class Queue<T> {
  // 实现以下方法：
  // enqueue(item: T): void          —— 入队
  // dequeue(): T | undefined        —— 出队（队头）
  // peek(): T | undefined           —— 查看队头（不移除）
  // isEmpty(): boolean              —— 是否为空
  // get size(): number              —— 队列长度
  // toArray(): T[]                  —— 转换为数组（保持顺序）
}

// 测试：
const queue = new Queue<string>();
queue.enqueue("任务A");
queue.enqueue("任务B");
queue.enqueue("任务C");
console.log(queue.size);    // 3
console.log(queue.dequeue()); // "任务A"
console.log(queue.peek());    // "任务B"
console.log(queue.size);    // 2
```

---

### 中级题

**练习 6-3**：泛型约束——实现类型安全的事件系统

```typescript
// 定义一个类型安全的事件总线，要求：
// 1. on<K>(event: K, handler) 中，handler 参数类型由 K 决定
// 2. emit<K>(event: K, data) 中，data 类型必须与事件定义匹配
// 3. off<K>(event: K, handler) 取消特定处理器
// 4. once<K>(event: K, handler) 只执行一次

interface AppEvents {
  "model:selected": { modelId: string; provider: string };
  "chat:message": { role: "user" | "assistant"; content: string; timestamp: number };
  "chat:complete": { messageCount: number; totalTokens: number };
  "error": { code: number; message: string };
}

class EventBus<TEvents extends Record<string, unknown>> {
  // 实现类型安全的 on / off / once / emit
}

// 使用示例（应类型安全）：
const bus = new EventBus<AppEvents>();

bus.on("model:selected", (data) => {
  // data: { modelId: string; provider: string } ✅
  console.log(`切换到 ${data.provider} 的 ${data.modelId}`);
});

bus.once("chat:complete", (data) => {
  // data: { messageCount: number; totalTokens: number } ✅
  console.log(`对话完成，共 ${data.messageCount} 条消息`);
});

bus.emit("chat:message", {
  role: "user",
  content: "你好",
  timestamp: Date.now(),
});
// bus.emit("error", { msg: "..." }); // ❌ 编译错误：应为 { code, message }
```

---

**练习 6-4**：实现泛型管道（Pipeline）

```typescript
// 实现一个数据处理管道，每个步骤的输出类型是下一个步骤的输入类型
// 要求：TypeScript 能推断出最终输出的类型

// 使用示例：
const pipeline = createPipeline("你好，世界！")  // 初始值: string
  .pipe((s: string) => s.split(""))             // string -> string[]
  .pipe((arr: string[]) => arr.length)           // string[] -> number
  .pipe((n: number) => `字符数: ${n}`);          // number -> string

const result = pipeline.run(); // result: string ✅ TypeScript 推断出类型
console.log(result); // "字符数: 7"

// 实现 createPipeline 函数和 Pipeline 类：
class Pipeline<T> {
  // 你的代码
}

function createPipeline<T>(initialValue: T): Pipeline<T> {
  // 你的代码
}
```

---

### 提高题

**练习 6-5**：实现带重试和缓存的泛型请求装饰器

实现两个泛型高阶函数：

```typescript
// (1) withRetry：为任意异步函数添加重试逻辑
// - 失败时自动重试，最多 maxRetries 次
// - 每次重试前等待 delay * attempt 毫秒（线性退避）
// - 返回与原函数相同的类型签名（包含所有参数类型和返回类型）
function withRetry<TArgs extends unknown[], TReturn>(
  fn: (...args: TArgs) => Promise<TReturn>,
  maxRetries: number,
  delay: number
): (...args: TArgs) => Promise<TReturn> {
  // 你的代码
}

// (2) withCache：为任意异步函数添加缓存
// - 以序列化的参数作为缓存键
// - ttl（毫秒）控制缓存有效期
// - 返回与原函数相同的类型签名
function withCache<TArgs extends unknown[], TReturn>(
  fn: (...args: TArgs) => Promise<TReturn>,
  ttl: number
): (...args: TArgs) => Promise<TReturn> {
  // 你的代码
}

// 测试：组合使用两个装饰器
async function fetchModelInfo(modelId: string): Promise<{ name: string; version: string }> {
  // 模拟 API 调用
  if (Math.random() < 0.5) throw new Error("随机网络错误");
  return { name: "GPT-4", version: "turbo" };
}

// 组合：先缓存（60秒），缓存未命中时带重试
const cachedFetch = withCache(
  withRetry(fetchModelInfo, 3, 500),
  60000
);

const info = await cachedFetch("gpt-4"); // info: { name: string; version: string } ✅
```

---

## 练习答案

### 答案 6-1

```typescript
// (1) unique
function unique<T>(arr: T[]): T[] {
  return [...new Set(arr)];
}

// (2) groupBy
function groupBy<T, K extends string>(
  arr: T[],
  keyFn: (item: T) => K
): Map<K, T[]> {
  const map = new Map<K, T[]>();
  for (const item of arr) {
    const key = keyFn(item);
    const group = map.get(key) ?? [];
    group.push(item);
    map.set(key, group);
  }
  return map;
}

// (3) safeGet（重载实现）
function safeGet(obj: unknown, ...keys: string[]): unknown {
  let current: unknown = obj;
  for (const key of keys) {
    if (current === null || current === undefined || typeof current !== "object") {
      return undefined;
    }
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

// 测试
console.log(unique([1, 2, 2, 3, 3, 3]));     // [1, 2, 3]
console.log(unique(["a", "b", "a", "c"]));   // ["a", "b", "c"]

const models = [
  { provider: "openai", name: "gpt-4" },
  { provider: "anthropic", name: "claude-3" },
  { provider: "openai", name: "gpt-3.5" },
];
console.log(groupBy(models, m => m.provider));
// Map { "openai" => [{...}, {...}], "anthropic" => [{...}] }

const config = { db: { host: "localhost", port: 5432 } };
console.log(safeGet(config, "db", "host")); // "localhost"
console.log(safeGet(config, "db", "password")); // undefined（不抛出错误）
```

---

### 答案 6-2

```typescript
class Queue<T> {
  private items: T[] = [];

  enqueue(item: T): void {
    this.items.push(item);
  }

  dequeue(): T | undefined {
    return this.items.shift();
  }

  peek(): T | undefined {
    return this.items[0];
  }

  isEmpty(): boolean {
    return this.items.length === 0;
  }

  get size(): number {
    return this.items.length;
  }

  toArray(): T[] {
    return [...this.items];
  }
}

// 测试
const queue = new Queue<string>();
queue.enqueue("任务A");
queue.enqueue("任务B");
queue.enqueue("任务C");
console.log(queue.size);      // 3
console.log(queue.dequeue()); // "任务A"
console.log(queue.peek());    // "任务B"
console.log(queue.size);      // 2
console.log(queue.toArray()); // ["任务B", "任务C"]
```

---

### 答案 6-3

```typescript
type EventHandler<TEvents extends Record<string, unknown>, K extends keyof TEvents> =
  (data: TEvents[K]) => void;

class EventBus<TEvents extends Record<string, unknown>> {
  private handlers: {
    [K in keyof TEvents]?: Set<EventHandler<TEvents, K>>;
  } = {};

  on<K extends keyof TEvents>(
    event: K,
    handler: EventHandler<TEvents, K>
  ): this {
    if (!this.handlers[event]) {
      this.handlers[event] = new Set();
    }
    this.handlers[event]!.add(handler);
    return this;
  }

  off<K extends keyof TEvents>(
    event: K,
    handler: EventHandler<TEvents, K>
  ): this {
    this.handlers[event]?.delete(handler);
    return this;
  }

  once<K extends keyof TEvents>(
    event: K,
    handler: EventHandler<TEvents, K>
  ): this {
    const wrapper: EventHandler<TEvents, K> = (data) => {
      handler(data);
      this.off(event, wrapper);
    };
    return this.on(event, wrapper);
  }

  emit<K extends keyof TEvents>(event: K, data: TEvents[K]): void {
    const handlers = this.handlers[event];
    if (handlers) {
      for (const handler of handlers) {
        handler(data);
      }
    }
  }
}

// 测试
interface AppEvents {
  "model:selected": { modelId: string; provider: string };
  "chat:message": { role: "user" | "assistant"; content: string; timestamp: number };
  "chat:complete": { messageCount: number; totalTokens: number };
  "error": { code: number; message: string };
}

const bus = new EventBus<AppEvents>();

bus.on("model:selected", (data) => {
  console.log(`切换到 ${data.provider} 的 ${data.modelId}`);
});

bus.once("chat:complete", (data) => {
  console.log(`对话完成，共 ${data.messageCount} 条消息，消耗 ${data.totalTokens} Token`);
});

bus.emit("model:selected", { modelId: "gpt-4o", provider: "openai" });
// 输出: 切换到 openai 的 gpt-4o

bus.emit("chat:complete", { messageCount: 5, totalTokens: 1234 });
// 输出: 对话完成，共 5 条消息，消耗 1234 Token

bus.emit("chat:complete", { messageCount: 3, totalTokens: 456 });
// 无输出（once 已移除处理器）
```

---

### 答案 6-4

```typescript
class Pipeline<T> {
  constructor(private value: T) {}

  // pipe 的返回类型由传入函数的返回类型决定
  pipe<U>(fn: (value: T) => U): Pipeline<U> {
    return new Pipeline<U>(fn(this.value));
  }

  run(): T {
    return this.value;
  }
}

function createPipeline<T>(initialValue: T): Pipeline<T> {
  return new Pipeline(initialValue);
}

// 测试
const result = createPipeline("你好，世界！")
  .pipe((s: string) => s.split(""))
  .pipe((arr: string[]) => arr.length)
  .pipe((n: number) => `字符数: ${n}`)
  .run();

console.log(result); // "字符数: 7"

// TypeScript 能推断出每一步的类型：
// createPipeline("你好，世界！") -> Pipeline<string>
// .pipe(s => s.split(""))       -> Pipeline<string[]>
// .pipe(arr => arr.length)      -> Pipeline<number>
// .pipe(n => `字符数: ${n}`)    -> Pipeline<string>
// .run()                         -> string
```

---

### 答案 6-5

```typescript
function withRetry<TArgs extends unknown[], TReturn>(
  fn: (...args: TArgs) => Promise<TReturn>,
  maxRetries: number,
  delay: number
): (...args: TArgs) => Promise<TReturn> {
  return async (...args: TArgs): Promise<TReturn> => {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      if (attempt > 0) {
        await new Promise(resolve => setTimeout(resolve, delay * attempt));
        console.log(`第 ${attempt} 次重试...`);
      }

      try {
        return await fn(...args);
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        console.warn(`尝试 ${attempt + 1}/${maxRetries + 1} 失败:`, lastError.message);
      }
    }

    throw lastError ?? new Error("所有重试均失败");
  };
}

function withCache<TArgs extends unknown[], TReturn>(
  fn: (...args: TArgs) => Promise<TReturn>,
  ttl: number
): (...args: TArgs) => Promise<TReturn> {
  const cache = new Map<string, { value: TReturn; expiresAt: number }>();

  return async (...args: TArgs): Promise<TReturn> => {
    const key = JSON.stringify(args);
    const cached = cache.get(key);

    if (cached && Date.now() < cached.expiresAt) {
      console.log("缓存命中:", key);
      return cached.value;
    }

    const value = await fn(...args);
    cache.set(key, { value, expiresAt: Date.now() + ttl });
    return value;
  };
}

// 测试
async function fetchModelInfo(
  modelId: string
): Promise<{ name: string; version: string }> {
  // 模拟不稳定的网络请求
  if (Math.random() < 0.5) {
    throw new Error("随机网络错误");
  }
  return { name: "GPT-4", version: "turbo" };
}

const cachedFetch = withCache(
  withRetry(fetchModelInfo, 3, 200),
  60000
);

// 测试调用（两次调用，第二次应命中缓存）
try {
  const info1 = await cachedFetch("gpt-4");
  console.log("第一次调用:", info1);
  // info1: { name: string; version: string } ✅

  const info2 = await cachedFetch("gpt-4");
  console.log("第二次调用（来自缓存）:", info2);
} catch (error) {
  console.error("最终失败:", error);
}
```

---

[上一章：类型守卫与类型收窄](./05-type-guards.md) | [下一章：条件类型与映射类型](./07-conditional-mapped-types.md)
