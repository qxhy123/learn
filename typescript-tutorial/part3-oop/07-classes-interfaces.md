# 第7章：类与接口

> **本章定位**：掌握TypeScript面向对象编程的核心机制，通过类和接口构建可扩展的AI应用架构。

---

## 学习目标

完成本章学习后，你将能够：

1. 理解TypeScript类的完整结构，包括构造函数、属性声明和方法定义，并与ES6类的区别建立清晰认知
2. 熟练运用 `public`、`private`、`protected`、`readonly` 四种访问修饰符控制成员的可见性与可变性
3. 使用 `interface` 定义契约并通过 `implements` 让类履行约定，理解接口与类型别名的使用场景差异
4. 掌握 `extends` 继承和 `super` 调用，构建合理的类继承层次，避免常见的继承陷阱
5. 运用接口继承与混入（Mixin）模式实现灵活的多重能力组合，设计统一的LLM客户端抽象层

---

## 7.1 类的基础

### 7.1.1 class 声明

TypeScript的类语法建立在ES6类的基础上，增加了类型注解和访问控制。一个最简单的类声明如下：

```typescript
class ModelConfig {
  name: string;
  maxTokens: number;
  temperature: number;

  constructor(name: string, maxTokens: number, temperature: number) {
    this.name = name;
    this.maxTokens = maxTokens;
    this.temperature = temperature;
  }

  describe(): string {
    return `${this.name} (maxTokens=${this.maxTokens}, temperature=${this.temperature})`;
  }
}

const config = new ModelConfig("gpt-4o", 4096, 0.7);
console.log(config.describe()); // gpt-4o (maxTokens=4096, temperature=0.7)
```

### 7.1.2 构造函数参数属性简写

TypeScript提供了一种语法糖——在构造函数参数上直接加访问修饰符，编译器会自动声明同名属性并完成赋值：

```typescript
// 普通写法（冗余）
class PromptTemplate {
  public name: string;
  public template: string;
  private variables: string[];

  constructor(name: string, template: string, variables: string[]) {
    this.name = name;
    this.template = template;
    this.variables = variables;
  }
}

// 参数属性简写（推荐）
class PromptTemplate {
  constructor(
    public name: string,
    public template: string,
    private variables: string[]
  ) {}

  render(values: Record<string, string>): string {
    let result = this.template;
    for (const variable of this.variables) {
      const value = values[variable] ?? `{${variable}}`;
      result = result.replaceAll(`{${variable}}`, value);
    }
    return result;
  }
}

const tmpl = new PromptTemplate(
  "qa",
  "问题：{question}\n请用{language}回答。",
  ["question", "language"]
);

console.log(tmpl.render({ question: "TypeScript有什么优点？", language: "中文" }));
// 问题：TypeScript有什么优点？
// 请用中文回答。
```

### 7.1.3 属性初始化与可选属性

类的属性可以在声明时直接初始化，也可以标记为可选：

```typescript
class LLMRequest {
  // 直接初始化
  model: string = "gpt-4o-mini";
  temperature: number = 1.0;
  maxTokens: number = 2048;
  stream: boolean = false;

  // 可选属性（可能是 undefined）
  systemPrompt?: string;
  stopSequences?: string[];

  // 断言非空（你保证会在构造函数中赋值，但TS无法推断）
  messages!: { role: string; content: string }[];

  constructor(userMessage: string) {
    this.messages = [{ role: "user", content: userMessage }];
  }
}

const req = new LLMRequest("你好，介绍一下TypeScript");
req.systemPrompt = "你是一位专业的编程导师";
req.stream = true;

console.log(req.model);       // "gpt-4o-mini"
console.log(req.systemPrompt); // "你是一位专业的编程导师"
console.log(req.stopSequences); // undefined
```

> **`!` 非空断言的使用场景**：当你确定某个属性会在类被使用前完成赋值（如在构造函数中或通过工厂方法），但TypeScript无法静态推断时，可以使用 `!` 告知编译器"我保证这个属性不会是 undefined"。过度使用会降低类型安全性，应谨慎。

### 7.1.4 方法定义

类中的方法分为实例方法、静态方法和访问器（getter/setter）：

```typescript
class TokenCounter {
  private static readonly CHARS_PER_TOKEN = 4; // 粗略估算
  private count: number = 0;

  // 实例方法
  add(text: string): void {
    this.count += Math.ceil(text.length / TokenCounter.CHARS_PER_TOKEN);
  }

  reset(): void {
    this.count = 0;
  }

  // getter：像属性一样访问，但实际是方法
  get total(): number {
    return this.count;
  }

  get estimatedCost(): number {
    // 假设 $0.002 per 1k tokens
    return (this.count / 1000) * 0.002;
  }

  // 静态方法：不需要实例即可调用
  static estimate(text: string): number {
    return Math.ceil(text.length / TokenCounter.CHARS_PER_TOKEN);
  }
}

const counter = new TokenCounter();
counter.add("TypeScript是一种强类型的JavaScript超集");
counter.add("它能在开发阶段捕获大量错误");

console.log(counter.total);           // 约 18
console.log(counter.estimatedCost);   // 约 0.000036

// 静态方法：直接在类上调用
console.log(TokenCounter.estimate("Hello, world!")); // 约 4
```

---

## 7.2 访问修饰符

TypeScript提供四种访问修饰符，控制类成员对外部代码的可见性：

| 修饰符 | 类内部 | 子类 | 类外部 |
|---|:---:|:---:|:---:|
| `public`（默认） | 可访问 | 可访问 | 可访问 |
| `protected` | 可访问 | 可访问 | 不可访问 |
| `private` | 可访问 | 不可访问 | 不可访问 |
| `readonly` | 只读 | 只读 | 只读 |

### 7.2.1 public

`public` 是默认修饰符，可以省略。所有外部代码均可访问：

```typescript
class ChatSession {
  public sessionId: string;
  public createdAt: Date;

  constructor(sessionId: string) {
    this.sessionId = sessionId;
    this.createdAt = new Date();
  }

  public getInfo(): string {
    return `Session ${this.sessionId} created at ${this.createdAt.toISOString()}`;
  }
}

const session = new ChatSession("sess_abc123");
console.log(session.sessionId);  // 可以直接访问
console.log(session.getInfo());  // 可以直接调用
```

### 7.2.2 private

`private` 成员只能在类的内部访问。这是实现**封装**的核心工具——隐藏实现细节，只暴露必要的接口：

```typescript
class ApiKeyManager {
  // 密钥绝对不应该对外暴露
  private apiKey: string;
  private callCount: number = 0;
  private readonly maxCallsPerMinute: number;

  constructor(apiKey: string, maxCallsPerMinute: number = 60) {
    this.apiKey = apiKey;
    this.maxCallsPerMinute = maxCallsPerMinute;
  }

  // 通过方法间接使用密钥，而不是暴露密钥本身
  createAuthHeader(): Record<string, string> {
    return {
      Authorization: `Bearer ${this.apiKey}`,
    };
  }

  private incrementCounter(): void {
    this.callCount += 1;
  }

  checkRateLimit(): boolean {
    return this.callCount < this.maxCallsPerMinute;
  }

  async makeRequest(endpoint: string, body: unknown): Promise<Response> {
    if (!this.checkRateLimit()) {
      throw new Error("超出速率限制");
    }
    this.incrementCounter();

    return fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...this.createAuthHeader(),
      },
      body: JSON.stringify(body),
    });
  }
}

const manager = new ApiKeyManager("sk-proj-xxx");
// manager.apiKey           // ❌ 编译错误：私有属性
// manager.incrementCounter() // ❌ 编译错误：私有方法
console.log(manager.checkRateLimit()); // ✅
```

> **TypeScript `private` vs JavaScript `#`**：TypeScript的 `private` 只在编译时强制检查，运行时仍可通过 `(obj as any).field` 访问。JavaScript原生私有字段使用 `#` 语法（如 `#apiKey`），在运行时也完全隔离。在安全性要求高的场景（如密钥管理）推荐使用 `#`。

### 7.2.3 protected

`protected` 成员可以在类本身及其**子类**中访问，但不能从外部访问。专为继承场景设计：

```typescript
class BaseClient {
  protected baseUrl: string;
  protected defaultHeaders: Record<string, string>;
  private requestCount: number = 0;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
    this.defaultHeaders = {
      "Content-Type": "application/json",
      "User-Agent": "TypeScript-AI-Client/1.0",
    };
  }

  // protected 方法：子类可以调用，外部不能
  protected async request<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<T> {
    this.requestCount += 1;
    const response = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers: this.defaultHeaders,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json() as Promise<T>;
  }

  getRequestCount(): number {
    return this.requestCount;
  }
}

class OpenAIClient extends BaseClient {
  constructor(apiKey: string) {
    super("https://api.openai.com/v1");
    // 子类可以访问和修改 protected 成员
    this.defaultHeaders["Authorization"] = `Bearer ${apiKey}`;
  }

  async chat(messages: { role: string; content: string }[]): Promise<string> {
    // 子类调用 protected 方法
    const response = await this.request<{ choices: { message: { content: string } }[] }>(
      "POST",
      "/chat/completions",
      { model: "gpt-4o-mini", messages }
    );
    return response.choices[0].message.content;
  }
}

const client = new OpenAIClient("sk-xxx");
// client.baseUrl   // ❌ 编译错误：protected 外部不可访问
// client.request() // ❌ 编译错误：protected 外部不可访问
client.getRequestCount(); // ✅
```

### 7.2.4 readonly

`readonly` 属性只能在声明时或构造函数中赋值，之后不可修改。与 `const` 的区别在于 `readonly` 作用于类成员，`const` 作用于变量：

```typescript
class ModelMetadata {
  readonly id: string;
  readonly provider: "openai" | "anthropic" | "gemini";
  readonly contextWindow: number;
  readonly createdAt: Date;

  // 可读写属性
  displayName: string;
  isDeprecated: boolean = false;

  constructor(
    id: string,
    provider: "openai" | "anthropic" | "gemini",
    contextWindow: number,
    displayName: string
  ) {
    this.id = id;
    this.provider = provider;
    this.contextWindow = contextWindow;
    this.createdAt = new Date();
    this.displayName = displayName;
  }

  deprecate(): void {
    // this.id = "xxx";  // ❌ 编译错误：readonly 属性不可修改
    this.isDeprecated = true; // ✅
  }
}

const meta = new ModelMetadata("gpt-4o", "openai", 128000, "GPT-4o");
// meta.id = "other"; // ❌ 编译错误
meta.displayName = "GPT-4o (Latest)"; // ✅
meta.deprecate();
```

### 7.2.5 静态成员与访问修饰符

静态成员（`static`）属于类本身而非实例，可以与访问修饰符组合使用：

```typescript
class LLMRegistry {
  private static instance: LLMRegistry | null = null;
  private models: Map<string, ModelMetadata> = new Map();

  // 私有构造函数：禁止外部直接 new，强制使用 getInstance
  private constructor() {}

  // 单例模式：整个应用共享同一个注册表
  static getInstance(): LLMRegistry {
    if (LLMRegistry.instance === null) {
      LLMRegistry.instance = new LLMRegistry();
    }
    return LLMRegistry.instance;
  }

  register(metadata: ModelMetadata): void {
    this.models.set(metadata.id, metadata);
  }

  get(id: string): ModelMetadata | undefined {
    return this.models.get(id);
  }

  list(): ModelMetadata[] {
    return Array.from(this.models.values());
  }
}

const registry = LLMRegistry.getInstance();
// new LLMRegistry(); // ❌ 编译错误：构造函数是私有的
```

---

## 7.3 接口定义与实现

### 7.3.1 interface 的作用

接口（`interface`）定义了一份**契约**——规定一个对象必须具备哪些属性和方法，但不关心具体实现。这是TypeScript结构类型系统的核心体现：

```typescript
// 定义 LLM 客户端应具备的能力契约
interface LLMClient {
  // 属性契约
  readonly modelId: string;
  readonly provider: string;

  // 方法契约
  complete(prompt: string): Promise<string>;
  countTokens(text: string): number;
}

// 实现契约的任何对象都可以当作 LLMClient 使用
function runBenchmark(client: LLMClient, prompts: string[]): Promise<string[]> {
  return Promise.all(prompts.map(p => client.complete(p)));
}
```

### 7.3.2 接口的属性修饰

接口中的属性也支持可选（`?`）和只读（`readonly`）修饰：

```typescript
interface CompletionRequest {
  // 必填
  model: string;
  messages: { role: string; content: string }[];

  // 可选（有默认值的参数）
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
  stopSequences?: string[];

  // 只读（响应中的字段，不应被修改）
  readonly requestId?: string;
}

interface CompletionResponse {
  readonly id: string;
  readonly model: string;
  readonly content: string;
  readonly usage: {
    readonly inputTokens: number;
    readonly outputTokens: number;
    readonly totalTokens: number;
  };
  readonly finishReason: "stop" | "length" | "tool_calls";
}
```

### 7.3.3 implements 关键字

使用 `implements` 关键字让类声明它实现了某个接口。TypeScript会检查类是否满足接口的所有要求：

```typescript
interface Configurable {
  configure(options: Record<string, unknown>): void;
  getConfig(): Record<string, unknown>;
}

interface Loggable {
  log(level: "info" | "warn" | "error", message: string): void;
}

// 一个类可以同时实现多个接口
class ProductionClient implements LLMClient, Configurable, Loggable {
  readonly modelId: string;
  readonly provider: string;
  private config: Record<string, unknown> = {};

  constructor(modelId: string, provider: string) {
    this.modelId = modelId;
    this.provider = provider;
  }

  // 实现 LLMClient 接口
  async complete(prompt: string): Promise<string> {
    this.log("info", `发送请求: ${prompt.slice(0, 50)}...`);
    // 实际实现...
    return `[${this.modelId}] 响应内容`;
  }

  countTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }

  // 实现 Configurable 接口
  configure(options: Record<string, unknown>): void {
    this.config = { ...this.config, ...options };
  }

  getConfig(): Record<string, unknown> {
    return { ...this.config };
  }

  // 实现 Loggable 接口
  log(level: "info" | "warn" | "error", message: string): void {
    const prefix = `[${level.toUpperCase()}][${this.modelId}]`;
    console.log(`${prefix} ${message}`);
  }
}

const client = new ProductionClient("gpt-4o", "openai");
client.configure({ timeout: 30000, retries: 3 });
client.complete("解释TypeScript的接口").then(console.log);
```

### 7.3.4 接口与类型别名的选择

`interface` 和 `type` 都可以描述对象形状，但有几处关键区别：

```typescript
// interface：可以多次声明并自动合并（声明合并）
interface ClientOptions {
  timeout: number;
}
interface ClientOptions {
  retries: number;
}
// 合并后等价于：{ timeout: number; retries: number }

// type：不支持声明合并
// type ClientOptions = { timeout: number };
// type ClientOptions = { retries: number }; // ❌ 重复标识符

// interface：可以被 implements
interface Serializable {
  serialize(): string;
  deserialize(data: string): void;
}

class JsonModel implements Serializable {
  private data: unknown = null;

  serialize(): string {
    return JSON.stringify(this.data);
  }

  deserialize(data: string): void {
    this.data = JSON.parse(data);
  }
}

// type：更适合联合类型、交叉类型、工具类型等复杂场景
type Provider = "openai" | "anthropic" | "gemini";
type ClientOrNull = LLMClient | null;
type ReadonlyClient = Readonly<LLMClient>;
```

> **实践建议**：对象形状和类契约优先用 `interface`（支持声明合并、extends更清晰）；联合类型、元组类型、工具类型等复杂类型操作用 `type`。

---

## 7.4 类的继承

### 7.4.1 extends 基本用法

继承让子类获得父类的属性和方法，并可以扩展或覆盖：

```typescript
// 基类：所有 LLM 客户端的通用逻辑
abstract class BaseLLMClient {
  readonly modelId: string;
  readonly provider: string;
  protected requestCount: number = 0;

  constructor(modelId: string, provider: string) {
    this.modelId = modelId;
    this.provider = provider;
  }

  // 模板方法：定义算法骨架，细节由子类实现
  async complete(prompt: string): Promise<string> {
    this.requestCount += 1;
    const messages = this.buildMessages(prompt);
    const raw = await this.sendRequest(messages);
    return this.extractContent(raw);
  }

  // 抽象方法：子类必须实现
  protected abstract buildMessages(prompt: string): { role: string; content: string }[];
  protected abstract sendRequest(messages: { role: string; content: string }[]): Promise<unknown>;
  protected abstract extractContent(raw: unknown): string;

  // 普通方法：子类可以继承直接使用
  countTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }

  getStats(): { model: string; requests: number } {
    return { model: this.modelId, requests: this.requestCount };
  }
}
```

### 7.4.2 super 关键字

`super` 有两种用法：在构造函数中调用父类构造函数，在方法中调用父类的同名方法：

```typescript
class OpenAIChatClient extends BaseLLMClient {
  private apiKey: string;
  private systemPrompt: string;

  constructor(apiKey: string, model: string = "gpt-4o-mini", systemPrompt: string = "") {
    // 必须在访问 this 之前调用 super
    super(model, "openai");
    this.apiKey = apiKey;
    this.systemPrompt = systemPrompt;
  }

  protected buildMessages(prompt: string): { role: string; content: string }[] {
    const messages: { role: string; content: string }[] = [];

    if (this.systemPrompt) {
      messages.push({ role: "system", content: this.systemPrompt });
    }

    messages.push({ role: "user", content: prompt });
    return messages;
  }

  protected async sendRequest(
    messages: { role: string; content: string }[]
  ): Promise<unknown> {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ model: this.modelId, messages }),
    });

    if (!response.ok) {
      throw new Error(`OpenAI API错误: ${response.status}`);
    }

    return response.json();
  }

  protected extractContent(raw: unknown): string {
    const data = raw as { choices: { message: { content: string } }[] };
    return data.choices[0].message.content;
  }

  // 覆盖父类方法，并调用 super 保留原有逻辑
  getStats(): { model: string; requests: number; provider: string } {
    const base = super.getStats(); // 调用父类方法
    return { ...base, provider: this.provider };
  }
}

class AnthropicClient extends BaseLLMClient {
  private apiKey: string;

  constructor(apiKey: string, model: string = "claude-3-5-sonnet-20241022") {
    super(model, "anthropic");
    this.apiKey = apiKey;
  }

  protected buildMessages(prompt: string): { role: string; content: string }[] {
    return [{ role: "user", content: prompt }];
  }

  protected async sendRequest(
    messages: { role: string; content: string }[]
  ): Promise<unknown> {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": this.apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: this.modelId,
        max_tokens: 2048,
        messages,
      }),
    });

    if (!response.ok) {
      throw new Error(`Anthropic API错误: ${response.status}`);
    }

    return response.json();
  }

  protected extractContent(raw: unknown): string {
    const data = raw as { content: { type: string; text: string }[] };
    return data.content
      .filter(block => block.type === "text")
      .map(block => block.text)
      .join("");
  }
}
```

### 7.4.3 abstract 类与方法

抽象类不能被直接实例化，只能作为基类使用。抽象方法没有实现体，强制要求子类提供具体实现：

```typescript
// abstract 类不能直接 new
// const client = new BaseLLMClient("gpt-4o", "openai"); // ❌ 编译错误

// 只能通过子类实例化
const openaiClient = new OpenAIChatClient("sk-xxx");
const anthropicClient = new AnthropicClient("sk-ant-xxx");

// 可以用父类类型持有子类实例（多态）
const clients: BaseLLMClient[] = [openaiClient, anthropicClient];

for (const client of clients) {
  // 调用公共方法，运行时分发到各自的实现
  console.log(client.getStats());
}
```

### 7.4.4 方法覆盖与 override 关键字

TypeScript 4.3引入了 `override` 关键字，明确标注"这个方法是在覆盖父类方法"。如果父类没有对应方法，编译器会报错——防止因拼写错误导致的静默失败：

```typescript
class LoggingClient extends OpenAIChatClient {
  private logs: string[] = [];

  // override 明确表示覆盖父类方法
  override async complete(prompt: string): Promise<string> {
    const start = Date.now();
    const result = await super.complete(prompt); // 调用父类实现
    const elapsed = Date.now() - start;
    this.logs.push(`[${elapsed}ms] ${prompt.slice(0, 30)}...`);
    return result;
  }

  // override countTokns(text: string): number {} // ❌ 拼写错误，编译器会报错
  override countTokens(text: string): number {
    // 使用更精确的估算
    const chineseChars = (text.match(/[\u4e00-\u9fff]/g) ?? []).length;
    const otherChars = text.length - chineseChars;
    return chineseChars * 2 + Math.ceil(otherChars / 4);
  }

  getLogs(): string[] {
    return [...this.logs];
  }
}
```

---

## 7.5 接口继承与混入模式

### 7.5.1 接口继承

接口可以通过 `extends` 继承一个或多个其他接口，组合出新的契约：

```typescript
// 基础能力接口
interface Identifiable {
  readonly id: string;
}

interface Timestamped {
  readonly createdAt: Date;
  updatedAt: Date;
}

interface HasMetadata {
  metadata: Record<string, unknown>;
}

// 接口继承：组合多个基础接口
interface LLMModel extends Identifiable, Timestamped, HasMetadata {
  name: string;
  provider: string;
  contextWindow: number;
  supportsFunctionCalling: boolean;
}

// 进一步继承，添加价格信息
interface CommercialLLMModel extends LLMModel {
  pricePer1kInputTokens: number;
  pricePer1kOutputTokens: number;
  calculateCost(inputTokens: number, outputTokens: number): number;
}

// 实现时必须满足完整的继承链
class GPT4oModel implements CommercialLLMModel {
  readonly id = "gpt-4o";
  readonly createdAt = new Date("2024-05-13");
  updatedAt = new Date();
  metadata: Record<string, unknown> = {};
  name = "GPT-4o";
  provider = "openai";
  contextWindow = 128000;
  supportsFunctionCalling = true;
  pricePer1kInputTokens = 0.005;
  pricePer1kOutputTokens = 0.015;

  calculateCost(inputTokens: number, outputTokens: number): number {
    return (
      (inputTokens / 1000) * this.pricePer1kInputTokens +
      (outputTokens / 1000) * this.pricePer1kOutputTokens
    );
  }
}

const model = new GPT4oModel();
console.log(model.calculateCost(1000, 500)); // 0.0125
```

### 7.5.2 混入模式（Mixin Pattern）

TypeScript不支持类的多重继承，但可以通过混入模式模拟——将多个独立能力"混入"一个类中：

```typescript
// 定义能力接口
interface RetryCapable {
  maxRetries: number;
  retryDelay: number;
  withRetry<T>(fn: () => Promise<T>): Promise<T>;
}

interface CacheCapable {
  cacheEnabled: boolean;
  cache: Map<string, { value: string; expireAt: number }>;
  getCached(key: string): string | null;
  setCached(key: string, value: string, ttlMs: number): void;
}

// 混入函数：为类添加重试能力
function withRetryMixin<TBase extends new (...args: unknown[]) => object>(
  Base: TBase
) {
  return class extends Base implements RetryCapable {
    maxRetries = 3;
    retryDelay = 1000;

    async withRetry<T>(fn: () => Promise<T>): Promise<T> {
      let lastError: unknown;

      for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
        try {
          return await fn();
        } catch (error) {
          lastError = error;
          if (attempt < this.maxRetries) {
            console.warn(`第${attempt + 1}次失败，${this.retryDelay}ms后重试...`);
            await new Promise(resolve => setTimeout(resolve, this.retryDelay));
          }
        }
      }

      throw lastError;
    }
  };
}

// 混入函数：为类添加缓存能力
function withCacheMixin<TBase extends new (...args: unknown[]) => object>(
  Base: TBase
) {
  return class extends Base implements CacheCapable {
    cacheEnabled = true;
    cache = new Map<string, { value: string; expireAt: number }>();

    getCached(key: string): string | null {
      if (!this.cacheEnabled) {
        return null;
      }
      const entry = this.cache.get(key);
      if (entry === undefined) {
        return null;
      }
      if (Date.now() > entry.expireAt) {
        this.cache.delete(key);
        return null;
      }
      return entry.value;
    }

    setCached(key: string, value: string, ttlMs: number): void {
      if (this.cacheEnabled) {
        this.cache.set(key, { value, expireAt: Date.now() + ttlMs });
      }
    }
  };
}

// 基础客户端类
class SimpleClient {
  constructor(public readonly modelId: string) {}

  async complete(prompt: string): Promise<string> {
    // 模拟API调用
    await new Promise(resolve => setTimeout(resolve, 100));
    return `[${this.modelId}] ${prompt}的回复`;
  }
}

// 组合多个混入，得到具备重试+缓存能力的增强客户端
const EnhancedClient = withCacheMixin(withRetryMixin(SimpleClient));

class FullFeaturedClient extends EnhancedClient {
  constructor(modelId: string) {
    super(modelId);
    this.maxRetries = 5;       // 来自 RetryCapable
    this.cacheEnabled = true;  // 来自 CacheCapable
  }

  // 带缓存和重试的完整请求
  async cachedComplete(prompt: string, ttlMs: number = 60_000): Promise<string> {
    const cacheKey = `${this.modelId}:${prompt}`;

    // 先查缓存
    const cached = this.getCached(cacheKey);
    if (cached !== null) {
      console.log("[缓存命中]", prompt.slice(0, 20));
      return cached;
    }

    // 缓存未命中，带重试发起请求
    const result = await this.withRetry(() => this.complete(prompt));

    // 写入缓存
    this.setCached(cacheKey, result, ttlMs);
    return result;
  }
}

// 使用示例
const client2 = new FullFeaturedClient("gpt-4o-mini");
client2.cachedComplete("解释什么是混入模式").then(result => {
  console.log(result);
});
```

### 7.5.3 接口与混入的配合使用

将接口与混入结合，可以实现编译期的能力检查：

```typescript
// 声明混入结果的完整类型
type RetryMixin = RetryCapable;
type CacheMixin = CacheCapable;

// 使用接口交叉类型描述混入后的完整能力
function createCachedClient(
  client: LLMClient & RetryCapable & CacheCapable,
  prompt: string
): Promise<string> {
  const cacheKey = `${client.modelId}:${prompt}`;
  const cached = client.getCached(cacheKey);

  if (cached !== null) {
    return Promise.resolve(cached);
  }

  return client.withRetry(async () => {
    const result = await client.complete(prompt);
    client.setCached(cacheKey, result, 5 * 60 * 1000); // 缓存5分钟
    return result;
  });
}
```

---

## 本章小结

| 概念 | 语法 | 核心用途 | 典型场景 |
|---|---|---|---|
| **类声明** | `class Foo { ... }` | 封装数据和行为 | 客户端、模型、会话 |
| **参数属性** | `constructor(public x: T)` | 简化属性声明与赋值 | 数据类、配置类 |
| **public** | `public field` / 默认 | 对外暴露接口 | 公共方法和属性 |
| **private** | `private field` | 隐藏实现细节 | API密钥、内部状态 |
| **protected** | `protected field` | 供子类访问 | 基类的可扩展实现细节 |
| **readonly** | `readonly field` | 防止意外修改 | ID、创建时间、配置 |
| **抽象类** | `abstract class` | 定义算法骨架 | 模板方法模式 |
| **接口** | `interface Foo { ... }` | 定义能力契约 | LLM客户端抽象 |
| **implements** | `class X implements I` | 声明实现契约 | 多态、依赖注入 |
| **继承** | `class X extends Y` | 复用并扩展 | 客户端特化 |
| **super** | `super()` / `super.method()` | 调用父类逻辑 | 构造函数、方法覆盖 |
| **override** | `override method()` | 明确标注覆盖意图 | 防止拼写错误 |
| **接口继承** | `interface A extends B, C` | 组合能力契约 | 复合模型描述 |
| **混入模式** | 高阶函数 + `extends Base` | 多重能力组合 | 重试、缓存、日志 |

---

## AI应用实战：LLM客户端抽象设计

### 场景描述

在实际AI应用中，我们往往需要同时支持多个LLM提供商（OpenAI、Anthropic、本地模型等），并在它们之间无缝切换。良好的抽象设计应该做到：

- 业务代码只依赖接口，不依赖具体实现（依赖倒置）
- 不同提供商的差异完全封装在各自的实现类中
- 支持功能扩展（重试、缓存、限流）而不修改业务代码

### 完整实现

```typescript
// ============================================================
// 核心接口定义
// ============================================================

interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

interface CompletionOptions {
  maxTokens?: number;
  temperature?: number;
  stream?: boolean;
  stopSequences?: string[];
}

interface UsageStats {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

interface CompletionResult {
  content: string;
  model: string;
  usage: UsageStats;
  finishReason: "stop" | "length" | "tool_calls";
}

// LLM客户端的核心契约
interface ILLMClient {
  readonly modelId: string;
  readonly provider: string;

  complete(
    messages: Message[],
    options?: CompletionOptions
  ): Promise<CompletionResult>;

  countTokens(text: string): number;
  isAvailable(): Promise<boolean>;
}

// 支持流式输出的扩展契约
interface IStreamingClient extends ILLMClient {
  stream(
    messages: Message[],
    options?: CompletionOptions
  ): AsyncGenerator<string>;
}

// ============================================================
// 抽象基类：封装通用逻辑
// ============================================================

abstract class AbstractLLMClient implements ILLMClient {
  readonly modelId: string;
  readonly provider: string;
  protected options: Required<CompletionOptions>;

  constructor(
    modelId: string,
    provider: string,
    defaultOptions: CompletionOptions = {}
  ) {
    this.modelId = modelId;
    this.provider = provider;
    this.options = {
      maxTokens: defaultOptions.maxTokens ?? 2048,
      temperature: defaultOptions.temperature ?? 1.0,
      stream: defaultOptions.stream ?? false,
      stopSequences: defaultOptions.stopSequences ?? [],
    };
  }

  // 模板方法：公共流程
  async complete(
    messages: Message[],
    options?: CompletionOptions
  ): Promise<CompletionResult> {
    const mergedOptions = { ...this.options, ...options };
    const validated = this.validateMessages(messages);
    const raw = await this.doComplete(validated, mergedOptions);
    return this.parseResponse(raw);
  }

  // 通用的token估算（子类可覆盖以提供更精确的实现）
  countTokens(text: string): number {
    const chineseChars = (text.match(/[\u4e00-\u9fff]/g) ?? []).length;
    const otherChars = text.length - chineseChars;
    return chineseChars * 2 + Math.ceil(otherChars / 4);
  }

  async isAvailable(): Promise<boolean> {
    try {
      await this.complete([{ role: "user", content: "hi" }], { maxTokens: 1 });
      return true;
    } catch {
      return false;
    }
  }

  // 校验消息格式
  private validateMessages(messages: Message[]): Message[] {
    if (messages.length === 0) {
      throw new Error("消息列表不能为空");
    }
    return messages;
  }

  // 子类实现：发送实际请求
  protected abstract doComplete(
    messages: Message[],
    options: Required<CompletionOptions>
  ): Promise<unknown>;

  // 子类实现：解析响应为统一格式
  protected abstract parseResponse(raw: unknown): CompletionResult;
}

// ============================================================
// OpenAI 实现
// ============================================================

interface OpenAIRawResponse {
  id: string;
  model: string;
  choices: Array<{
    message: { content: string };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

class OpenAILLMClient extends AbstractLLMClient implements IStreamingClient {
  private readonly apiKey: string;
  private readonly baseUrl: string;

  constructor(
    apiKey: string,
    model: string = "gpt-4o-mini",
    baseUrl: string = "https://api.openai.com/v1",
    defaultOptions?: CompletionOptions
  ) {
    super(model, "openai", defaultOptions);
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  protected async doComplete(
    messages: Message[],
    options: Required<CompletionOptions>
  ): Promise<unknown> {
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.modelId,
        messages,
        max_tokens: options.maxTokens,
        temperature: options.temperature,
        stop: options.stopSequences.length > 0 ? options.stopSequences : undefined,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OpenAI API错误 [${response.status}]: ${error}`);
    }

    return response.json();
  }

  protected parseResponse(raw: unknown): CompletionResult {
    const data = raw as OpenAIRawResponse;
    return {
      content: data.choices[0].message.content,
      model: data.model,
      usage: {
        inputTokens: data.usage.prompt_tokens,
        outputTokens: data.usage.completion_tokens,
        totalTokens: data.usage.total_tokens,
      },
      finishReason: data.choices[0].finish_reason as CompletionResult["finishReason"],
    };
  }

  // 实现流式输出
  async *stream(
    messages: Message[],
    options?: CompletionOptions
  ): AsyncGenerator<string> {
    const mergedOptions = { ...this.options, ...options };

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.modelId,
        messages,
        max_tokens: mergedOptions.maxTokens,
        temperature: mergedOptions.temperature,
        stream: true,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`OpenAI流式API错误 [${response.status}]`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) {
            continue;
          }
          const data = trimmed.slice(6);
          if (data === "[DONE]") {
            return;
          }

          try {
            const chunk = JSON.parse(data) as {
              choices: Array<{ delta: { content?: string } }>;
            };
            const text = chunk.choices[0]?.delta?.content;
            if (text !== undefined && text !== "") {
              yield text;
            }
          } catch {
            // 跳过无效的JSON行
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}

// ============================================================
// Anthropic 实现
// ============================================================

interface AnthropicRawResponse {
  id: string;
  model: string;
  content: Array<{ type: string; text: string }>;
  stop_reason: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

class AnthropicLLMClient extends AbstractLLMClient {
  private readonly apiKey: string;

  constructor(
    apiKey: string,
    model: string = "claude-3-5-sonnet-20241022",
    defaultOptions?: CompletionOptions
  ) {
    super(model, "anthropic", defaultOptions);
    this.apiKey = apiKey;
  }

  protected async doComplete(
    messages: Message[],
    options: Required<CompletionOptions>
  ): Promise<unknown> {
    // Anthropic API要求system消息单独传递
    const systemMsg = messages.find(m => m.role === "system");
    const userMessages = messages.filter(m => m.role !== "system");

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": this.apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: this.modelId,
        max_tokens: options.maxTokens,
        temperature: options.temperature,
        system: systemMsg?.content,
        messages: userMessages,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Anthropic API错误 [${response.status}]: ${error}`);
    }

    return response.json();
  }

  protected parseResponse(raw: unknown): CompletionResult {
    const data = raw as AnthropicRawResponse;
    const textBlocks = data.content.filter(b => b.type === "text");
    return {
      content: textBlocks.map(b => b.text).join(""),
      model: data.model,
      usage: {
        inputTokens: data.usage.input_tokens,
        outputTokens: data.usage.output_tokens,
        totalTokens: data.usage.input_tokens + data.usage.output_tokens,
      },
      finishReason: data.stop_reason === "end_turn" ? "stop" : "length",
    };
  }
}

// ============================================================
// 模拟客户端：用于测试，不发起真实请求
// ============================================================

class MockLLMClient extends AbstractLLMClient {
  private responses: string[];
  private callIndex: number = 0;

  constructor(
    responses: string[] = ["这是模拟响应"],
    modelId: string = "mock-model"
  ) {
    super(modelId, "mock");
    this.responses = responses;
  }

  protected async doComplete(
    messages: Message[],
    _options: Required<CompletionOptions>
  ): Promise<unknown> {
    const response = this.responses[this.callIndex % this.responses.length];
    this.callIndex += 1;
    return {
      content: response,
      prompt: messages.at(-1)?.content ?? "",
    };
  }

  protected parseResponse(raw: unknown): CompletionResult {
    const data = raw as { content: string; prompt: string };
    return {
      content: data.content,
      model: this.modelId,
      usage: {
        inputTokens: this.countTokens(data.prompt),
        outputTokens: this.countTokens(data.content),
        totalTokens:
          this.countTokens(data.prompt) + this.countTokens(data.content),
      },
      finishReason: "stop",
    };
  }
}

// ============================================================
// 客户端工厂：根据配置创建合适的客户端
// ============================================================

type ClientConfig =
  | { provider: "openai"; apiKey: string; model?: string; baseUrl?: string }
  | { provider: "anthropic"; apiKey: string; model?: string }
  | { provider: "mock"; responses?: string[] };

function createLLMClient(config: ClientConfig): ILLMClient {
  switch (config.provider) {
    case "openai":
      return new OpenAILLMClient(config.apiKey, config.model, config.baseUrl);
    case "anthropic":
      return new AnthropicLLMClient(config.apiKey, config.model);
    case "mock":
      return new MockLLMClient(config.responses);
    default: {
      const _exhaustive: never = config;
      throw new Error(`未知的提供商: ${(_exhaustive as ClientConfig).provider}`);
    }
  }
}

// ============================================================
// 业务代码：只依赖接口，不依赖具体实现
// ============================================================

class ConversationManager {
  private history: Message[] = [];
  private client: ILLMClient;

  constructor(client: ILLMClient, systemPrompt?: string) {
    this.client = client;
    if (systemPrompt !== undefined) {
      this.history.push({ role: "system", content: systemPrompt });
    }
  }

  async chat(userMessage: string): Promise<string> {
    this.history.push({ role: "user", content: userMessage });

    const result = await this.client.complete(this.history);

    this.history.push({ role: "assistant", content: result.content });

    console.log(
      `[${result.model}] Token使用: ${result.usage.inputTokens}入 + ${result.usage.outputTokens}出`
    );

    return result.content;
  }

  clearHistory(): void {
    this.history = this.history.filter(m => m.role === "system");
  }

  // 切换客户端，不影响历史记录
  switchClient(newClient: ILLMClient): void {
    this.client = newClient;
  }
}

// ============================================================
// 使用示例
// ============================================================

async function demo(): Promise<void> {
  // 开发/测试阶段：使用Mock客户端
  const mockClient = createLLMClient({
    provider: "mock",
    responses: [
      "TypeScript是JavaScript的超集，提供静态类型检查。",
      "接口定义了对象的形状契约，非常适合多态场景。",
      "类继承让子类复用父类的逻辑，同时可以扩展和覆盖。",
    ],
  });

  const conversation = new ConversationManager(
    mockClient,
    "你是一位TypeScript专家，用简洁的中文回答问题。"
  );

  console.log("=== 使用Mock客户端 ===");
  console.log(await conversation.chat("TypeScript是什么？"));
  console.log(await conversation.chat("什么是接口？"));

  // 生产阶段：切换到真实客户端，历史记录保留
  if (process.env.OPENAI_API_KEY) {
    const realClient = createLLMClient({
      provider: "openai",
      apiKey: process.env.OPENAI_API_KEY,
      model: "gpt-4o-mini",
    });

    conversation.switchClient(realClient);
    console.log("\n=== 切换到OpenAI客户端 ===");
    console.log(await conversation.chat("继续之前的话题，什么是类继承？"));
  }
}

// demo().catch(console.error);
```

### 设计要点总结

1. **接口优先（Interface-First）**：`ILLMClient` 是核心，业务代码（`ConversationManager`）只依赖接口，与具体实现完全解耦。新增提供商只需实现接口，无需修改任何业务代码。

2. **模板方法模式**：`AbstractLLMClient.complete()` 定义了"验证 → 请求 → 解析"的固定流程，子类只需填充 `doComplete()` 和 `parseResponse()` 两个"槽位"。

3. **策略模式通过接口实现**：`ConversationManager.switchClient()` 体现了策略模式——运行时替换算法（LLM客户端）而不改变上下文（对话管理器）。

4. **Mock可测试性**：`MockLLMClient` 无需真实API密钥就能运行，单元测试和CI/CD流水线可以直接使用。

5. **工厂函数 + 穷举检查**：`createLLMClient` 中的 `never` 穷举检查确保每次新增 `ClientConfig` 类型时，编译器会提醒更新工厂逻辑。

---

## 练习题

### 基础题

**练习 7-1**：实现一个 `ModelProfile` 类

定义一个 `ModelProfile` 类，满足以下要求：

- `readonly` 属性：`id`（string）、`provider`（string）、`createdAt`（Date）
- `public` 属性：`displayName`（string）、`tags`（string[]）
- `private` 属性：`_callCount`（number，初始为0）
- 静态方法 `fromJSON(json: string): ModelProfile`，解析JSON字符串创建实例
- 实例方法 `recordCall(): void`，每次调用将 `_callCount` 加1
- 只读 getter `callCount`，返回当前调用次数
- 实例方法 `toJSON(): string`，返回JSON字符串（不包含 `_callCount`）

```typescript
class ModelProfile {
  // 你的代码
}

// 测试用例
const profile = new ModelProfile("gpt-4o", "openai", "GPT-4o", ["chat", "vision"]);
profile.recordCall();
profile.recordCall();
console.log(profile.callCount); // 2
// profile.id = "other"; // ❌ 应该报错

const json = profile.toJSON();
const restored = ModelProfile.fromJSON(json);
console.log(restored.displayName); // "GPT-4o"
```

---

**练习 7-2**：实现接口并验证

定义以下接口并创建实现类：

```typescript
interface RateLimiter {
  readonly limit: number;       // 每分钟最多请求次数
  readonly windowMs: number;    // 时间窗口（毫秒）
  tryAcquire(): boolean;        // 尝试获取令牌，返回是否成功
  getRemainingCapacity(): number; // 返回剩余可用次数
  reset(): void;                // 重置计数器
}

// 实现 TokenBucketRateLimiter 类
class TokenBucketRateLimiter implements RateLimiter {
  // 你的代码
}

// 测试用例
const limiter = new TokenBucketRateLimiter(3, 60_000); // 每分钟3次
console.log(limiter.tryAcquire()); // true（剩余2次）
console.log(limiter.tryAcquire()); // true（剩余1次）
console.log(limiter.tryAcquire()); // true（剩余0次）
console.log(limiter.tryAcquire()); // false（超限）
console.log(limiter.getRemainingCapacity()); // 0
limiter.reset();
console.log(limiter.getRemainingCapacity()); // 3
```

---

### 中级题

**练习 7-3**：类继承与 super

基于本章的 `BaseLLMClient`（或 `AbstractLLMClient`），实现一个 `CostTrackingClient` 装饰器类，它：

- 继承自 `AbstractLLMClient`（通过组合而非继承来包装任意客户端）
- 记录每次请求的费用（按 `$0.002/1k tokens` 估算）
- 提供 `getTotalCost(): number` 方法返回累计费用
- 提供 `getCostReport(): string` 方法返回格式化的费用报告
- 所有实际请求委托给内部的 `ILLMClient` 实例

```typescript
class CostTrackingClient implements ILLMClient {
  // 你的代码
}

// 测试用例
const mock = new MockLLMClient(["回复内容"]);
const tracked = new CostTrackingClient(mock);
await tracked.complete([{ role: "user", content: "你好" }]);
await tracked.complete([{ role: "user", content: "再次你好，这是一条更长的消息" }]);
console.log(tracked.getCostReport());
// 示例输出:
// === 费用报告 ===
// 总请求次数: 2
// 总Token数: 约 XX
// 预估总费用: $0.000XX
```

---

**练习 7-4**：接口继承设计

为一个AI工作流系统设计接口继承层次：

```typescript
// 所有工作流节点的基础接口
interface WorkflowNode {
  readonly nodeId: string;
  readonly nodeType: string;
  execute(input: unknown): Promise<unknown>;
}

// 继承 WorkflowNode，添加LLM特有的属性
interface LLMNode extends WorkflowNode {
  // 你的代码：添加 client、systemPrompt 等属性
}

// 继承 WorkflowNode，添加工具调用特有的属性
interface ToolNode extends WorkflowNode {
  // 你的代码：添加 toolName、parameters schema 等属性
}

// 继承 WorkflowNode，添加条件分支特有的属性
interface ConditionNode extends WorkflowNode {
  // 你的代码：添加 condition、trueNode、falseNode 等属性
}

// 实现至少两种节点类型
class LLMWorkflowNode implements LLMNode {
  // 你的代码
}

class EchoToolNode implements ToolNode {
  // 你的代码：execute 直接返回输入，用于测试
}
```

---

### 提高题

**练习 7-5**：混入模式实战

实现一个可观测（Observable）的LLM客户端，使用混入模式为任意客户端添加事件订阅能力：

```typescript
// 事件类型定义
type ClientEvent =
  | { type: "request_start"; messages: Message[]; timestamp: number }
  | { type: "request_end"; result: CompletionResult; durationMs: number }
  | { type: "error"; error: Error; timestamp: number };

// 混入接口
interface Observable {
  on(event: ClientEvent["type"], handler: (event: ClientEvent) => void): void;
  off(event: ClientEvent["type"], handler: (event: ClientEvent) => void): void;
  emit(event: ClientEvent): void;
}

// 混入函数：为任意类添加事件订阅能力
function withObservable<TBase extends new (...args: unknown[]) => object>(
  Base: TBase
) {
  return class extends Base implements Observable {
    // 你的代码
  };
}

// 将 Observable 混入 MockLLMClient
const ObservableMock = withObservable(MockLLMClient);

// 测试用例
const observableClient = new ObservableMock(["回复内容"]);

observableClient.on("request_start", (event) => {
  if (event.type === "request_start") {
    console.log(`请求开始: ${event.messages.length} 条消息`);
  }
});

observableClient.on("request_end", (event) => {
  if (event.type === "request_end") {
    console.log(`请求完成: ${event.durationMs}ms, ${event.result.usage.totalTokens} tokens`);
  }
});

// 调用时自动触发事件
await observableClient.complete([{ role: "user", content: "你好" }]);
// 输出:
// 请求开始: 1 条消息
// 请求完成: Xms, XX tokens
```

---

## 练习答案

### 答案 7-1

```typescript
class ModelProfile {
  readonly id: string;
  readonly provider: string;
  readonly createdAt: Date;
  displayName: string;
  tags: string[];
  private _callCount: number = 0;

  constructor(id: string, provider: string, displayName: string, tags: string[] = []) {
    this.id = id;
    this.provider = provider;
    this.createdAt = new Date();
    this.displayName = displayName;
    this.tags = tags;
  }

  static fromJSON(json: string): ModelProfile {
    const data = JSON.parse(json) as {
      id: string;
      provider: string;
      displayName: string;
      tags: string[];
    };
    return new ModelProfile(data.id, data.provider, data.displayName, data.tags);
  }

  recordCall(): void {
    this._callCount += 1;
  }

  get callCount(): number {
    return this._callCount;
  }

  toJSON(): string {
    return JSON.stringify({
      id: this.id,
      provider: this.provider,
      displayName: this.displayName,
      tags: this.tags,
      createdAt: this.createdAt.toISOString(),
    });
  }
}

// 测试
const profile = new ModelProfile("gpt-4o", "openai", "GPT-4o", ["chat", "vision"]);
profile.recordCall();
profile.recordCall();
console.log(profile.callCount); // 2

const json = profile.toJSON();
const restored = ModelProfile.fromJSON(json);
console.log(restored.displayName); // "GPT-4o"
```

---

### 答案 7-2

```typescript
class TokenBucketRateLimiter implements RateLimiter {
  readonly limit: number;
  readonly windowMs: number;
  private tokens: number;
  private lastResetAt: number;

  constructor(limit: number, windowMs: number) {
    this.limit = limit;
    this.windowMs = windowMs;
    this.tokens = limit;
    this.lastResetAt = Date.now();
  }

  private checkAndRefill(): void {
    const now = Date.now();
    if (now - this.lastResetAt >= this.windowMs) {
      this.tokens = this.limit;
      this.lastResetAt = now;
    }
  }

  tryAcquire(): boolean {
    this.checkAndRefill();
    if (this.tokens > 0) {
      this.tokens -= 1;
      return true;
    }
    return false;
  }

  getRemainingCapacity(): number {
    this.checkAndRefill();
    return this.tokens;
  }

  reset(): void {
    this.tokens = this.limit;
    this.lastResetAt = Date.now();
  }
}

// 测试
const limiter = new TokenBucketRateLimiter(3, 60_000);
console.log(limiter.tryAcquire()); // true
console.log(limiter.tryAcquire()); // true
console.log(limiter.tryAcquire()); // true
console.log(limiter.tryAcquire()); // false
console.log(limiter.getRemainingCapacity()); // 0
limiter.reset();
console.log(limiter.getRemainingCapacity()); // 3
```

---

### 答案 7-3

```typescript
class CostTrackingClient implements ILLMClient {
  readonly modelId: string;
  readonly provider: string;

  private inner: ILLMClient;
  private totalTokens: number = 0;
  private requestCount: number = 0;
  private readonly costPer1kTokens = 0.002;

  constructor(inner: ILLMClient) {
    this.inner = inner;
    this.modelId = inner.modelId;
    this.provider = inner.provider;
  }

  async complete(
    messages: Message[],
    options?: CompletionOptions
  ): Promise<CompletionResult> {
    const result = await this.inner.complete(messages, options);
    this.totalTokens += result.usage.totalTokens;
    this.requestCount += 1;
    return result;
  }

  countTokens(text: string): number {
    return this.inner.countTokens(text);
  }

  async isAvailable(): Promise<boolean> {
    return this.inner.isAvailable();
  }

  getTotalCost(): number {
    return (this.totalTokens / 1000) * this.costPer1kTokens;
  }

  getCostReport(): string {
    const cost = this.getTotalCost();
    return [
      "=== 费用报告 ===",
      `总请求次数: ${this.requestCount}`,
      `总Token数: 约 ${this.totalTokens}`,
      `预估总费用: $${cost.toFixed(6)}`,
    ].join("\n");
  }
}

// 测试
const mock = new MockLLMClient(["回复内容"]);
const tracked = new CostTrackingClient(mock);

await tracked.complete([{ role: "user", content: "你好" }]);
await tracked.complete([{ role: "user", content: "再次你好，这是一条更长的消息" }]);
console.log(tracked.getCostReport());
```

---

### 答案 7-4

```typescript
interface WorkflowNode {
  readonly nodeId: string;
  readonly nodeType: string;
  execute(input: unknown): Promise<unknown>;
}

interface LLMNode extends WorkflowNode {
  readonly nodeType: "llm";
  readonly client: ILLMClient;
  systemPrompt?: string;
}

interface ToolNode extends WorkflowNode {
  readonly nodeType: "tool";
  readonly toolName: string;
  readonly parametersSchema: Record<string, unknown>;
}

interface ConditionNode extends WorkflowNode {
  readonly nodeType: "condition";
  condition(input: unknown): boolean;
  readonly trueNode: WorkflowNode;
  readonly falseNode: WorkflowNode;
}

class LLMWorkflowNode implements LLMNode {
  readonly nodeType = "llm" as const;
  readonly client: ILLMClient;
  systemPrompt?: string;

  constructor(
    readonly nodeId: string,
    client: ILLMClient,
    systemPrompt?: string
  ) {
    this.client = client;
    this.systemPrompt = systemPrompt;
  }

  async execute(input: unknown): Promise<unknown> {
    const userContent = typeof input === "string" ? input : JSON.stringify(input);
    const messages: Message[] = [];

    if (this.systemPrompt !== undefined) {
      messages.push({ role: "system", content: this.systemPrompt });
    }

    messages.push({ role: "user", content: userContent });

    const result = await this.client.complete(messages);
    return result.content;
  }
}

class EchoToolNode implements ToolNode {
  readonly nodeType = "tool" as const;
  readonly toolName = "echo";
  readonly parametersSchema = { type: "object", properties: { input: { type: "string" } } };

  constructor(readonly nodeId: string) {}

  async execute(input: unknown): Promise<unknown> {
    return input; // 直接回传输入，用于测试
  }
}

// 测试
const mockForWorkflow = new MockLLMClient(["这是工作流的LLM节点响应"]);
const llmNode = new LLMWorkflowNode("node-1", mockForWorkflow, "你是一个助手");
const echoNode = new EchoToolNode("node-2");

console.log(await llmNode.execute("你好"));   // 工作流LLM响应
console.log(await echoNode.execute("测试"));  // 测试（原样返回）
```

---

### 答案 7-5

```typescript
type ClientEvent =
  | { type: "request_start"; messages: Message[]; timestamp: number }
  | { type: "request_end"; result: CompletionResult; durationMs: number }
  | { type: "error"; error: Error; timestamp: number };

interface Observable {
  on(event: ClientEvent["type"], handler: (event: ClientEvent) => void): void;
  off(event: ClientEvent["type"], handler: (event: ClientEvent) => void): void;
  emit(event: ClientEvent): void;
}

function withObservable<TBase extends new (...args: unknown[]) => object>(
  Base: TBase
) {
  return class extends Base implements Observable {
    private eventHandlers = new Map<
      ClientEvent["type"],
      Set<(event: ClientEvent) => void>
    >();

    on(eventType: ClientEvent["type"], handler: (event: ClientEvent) => void): void {
      const handlers = this.eventHandlers.get(eventType) ?? new Set();
      handlers.add(handler);
      this.eventHandlers.set(eventType, handlers);
    }

    off(eventType: ClientEvent["type"], handler: (event: ClientEvent) => void): void {
      this.eventHandlers.get(eventType)?.delete(handler);
    }

    emit(event: ClientEvent): void {
      const handlers = this.eventHandlers.get(event.type) ?? new Set();
      for (const handler of handlers) {
        handler(event);
      }
    }

    // 覆盖 complete，在原有逻辑前后发射事件
    async complete(
      messages: Message[],
      options?: CompletionOptions
    ): Promise<CompletionResult> {
      this.emit({ type: "request_start", messages, timestamp: Date.now() });
      const start = Date.now();

      try {
        // 调用父类的 complete
        const result = await (
          super as unknown as { complete: typeof AbstractLLMClient.prototype.complete }
        ).complete.call(this, messages, options);

        this.emit({
          type: "request_end",
          result,
          durationMs: Date.now() - start,
        });

        return result;
      } catch (error) {
        this.emit({
          type: "error",
          error: error instanceof Error ? error : new Error(String(error)),
          timestamp: Date.now(),
        });
        throw error;
      }
    }
  };
}

// 测试
const ObservableMock = withObservable(MockLLMClient);
const observableClient = new ObservableMock(["回复内容"]);

observableClient.on("request_start", (event) => {
  if (event.type === "request_start") {
    console.log(`请求开始: ${event.messages.length} 条消息`);
  }
});

observableClient.on("request_end", (event) => {
  if (event.type === "request_end") {
    console.log(`请求完成: ${event.durationMs}ms, ${event.result.usage.totalTokens} tokens`);
  }
});

await observableClient.complete([{ role: "user", content: "你好" }]);
// 输出:
// 请求开始: 1 条消息
// 请求完成: Xms, XX tokens
```

---

[上一章：泛型编程](./06-generics.md) | [下一章：装饰器模式](./08-decorators.md)
