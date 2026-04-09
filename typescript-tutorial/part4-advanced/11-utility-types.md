# 第11章：实用工具类型

> **适用读者**：已完成第10章映射类型与条件类型学习，希望掌握 TypeScript 内置工具类型并能在 AI 应用开发中灵活运用的开发者。

---

## 学习目标

完成本章学习后，你将能够：

1. 熟练使用 `Partial<T>` 与 `Required<T>`，在 AI 配置接口中灵活控制字段的可选性
2. 掌握 `Pick<T, K>` 与 `Omit<T, K>` 的使用场景，精确裁剪复杂类型以满足不同调用层的需求
3. 综合运用 `Record<K, V>`、`Extract<T, U>` 与 `Exclude<T, U>` 构建类型安全的映射结构与联合类型过滤器
4. 使用 `ReturnType<T>` 与 `Parameters<T>` 从已有函数中自动提取类型信息，减少重复类型定义
5. 组合多个内置工具类型实现自定义工具类型，设计灵活的 AI API 配置接口

---

## 11.1 Partial 与 Required

### 11.1.1 Partial\<T\>：将所有属性变为可选

`Partial<T>` 将类型 `T` 的所有属性变为**可选**（即每个属性都加上 `?`）。其内部实现是一个简单的映射类型：

```typescript
// TypeScript 内部实现（仅供参考）
type Partial<T> = {
  [K in keyof T]?: T[K];
};
```

最常见的使用场景是**配置更新**——当你只想修改对象的部分字段时：

```typescript
interface ModelConfig {
  model: string;
  temperature: number;
  maxTokens: number;
  topP: number;
  stream: boolean;
}

// 更新函数：只传入需要修改的字段
function updateConfig(
  current: ModelConfig,
  patch: Partial<ModelConfig>
): ModelConfig {
  return { ...current, ...patch };
}

const defaultConfig: ModelConfig = {
  model: "claude-sonnet-4-6",
  temperature: 0.7,
  maxTokens: 2048,
  topP: 1.0,
  stream: false,
};

// 只修改 temperature 和 stream，其他字段保持不变
const streamingConfig = updateConfig(defaultConfig, {
  temperature: 0.9,
  stream: true,
});
// { model: "claude-sonnet-4-6", temperature: 0.9, maxTokens: 2048, topP: 1.0, stream: true }
```

`Partial<T>` 也常用于函数参数的默认值合并：

```typescript
interface ChatOptions {
  model: string;
  systemPrompt: string;
  temperature: number;
  maxTokens: number;
  retries: number;
}

const CHAT_DEFAULTS: ChatOptions = {
  model: "claude-sonnet-4-6",
  systemPrompt: "你是一个有帮助的助手。",
  temperature: 0.7,
  maxTokens: 1024,
  retries: 2,
};

// 调用方只需传入与默认值不同的字段
async function chat(
  prompt: string,
  options: Partial<ChatOptions> = {}
): Promise<string> {
  const resolved: ChatOptions = { ...CHAT_DEFAULTS, ...options };
  console.log(`使用模型：${resolved.model}，温度：${resolved.temperature}`);
  // 实际调用 AI API...
  return `回复：${prompt}`;
}

// 只覆盖 model 字段
const result = await chat("你好", { model: "claude-opus-4-5-20251101" });
```

### 11.1.2 Required\<T\>：将所有属性变为必填

`Required<T>` 是 `Partial<T>` 的逆操作，它移除所有属性的 `?` 修饰符，使每个属性都变为**必填**：

```typescript
// TypeScript 内部实现（仅供参考）
type Required<T> = {
  [K in keyof T]-?: T[K];
};
// 注意 -? 表示"移除可选修饰符"
```

实际应用场景：在运行时完成配置合并后，对配置对象做类型断言：

```typescript
interface ProviderConfig {
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  timeout?: number;
}

// 合并默认值后，所有字段都已填充
function buildProviderConfig(
  overrides: ProviderConfig
): Required<ProviderConfig> {
  return {
    apiKey: overrides.apiKey ?? process.env.OPENAI_API_KEY ?? "",
    baseUrl: overrides.baseUrl ?? "https://api.openai.com/v1",
    model: overrides.model ?? "gpt-4o",
    timeout: overrides.timeout ?? 30000,
  };
}

const config = buildProviderConfig({ model: "gpt-4o-mini" });
// config.apiKey 类型为 string（非 string | undefined）✓
// config.timeout 类型为 number（非 number | undefined）✓
console.log(config.timeout.toFixed(0)); // 安全访问，无需空值检查
```

### 11.1.3 Partial 与 Required 的深层版本

标准的 `Partial<T>` 和 `Required<T>` 只作用于**顶层属性**，不会递归处理嵌套对象。如果需要深层处理，可以自定义递归版本：

```typescript
// 深度可选：递归将所有嵌套属性变为可选
type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object
    ? T[K] extends Array<infer E>
      ? Array<DeepPartial<E>>
      : DeepPartial<T[K]>
    : T[K];
};

interface FullAIConfig {
  provider: {
    name: string;
    baseUrl: string;
    auth: {
      type: "bearer" | "apikey";
      token: string;
    };
  };
  model: {
    id: string;
    params: {
      temperature: number;
      maxTokens: number;
    };
  };
}

// 深度可选：可以只更新任意层级的任意字段
type PartialAIConfig = DeepPartial<FullAIConfig>;

const patch: PartialAIConfig = {
  provider: {
    auth: { token: "new-token" }, // 只更新 token，其他保持不变
  },
  // model 整体省略也没问题
};
```

---

## 11.2 Pick 与 Omit

### 11.2.1 Pick\<T, K\>：选取指定属性

`Pick<T, K>` 从类型 `T` 中选取键集合 `K` 对应的属性，构成一个新类型：

```typescript
// TypeScript 内部实现（仅供参考）
type Pick<T, K extends keyof T> = {
  [P in K]: T[P];
};
```

典型用途是**按需裁剪类型**，只暴露调用层需要的字段：

```typescript
interface FullModelInfo {
  id: string;
  name: string;
  provider: string;
  contextWindow: number;
  inputCostPer1k: number;
  outputCostPer1k: number;
  supportedFeatures: string[];
  deprecated: boolean;
  releaseDate: string;
}

// 列表展示：只需要 id、name、provider
type ModelListItem = Pick<FullModelInfo, "id" | "name" | "provider">;
// { id: string; name: string; provider: string }

// 价格展示：只需要 id 和成本字段
type ModelPricing = Pick<FullModelInfo, "id" | "inputCostPer1k" | "outputCostPer1k">;
// { id: string; inputCostPer1k: number; outputCostPer1k: number }

// 使用示例
function renderModelList(models: ModelListItem[]): void {
  models.forEach(({ id, name, provider }) => {
    console.log(`[${provider}] ${name} (${id})`);
  });
}

function calculateCost(pricing: ModelPricing, inputTokens: number, outputTokens: number): number {
  return (inputTokens / 1000) * pricing.inputCostPer1k
       + (outputTokens / 1000) * pricing.outputCostPer1k;
}
```

`Pick` 也常配合泛型使用，构建通用的字段选取工具：

```typescript
// 从对象中提取指定键的值，返回裁剪后的对象
function pickFields<T, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
  const result = {} as Pick<T, K>;
  for (const key of keys) {
    result[key] = obj[key];
  }
  return result;
}

const fullInfo: FullModelInfo = {
  id: "claude-opus-4-5-20251101",
  name: "Claude Opus",
  provider: "anthropic",
  contextWindow: 200000,
  inputCostPer1k: 0.015,
  outputCostPer1k: 0.075,
  supportedFeatures: ["vision", "tool_use"],
  deprecated: false,
  releaseDate: "2025-11-01",
};

const listItem = pickFields(fullInfo, ["id", "name", "provider"]);
// listItem: Pick<FullModelInfo, "id" | "name" | "provider">
// TypeScript 自动推断类型，调用方无需手动标注 ✓
```

### 11.2.2 Omit\<T, K\>：排除指定属性

`Omit<T, K>` 从类型 `T` 中**排除**键集合 `K` 对应的属性，保留其余所有属性：

```typescript
// TypeScript 内部实现（仅供参考）
type Omit<T, K extends keyof any> = Pick<T, Exclude<keyof T, K>>;
```

当需要**隐藏敏感字段**或**去除特定属性**时，`Omit` 比 `Pick` 更方便（无需列举所有保留的键）：

```typescript
interface UserWithSecret {
  id: string;
  name: string;
  email: string;
  apiKey: string;         // 敏感：不应暴露给前端
  passwordHash: string;   // 敏感：不应暴露给前端
  createdAt: Date;
  role: "admin" | "user";
}

// 安全的公开用户类型：去掉所有敏感字段
type PublicUser = Omit<UserWithSecret, "apiKey" | "passwordHash">;
// { id: string; name: string; email: string; createdAt: Date; role: "admin" | "user" }

// API 响应中只返回公开信息
function toPublicUser(user: UserWithSecret): PublicUser {
  const { apiKey, passwordHash, ...publicFields } = user;
  return publicFields;
}
```

`Omit` 在扩展接口时也非常实用——覆盖父类型中某个属性的类型：

```typescript
interface BaseRequest {
  model: string;
  messages: Array<{ role: string; content: string }>;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
}

// 覆盖 stream 属性：将其从 boolean | undefined 收窄为 true
type StreamRequest = Omit<BaseRequest, "stream"> & { stream: true };

// 覆盖 messages 属性：限制更严格的 role 类型
type StrictRequest = Omit<BaseRequest, "messages"> & {
  messages: Array<{
    role: "system" | "user" | "assistant";
    content: string;
  }>;
};
```

### 11.2.3 Pick 与 Omit 的选择策略

| 场景 | 推荐 | 原因 |
|------|------|------|
| 需要保留的字段少（2-3个） | `Pick` | 列举保留字段更直观 |
| 需要排除的字段少（1-2个） | `Omit` | 列举排除字段更简洁 |
| 去除敏感字段 | `Omit` | 明确表达"去掉什么"的意图 |
| 构建最小接口 | `Pick` | 明确表达"只需要什么"的意图 |
| 覆盖某个属性的类型 | `Omit + &` | 先排除旧属性，再交叉合并新属性 |

---

## 11.3 Record、Extract 与 Exclude

### 11.3.1 Record\<K, V\>：构建键值映射类型

`Record<K, V>` 构造一个以 `K` 中的类型为键、以 `V` 为值的对象类型：

```typescript
// TypeScript 内部实现（仅供参考）
type Record<K extends keyof any, T> = {
  [P in K]: T;
};
```

当所有键的值类型相同时，`Record` 是最简洁的表达方式：

```typescript
type Provider = "openai" | "anthropic" | "google" | "cohere";

// 每个 provider 对应一个 API 基础 URL
const BASE_URLS: Record<Provider, string> = {
  openai: "https://api.openai.com/v1",
  anthropic: "https://api.anthropic.com/v1",
  google: "https://generativelanguage.googleapis.com/v1beta",
  cohere: "https://api.cohere.ai/v1",
};
// 如果漏掉任意一个 provider，TypeScript 会报错 ✓

// 每个 provider 对应的默认模型名称
const DEFAULT_MODELS: Record<Provider, string> = {
  openai: "gpt-4o",
  anthropic: "claude-sonnet-4-6",
  google: "gemini-1.5-pro",
  cohere: "command-r-plus",
};

// 每个 provider 对应的请求次数统计
const requestCounts: Record<Provider, number> = {
  openai: 0,
  anthropic: 0,
  google: 0,
  cohere: 0,
};

function trackRequest(provider: Provider): void {
  requestCounts[provider]++;
}
```

`Record` 的值类型也可以是复杂类型：

```typescript
interface ProviderStats {
  totalRequests: number;
  successRate: number;
  avgLatencyMs: number;
  lastCalledAt: Date | null;
}

// 每个 provider 对应一组统计数据
type AllProviderStats = Record<Provider, ProviderStats>;

const stats: AllProviderStats = {
  openai: { totalRequests: 1250, successRate: 0.99, avgLatencyMs: 420, lastCalledAt: new Date() },
  anthropic: { totalRequests: 840, successRate: 0.998, avgLatencyMs: 380, lastCalledAt: new Date() },
  google: { totalRequests: 320, successRate: 0.995, avgLatencyMs: 510, lastCalledAt: null },
  cohere: { totalRequests: 90, successRate: 0.97, avgLatencyMs: 290, lastCalledAt: null },
};
```

### 11.3.2 Extract\<T, U\>：提取联合类型中的子集

`Extract<T, U>` 从联合类型 `T` 中提取**可赋值给 `U`** 的类型成员：

```typescript
// TypeScript 内部实现（仅供参考）
type Extract<T, U> = T extends U ? T : never;
// 利用分布式条件类型，对 T 的每个成员分别检查
```

```typescript
type AllEvents =
  | "chat:start"
  | "chat:message"
  | "chat:end"
  | "tool:call"
  | "tool:result"
  | "error:network"
  | "error:rateLimit"
  | "error:auth";

// 提取所有聊天相关事件
type ChatEvents = Extract<AllEvents, `chat:${string}`>;
// "chat:start" | "chat:message" | "chat:end"

// 提取所有工具相关事件
type ToolEvents = Extract<AllEvents, `tool:${string}`>;
// "tool:call" | "tool:result"

// 提取所有错误事件
type ErrorEvents = Extract<AllEvents, `error:${string}`>;
// "error:network" | "error:rateLimit" | "error:auth"

// 提取多个具体值
type CriticalEvents = Extract<AllEvents, "error:auth" | "error:rateLimit">;
// "error:auth" | "error:rateLimit"
```

`Extract` 与对象类型结合：

```typescript
type MessageRole = "system" | "user" | "assistant" | "tool";

interface Message {
  role: MessageRole;
  content: string;
}

// 只允许用户和助手的对话消息（不含 system 和 tool）
type ConversationMessage = Omit<Message, "role"> & {
  role: Extract<MessageRole, "user" | "assistant">;
};
// { role: "user" | "assistant"; content: string }

function buildHistory(messages: ConversationMessage[]): string {
  return messages
    .map(m => `${m.role === "user" ? "用户" : "助手"}：${m.content}`)
    .join("\n");
}
```

### 11.3.3 Exclude\<T, U\>：从联合类型中排除成员

`Exclude<T, U>` 从联合类型 `T` 中**排除**可赋值给 `U` 的类型成员，与 `Extract` 互为逆操作：

```typescript
// TypeScript 内部实现（仅供参考）
type Exclude<T, U> = T extends U ? never : T;
```

```typescript
type ModelTier = "free" | "basic" | "pro" | "enterprise" | "internal" | "deprecated";

// 排除内部和废弃的模型层级，得到对外公开的层级
type PublicTier = Exclude<ModelTier, "internal" | "deprecated">;
// "free" | "basic" | "pro" | "enterprise"

// 排除 free 层级，得到付费层级
type PaidTier = Exclude<PublicTier, "free">;
// "basic" | "pro" | "enterprise"

// 与 Record 结合：为每个公开层级设置价格
const tierPrices: Record<PublicTier, number> = {
  free: 0,
  basic: 9.99,
  pro: 29.99,
  enterprise: 99.99,
};
```

`Exclude` 也用于从函数参数类型中剔除不需要的成员：

```typescript
type SupportedLocale = "zh-CN" | "zh-TW" | "en-US" | "en-GB" | "ja-JP" | "ko-KR";

// 移除已弃用的地区代码
type ActiveLocale = Exclude<SupportedLocale, "en-GB">;
// "zh-CN" | "zh-TW" | "en-US" | "ja-JP" | "ko-KR"

function setLocale(locale: ActiveLocale): void {
  console.log(`已切换语言为：${locale}`);
}

setLocale("zh-CN");   // ✓
// setLocale("en-GB"); // 编译错误：已被排除 ✓
```

### 11.3.4 三者组合：构建类型安全的能力映射

```typescript
type Capability =
  | "chat"
  | "embedding"
  | "vision"
  | "tool_use"
  | "image_generation"
  | "audio_transcription"
  | "fine_tuning";

// 按功能分组：文本类能力
type TextCapability = Extract<Capability, "chat" | "embedding" | "fine_tuning">;
// "chat" | "embedding" | "fine_tuning"

// 多模态能力（排除纯文本能力）
type MultimodalCapability = Exclude<Capability, TextCapability>;
// "vision" | "tool_use" | "image_generation" | "audio_transcription"

// 为每个多模态能力记录是否需要额外计费
const multimodalPricing: Record<MultimodalCapability, boolean> = {
  vision: false,             // 包含在对话价格中
  tool_use: false,           // 包含在对话价格中
  image_generation: true,    // 单独计费
  audio_transcription: true, // 单独计费
};
```

---

## 11.4 ReturnType 与 Parameters

### 11.4.1 ReturnType\<T\>：提取函数返回值类型

`ReturnType<T>` 从函数类型 `T` 中提取其**返回值类型**。它的内部实现使用了 `infer` 关键字：

```typescript
// TypeScript 内部实现（仅供参考）
type ReturnType<T extends (...args: any) => any> =
  T extends (...args: any) => infer R ? R : any;
```

当你不想重复定义函数的返回类型，或者返回类型很复杂时，`ReturnType` 非常有用：

```typescript
// 已有的函数定义（返回类型复杂，不想手写）
async function fetchChatCompletion(prompt: string) {
  // 实际调用 AI API 并返回结构化数据...
  return {
    id: "msg-001",
    content: "这是回复内容",
    model: "claude-sonnet-4-6",
    usage: {
      inputTokens: 15,
      outputTokens: 42,
      totalTokens: 57,
    },
    finishReason: "end_turn" as const,
    createdAt: new Date(),
  };
}

// 自动提取返回类型，无需手动定义
type ChatCompletion = Awaited<ReturnType<typeof fetchChatCompletion>>;
// {
//   id: string;
//   content: string;
//   model: string;
//   usage: { inputTokens: number; outputTokens: number; totalTokens: number };
//   finishReason: "end_turn";
//   createdAt: Date;
// }

// 后续代码直接使用提取到的类型
function formatCompletion(completion: ChatCompletion): string {
  return `[${completion.finishReason}] ${completion.content} (${completion.usage.totalTokens} tokens)`;
}
```

`ReturnType` 在**工厂函数**场景中特别有价值：

```typescript
// 工厂函数：根据配置创建不同类型的 AI 客户端
function createOpenAIClient(apiKey: string) {
  return {
    chat: async (prompt: string) => ({ content: "", model: "gpt-4o" }),
    embed: async (text: string) => ({ vector: [] as number[] }),
    moderations: async (text: string) => ({ flagged: false, categories: {} as Record<string, boolean> }),
  };
}

// 提取工厂函数的返回类型，作为客户端的类型
type OpenAIClient = ReturnType<typeof createOpenAIClient>;

// 泛型函数：接受任意 AI 客户端，只要它有 chat 方法
async function runChat<T extends Pick<OpenAIClient, "chat">>(
  client: T,
  prompt: string
): Promise<string> {
  const response = await client.chat(prompt);
  return response.content;
}
```

### 11.4.2 Parameters\<T\>：提取函数参数类型

`Parameters<T>` 从函数类型 `T` 中提取其**参数类型**，返回一个**元组类型**：

```typescript
// TypeScript 内部实现（仅供参考）
type Parameters<T extends (...args: any) => any> =
  T extends (...args: infer P) => any ? P : never;
```

```typescript
// 已有的函数
function createChatSession(
  userId: string,
  modelId: string,
  systemPrompt: string,
  options: {
    maxTurns: number;
    timeout: number;
    enableLogging: boolean;
  }
): { sessionId: string; startedAt: Date } {
  return { sessionId: `session-${Date.now()}`, startedAt: new Date() };
}

// 提取参数类型（元组）
type CreateSessionParams = Parameters<typeof createChatSession>;
// [
//   userId: string,
//   modelId: string,
//   systemPrompt: string,
//   options: { maxTurns: number; timeout: number; enableLogging: boolean }
// ]

// 提取第一个参数类型
type UserIdParam = Parameters<typeof createChatSession>[0];  // string

// 提取第四个参数（options）类型
type SessionOptions = Parameters<typeof createChatSession>[3];
// { maxTurns: number; timeout: number; enableLogging: boolean }

// 复用提取到的参数类型，避免重复定义
function createSessionWithDefaults(
  userId: string,
  modelId: string,
  options: Partial<SessionOptions> = {}
): ReturnType<typeof createChatSession> {
  return createChatSession(userId, modelId, "你是一个有帮助的助手。", {
    maxTurns: 50,
    timeout: 30000,
    enableLogging: true,
    ...options,
  });
}
```

### 11.4.3 配合展开运算符转发参数

`Parameters<T>` 与展开运算符结合，可以实现**完美的参数转发**（perfect forwarding）：

```typescript
// 原始 API 调用函数（参数复杂）
async function callAnthropicAPI(
  apiKey: string,
  model: string,
  messages: Array<{ role: "user" | "assistant"; content: string }>,
  maxTokens: number,
  temperature?: number,
  systemPrompt?: string
): Promise<{ content: string; usage: { inputTokens: number; outputTokens: number } }> {
  // 实际调用...
  return { content: "模拟回复", usage: { inputTokens: 10, outputTokens: 5 } };
}

// 带日志的包装函数：参数类型完全与原函数一致
function callAnthropicAPIWithLogging(
  ...args: Parameters<typeof callAnthropicAPI>
): ReturnType<typeof callAnthropicAPI> {
  const [apiKey, model, messages, maxTokens] = args;
  console.log(`[${new Date().toISOString()}] 调用 Anthropic API`);
  console.log(`  模型：${model}，最大 Token：${maxTokens}，消息数：${messages.length}`);
  const startTime = Date.now();
  return callAnthropicAPI(...args).then(result => {
    console.log(`  耗时：${Date.now() - startTime}ms`);
    return result;
  });
}

// 带重试的包装函数
function withRetry<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  maxRetries: number = 3
): (...args: Parameters<T>) => ReturnType<T> {
  return async (...args: Parameters<T>): ReturnType<T> => {
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await fn(...args);
      } catch (error) {
        if (i === maxRetries - 1) throw error;
        console.warn(`第 ${i + 1} 次重试...`);
        await new Promise(resolve => setTimeout(resolve, 500 * (i + 1)));
      }
    }
    throw new Error("不可达");
  };
}

const resilientCall = withRetry(callAnthropicAPI, 3);
// resilientCall 的参数类型与 callAnthropicAPI 完全一致 ✓
```

### 11.4.4 ConstructorParameters 与 InstanceType

除了 `ReturnType` 和 `Parameters`，TypeScript 还提供了针对**构造函数**的工具类型：

```typescript
// 提取构造函数的参数类型
class AIProvider {
  constructor(
    public readonly name: string,
    private apiKey: string,
    private options: { timeout: number; retries: number }
  ) {}

  async chat(prompt: string): Promise<string> {
    return `[${this.name}] 回复：${prompt}`;
  }
}

// 提取构造函数参数类型
type AIProviderParams = ConstructorParameters<typeof AIProvider>;
// [name: string, apiKey: string, options: { timeout: number; retries: number }]

// 提取实例类型
type AIProviderInstance = InstanceType<typeof AIProvider>;
// AIProvider（等价于直接写 AIProvider，但在泛型场景中很有用）

// 泛型工厂：接受任意构造函数，返回其实例
function createInstance<T extends new (...args: any[]) => any>(
  Ctor: T,
  ...args: ConstructorParameters<T>
): InstanceType<T> {
  return new Ctor(...args);
}

const provider = createInstance(
  AIProvider,
  "Anthropic",
  "sk-ant-xxx",
  { timeout: 30000, retries: 2 }
);
// provider 的类型自动推断为 AIProvider ✓
```

---

## 11.5 自定义工具类型

内置工具类型只覆盖了最常见的场景，实际项目中经常需要组合它们来创建**自定义工具类型**。本节展示几个在 AI 应用中高频使用的自定义工具类型。

### 11.5.1 PickByValue：按值类型筛选属性

内置的 `Pick` 按键名筛选，而 `PickByValue` 按值类型筛选：

```typescript
// 选出值类型可赋值给 V 的所有属性
type PickByValue<T, V> = {
  [K in keyof T as T[K] extends V ? K : never]: T[K];
};

// 排除值类型可赋值给 V 的所有属性
type OmitByValue<T, V> = {
  [K in keyof T as T[K] extends V ? never : K]: T[K];
};

interface ModelMetrics {
  modelId: string;
  provider: string;
  requestCount: number;
  successCount: number;
  failureCount: number;
  avgLatencyMs: number;
  isActive: boolean;
  isDeprecated: boolean;
  lastUpdated: Date;
}

// 提取所有 number 类型的指标字段
type NumericMetrics = PickByValue<ModelMetrics, number>;
// { requestCount: number; successCount: number; failureCount: number; avgLatencyMs: number }

// 提取所有 boolean 类型的标志字段
type FlagMetrics = PickByValue<ModelMetrics, boolean>;
// { isActive: boolean; isDeprecated: boolean }

// 排除所有 boolean 和 Date 类型
type DisplayMetrics = OmitByValue<ModelMetrics, boolean | Date>;
// { modelId: string; provider: string; requestCount: number; successCount: number; failureCount: number; avgLatencyMs: number }

// 用途：计算所有数字指标的总和（类型安全）
function sumMetrics(metrics: NumericMetrics): number {
  return Object.values(metrics).reduce((sum, val) => sum + val, 0);
}
```

### 11.5.2 Mutable：移除 readonly 修饰符

```typescript
// 移除所有属性的 readonly 修饰符
type Mutable<T> = {
  -readonly [K in keyof T]: T[K];
};

// 深度移除 readonly
type DeepMutable<T> = {
  -readonly [K in keyof T]: T[K] extends object ? DeepMutable<T[K]> : T[K];
};

// 使用场景：将只读的配置对象转为可修改副本
interface ReadonlyConfig {
  readonly model: string;
  readonly temperature: number;
  readonly maxTokens: number;
  readonly systemPrompt: string;
}

// 在配置编辑界面中需要可修改副本
type EditableConfig = Mutable<ReadonlyConfig>;
// { model: string; temperature: number; maxTokens: number; systemPrompt: string }

function createEditableConfig(config: ReadonlyConfig): EditableConfig {
  return { ...config }; // 浅拷贝，返回类型为 EditableConfig
}
```

### 11.5.3 NonNullableProperties：移除所有可空性

```typescript
// 将对象所有属性的值类型中的 null 和 undefined 移除
type NonNullableProperties<T> = {
  [K in keyof T]: NonNullable<T[K]>;
};

// 使用场景：AI API 响应经过验证后，断言所有字段非空
interface RawAPIResponse {
  id: string | null;
  content: string | null | undefined;
  model: string;
  finishReason: string | null;
  usage: {
    inputTokens: number | null;
    outputTokens: number | null;
  } | null;
}

type ValidatedResponse = NonNullableProperties<RawAPIResponse>;
// {
//   id: string;
//   content: string;
//   model: string;
//   finishReason: string;
//   usage: { inputTokens: number | null; outputTokens: number | null };
// }
// 注意：只处理顶层属性，嵌套的 usage 内部仍保留 null

function validateResponse(raw: RawAPIResponse): ValidatedResponse {
  if (!raw.id || !raw.content || !raw.finishReason || !raw.usage) {
    throw new Error("API 响应缺少必要字段");
  }
  return raw as ValidatedResponse;
}
```

### 11.5.4 Prettify：展开交叉类型以提高可读性

多个交叉类型 `A & B & C` 在 IDE 中悬浮提示时经常显示为 `A & B & C` 而非展开后的形状，`Prettify` 能强制展开：

```typescript
// 展开交叉类型，让 IDE 悬浮提示显示完整的对象形状
type Prettify<T> = {
  [K in keyof T]: T[K];
} & {};

// 示例：三个配置片段的交叉类型
type BaseConfig = { model: string; apiKey: string };
type RetryConfig = { maxRetries: number; retryDelay: number };
type LogConfig = { enableLogging: boolean; logLevel: "debug" | "info" | "warn" | "error" };

type FullConfig = BaseConfig & RetryConfig & LogConfig;
// IDE 悬浮提示：BaseConfig & RetryConfig & LogConfig（不直观）

type PrettyFullConfig = Prettify<FullConfig>;
// IDE 悬浮提示：
// {
//   model: string;
//   apiKey: string;
//   maxRetries: number;
//   retryDelay: number;
//   enableLogging: boolean;
//   logLevel: "debug" | "info" | "warn" | "error";
// }
```

### 11.5.5 组合多个工具类型

实际项目中，通常需要将多个工具类型组合使用：

```typescript
// 场景：AI 聊天函数的完整接口设计

interface FullChatParams {
  // 必填核心参数
  model: string;
  messages: Array<{ role: "system" | "user" | "assistant"; content: string }>;
  apiKey: string;

  // 可调节参数
  temperature: number;
  maxTokens: number;
  topP: number;
  topK: number;

  // 高级功能
  tools: Array<{ name: string; description: string }>;
  toolChoice: "auto" | "none" | "required";
  stream: boolean;

  // 基础设施参数
  timeout: number;
  retries: number;
  baseUrl: string;
}

// 1. 最小必填接口：只包含核心必要参数
type MinimalChatParams = Pick<FullChatParams, "model" | "messages" | "apiKey">;

// 2. 用户可配置的参数（排除基础设施参数）
type UserChatParams = Omit<FullChatParams, "timeout" | "retries" | "baseUrl">;

// 3. 带默认值的公开接口：核心必填 + 其余可选
type PublicChatParams = MinimalChatParams & Partial<Omit<FullChatParams, keyof MinimalChatParams>>;

// 4. 展开显示
type PrettyChatParams = Prettify<PublicChatParams>;
// {
//   model: string;             ← 必填
//   messages: [...];           ← 必填
//   apiKey: string;            ← 必填
//   temperature?: number;      ← 可选
//   maxTokens?: number;        ← 可选
//   ...其他可选字段
// }

// 实现函数签名
async function chat(params: PrettyChatParams): Promise<string> {
  const resolved: FullChatParams = {
    temperature: 0.7,
    maxTokens: 1024,
    topP: 1.0,
    topK: 40,
    tools: [],
    toolChoice: "auto",
    stream: false,
    timeout: 30000,
    retries: 2,
    baseUrl: "https://api.anthropic.com/v1",
    ...params,
  };
  // 实际调用...
  return "模拟回复";
}

// 调用方只需提供必填字段，其他字段使用默认值
const reply = await chat({
  model: "claude-sonnet-4-6",
  messages: [{ role: "user", content: "你好" }],
  apiKey: process.env.ANTHROPIC_API_KEY ?? "",
  // temperature, maxTokens 等均为可选
});
```

---

## 本章小结

| 工具类型 | 内部实现原理 | 典型用途 | AI 应用场景举例 |
|----------|------------|---------|----------------|
| `Partial<T>` | 映射类型 + `?` | 配置更新、可选参数 | 覆盖默认模型参数 |
| `Required<T>` | 映射类型 + `-?` | 参数验证后的类型收窄 | 合并默认值后断言完整配置 |
| `Pick<T, K>` | 映射类型 + 键过滤 | 裁剪类型、最小接口 | 为列表视图提取展示字段 |
| `Omit<T, K>` | `Pick` + `Exclude` | 去除敏感字段、属性覆盖 | 响应中移除 apiKey 等私密字段 |
| `Record<K, V>` | 映射类型 | 键值映射、枚举到值的映射 | provider 到 baseUrl 的映射表 |
| `Extract<T, U>` | 分布式条件类型 | 提取联合类型子集 | 过滤出聊天类事件名称 |
| `Exclude<T, U>` | 分布式条件类型 | 排除联合类型成员 | 剔除废弃/内部模型层级 |
| `ReturnType<T>` | `infer` + 条件类型 | 从函数提取返回类型 | 从工厂函数自动推断客户端类型 |
| `Parameters<T>` | `infer` + 条件类型 | 从函数提取参数类型元组 | 实现完美参数转发的包装函数 |

**实践建议**：
- 优先使用内置工具类型——它们经过优化，IDE 支持最好
- `Pick` vs `Omit`：保留字段少用 `Pick`，排除字段少用 `Omit`
- 需要深层处理时，自定义 `DeepPartial`、`DeepReadonly` 等递归版本
- 用 `Prettify<T>` 展开交叉类型，改善 IDE 提示体验
- `ReturnType` + `Parameters` 是减少重复类型定义的利器

---

## AI 应用实战：可选配置与必选参数

在 AI 应用开发中，一个常见的架构挑战是设计**灵活的 API 配置接口**：既要允许调用方按需传入参数（可选），又要在运行时确保关键参数存在（必选）。本节综合运用本章的工具类型，实现一套完整的、类型安全的配置体系。

### 需求场景

设计一个支持多家 AI 提供商的统一调用层，要求：

1. 调用方只需提供**最少必要参数**，其余参数有合理默认值
2. **运行时**完成参数合并后，通过类型系统保证所有字段非空
3. 不同调用场景（对话/嵌入/图像）**共享**基础参数，各自扩展专属参数
4. 敏感字段（`apiKey`）在日志和序列化时**自动隐藏**

### 完整实现

```typescript
// ============================================================
// 1. 基础类型定义：提供商与能力枚举
// ============================================================

type AIProvider = "openai" | "anthropic" | "google" | "cohere";

type ModelCapability =
  | "chat"
  | "embedding"
  | "image_generation"
  | "audio_transcription"
  | "vision"
  | "tool_use";

// ============================================================
// 2. 分层配置接口：基础、提供商、任务三层
// ============================================================

// 层 1：所有调用都需要的基础配置（全部可选，有默认值）
interface BaseInfraConfig {
  timeout: number;         // HTTP 超时（毫秒）
  retries: number;         // 失败重试次数
  retryDelay: number;      // 重试间隔（毫秒）
  enableLogging: boolean;  // 是否开启请求日志
  logLevel: "debug" | "info" | "warn" | "error";
}

// 层 2：提供商身份配置（apiKey 必填，其余可选）
interface ProviderIdentityConfig {
  provider: AIProvider;
  apiKey: string;           // 必填
  baseUrl?: string;         // 可选：覆盖默认端点
  organizationId?: string;  // 可选：OpenAI 专用
  projectId?: string;       // 可选：Google 专用
}

// 层 3a：对话任务配置
interface ChatTaskConfig {
  model: string;
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  stream?: boolean;
  tools?: Array<{
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  }>;
}

// 层 3b：嵌入任务配置
interface EmbeddingTaskConfig {
  model: string;
  encodingFormat?: "float" | "base64";
  dimensions?: number;
}

// 层 3c：图像生成任务配置
interface ImageTaskConfig {
  model: string;
  size?: "256x256" | "512x512" | "1024x1024" | "1024x1792" | "1792x1024";
  quality?: "standard" | "hd";
  style?: "vivid" | "natural";
  n?: number;
}

// ============================================================
// 3. 公开接口：必填 + 可选分离
// ============================================================

// 对话任务的必填字段
type RequiredChatFields = Pick<ProviderIdentityConfig, "provider" | "apiKey">
  & Pick<ChatTaskConfig, "model">;

// 对话任务的完整公开接口
type PublicChatConfig = RequiredChatFields
  & Partial<Omit<ProviderIdentityConfig, keyof RequiredChatFields>>
  & Partial<Omit<ChatTaskConfig, "model">>
  & Partial<BaseInfraConfig>;

// 展开以改善 IDE 提示
type ChatConfig = Prettify<PublicChatConfig>;

// 嵌入任务的必填字段
type RequiredEmbeddingFields = Pick<ProviderIdentityConfig, "provider" | "apiKey">
  & Pick<EmbeddingTaskConfig, "model">;

type EmbeddingConfig = Prettify<
  RequiredEmbeddingFields
  & Partial<Omit<ProviderIdentityConfig, keyof RequiredEmbeddingFields>>
  & Partial<Omit<EmbeddingTaskConfig, "model">>
  & Partial<BaseInfraConfig>
>;

// ============================================================
// 4. 内部完整配置：所有字段必填（合并默认值后使用）
// ============================================================

type ResolvedChatConfig = Required<
  Omit<ProviderIdentityConfig, "organizationId" | "projectId">
  & ChatTaskConfig
  & BaseInfraConfig
> & Pick<ProviderIdentityConfig, "organizationId" | "projectId">;
// 注意：organizationId 和 projectId 保持可选

// ============================================================
// 5. 默认值：按提供商区分
// ============================================================

const PROVIDER_DEFAULTS: Record<AIProvider, Pick<ProviderIdentityConfig, "baseUrl"> & Pick<ChatTaskConfig, "model">> = {
  openai:    { baseUrl: "https://api.openai.com/v1",                              model: "gpt-4o" },
  anthropic: { baseUrl: "https://api.anthropic.com/v1",                           model: "claude-sonnet-4-6" },
  google:    { baseUrl: "https://generativelanguage.googleapis.com/v1beta",       model: "gemini-1.5-pro" },
  cohere:    { baseUrl: "https://api.cohere.ai/v1",                               model: "command-r-plus" },
};

const INFRA_DEFAULTS: Required<BaseInfraConfig> = {
  timeout: 30000,
  retries: 2,
  retryDelay: 1000,
  enableLogging: false,
  logLevel: "info",
};

const CHAT_DEFAULTS: Required<Omit<ChatTaskConfig, "model" | "tools">> & Pick<ChatTaskConfig, "tools"> = {
  systemPrompt: "你是一个有帮助的 AI 助手。",
  temperature: 0.7,
  maxTokens: 1024,
  topP: 1.0,
  stream: false,
  tools: undefined,
};

// ============================================================
// 6. 配置解析函数：将公开接口 → 内部完整配置
// ============================================================

function resolveChatConfig(input: ChatConfig): ResolvedChatConfig {
  const providerDefaults = PROVIDER_DEFAULTS[input.provider];

  return {
    // 基础设施（优先用传入值，否则用默认值）
    ...INFRA_DEFAULTS,
    ...pick(input, ["timeout", "retries", "retryDelay", "enableLogging", "logLevel"]),

    // 提供商身份
    provider: input.provider,
    apiKey: input.apiKey,
    baseUrl: input.baseUrl ?? providerDefaults.baseUrl,
    organizationId: input.organizationId,
    projectId: input.projectId,

    // 对话任务参数
    model: input.model ?? providerDefaults.model,
    ...CHAT_DEFAULTS,
    ...pick(input, ["systemPrompt", "temperature", "maxTokens", "topP", "stream", "tools"]),
  } as ResolvedChatConfig;
}

// 工具：从对象中提取指定键（排除 undefined 值）
function pick<T extends object, K extends keyof T>(
  obj: T,
  keys: K[]
): Partial<Pick<T, K>> {
  const result = {} as Partial<Pick<T, K>>;
  for (const key of keys) {
    if (obj[key] !== undefined) {
      result[key] = obj[key];
    }
  }
  return result;
}

// ============================================================
// 7. 安全序列化：自动隐藏敏感字段
// ============================================================

// 去掉 apiKey 和 organizationId 的类型
type SafeLogConfig = Omit<ResolvedChatConfig, "apiKey" | "organizationId" | "projectId"> & {
  apiKey: "[REDACTED]";
  organizationId?: "[REDACTED]";
  projectId?: "[REDACTED]";
};

function toSafeLogConfig(config: ResolvedChatConfig): SafeLogConfig {
  return {
    ...config,
    apiKey: "[REDACTED]",
    organizationId: config.organizationId ? "[REDACTED]" : undefined,
    projectId: config.projectId ? "[REDACTED]" : undefined,
  };
}

// ============================================================
// 8. 类型安全的对话函数
// ============================================================

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface ChatResult {
  content: string;
  model: string;
  provider: AIProvider;
  usage: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
  finishReason: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use";
  resolvedConfig: SafeLogConfig; // 供调试用，apiKey 已脱敏
}

async function chatWithConfig(
  messages: ChatMessage[],
  config: ChatConfig
): Promise<ChatResult> {
  // 1. 解析配置（所有字段确保非 undefined）
  const resolved = resolveChatConfig(config);

  // 2. 记录日志（使用脱敏配置）
  if (resolved.enableLogging) {
    const safeConfig = toSafeLogConfig(resolved);
    console.log(`[${safeConfig.logLevel.toUpperCase()}] 发起对话请求`, {
      provider: safeConfig.provider,
      model: safeConfig.model,
      temperature: safeConfig.temperature,
      maxTokens: safeConfig.maxTokens,
      messageCount: messages.length,
    });
  }

  // 3. 注入 system prompt（如果消息中没有）
  const finalMessages: ChatMessage[] = messages[0]?.role === "system"
    ? messages
    : [{ role: "system", content: resolved.systemPrompt }, ...messages];

  // 4. 模拟 API 调用（实际项目中替换为真实 HTTP 请求）
  console.log(`调用 ${resolved.provider} API: ${resolved.baseUrl}/chat/completions`);
  console.log(`模型：${resolved.model}，消息数：${finalMessages.length}`);

  await new Promise(resolve => setTimeout(resolve, 100)); // 模拟网络延迟

  const mockResult: ChatResult = {
    content: `[${resolved.provider}/${resolved.model}] 这是模拟的回复内容。`,
    model: resolved.model,
    provider: resolved.provider,
    usage: {
      inputTokens: finalMessages.reduce((n, m) => n + m.content.length, 0),
      outputTokens: 42,
      totalTokens: finalMessages.reduce((n, m) => n + m.content.length, 0) + 42,
    },
    finishReason: "end_turn",
    resolvedConfig: toSafeLogConfig(resolved),
  };

  return mockResult;
}

// ============================================================
// 9. 使用示例：演示不同调用场景
// ============================================================

async function demo(): Promise<void> {
  // 场景 A：最简调用——只提供三个必填字段
  const resultA = await chatWithConfig(
    [{ role: "user", content: "TypeScript 的实用工具类型有哪些？" }],
    {
      provider: "anthropic",
      apiKey: process.env.ANTHROPIC_API_KEY ?? "sk-ant-demo",
      model: "claude-sonnet-4-6",
      // 其余参数使用默认值
    }
  );
  console.log("场景 A 回复：", resultA.content);
  console.log("Token 消耗：", resultA.usage.totalTokens);

  // 场景 B：定制化调用——覆盖部分默认值
  const resultB = await chatWithConfig(
    [
      { role: "system", content: "你是一个严谨的代码审查员，用简洁的中文回复。" },
      { role: "user", content: "请审查以下代码片段：const x = require('fs')" },
    ],
    {
      provider: "openai",
      apiKey: process.env.OPENAI_API_KEY ?? "sk-demo",
      model: "gpt-4o",
      temperature: 0.2,       // 更低的随机性，适合代码审查
      maxTokens: 512,
      enableLogging: true,    // 开启日志
      logLevel: "debug",
    }
  );
  console.log("\n场景 B 回复：", resultB.content);
  console.log("使用的配置（脱敏）：", JSON.stringify(resultB.resolvedConfig, null, 2));

  // 场景 C：自定义 baseUrl（使用代理或本地模型）
  const resultC = await chatWithConfig(
    [{ role: "user", content: "你好！" }],
    {
      provider: "openai",
      apiKey: "local-key",
      model: "llama3.1:8b",
      baseUrl: "http://localhost:11434/v1", // Ollama 本地服务
      timeout: 60000,                       // 本地模型响应慢，增大超时
    }
  );
  console.log("\n场景 C 回复：", resultC.content);
}

// ============================================================
// 10. 类型工具验证（编译期检查）
// ============================================================

// 验证：缺少必填字段应报错
// const badConfig: ChatConfig = {
//   provider: "anthropic",
//   // 缺少 apiKey 和 model → 编译错误 ✓
// };

// 验证：多余字段应报错
// const badConfig2: ChatConfig = {
//   provider: "anthropic",
//   apiKey: "sk-xxx",
//   model: "claude-sonnet-4-6",
//   unknownField: true, // 编译错误：对象字面量不允许多余属性 ✓
// };

// 验证：ResolvedChatConfig 所有字段都是非可选的
type CheckRequired = keyof ResolvedChatConfig extends string ? true : false;
// 通过类型验证：ResolvedChatConfig 无任何可选属性（除 organizationId/projectId）✓
```

### 设计要点总结

| 工具类型 | 在本案例中的作用 |
|---------|----------------|
| `Pick<T, K>` | 提取 `provider`、`apiKey`、`model` 三个必填字段，构成 `RequiredXxxFields` |
| `Omit<T, K>` | 排除必填字段后对其余字段应用 `Partial`，避免重复声明 |
| `Partial<T>` | 将基础设施配置和任务配置的所有字段变为可选，供调用方按需传入 |
| `Required<T>` | 标记 `ResolvedChatConfig` 内部完整配置，确保合并默认值后不存在 `undefined` |
| `Record<K, V>` | 为每个 AI 提供商记录默认的 `baseUrl` 和 `model`，键穷举由编译器保证 |
| `Prettify<T>` | 展开交叉类型，改善 `ChatConfig` 在 IDE 中的悬浮提示显示 |

---

## 练习题

### 基础题

**练习 11-1**：使用 `Partial`、`Required` 和 `Readonly` 实现一个 `ModelConfigManager` 类：

- 构造函数接受 `Partial<ModelConfig>` 作为初始配置（字段全部可选）
- `update(patch: Partial<ModelConfig>): void` 方法合并更新配置
- `get(): Readonly<Required<ModelConfig>>` 方法返回当前完整配置（所有字段非空且只读）
- `reset(): void` 方法恢复默认配置

```typescript
interface ModelConfig {
  model: string;
  temperature: number;
  maxTokens: number;
  topP: number;
  systemPrompt: string;
}

// 默认值
const MODEL_DEFAULTS: Required<ModelConfig> = {
  model: "claude-sonnet-4-6",
  temperature: 0.7,
  maxTokens: 1024,
  topP: 1.0,
  systemPrompt: "你是一个有帮助的助手。",
};

class ModelConfigManager {
  // 在此实现...
}

// 测试：
const manager = new ModelConfigManager({ model: "gpt-4o" });
console.log(manager.get().model);        // "gpt-4o"
console.log(manager.get().temperature);  // 0.7（使用默认值）
manager.update({ temperature: 0.9, maxTokens: 2048 });
console.log(manager.get().temperature);  // 0.9
manager.reset();
console.log(manager.get().model);        // "claude-sonnet-4-6"（恢复默认）
```

**练习 11-2**：使用 `Pick`、`Omit` 和 `Record` 完成下列类型操作：

```typescript
interface APIEndpoint {
  path: string;
  method: "GET" | "POST" | "PUT" | "DELETE";
  requiresAuth: boolean;
  rateLimit: number;         // 每分钟请求次数
  deprecated: boolean;
  version: "v1" | "v2" | "v3";
}

// (1) 定义 PublicEndpointInfo 类型：只包含 path、method、version 三个字段
type PublicEndpointInfo = /* 你的答案 */;

// (2) 定义 EndpointMetadata 类型：排除 path 和 method，保留其余所有字段
type EndpointMetadata = /* 你的答案 */;

// (3) 定义 EndpointRegistry 类型：
//     键为端点名称（字符串），值为 APIEndpoint
type EndpointRegistry = /* 你的答案 */;

// (4) 创建一个符合 EndpointRegistry 的注册表对象 registry，包含至少两个端点
```

---

### 进阶题

**练习 11-3**：实现 `createTypedStorage<T>()` 函数，返回一个类型安全的本地存储封装：

```typescript
// 接受泛型参数 T（描述存储的数据结构）
// 返回一个对象，包含：
// - set<K extends keyof T>(key: K, value: T[K]): void
// - get<K extends keyof T>(key: K): T[K] | null
// - remove<K extends keyof T>(key: K): void
// - clear(): void
// - getAll(): Partial<T>

interface AppStorage {
  userId: string;
  authToken: string;
  preferredModel: string;
  chatHistory: Array<{ role: string; content: string }>;
  settings: { theme: "light" | "dark"; language: string };
}

const storage = createTypedStorage<AppStorage>();

storage.set("userId", "user-123");          // ✓
storage.set("preferredModel", "gpt-4o");    // ✓
// storage.set("unknownKey", "value");       // 编译错误 ✓
// storage.set("userId", 123);               // 编译错误：类型不匹配 ✓

const userId = storage.get("userId");       // 类型为 string | null ✓
const settings = storage.get("settings");   // 类型为 { theme: "light" | "dark"; language: string } | null ✓
```

**练习 11-4**：使用 `ReturnType`、`Parameters` 和 `Record` 实现一个**函数注册表**（Function Registry）：

```typescript
// 定义几个 AI 相关的处理函数
async function summarize(text: string, maxLength: number): Promise<string> {
  return text.slice(0, maxLength) + "...";
}

async function translate(text: string, targetLang: string, formality: "formal" | "informal"): Promise<string> {
  return `[${targetLang}] ${text}`;
}

async function classify(text: string, categories: string[]): Promise<{ category: string; confidence: number }> {
  return { category: categories[0], confidence: 0.95 };
}

// 实现 FunctionRegistry 类：
// - register<N extends string, F extends (...args: any[]) => any>(name: N, fn: F): void
//   注册函数后，后续 call(name, ...args) 的参数和返回类型应与 F 完全一致
// - call<N extends keyof Registry>(name: N, ...args: Parameters<Registry[N]>): ReturnType<Registry[N]>
//   调用已注册函数，参数类型和返回类型由注册时的函数决定

// 提示：这道题需要用到 ReturnType、Parameters，以及泛型类型推断
```

---

### 挑战题

**练习 11-5**：设计一个**类型安全的 AI Pipeline（处理管道）**系统。

需求描述：

1. 定义若干处理步骤（Step）类型，每个 Step 有输入类型和输出类型
2. 使用工具类型设计 `Pipeline<TInput, TOutput>` 接口，保证链式调用时相邻步骤的输入输出类型匹配
3. 实现 `createPipeline<T>(initialValue: T)` 工厂函数，返回可链式调用的管道对象
4. 管道中每个 `step` 方法接受一个处理函数，TypeScript 自动推断下一步的输入类型
5. 最终 `run()` 方法返回管道的最终输出，其类型由最后一个 step 的输出决定

验证示例：

```typescript
// 定义步骤
const tokenize = (text: string): string[] => text.split(" ");
const countWords = (words: string[]): number => words.length;
const formatResult = (count: number): { wordCount: number; category: string } => ({
  wordCount: count,
  category: count > 100 ? "长文本" : "短文本",
});

// 创建并运行管道
const result = createPipeline("TypeScript 的工具类型是一个非常强大的特性")
  .step(tokenize)      // string -> string[]
  .step(countWords)    // string[] -> number
  .step(formatResult)  // number -> { wordCount: number; category: string }
  .run();

// result 类型自动推断为 { wordCount: number; category: string } ✓
console.log(result.wordCount);  // 数字
console.log(result.category);   // "短文本"

// 类型错误应在编译期捕获：
// createPipeline("hello")
//   .step((s: string) => s.length)
//   .step((words: string[]) => words.join(",")); // 编译错误：number 不能赋给 string[] ✓
```

附加要求：使用 `Parameters<T>` 和 `ReturnType<T>` 在 `step` 方法的类型签名中推断输入输出类型。

---

## 练习答案

### 练习 11-1 答案

```typescript
class ModelConfigManager {
  private config: Required<ModelConfig>;

  constructor(initial: Partial<ModelConfig> = {}) {
    this.config = { ...MODEL_DEFAULTS, ...initial };
  }

  update(patch: Partial<ModelConfig>): void {
    this.config = { ...this.config, ...patch };
  }

  get(): Readonly<Required<ModelConfig>> {
    // 返回只读副本，防止外部直接修改
    return Object.freeze({ ...this.config });
  }

  reset(): void {
    this.config = { ...MODEL_DEFAULTS };
  }
}

// 测试
const manager = new ModelConfigManager({ model: "gpt-4o" });
console.log(manager.get().model);        // "gpt-4o"
console.log(manager.get().temperature);  // 0.7

manager.update({ temperature: 0.9, maxTokens: 2048 });
console.log(manager.get().temperature);  // 0.9
console.log(manager.get().maxTokens);    // 2048

manager.reset();
console.log(manager.get().model);        // "claude-sonnet-4-6"
console.log(manager.get().temperature);  // 0.7

// 验证只读性（运行时不会抛出错误，但 TypeScript 编译期会报错）
// manager.get().model = "test"; // 编译错误：只读属性 ✓
```

### 练习 11-2 答案

```typescript
// (1) 只包含 path、method、version
type PublicEndpointInfo = Pick<APIEndpoint, "path" | "method" | "version">;
// { path: string; method: "GET" | "POST" | "PUT" | "DELETE"; version: "v1" | "v2" | "v3" }

// (2) 排除 path 和 method
type EndpointMetadata = Omit<APIEndpoint, "path" | "method">;
// { requiresAuth: boolean; rateLimit: number; deprecated: boolean; version: "v1" | "v2" | "v3" }

// (3) 键为字符串，值为 APIEndpoint
type EndpointRegistry = Record<string, APIEndpoint>;

// (4) 示例注册表
const registry: EndpointRegistry = {
  "chat.completions": {
    path: "/v1/chat/completions",
    method: "POST",
    requiresAuth: true,
    rateLimit: 60,
    deprecated: false,
    version: "v1",
  },
  "embeddings": {
    path: "/v1/embeddings",
    method: "POST",
    requiresAuth: true,
    rateLimit: 120,
    deprecated: false,
    version: "v1",
  },
  "images.generate.legacy": {
    path: "/v1/images/generations",
    method: "POST",
    requiresAuth: true,
    rateLimit: 10,
    deprecated: true,
    version: "v1",
  },
};

// 过滤出未废弃的端点
const activeEndpoints = Object.entries(registry)
  .filter(([, endpoint]) => !endpoint.deprecated)
  .map(([name, endpoint]): [string, PublicEndpointInfo] => [
    name,
    { path: endpoint.path, method: endpoint.method, version: endpoint.version },
  ]);

console.log("活跃端点：", activeEndpoints);
```

### 练习 11-3 答案

```typescript
function createTypedStorage<T extends Record<string, unknown>>() {
  return {
    set<K extends keyof T>(key: K, value: T[K]): void {
      try {
        localStorage.setItem(String(key), JSON.stringify(value));
      } catch {
        // 非浏览器环境的兼容处理
        console.warn(`storage.set: localStorage 不可用`);
      }
    },

    get<K extends keyof T>(key: K): T[K] | null {
      try {
        const raw = localStorage.getItem(String(key));
        if (raw === null) return null;
        return JSON.parse(raw) as T[K];
      } catch {
        return null;
      }
    },

    remove<K extends keyof T>(key: K): void {
      try {
        localStorage.removeItem(String(key));
      } catch {
        console.warn(`storage.remove: localStorage 不可用`);
      }
    },

    clear(): void {
      try {
        localStorage.clear();
      } catch {
        console.warn(`storage.clear: localStorage 不可用`);
      }
    },

    getAll(): Partial<T> {
      const result: Partial<T> = {};
      try {
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i);
          if (key !== null) {
            const raw = localStorage.getItem(key);
            if (raw !== null) {
              try {
                (result as Record<string, unknown>)[key] = JSON.parse(raw);
              } catch {
                // 忽略无法解析的项
              }
            }
          }
        }
      } catch {
        console.warn(`storage.getAll: localStorage 不可用`);
      }
      return result;
    },
  };
}

// 测试（Node.js 环境无 localStorage，仅验证类型）
const storage = createTypedStorage<AppStorage>();

// 以下类型检查在编译期生效：
// storage.set("userId", "user-123");          // ✓ string
// storage.set("chatHistory", []);             // ✓ Array<...>
// storage.set("settings", { theme: "dark", language: "zh-CN" }); // ✓
// storage.set("unknownKey", "value");         // 编译错误 ✓
// storage.set("userId", 123);                 // 编译错误：number 不是 string ✓

// 返回类型验证：
// const userId = storage.get("userId");       // string | null ✓
// const settings = storage.get("settings");   // { theme: ...; language: string } | null ✓
```

### 练习 11-4 答案

```typescript
class FunctionRegistry<
  TRegistry extends Record<string, (...args: any[]) => any> = {}
> {
  private fns: Record<string, (...args: any[]) => any> = {};

  register<N extends string, F extends (...args: any[]) => any>(
    name: N,
    fn: F
  ): FunctionRegistry<TRegistry & Record<N, F>> {
    this.fns[name] = fn;
    // 返回新的实例类型（通过类型断言），实际指向同一个对象
    return this as unknown as FunctionRegistry<TRegistry & Record<N, F>>;
  }

  async call<N extends keyof TRegistry>(
    name: N,
    ...args: Parameters<TRegistry[N]>
  ): ReturnType<TRegistry[N]> {
    const fn = this.fns[name as string];
    if (!fn) throw new Error(`未注册函数：${String(name)}`);
    return fn(...args) as ReturnType<TRegistry[N]>;
  }

  has(name: string): boolean {
    return name in this.fns;
  }

  list(): string[] {
    return Object.keys(this.fns);
  }
}

// 使用示例
const registry = new FunctionRegistry()
  .register("summarize", summarize)
  .register("translate", translate)
  .register("classify", classify);

// 类型安全调用：参数和返回类型自动推断
const summary = await registry.call("summarize", "这是一段很长的文本...", 50);
// summary: string ✓

const translation = await registry.call("translate", "Hello", "zh-CN", "formal");
// translation: string ✓

const classification = await registry.call("classify", "这是一个问题", ["问题", "陈述", "命令"]);
// classification: { category: string; confidence: number } ✓

// 编译错误示例：
// await registry.call("summarize", 123, 50);     // 错误：第一个参数应为 string ✓
// await registry.call("unknownFn", "text", 50);  // 错误：未注册的函数名 ✓

console.log("已注册函数：", registry.list());
// ["summarize", "translate", "classify"]
```

### 练习 11-5 答案

```typescript
// Pipeline 类：用泛型参数 T 追踪当前值的类型
class Pipeline<T> {
  private constructor(private readonly value: T) {}

  // 工厂方法（避免 new 关键字暴露给使用者）
  static of<T>(value: T): Pipeline<T> {
    return new Pipeline(value);
  }

  // step：接受一个函数，将当前值转换为新类型
  // 利用 ReturnType 推断下一步的类型
  step<F extends (input: T) => any>(
    fn: F
  ): Pipeline<ReturnType<F>> {
    return new Pipeline<ReturnType<F>>(fn(this.value));
  }

  // 异步版本的 step
  asyncStep<F extends (input: T) => Promise<any>>(
    fn: F
  ): Promise<Pipeline<Awaited<ReturnType<F>>>> {
    return fn(this.value).then(
      result => new Pipeline<Awaited<ReturnType<F>>>(result)
    );
  }

  // 运行管道，返回最终值
  run(): T {
    return this.value;
  }

  // 调试：查看当前值（不改变管道状态）
  tap(fn: (value: T) => void): Pipeline<T> {
    fn(this.value);
    return this;
  }
}

// 工厂函数
function createPipeline<T>(initialValue: T): Pipeline<T> {
  return Pipeline.of(initialValue);
}

// 测试
const tokenize = (text: string): string[] => text.split(" ");
const countWords = (words: string[]): number => words.length;
const formatResult = (count: number): { wordCount: number; category: string } => ({
  wordCount: count,
  category: count > 10 ? "长文本" : "短文本",
});

const result = createPipeline("TypeScript 的工具类型是一个非常强大的特性")
  .step(tokenize)       // Pipeline<string[]>
  .step(countWords)     // Pipeline<number>
  .step(formatResult)   // Pipeline<{ wordCount: number; category: string }>
  .run();               // { wordCount: number; category: string }

console.log(result.wordCount);  // 8
console.log(result.category);   // "短文本"

// 带调试输出的管道
const debugResult = createPipeline("Hello TypeScript World")
  .step((s: string) => s.toLowerCase())
  .tap(s => console.log("小写后：", s))
  .step((s: string) => s.split(" "))
  .tap(words => console.log("分词后：", words))
  .step((words: string[]) => words.filter(w => w.length > 4))
  .run();

console.log("过滤后：", debugResult); // ["hello", "typescript", "world"] 中长度 > 4 的

// 类型错误验证（编译期）：
// createPipeline("hello")
//   .step((s: string) => s.length)
//   .step((words: string[]) => words.join(","));
// 编译错误：number 类型不能赋给 string[]，类型不匹配 ✓

// AI 应用示例：文本处理管道
const aiPipeline = createPipeline("  这是一段需要处理的 AI 生成文本，包含一些多余的空格。  ")
  .step((raw: string) => raw.trim())
  .step((text: string) => text.replace(/\s+/g, " "))
  .step((text: string) => ({
    text,
    wordCount: text.split(" ").length,
    charCount: text.length,
    preview: text.slice(0, 20) + (text.length > 20 ? "..." : ""),
  }))
  .run();

console.log("AI 文本处理结果：", aiPipeline);
// { text: "这是一段需要处理的 AI 生成文本，包含一些多余的空格。", wordCount: ..., charCount: ..., preview: "..." }
```

---

*下一章：[第12章：声明文件与类型定义](./12-declaration-files.md)*
