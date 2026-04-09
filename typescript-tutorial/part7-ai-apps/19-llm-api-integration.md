# 第19章：LLM API集成

> **适用读者**：已具备 TypeScript 基础与异步编程知识，希望在实际项目中集成 OpenAI、Anthropic 等主流大语言模型 API，并掌握类型安全调用方式的开发者。

---

## 学习目标

完成本章学习后，你将能够：

1. 使用 `openai` 官方 SDK 完成文本生成、流式输出与函数调用，理解其 TypeScript 类型体系
2. 使用 `@anthropic-ai/sdk` 调用 Claude 系列模型，掌握 Messages API 的请求与响应结构
3. 通过环境变量与 `dotenv` 安全管理 API 密钥，避免密钥泄露到版本控制系统
4. 理解各 SDK 的请求参数类型与响应类型，编写完全类型安全的 LLM 调用代码
5. 实现指数退避重试策略与错误分类处理，构建生产级别的容错 LLM 客户端

---

## 19.1 OpenAI SDK 使用（openai 包）

### 19.1.1 安装与初始化

OpenAI 官方 Node.js SDK 提供了完整的 TypeScript 类型声明，直接安装即可使用：

```bash
npm install openai
# 或
pnpm add openai
```

初始化客户端时，SDK 会自动从 `OPENAI_API_KEY` 环境变量读取密钥（如果不手动传入）：

```typescript
import OpenAI from "openai";

// 方式1：自动从环境变量 OPENAI_API_KEY 读取
const client = new OpenAI();

// 方式2：显式传入（适合多账号场景）
const clientExplicit = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  // 可选：自定义 base URL（兼容 OpenAI 协议的第三方服务）
  baseURL: "https://api.openai.com/v1",
  // 可选：请求超时（毫秒）
  timeout: 30_000,
  // 可选：最大重试次数（SDK 内置简单重试）
  maxRetries: 2,
});
```

### 19.1.2 基础文本生成

最常用的接口是 `chat.completions.create`，对应 Chat Completions API：

```typescript
import OpenAI from "openai";

const openai = new OpenAI();

async function generateText(prompt: string): Promise<string> {
  const completion = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: "你是一个专业的 TypeScript 技术顾问，回答要简洁准确。",
      },
      {
        role: "user",
        content: prompt,
      },
    ],
    temperature: 0.7,
    max_tokens: 1024,
  });

  // completion.choices[0].message.content 类型为 string | null
  const content = completion.choices[0].message.content;
  if (content === null) {
    throw new Error("模型返回了空内容");
  }
  return content;
}

// 使用示例
const answer = await generateText("解释 TypeScript 中的 never 类型");
console.log(answer);
```

`completion` 对象的完整类型是 `OpenAI.Chat.Completions.ChatCompletion`，其关键字段包括：

```typescript
interface ChatCompletion {
  id: string;                          // 请求唯一 ID
  object: "chat.completion";
  created: number;                     // Unix 时间戳
  model: string;                       // 实际使用的模型
  choices: Array<{
    index: number;
    message: {
      role: "assistant";
      content: string | null;
      tool_calls?: ToolCall[];          // 工具调用（如有）
    };
    finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  } | undefined;
}
```

### 19.1.3 流式输出（Streaming）

流式输出可以实现"打字机"效果，减少用户等待感。OpenAI SDK 通过异步迭代器暴露流：

```typescript
import OpenAI from "openai";

const openai = new OpenAI();

async function streamText(prompt: string): Promise<void> {
  // stream: true 时返回 Stream<ChatCompletionChunk>
  const stream = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: prompt }],
    stream: true,
  });

  process.stdout.write("助手：");

  for await (const chunk of stream) {
    // 每个 chunk 类型为 OpenAI.Chat.Completions.ChatCompletionChunk
    const delta = chunk.choices[0]?.delta;
    if (delta?.content) {
      process.stdout.write(delta.content);
    }
  }

  console.log(); // 换行
}

await streamText("用三句话介绍 TypeScript 的核心优势");
```

如果需要同时收集完整内容，可以用辅助方法 `streamText` 累积结果：

```typescript
async function streamAndCollect(prompt: string): Promise<string> {
  const stream = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: prompt }],
    stream: true,
  });

  const chunks: string[] = [];

  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) {
      chunks.push(content);
      process.stdout.write(content); // 实时输出
    }
  }

  console.log();
  return chunks.join("");
}
```

### 19.1.4 函数调用（Tool Calls）

函数调用（Function Calling）让模型能够以结构化 JSON 输出调用你定义的工具：

```typescript
import OpenAI from "openai";

const openai = new OpenAI();

// 定义工具的类型
const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "获取指定城市的当前天气信息",
      parameters: {
        type: "object",
        properties: {
          city: {
            type: "string",
            description: "城市名称，例如：北京、上海",
          },
          unit: {
            type: "string",
            enum: ["celsius", "fahrenheit"],
            description: "温度单位",
          },
        },
        required: ["city"],
      },
    },
  },
];

// 模拟的天气查询函数
function getWeather(city: string, unit: string = "celsius"): string {
  return JSON.stringify({ city, temperature: 22, unit, condition: "晴天" });
}

async function chatWithTools(userMessage: string): Promise<string> {
  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: "user", content: userMessage },
  ];

  // 第一轮：让模型决定是否调用工具
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages,
    tools,
    tool_choice: "auto",
  });

  const assistantMessage = response.choices[0].message;

  // 检查是否触发了工具调用
  if (assistantMessage.tool_calls && assistantMessage.tool_calls.length > 0) {
    messages.push(assistantMessage);

    // 执行工具调用并收集结果
    for (const toolCall of assistantMessage.tool_calls) {
      const args = JSON.parse(toolCall.function.arguments) as {
        city: string;
        unit?: string;
      };
      const result = getWeather(args.city, args.unit);

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: result,
      });
    }

    // 第二轮：将工具结果回传给模型
    const finalResponse = await openai.chat.completions.create({
      model: "gpt-4o",
      messages,
    });

    return finalResponse.choices[0].message.content ?? "";
  }

  return assistantMessage.content ?? "";
}

const result = await chatWithTools("北京今天天气怎么样？");
console.log(result);
```

---

## 19.2 Anthropic SDK 使用（@anthropic-ai/sdk）

### 19.2.1 安装与初始化

```bash
npm install @anthropic-ai/sdk
```

Anthropic SDK 同样提供完整的 TypeScript 类型支持：

```typescript
import Anthropic from "@anthropic-ai/sdk";

// 自动读取 ANTHROPIC_API_KEY 环境变量
const client = new Anthropic();

// 显式配置
const clientExplicit = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  // 可选：自定义 base URL
  baseURL: "https://api.anthropic.com",
  // 可选：超时（毫秒）
  timeout: 60_000,
  // 可选：最大重试次数
  maxRetries: 2,
  // 可选：自定义请求头
  defaultHeaders: {
    "anthropic-beta": "prompt-caching-2024-07-31",
  },
});
```

### 19.2.2 Messages API 基础调用

Anthropic 使用 Messages API，与 OpenAI 的 Chat Completions 在结构上有所不同：

```typescript
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic();

async function callClaude(userMessage: string): Promise<string> {
  // 返回类型为 Anthropic.Message
  const message = await anthropic.messages.create({
    model: "claude-sonnet-4-6-20251101",
    max_tokens: 1024,
    // system 是顶层字段，不在 messages 数组中
    system: "你是一个专业的 TypeScript 开发者，提供准确的技术解答。",
    messages: [
      {
        role: "user",
        content: userMessage,
      },
    ],
  });

  // content 是 ContentBlock 数组
  const firstBlock = message.content[0];
  if (firstBlock.type === "text") {
    return firstBlock.text;
  }
  throw new Error(`意外的内容类型: ${firstBlock.type}`);
}

const response = await callClaude("TypeScript 中 interface 和 type 有什么区别？");
console.log(response);
```

### 19.2.3 Anthropic 响应类型详解

`Anthropic.Message` 的完整结构如下：

```typescript
interface Message {
  id: string;                         // 请求唯一 ID，如 "msg_01XFDUDYJgAACzvnptvVoYEL"
  type: "message";
  role: "assistant";
  content: Array<ContentBlock>;       // 内容块数组
  model: string;                      // 使用的模型
  stop_reason: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use" | null;
  stop_sequence: string | null;       // 触发停止的序列（如有）
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

// ContentBlock 是联合类型
type ContentBlock = TextBlock | ToolUseBlock;

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
```

### 19.2.4 多轮对话

Anthropic Messages API 通过在 `messages` 数组中交替传入 `user` 和 `assistant` 消息实现多轮对话：

```typescript
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic();

type ConversationMessage = {
  role: "user" | "assistant";
  content: string;
};

class ClaudeConversation {
  private history: ConversationMessage[] = [];
  private readonly systemPrompt: string;

  constructor(systemPrompt: string) {
    this.systemPrompt = systemPrompt;
  }

  async chat(userMessage: string): Promise<string> {
    // 添加用户消息
    this.history.push({ role: "user", content: userMessage });

    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-6-20251101",
      max_tokens: 2048,
      system: this.systemPrompt,
      messages: this.history,
    });

    const assistantText = response.content
      .filter((block): block is Anthropic.TextBlock => block.type === "text")
      .map((block) => block.text)
      .join("");

    // 保存助手回复到历史
    this.history.push({ role: "assistant", content: assistantText });

    return assistantText;
  }

  clearHistory(): void {
    this.history = [];
  }
}

// 使用示例
const conversation = new ClaudeConversation("你是一个 TypeScript 专家助手。");

const reply1 = await conversation.chat("什么是泛型？");
console.log("第一轮：", reply1);

const reply2 = await conversation.chat("能给我一个实际例子吗？");
console.log("第二轮：", reply2);
```

### 19.2.5 流式输出

```typescript
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic();

async function streamClaude(prompt: string): Promise<string> {
  const collected: string[] = [];

  // stream() 方法返回 Stream<MessageStreamEvent>
  const stream = anthropic.messages.stream({
    model: "claude-sonnet-4-6-20251101",
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
  });

  // 监听文本增量事件
  stream.on("text", (text) => {
    process.stdout.write(text);
    collected.push(text);
  });

  // 等待流结束并获取最终消息
  const finalMessage = await stream.finalMessage();
  console.log("\n--- 用量统计 ---");
  console.log(`输入 tokens: ${finalMessage.usage.input_tokens}`);
  console.log(`输出 tokens: ${finalMessage.usage.output_tokens}`);

  return collected.join("");
}

await streamClaude("请解释 TypeScript 中的协变与逆变");
```

也可以使用异步迭代器风格：

```typescript
async function streamClaudeIterator(prompt: string): Promise<void> {
  const stream = await anthropic.messages.create({
    model: "claude-sonnet-4-6-20251101",
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
    stream: true,
  });

  for await (const event of stream) {
    if (
      event.type === "content_block_delta" &&
      event.delta.type === "text_delta"
    ) {
      process.stdout.write(event.delta.text);
    }
  }
  console.log();
}
```

---

## 19.3 API 密钥管理与环境变量

### 19.3.1 使用 dotenv 管理本地密钥

绝对不要将 API 密钥硬编码到源码或提交到版本控制。标准做法是使用 `.env` 文件：

```bash
# 安装 dotenv
npm install dotenv
npm install --save-dev @types/node
```

在项目根目录创建 `.env` 文件：

```bash
# .env（永远不要提交到 git！）
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_ORG_ID=org-xxxxxxxxxxxxxxxx

# 可选的应用配置
DEFAULT_MODEL=gpt-4o
MAX_TOKENS=2048
REQUEST_TIMEOUT_MS=30000
```

在 `.gitignore` 中添加：

```gitignore
# .gitignore
.env
.env.local
.env.*.local
```

在应用入口处加载环境变量：

```typescript
// src/config.ts
import * as dotenv from "dotenv";
import * as path from "path";

// 加载 .env 文件（在任何 import SDK 之前调用）
dotenv.config({ path: path.resolve(process.cwd(), ".env") });

// 定义配置类型
interface AppConfig {
  openaiApiKey: string;
  anthropicApiKey: string;
  defaultModel: string;
  maxTokens: number;
  requestTimeoutMs: number;
}

// 带校验的配置读取函数
function loadConfig(): AppConfig {
  const requiredKeys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"];

  for (const key of requiredKeys) {
    if (!process.env[key]) {
      throw new Error(
        `缺少必要的环境变量: ${key}\n` +
          `请在项目根目录创建 .env 文件并设置该变量。`
      );
    }
  }

  return {
    openaiApiKey: process.env.OPENAI_API_KEY!,
    anthropicApiKey: process.env.ANTHROPIC_API_KEY!,
    defaultModel: process.env.DEFAULT_MODEL ?? "gpt-4o",
    maxTokens: parseInt(process.env.MAX_TOKENS ?? "2048", 10),
    requestTimeoutMs: parseInt(process.env.REQUEST_TIMEOUT_MS ?? "30000", 10),
  };
}

export const config = loadConfig();
```

### 19.3.2 不同部署环境的密钥管理

生产环境不应使用 `.env` 文件，而是通过平台原生的密钥管理机制：

```typescript
// src/config/env-loader.ts

type Environment = "development" | "staging" | "production";

function detectEnvironment(): Environment {
  const nodeEnv = process.env.NODE_ENV?.toLowerCase();
  if (nodeEnv === "production") return "production";
  if (nodeEnv === "staging") return "staging";
  return "development";
}

// 仅在开发环境加载 .env 文件
export function setupEnvironment(): void {
  const env = detectEnvironment();

  if (env === "development") {
    // 开发环境：从 .env 文件加载
    const { config: dotenvConfig } = await import("dotenv");
    dotenvConfig();
    console.log("[Config] 已从 .env 文件加载配置");
  } else {
    // 生产/预发布：依赖平台注入的环境变量
    // AWS: Secrets Manager / Parameter Store
    // GCP: Secret Manager
    // Azure: Key Vault
    // Vercel/Railway: 平台环境变量配置界面
    console.log(`[Config] 使用平台环境变量（${env} 环境）`);
  }

  validateRequiredEnvVars(["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]);
}

function validateRequiredEnvVars(keys: string[]): void {
  const missing = keys.filter((k) => !process.env[k]);
  if (missing.length > 0) {
    throw new Error(`缺少环境变量: ${missing.join(", ")}`);
  }
}
```

### 19.3.3 密钥安全最佳实践

```typescript
// src/utils/api-key-validator.ts

/**
 * 验证 OpenAI API 密钥格式（不校验有效性，只校验格式）
 */
export function validateOpenAIKeyFormat(key: string): boolean {
  // OpenAI 项目密钥格式: sk-proj-...
  // 旧格式: sk-...
  return /^sk-(proj-)?[A-Za-z0-9_-]{32,}$/.test(key);
}

/**
 * 验证 Anthropic API 密钥格式
 */
export function validateAnthropicKeyFormat(key: string): boolean {
  return /^sk-ant-api\d{2}-[A-Za-z0-9_-]{32,}$/.test(key);
}

/**
 * 脱敏展示密钥（用于日志）
 * 例如：sk-proj-abc...xyz
 */
export function maskApiKey(key: string): string {
  if (key.length <= 12) return "***";
  const prefix = key.substring(0, 8);
  const suffix = key.substring(key.length - 4);
  return `${prefix}...${suffix}`;
}

// 使用示例
const key = process.env.OPENAI_API_KEY ?? "";
console.log(`使用 API 密钥: ${maskApiKey(key)}`);
// 输出: 使用 API 密钥: sk-proj-...wxyz
```

---

## 19.4 请求参数类型与响应类型

### 19.4.1 OpenAI 请求参数类型

OpenAI SDK 为所有请求参数提供了精确的类型定义：

```typescript
import OpenAI from "openai";

// 完整的 chat.completions.create 参数类型
type ChatCompletionCreateParams =
  OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming;

// 消息参数类型（联合类型）
type MessageParam = OpenAI.Chat.Completions.ChatCompletionMessageParam;
// 等价于：
// | ChatCompletionSystemMessageParam
// | ChatCompletionUserMessageParam
// | ChatCompletionAssistantMessageParam
// | ChatCompletionToolMessageParam
// | ChatCompletionFunctionMessageParam

// 构建类型安全的请求参数
function buildChatRequest(
  userMessage: string,
  options: Partial<{
    model: string;
    temperature: number;
    maxTokens: number;
    systemPrompt: string;
  }> = {}
): ChatCompletionCreateParams {
  const {
    model = "gpt-4o",
    temperature = 0.7,
    maxTokens = 1024,
    systemPrompt,
  } = options;

  const messages: MessageParam[] = [];

  if (systemPrompt) {
    messages.push({ role: "system", content: systemPrompt });
  }
  messages.push({ role: "user", content: userMessage });

  return {
    model,
    messages,
    temperature,
    max_tokens: maxTokens,
  };
}
```

### 19.4.2 Anthropic 请求参数类型

```typescript
import Anthropic from "@anthropic-ai/sdk";

// Messages API 请求参数类型
type MessageCreateParams = Anthropic.MessageCreateParamsNonStreaming;

// 消息参数类型
type AnthropicMessageParam = Anthropic.MessageParam;
// role: "user" | "assistant"
// content: string | ContentBlockParam[]

// 内容块类型
type ContentBlockParam =
  | Anthropic.TextBlockParam      // { type: "text"; text: string }
  | Anthropic.ImageBlockParam     // { type: "image"; source: ... }
  | Anthropic.ToolUseBlockParam   // { type: "tool_use"; ... }
  | Anthropic.ToolResultBlockParam; // { type: "tool_result"; ... }

// 构建带图像的消息（多模态）
function buildVisionRequest(
  imageBase64: string,
  textPrompt: string
): MessageCreateParams {
  return {
    model: "claude-sonnet-4-6-20251101",
    max_tokens: 1024,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/jpeg",
              data: imageBase64,
            },
          },
          {
            type: "text",
            text: textPrompt,
          },
        ],
      },
    ],
  };
}
```

### 19.4.3 响应类型的类型守卫

处理响应时，需要用类型守卫区分不同的内容块类型：

```typescript
import Anthropic from "@anthropic-ai/sdk";

// 类型守卫函数
function isTextBlock(
  block: Anthropic.ContentBlock
): block is Anthropic.TextBlock {
  return block.type === "text";
}

function isToolUseBlock(
  block: Anthropic.ContentBlock
): block is Anthropic.ToolUseBlock {
  return block.type === "tool_use";
}

// 从响应中提取不同类型的内容
function parseAnthropicResponse(message: Anthropic.Message): {
  texts: string[];
  toolCalls: Array<{ name: string; input: Record<string, unknown> }>;
} {
  const texts: string[] = [];
  const toolCalls: Array<{ name: string; input: Record<string, unknown> }> = [];

  for (const block of message.content) {
    if (isTextBlock(block)) {
      texts.push(block.text);
    } else if (isToolUseBlock(block)) {
      toolCalls.push({
        name: block.name,
        input: block.input as Record<string, unknown>,
      });
    }
  }

  return { texts, toolCalls };
}
```

### 19.4.4 使用工具类型裁剪参数

在不同场景下，我们经常只需要请求参数的子集：

```typescript
import OpenAI from "openai";

type FullParams = OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming;

// 只暴露常用参数的简化接口
type SimpleChatParams = Pick<
  FullParams,
  "model" | "messages" | "temperature" | "max_tokens"
>;

// 创建预设模型的工厂函数
function createModelCaller(
  model: string,
  defaults: Omit<SimpleChatParams, "model" | "messages">
) {
  const openai = new OpenAI();

  return async function call(
    messages: FullParams["messages"],
    overrides: Partial<Omit<SimpleChatParams, "model" | "messages">> = {}
  ): Promise<string> {
    const completion = await openai.chat.completions.create({
      model,
      messages,
      ...defaults,
      ...overrides,
    });
    return completion.choices[0].message.content ?? "";
  };
}

// 预配置不同场景的调用器
const creativeCaller = createModelCaller("gpt-4o", {
  temperature: 1.2,
  max_tokens: 2048,
});

const preciseCaller = createModelCaller("gpt-4o", {
  temperature: 0.1,
  max_tokens: 512,
});

// 使用
const story = await creativeCaller([
  { role: "user", content: "写一个关于 AI 的短篇故事" },
]);

const code = await preciseCaller([
  { role: "user", content: "用 TypeScript 写一个二分查找函数" },
]);
```

---

## 19.5 错误处理与重试策略

### 19.5.1 OpenAI 错误类型

OpenAI SDK 定义了专用的错误类，可以精确捕获不同类型的错误：

```typescript
import OpenAI from "openai";

const openai = new OpenAI();

async function callWithErrorHandling(prompt: string): Promise<string> {
  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: prompt }],
    });
    return completion.choices[0].message.content ?? "";
  } catch (error) {
    if (error instanceof OpenAI.APIError) {
      // 所有 API 错误的基类
      console.error(`API 错误 [${error.status}]: ${error.message}`);

      switch (error.status) {
        case 400:
          throw new Error(`请求参数无效: ${error.message}`);
        case 401:
          throw new Error("API 密钥无效或已过期，请检查 OPENAI_API_KEY");
        case 403:
          throw new Error("没有权限访问该资源");
        case 404:
          throw new Error(`资源不存在: ${error.message}`);
        case 422:
          throw new Error(`请求格式错误: ${error.message}`);
        case 429:
          // 速率限制或配额耗尽
          throw new RateLimitError(error.message, error.headers);
        case 500:
        case 502:
        case 503:
          throw new ServiceUnavailableError(error.message);
        default:
          throw error;
      }
    }

    if (error instanceof OpenAI.APIConnectionError) {
      throw new Error(`网络连接失败: ${error.message}`);
    }

    if (error instanceof OpenAI.APIConnectionTimeoutError) {
      throw new Error("请求超时，请检查网络或增大 timeout 配置");
    }

    throw error;
  }
}

class RateLimitError extends Error {
  constructor(
    message: string,
    public readonly headers: Record<string, string> | null | undefined
  ) {
    super(`速率限制: ${message}`);
    this.name = "RateLimitError";
  }
}

class ServiceUnavailableError extends Error {
  constructor(message: string) {
    super(`服务不可用: ${message}`);
    this.name = "ServiceUnavailableError";
  }
}
```

### 19.5.2 Anthropic 错误类型

```typescript
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic();

async function callAnthropicWithErrorHandling(
  prompt: string
): Promise<string> {
  try {
    const message = await anthropic.messages.create({
      model: "claude-sonnet-4-6-20251101",
      max_tokens: 1024,
      messages: [{ role: "user", content: prompt }],
    });

    const textBlock = message.content.find(
      (b): b is Anthropic.TextBlock => b.type === "text"
    );
    return textBlock?.text ?? "";
  } catch (error) {
    if (error instanceof Anthropic.APIError) {
      switch (error.status) {
        case 400:
          throw new Error(`请求无效: ${error.message}`);
        case 401:
          throw new Error("Anthropic API 密钥无效");
        case 403:
          throw new Error(`权限被拒绝: ${error.message}`);
        case 404:
          throw new Error(`资源未找到: ${error.message}`);
        case 429:
          throw new Error(`超出速率限制: ${error.message}`);
        case 529:
          // Anthropic 过载错误
          throw new Error("Anthropic 服务过载，请稍后重试");
        case 500:
          throw new Error(`Anthropic 内部错误: ${error.message}`);
        default:
          throw error;
      }
    }
    throw error;
  }
}
```

### 19.5.3 指数退避重试策略

生产环境中需要对可重试错误（如速率限制、服务暂时不可用）实施自动重试：

```typescript
// src/utils/retry.ts

interface RetryOptions {
  /** 最大重试次数 */
  maxRetries: number;
  /** 初始等待时间（毫秒） */
  initialDelayMs: number;
  /** 最大等待时间（毫秒） */
  maxDelayMs: number;
  /** 指数退避基数 */
  backoffFactor: number;
  /** 判断错误是否可重试 */
  isRetryable: (error: unknown) => boolean;
  /** 重试回调（用于日志） */
  onRetry?: (attempt: number, error: unknown, delayMs: number) => void;
}

const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  maxRetries: 3,
  initialDelayMs: 1000,
  maxDelayMs: 60_000,
  backoffFactor: 2,
  isRetryable: isRetryableError,
  onRetry: (attempt, error, delayMs) => {
    console.warn(
      `[Retry] 第 ${attempt} 次重试，等待 ${delayMs}ms... ` +
        `错误: ${error instanceof Error ? error.message : String(error)}`
    );
  },
};

function isRetryableError(error: unknown): boolean {
  if (error instanceof Error) {
    const msg = error.message.toLowerCase();
    // 可重试的错误特征
    if (msg.includes("rate limit")) return true;
    if (msg.includes("timeout")) return true;
    if (msg.includes("connection")) return true;
    if (msg.includes("503") || msg.includes("529")) return true;
  }

  // 检查 HTTP 状态码
  const statusCode = (error as { status?: number })?.status;
  if (statusCode) {
    return [429, 500, 502, 503, 504, 529].includes(statusCode);
  }

  return false;
}

function calculateDelay(
  attempt: number,
  options: RetryOptions
): number {
  // 指数退避 + 抖动（Jitter），避免"雷群效应"
  const exponentialDelay =
    options.initialDelayMs * Math.pow(options.backoffFactor, attempt - 1);
  const jitter = Math.random() * 0.1 * exponentialDelay; // 10% 随机抖动
  return Math.min(exponentialDelay + jitter, options.maxDelayMs);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * 带指数退避重试的异步函数包装器
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options: Partial<RetryOptions> = {}
): Promise<T> {
  const opts = { ...DEFAULT_RETRY_OPTIONS, ...options };
  let lastError: unknown;

  for (let attempt = 1; attempt <= opts.maxRetries + 1; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      const isLastAttempt = attempt > opts.maxRetries;
      const canRetry = opts.isRetryable(error);

      if (isLastAttempt || !canRetry) {
        throw error;
      }

      const delayMs = calculateDelay(attempt, opts);
      opts.onRetry?.(attempt, error, delayMs);
      await sleep(delayMs);
    }
  }

  throw lastError;
}
```

### 19.5.4 使用重试包装器

```typescript
import OpenAI from "openai";
import { withRetry } from "./utils/retry";

const openai = new OpenAI({ maxRetries: 0 }); // 禁用 SDK 内置重试，使用自定义逻辑

async function robustCompletion(prompt: string): Promise<string> {
  return withRetry(
    async () => {
      const completion = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: prompt }],
        max_tokens: 1024,
      });
      return completion.choices[0].message.content ?? "";
    },
    {
      maxRetries: 3,
      initialDelayMs: 2000,
      maxDelayMs: 30_000,
      backoffFactor: 2,
      onRetry: (attempt, error, delayMs) => {
        console.warn(`[OpenAI] 重试 ${attempt}/3，${delayMs}ms 后重试...`);
      },
    }
  );
}

// 批量处理时的并发控制与重试
async function processBatch(
  prompts: string[],
  concurrency: number = 5
): Promise<string[]> {
  const results: string[] = [];

  for (let i = 0; i < prompts.length; i += concurrency) {
    const batch = prompts.slice(i, i + concurrency);
    const batchResults = await Promise.all(
      batch.map((p) => robustCompletion(p))
    );
    results.push(...batchResults);
    console.log(`已处理 ${Math.min(i + concurrency, prompts.length)}/${prompts.length}`);
  }

  return results;
}
```

---

## 本章小结

| 知识点 | 核心 API / 类型 | 关键注意事项 |
|--------|----------------|-------------|
| OpenAI 初始化 | `new OpenAI({ apiKey, timeout, maxRetries })` | 优先从环境变量读取密钥 |
| OpenAI 文本生成 | `chat.completions.create()` → `ChatCompletion` | `content` 可能为 `null`，需处理 |
| OpenAI 流式输出 | `stream: true` → `for await (chunk of stream)` | 通过 `delta.content` 拼接增量文本 |
| OpenAI 函数调用 | `tools` + `tool_choice` → 检查 `tool_calls` | 需要两轮请求：决策 + 执行 |
| Anthropic 初始化 | `new Anthropic({ apiKey, timeout })` | `system` 是顶层字段，非 messages 成员 |
| Anthropic 消息 | `messages.create()` → `Message` | `content` 是 `ContentBlock[]`，需类型守卫 |
| Anthropic 流式 | `messages.stream()` → `.on("text", cb)` | `finalMessage()` 获取 usage 统计 |
| 密钥管理 | `dotenv` + `.env` + `.gitignore` | 生产环境用平台密钥管理服务 |
| 请求参数类型 | `ChatCompletionCreateParams` / `MessageCreateParams` | 用 `Pick`/`Omit` 裁剪暴露给上层的接口 |
| 错误分类 | `OpenAI.APIError` / `Anthropic.APIError` | 按 `status` 码区分可重试与不可重试错误 |
| 重试策略 | 指数退避 + Jitter | 禁用 SDK 内置重试，使用统一的自定义重试逻辑 |

---

## AI 应用实战：多模型统一调用接口

本节实现一个支持 OpenAI 和 Anthropic 的**统一 LLM 调用接口**，屏蔽底层 SDK 差异，让业务代码不依赖具体的 LLM 提供商。

### 设计目标

1. 统一的 `LLMProvider` 接口，支持普通调用与流式输出
2. 类型安全的请求参数，不同提供商的差异由适配器层处理
3. 内置重试与错误标准化
4. 支持运行时动态切换提供商

```typescript
// src/llm/types.ts

/**
 * 统一的消息格式
 */
export interface UnifiedMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/**
 * 统一的请求参数
 */
export interface LLMRequest {
  messages: UnifiedMessage[];
  model?: string;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
}

/**
 * 统一的响应格式
 */
export interface LLMResponse {
  content: string;
  model: string;
  usage: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
  provider: "openai" | "anthropic";
}

/**
 * 流式响应的事件
 */
export type StreamEvent =
  | { type: "delta"; content: string }
  | { type: "done"; response: LLMResponse };

/**
 * LLM 提供商接口
 */
export interface LLMProvider {
  readonly name: "openai" | "anthropic";
  readonly defaultModel: string;

  complete(request: LLMRequest): Promise<LLMResponse>;
  stream(request: LLMRequest): AsyncIterable<StreamEvent>;
}
```

```typescript
// src/llm/openai-provider.ts

import OpenAI from "openai";
import type { LLMProvider, LLMRequest, LLMResponse, StreamEvent, UnifiedMessage } from "./types";
import { withRetry } from "../utils/retry";

export class OpenAIProvider implements LLMProvider {
  readonly name = "openai" as const;
  readonly defaultModel = "gpt-4o";

  private client: OpenAI;

  constructor(apiKey?: string) {
    this.client = new OpenAI({
      apiKey: apiKey ?? process.env.OPENAI_API_KEY,
      maxRetries: 0, // 由 withRetry 统一控制
    });
  }

  private toOpenAIMessages(
    messages: UnifiedMessage[]
  ): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
    return messages.map((msg) => {
      if (msg.role === "system") {
        return { role: "system", content: msg.content };
      }
      if (msg.role === "user") {
        return { role: "user", content: msg.content };
      }
      return { role: "assistant", content: msg.content };
    });
  }

  async complete(request: LLMRequest): Promise<LLMResponse> {
    return withRetry(async () => {
      const completion = await this.client.chat.completions.create({
        model: request.model ?? this.defaultModel,
        messages: this.toOpenAIMessages(request.messages),
        temperature: request.temperature ?? 0.7,
        max_tokens: request.maxTokens ?? 2048,
      });

      const content = completion.choices[0].message.content ?? "";
      const usage = completion.usage;

      return {
        content,
        model: completion.model,
        usage: {
          inputTokens: usage?.prompt_tokens ?? 0,
          outputTokens: usage?.completion_tokens ?? 0,
          totalTokens: usage?.total_tokens ?? 0,
        },
        provider: "openai",
      };
    });
  }

  async *stream(request: LLMRequest): AsyncIterable<StreamEvent> {
    const streamResponse = await this.client.chat.completions.create({
      model: request.model ?? this.defaultModel,
      messages: this.toOpenAIMessages(request.messages),
      temperature: request.temperature ?? 0.7,
      max_tokens: request.maxTokens ?? 2048,
      stream: true,
    });

    let fullContent = "";

    for await (const chunk of streamResponse) {
      const delta = chunk.choices[0]?.delta?.content;
      if (delta) {
        fullContent += delta;
        yield { type: "delta", content: delta };
      }
    }

    yield {
      type: "done",
      response: {
        content: fullContent,
        model: request.model ?? this.defaultModel,
        usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
        provider: "openai",
      },
    };
  }
}
```

```typescript
// src/llm/anthropic-provider.ts

import Anthropic from "@anthropic-ai/sdk";
import type { LLMProvider, LLMRequest, LLMResponse, StreamEvent, UnifiedMessage } from "./types";
import { withRetry } from "../utils/retry";

export class AnthropicProvider implements LLMProvider {
  readonly name = "anthropic" as const;
  readonly defaultModel = "claude-sonnet-4-6-20251101";

  private client: Anthropic;

  constructor(apiKey?: string) {
    this.client = new Anthropic({
      apiKey: apiKey ?? process.env.ANTHROPIC_API_KEY,
      maxRetries: 0,
    });
  }

  private extractSystemAndMessages(messages: UnifiedMessage[]): {
    system: string | undefined;
    userMessages: Anthropic.MessageParam[];
  } {
    // Anthropic 的 system 是顶层参数，需要从 messages 中分离
    const systemMessage = messages.find((m) => m.role === "system");
    const nonSystemMessages = messages.filter((m) => m.role !== "system");

    const userMessages: Anthropic.MessageParam[] = nonSystemMessages.map(
      (msg) => ({
        role: msg.role as "user" | "assistant",
        content: msg.content,
      })
    );

    return {
      system: systemMessage?.content,
      userMessages,
    };
  }

  async complete(request: LLMRequest): Promise<LLMResponse> {
    return withRetry(async () => {
      const { system, userMessages } = this.extractSystemAndMessages(
        request.messages
      );

      const message = await this.client.messages.create({
        model: request.model ?? this.defaultModel,
        max_tokens: request.maxTokens ?? 2048,
        temperature: request.temperature ?? 0.7,
        ...(system ? { system } : {}),
        messages: userMessages,
      });

      const textBlocks = message.content.filter(
        (b): b is Anthropic.TextBlock => b.type === "text"
      );
      const content = textBlocks.map((b) => b.text).join("");

      return {
        content,
        model: message.model,
        usage: {
          inputTokens: message.usage.input_tokens,
          outputTokens: message.usage.output_tokens,
          totalTokens: message.usage.input_tokens + message.usage.output_tokens,
        },
        provider: "anthropic",
      };
    });
  }

  async *stream(request: LLMRequest): AsyncIterable<StreamEvent> {
    const { system, userMessages } = this.extractSystemAndMessages(
      request.messages
    );

    let fullContent = "";
    let inputTokens = 0;
    let outputTokens = 0;
    let finalModel = request.model ?? this.defaultModel;

    const anthropicStream = this.client.messages.stream({
      model: request.model ?? this.defaultModel,
      max_tokens: request.maxTokens ?? 2048,
      temperature: request.temperature ?? 0.7,
      ...(system ? { system } : {}),
      messages: userMessages,
    });

    for await (const event of anthropicStream) {
      if (
        event.type === "content_block_delta" &&
        event.delta.type === "text_delta"
      ) {
        fullContent += event.delta.text;
        yield { type: "delta", content: event.delta.text };
      }
      if (event.type === "message_delta" && event.usage) {
        outputTokens = event.usage.output_tokens;
      }
      if (event.type === "message_start") {
        inputTokens = event.message.usage.input_tokens;
        finalModel = event.message.model;
      }
    }

    yield {
      type: "done",
      response: {
        content: fullContent,
        model: finalModel,
        usage: {
          inputTokens,
          outputTokens,
          totalTokens: inputTokens + outputTokens,
        },
        provider: "anthropic",
      },
    };
  }
}
```

```typescript
// src/llm/llm-client.ts

import type { LLMProvider, LLMRequest, LLMResponse, StreamEvent } from "./types";
import { OpenAIProvider } from "./openai-provider";
import { AnthropicProvider } from "./anthropic-provider";

type ProviderName = "openai" | "anthropic";

/**
 * 统一 LLM 客户端 —— 门面模式（Facade Pattern）
 * 业务代码只需依赖此类，不感知底层 SDK 差异
 */
export class LLMClient {
  private providers: Map<ProviderName, LLMProvider>;
  private activeProvider: ProviderName;

  constructor(
    defaultProvider: ProviderName = "openai",
    config: {
      openaiApiKey?: string;
      anthropicApiKey?: string;
    } = {}
  ) {
    this.providers = new Map([
      ["openai", new OpenAIProvider(config.openaiApiKey)],
      ["anthropic", new AnthropicProvider(config.anthropicApiKey)],
    ]);
    this.activeProvider = defaultProvider;
  }

  /** 动态切换提供商 */
  useProvider(provider: ProviderName): this {
    if (!this.providers.has(provider)) {
      throw new Error(`未知提供商: ${provider}`);
    }
    this.activeProvider = provider;
    return this;
  }

  private getProvider(): LLMProvider {
    const provider = this.providers.get(this.activeProvider);
    if (!provider) throw new Error(`提供商未初始化: ${this.activeProvider}`);
    return provider;
  }

  /** 普通调用 */
  async complete(request: LLMRequest): Promise<LLMResponse> {
    const provider = this.getProvider();
    const startTime = Date.now();

    try {
      const response = await provider.complete(request);
      const elapsed = Date.now() - startTime;
      console.debug(
        `[LLM] ${provider.name}/${response.model} ` +
          `${response.usage.totalTokens} tokens ${elapsed}ms`
      );
      return response;
    } catch (error) {
      console.error(`[LLM] ${provider.name} 调用失败:`, error);
      throw error;
    }
  }

  /** 流式调用 */
  async *stream(request: LLMRequest): AsyncIterable<StreamEvent> {
    const provider = this.getProvider();
    yield* provider.stream(request);
  }

  /**
   * 简化的问答方法
   */
  async ask(
    question: string,
    options: Partial<LLMRequest> = {}
  ): Promise<string> {
    const response = await this.complete({
      messages: [{ role: "user", content: question }],
      ...options,
    });
    return response.content;
  }

  /**
   * A/B 测试：同时调用两个提供商，比较结果
   */
  async compare(
    request: LLMRequest
  ): Promise<{ openai: LLMResponse; anthropic: LLMResponse }> {
    const [openaiResult, anthropicResult] = await Promise.all([
      this.providers.get("openai")!.complete(request),
      this.providers.get("anthropic")!.complete(request),
    ]);
    return { openai: openaiResult, anthropic: anthropicResult };
  }
}
```

```typescript
// src/main.ts —— 使用示例

import * as dotenv from "dotenv";
dotenv.config();

import { LLMClient } from "./llm/llm-client";

async function main() {
  const llm = new LLMClient("anthropic");

  // --- 普通调用 ---
  console.log("=== 普通调用（Anthropic）===");
  const answer = await llm.ask("用一句话解释什么是依赖注入");
  console.log(answer);

  // --- 切换到 OpenAI ---
  console.log("\n=== 普通调用（OpenAI）===");
  llm.useProvider("openai");
  const answer2 = await llm.ask("用一句话解释什么是依赖注入");
  console.log(answer2);

  // --- 流式输出 ---
  console.log("\n=== 流式输出 ===");
  llm.useProvider("anthropic");
  const stream = llm.stream({
    messages: [
      { role: "system", content: "你是一个 TypeScript 专家。" },
      { role: "user", content: "解释 TypeScript 中的协变与逆变" },
    ],
    maxTokens: 512,
  });

  for await (const event of stream) {
    if (event.type === "delta") {
      process.stdout.write(event.content);
    } else if (event.type === "done") {
      console.log("\n");
      console.log("用量:", event.response.usage);
    }
  }

  // --- A/B 比较 ---
  console.log("\n=== A/B 比较两个提供商 ===");
  const comparison = await llm.compare({
    messages: [{ role: "user", content: "TypeScript 最大的优势是什么？" }],
    maxTokens: 200,
  });
  console.log("OpenAI:", comparison.openai.content);
  console.log("Anthropic:", comparison.anthropic.content);
}

main().catch(console.error);
```

---

## 练习题

### 基础题

**练习 1**：参数类型约束

`LLMRequest` 中的 `temperature` 取值范围是 `0.0` 到 `2.0`。请定义一个品牌类型（Branded Type）`Temperature`，让编译器在接受非法值时报错，并编写一个工厂函数 `createTemperature(value: number): Temperature`。

**练习 2**：流式响应收集器

基于 `LLMClient.stream()` 方法，实现一个 `collectStream` 工具函数，签名如下：

```typescript
async function collectStream(
  iterable: AsyncIterable<StreamEvent>
): Promise<{ text: string; response: LLMResponse }>
```

该函数应在控制台实时打印增量内容，并在流结束后返回完整文本和最终响应对象。

### 进阶题

**练习 3**：带回退的提供商链

实现一个 `withFallback(primary: LLMProvider, fallback: LLMProvider)` 函数，返回一个新的 `LLMProvider`，行为如下：
- 优先调用 `primary`
- 若 `primary` 抛出错误（任何类型），自动切换到 `fallback` 重试
- 在回退时打印警告日志，包含原始错误信息

**练习 4**：Token 用量追踪器

设计一个 `UsageTracker` 类，包装任意 `LLMProvider`，在每次调用后累计记录 token 用量。要求：
- 按模型名称分组统计 `inputTokens`、`outputTokens`、`totalTokens`
- 提供 `getReport(): Record<string, UsageStat>` 方法输出报告
- 提供 `reset()` 方法清空统计

### 挑战题

**练习 5**：带缓存的 LLM 客户端

为 `LLMClient.complete()` 实现语义缓存层：
- 对**完全相同**的请求（`messages` + `model` + `temperature` + `maxTokens`），返回缓存结果，不发起网络请求
- 缓存键使用 `JSON.stringify` + SHA-256 哈希（可使用 Node.js 内置 `crypto` 模块）
- 支持 TTL（存活时间），超时后缓存条目自动失效
- 接口签名：`new CachedLLMClient(provider: LLMProvider, options: { ttlMs: number })`
- 提供 `getCacheStats(): { hits: number; misses: number; size: number }` 方法

---

## 练习答案

### 答案 1：品牌类型 Temperature

```typescript
// 品牌类型：通过交叉类型添加不可见标记
type Temperature = number & { readonly __brand: "Temperature" };

function createTemperature(value: number): Temperature {
  if (value < 0 || value > 2) {
    throw new RangeError(
      `Temperature 必须在 [0, 2] 范围内，收到: ${value}`
    );
  }
  return value as Temperature;
}

// 更新 LLMRequest 使用品牌类型
interface LLMRequestStrict {
  messages: UnifiedMessage[];
  model?: string;
  temperature?: Temperature;
  maxTokens?: number;
}

// 使用示例
const t1 = createTemperature(0.7); // 合法
const t2 = createTemperature(2.5); // 抛出 RangeError

const request: LLMRequestStrict = {
  messages: [{ role: "user", content: "hello" }],
  temperature: t1,          // 类型检查通过
  // temperature: 0.7,      // 编译错误：number 不兼容 Temperature
};
```

### 答案 2：流式响应收集器

```typescript
import type { LLMResponse, StreamEvent } from "./llm/types";

async function collectStream(
  iterable: AsyncIterable<StreamEvent>
): Promise<{ text: string; response: LLMResponse }> {
  const parts: string[] = [];
  let finalResponse: LLMResponse | undefined;

  for await (const event of iterable) {
    if (event.type === "delta") {
      parts.push(event.content);
      process.stdout.write(event.content);
    } else if (event.type === "done") {
      finalResponse = event.response;
    }
  }

  console.log(); // 换行

  if (!finalResponse) {
    throw new Error("流结束但未收到 done 事件");
  }

  return {
    text: parts.join(""),
    response: finalResponse,
  };
}

// 使用示例
const llm = new LLMClient("openai");
const streamIterable = llm.stream({
  messages: [{ role: "user", content: "介绍 TypeScript 泛型" }],
});
const { text, response } = await collectStream(streamIterable);
console.log(`\n总 tokens: ${response.usage.totalTokens}`);
```

### 答案 3：带回退的提供商链

```typescript
import type { LLMProvider, LLMRequest, LLMResponse, StreamEvent } from "./llm/types";

function withFallback(
  primary: LLMProvider,
  fallback: LLMProvider
): LLMProvider {
  return {
    name: primary.name,
    defaultModel: primary.defaultModel,

    async complete(request: LLMRequest): Promise<LLMResponse> {
      try {
        return await primary.complete(request);
      } catch (primaryError) {
        console.warn(
          `[Fallback] ${primary.name} 失败，切换到 ${fallback.name}。` +
            `原始错误: ${primaryError instanceof Error ? primaryError.message : String(primaryError)}`
        );
        return fallback.complete(request);
      }
    },

    async *stream(request: LLMRequest): AsyncIterable<StreamEvent> {
      try {
        // 注意：流式场景下，需要先确认连接成功再 yield
        // 简化处理：降级为非流式调用
        const response = await primary.complete(request);
        yield { type: "delta", content: response.content };
        yield { type: "done", response };
      } catch (primaryError) {
        console.warn(
          `[Fallback] ${primary.name} 流式失败，切换到 ${fallback.name}`
        );
        yield* fallback.stream(request);
      }
    },
  };
}

// 使用示例
const openaiProvider = new OpenAIProvider();
const anthropicProvider = new AnthropicProvider();
const resilientProvider = withFallback(openaiProvider, anthropicProvider);

const result = await resilientProvider.complete({
  messages: [{ role: "user", content: "你好" }],
});
```

### 答案 4：Token 用量追踪器

```typescript
interface UsageStat {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  callCount: number;
}

class UsageTracker implements LLMProvider {
  readonly name: LLMProvider["name"];
  readonly defaultModel: string;

  private inner: LLMProvider;
  private stats: Map<string, UsageStat> = new Map();

  constructor(provider: LLMProvider) {
    this.inner = provider;
    this.name = provider.name;
    this.defaultModel = provider.defaultModel;
  }

  private recordUsage(model: string, usage: LLMResponse["usage"]): void {
    const existing = this.stats.get(model) ?? {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      callCount: 0,
    };
    this.stats.set(model, {
      inputTokens: existing.inputTokens + usage.inputTokens,
      outputTokens: existing.outputTokens + usage.outputTokens,
      totalTokens: existing.totalTokens + usage.totalTokens,
      callCount: existing.callCount + 1,
    });
  }

  async complete(request: LLMRequest): Promise<LLMResponse> {
    const response = await this.inner.complete(request);
    this.recordUsage(response.model, response.usage);
    return response;
  }

  async *stream(request: LLMRequest): AsyncIterable<StreamEvent> {
    for await (const event of this.inner.stream(request)) {
      if (event.type === "done") {
        this.recordUsage(event.response.model, event.response.usage);
      }
      yield event;
    }
  }

  getReport(): Record<string, UsageStat> {
    return Object.fromEntries(this.stats.entries());
  }

  reset(): void {
    this.stats.clear();
  }
}

// 使用示例
const tracker = new UsageTracker(new OpenAIProvider());
await tracker.complete({ messages: [{ role: "user", content: "你好" }] });
await tracker.complete({ messages: [{ role: "user", content: "再见" }] });
console.log(tracker.getReport());
// { "gpt-4o": { inputTokens: 42, outputTokens: 30, totalTokens: 72, callCount: 2 } }
```

### 答案 5：带缓存的 LLM 客户端

```typescript
import * as crypto from "crypto";
import type { LLMProvider, LLMRequest, LLMResponse, StreamEvent } from "./llm/types";

interface CacheEntry {
  response: LLMResponse;
  expiresAt: number;
}

interface CacheStats {
  hits: number;
  misses: number;
  size: number;
}

class CachedLLMClient implements LLMProvider {
  readonly name: LLMProvider["name"];
  readonly defaultModel: string;

  private inner: LLMProvider;
  private cache: Map<string, CacheEntry> = new Map();
  private ttlMs: number;
  private hits = 0;
  private misses = 0;

  constructor(provider: LLMProvider, options: { ttlMs: number }) {
    this.inner = provider;
    this.name = provider.name;
    this.defaultModel = provider.defaultModel;
    this.ttlMs = options.ttlMs;
  }

  private computeCacheKey(request: LLMRequest): string {
    const normalized = {
      messages: request.messages,
      model: request.model ?? this.defaultModel,
      temperature: request.temperature ?? 0.7,
      maxTokens: request.maxTokens ?? 2048,
    };
    const json = JSON.stringify(normalized);
    return crypto.createHash("sha256").update(json).digest("hex");
  }

  async complete(request: LLMRequest): Promise<LLMResponse> {
    const key = this.computeCacheKey(request);
    const now = Date.now();

    // 检查缓存
    const cached = this.cache.get(key);
    if (cached && cached.expiresAt > now) {
      this.hits++;
      console.debug(`[Cache] HIT  ${key.substring(0, 8)}...`);
      return cached.response;
    }

    // 缓存未命中：调用提供商
    this.misses++;
    console.debug(`[Cache] MISS ${key.substring(0, 8)}...`);
    const response = await this.inner.complete(request);

    this.cache.set(key, {
      response,
      expiresAt: now + this.ttlMs,
    });

    return response;
  }

  // 流式调用不缓存（结果是动态的）
  stream(request: LLMRequest): AsyncIterable<StreamEvent> {
    return this.inner.stream(request);
  }

  getCacheStats(): CacheStats {
    return {
      hits: this.hits,
      misses: this.misses,
      size: this.cache.size,
    };
  }

  clearCache(): void {
    this.cache.clear();
  }
}

// 使用示例
const cachedClient = new CachedLLMClient(new OpenAIProvider(), {
  ttlMs: 5 * 60 * 1000, // 缓存 5 分钟
});

const r1 = await cachedClient.complete({
  messages: [{ role: "user", content: "TypeScript 是什么？" }],
});
// [Cache] MISS ...

const r2 = await cachedClient.complete({
  messages: [{ role: "user", content: "TypeScript 是什么？" }],
});
// [Cache] HIT  ...（直接返回缓存，不发网络请求）

console.log(cachedClient.getCacheStats());
// { hits: 1, misses: 1, size: 1 }
```
