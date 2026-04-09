# 附录B：AI API参考

本附录提供 OpenAI 和 Anthropic 两大主流 AI API 的快速参考，包括请求参数、响应格式、类型定义及错误码说明，适合在开发过程中查阅使用。

---

## 目录

- [1. OpenAI API](#1-openai-api)
- [2. Anthropic API](#2-anthropic-api)
- [3. 通用类型定义](#3-通用类型定义)
- [4. 错误码参考](#4-错误码参考)
- [5. 速率限制与配额](#5-速率限制与配额)

---

## 1. OpenAI API

### 1.1 Chat Completions API

**端点**：`POST https://api.openai.com/v1/chat/completions`

**认证方式**：Bearer Token（HTTP 请求头 `Authorization: Bearer <API_KEY>`）

**最低请求示例**：

```typescript
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const response = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [
    { role: "system", content: "你是一个有帮助的助手。" },
    { role: "user", content: "你好！" },
  ],
});

console.log(response.choices[0].message.content);
```

---

### 1.2 请求参数说明

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `model` | `string` | 是 | 模型 ID，如 `gpt-4o`、`gpt-4o-mini`、`gpt-3.5-turbo` |
| `messages` | `Message[]` | 是 | 对话历史消息数组，详见 [Message类型](#31-message-类型) |
| `max_tokens` | `number` | 否 | 生成的最大 token 数量（新参数名：`max_completion_tokens`）|
| `temperature` | `number` | 否 | 采样温度，范围 `0.0`–`2.0`，默认 `1.0`；越高越随机 |
| `top_p` | `number` | 否 | 核采样概率，范围 `0.0`–`1.0`，默认 `1.0` |
| `n` | `number` | 否 | 每次请求生成的候选回复数量，默认 `1` |
| `stream` | `boolean` | 否 | 是否启用流式响应，默认 `false` |
| `stop` | `string \| string[]` | 否 | 停止生成的标记字符串（最多 4 个） |
| `presence_penalty` | `number` | 否 | 存在惩罚，范围 `-2.0`–`2.0`，鼓励引入新话题 |
| `frequency_penalty` | `number` | 否 | 频率惩罚，范围 `-2.0`–`2.0`，降低重复词频率 |
| `logit_bias` | `Record<string, number>` | 否 | 调整特定 token 的生成概率 |
| `user` | `string` | 否 | 终端用户标识符，用于滥用检测 |
| `tools` | `Tool[]` | 否 | 可调用工具的列表，详见 [Function Calling参数](#15-function-calling-参数) |
| `tool_choice` | `string \| object` | 否 | 工具调用策略：`"none"`、`"auto"`、`"required"` 或指定工具 |
| `response_format` | `object` | 否 | 响应格式，如 `{ type: "json_object" }` 强制 JSON 输出 |
| `seed` | `number` | 否 | 随机种子，用于可重复的结果 |

---

### 1.3 响应格式

```typescript
interface ChatCompletion {
  id: string;                    // 请求唯一标识，如 "chatcmpl-abc123"
  object: "chat.completion";
  created: number;               // Unix 时间戳（秒）
  model: string;                 // 实际使用的模型
  choices: Choice[];
  usage: Usage;
  system_fingerprint?: string;   // 系统配置指纹
}

interface Choice {
  index: number;
  message: AssistantMessage;
  finish_reason:
    | "stop"          // 正常结束（达到停止词或消息末尾）
    | "length"        // 达到 max_tokens 限制
    | "tool_calls"    // 模型调用了工具
    | "content_filter" // 内容被安全过滤器拦截
    | null;
  logprobs?: LogProbs | null;
}

interface AssistantMessage {
  role: "assistant";
  content: string | null;        // 文本内容（调用工具时可能为 null）
  tool_calls?: ToolCall[];       // 工具调用列表
  refusal?: string | null;       // 模型拒绝响应时的提示
}

interface Usage {
  prompt_tokens: number;         // 输入消耗的 token 数
  completion_tokens: number;     // 输出消耗的 token 数
  total_tokens: number;          // 总 token 数
  completion_tokens_details?: {
    reasoning_tokens: number;    // 推理消耗的 token（o1 系列模型）
  };
}
```

**响应示例**：

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1720000000,
  "model": "gpt-4o-2024-08-06",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！有什么我可以帮助你的吗？"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 12,
    "total_tokens": 32
  }
}
```

---

### 1.4 流式响应格式

启用 `stream: true` 后，服务器通过 Server-Sent Events（SSE）返回数据块。

```typescript
const stream = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "写一首诗" }],
  stream: true,
});

for await (const chunk of stream) {
  const delta = chunk.choices[0]?.delta;
  if (delta?.content) {
    process.stdout.write(delta.content);
  }
}
```

**流式数据块格式（`ChatCompletionChunk`）**：

```typescript
interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: ChunkChoice[];
  usage?: Usage | null;          // 仅最后一个块包含（需开启 stream_options）
}

interface ChunkChoice {
  index: number;
  delta: {
    role?: "assistant";          // 仅第一个块包含
    content?: string | null;     // 本次增量文本内容
    tool_calls?: ToolCallChunk[];
  };
  finish_reason: string | null;  // 最后一块不为 null
}
```

**原始 SSE 数据流示例**：

```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":""},"index":0}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":"你"},"index":0}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":"好"},"index":0}]}

data: [DONE]
```

---

### 1.5 Function Calling 参数

```typescript
interface Tool {
  type: "function";
  function: {
    name: string;              // 函数名称，只能包含字母、数字、下划线和连字符
    description?: string;      // 函数功能描述，帮助模型决定何时调用
    parameters?: {             // JSON Schema 格式的参数定义
      type: "object";
      properties: Record<string, {
        type: string;          // "string" | "number" | "boolean" | "array" | "object"
        description?: string;
        enum?: string[];       // 枚举值限制
        items?: object;        // 当 type 为 "array" 时定义数组元素类型
      }>;
      required?: string[];     // 必填参数名称列表
      additionalProperties?: boolean;
    };
    strict?: boolean;          // 启用严格模式，确保 JSON Schema 完全遵循
  };
}

// 工具调用结果（模型发起调用后，需要将结果回传）
interface ToolResultMessage {
  role: "tool";
  tool_call_id: string;        // 对应 ToolCall 的 id
  content: string;             // 工具执行结果（字符串形式）
}
```

**完整 Function Calling 示例**：

```typescript
const tools: Tool[] = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "获取指定城市的当前天气",
      parameters: {
        type: "object",
        properties: {
          city: { type: "string", description: "城市名称" },
          unit: { type: "string", enum: ["celsius", "fahrenheit"] },
        },
        required: ["city"],
      },
    },
  },
];

const response = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "北京今天天气怎么样？" }],
  tools,
  tool_choice: "auto",
});
```

---

## 2. Anthropic API

### 2.1 Messages API

**端点**：`POST https://api.anthropic.com/v1/messages`

**认证方式**：`x-api-key` 请求头

**必须包含的请求头**：

| 请求头 | 值 |
|--------|----|
| `x-api-key` | 你的 Anthropic API Key |
| `anthropic-version` | `2023-06-01`（当前最新版本） |
| `content-type` | `application/json` |

**最低请求示例**：

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const response = await client.messages.create({
  model: "claude-opus-4-5-20251101",
  max_tokens: 1024,
  messages: [{ role: "user", content: "你好！" }],
});

console.log(response.content[0].text);
```

---

### 2.2 请求参数说明

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `model` | `string` | 是 | 模型 ID，如 `claude-opus-4-5-20251101`、`claude-sonnet-4-6` |
| `messages` | `Message[]` | 是 | 对话历史消息数组（不含 system） |
| `max_tokens` | `number` | 是 | 最大输出 token 数（**Anthropic 中为必填项**） |
| `system` | `string \| SystemBlock[]` | 否 | 系统提示，可以是字符串或结构化内容块数组 |
| `temperature` | `number` | 否 | 采样温度，范围 `0.0`–`1.0`，默认 `1.0` |
| `top_p` | `number` | 否 | 核采样概率，范围 `0.0`–`1.0` |
| `top_k` | `number` | 否 | 仅保留概率最高的 k 个 token 进行采样 |
| `stop_sequences` | `string[]` | 否 | 停止序列列表，遇到时停止生成 |
| `stream` | `boolean` | 否 | 是否启用流式响应，默认 `false` |
| `tools` | `Tool[]` | 否 | 可调用工具列表，详见 [Tool Use参数](#25-tool-use-参数) |
| `tool_choice` | `object` | 否 | 工具调用策略，`{ type: "auto" \| "any" \| "tool" }` |
| `metadata` | `object` | 否 | 额外元数据，如 `{ user_id: "..." }` |

**主要 Claude 模型列表**：

| 模型 ID | 上下文窗口 | 说明 |
|---------|-----------|------|
| `claude-opus-4-5-20251101` | 200K tokens | 最强能力，适合复杂任务 |
| `claude-sonnet-4-6` | 200K tokens | 性能与速度均衡 |
| `claude-haiku-3-5` | 200K tokens | 速度最快，适合简单任务 |

---

### 2.3 响应格式

```typescript
interface Message {
  id: string;                    // 消息唯一标识，如 "msg_abc123"
  type: "message";
  role: "assistant";
  content: ContentBlock[];       // 内容块数组（可含文本和工具调用）
  model: string;                 // 实际使用的模型
  stop_reason:
    | "end_turn"       // 正常结束
    | "max_tokens"     // 达到 max_tokens 限制
    | "stop_sequence"  // 遇到停止序列
    | "tool_use"       // 模型调用了工具
    | null;
  stop_sequence: string | null;  // 触发停止的序列（如适用）
  usage: Usage;
}

type ContentBlock = TextBlock | ToolUseBlock;

interface TextBlock {
  type: "text";
  text: string;
}

interface ToolUseBlock {
  type: "tool_use";
  id: string;          // 工具调用唯一 ID
  name: string;        // 工具名称
  input: object;       // 工具调用参数（已解析的 JSON 对象）
}

interface Usage {
  input_tokens: number;    // 输入消耗的 token 数
  output_tokens: number;   // 输出消耗的 token 数
}
```

**响应示例**：

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "你好！我是 Claude，有什么可以帮助你的吗？"
    }
  ],
  "model": "claude-opus-4-5-20251101",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 15,
    "output_tokens": 18
  }
}
```

---

### 2.4 流式响应格式

```typescript
const stream = await client.messages.stream({
  model: "claude-opus-4-5-20251101",
  max_tokens: 1024,
  messages: [{ role: "user", content: "写一首诗" }],
});

for await (const event of stream) {
  if (
    event.type === "content_block_delta" &&
    event.delta.type === "text_delta"
  ) {
    process.stdout.write(event.delta.text);
  }
}

const finalMessage = await stream.finalMessage();
console.log("\n用量：", finalMessage.usage);
```

**流式事件类型（完整列表）**：

| 事件类型 | 说明 |
|----------|------|
| `message_start` | 流开始，包含初始 Message 对象（content 为空） |
| `content_block_start` | 新内容块开始，包含 `index` 和初始块对象 |
| `content_block_delta` | 内容块增量，包含 `text_delta` 或 `input_json_delta` |
| `content_block_stop` | 内容块结束 |
| `message_delta` | 消息级别的增量，包含 `stop_reason`、`stop_sequence`、`usage` |
| `message_stop` | 流结束 |
| `ping` | 心跳保活事件 |
| `error` | 发生错误 |

**原始 SSE 数据流示例**：

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_abc","type":"message","role":"assistant","content":[],"model":"claude-opus-4-5-20251101","stop_reason":null,"usage":{"input_tokens":15,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你好"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}
```

---

### 2.5 Tool Use 参数

```typescript
interface AnthropicTool {
  name: string;              // 工具名称
  description?: string;      // 工具功能描述
  input_schema: {            // JSON Schema 格式的输入参数定义
    type: "object";
    properties: Record<string, {
      type: string;
      description?: string;
      enum?: string[];
    }>;
    required?: string[];
  };
}

// 工具调用结果回传格式
interface ToolResultBlock {
  type: "tool_result";
  tool_use_id: string;       // 对应 ToolUseBlock 的 id
  content: string | ContentBlock[];  // 工具执行结果
  is_error?: boolean;        // 是否执行出错
}
```

**完整 Tool Use 示例**：

```typescript
const tools: AnthropicTool[] = [
  {
    name: "get_weather",
    description: "获取指定城市的当前天气",
    input_schema: {
      type: "object",
      properties: {
        city: { type: "string", description: "城市名称" },
        unit: { type: "string", enum: ["celsius", "fahrenheit"] },
      },
      required: ["city"],
    },
  },
];

// 第一轮：模型决定调用工具
const response = await client.messages.create({
  model: "claude-opus-4-5-20251101",
  max_tokens: 1024,
  tools,
  messages: [{ role: "user", content: "北京今天天气怎么样？" }],
});

// 第二轮：将工具结果回传给模型
if (response.stop_reason === "tool_use") {
  const toolUse = response.content.find((b) => b.type === "tool_use") as ToolUseBlock;
  const weatherResult = await fetchWeather(toolUse.input as { city: string });

  const followUp = await client.messages.create({
    model: "claude-opus-4-5-20251101",
    max_tokens: 1024,
    tools,
    messages: [
      { role: "user", content: "北京今天天气怎么样？" },
      { role: "assistant", content: response.content },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: toolUse.id,
            content: JSON.stringify(weatherResult),
          },
        ],
      },
    ],
  });
}
```

---

## 3. 通用类型定义

### 3.1 Message 类型

不同 API 的 Message 格式对比：

| 字段 | OpenAI | Anthropic | 说明 |
|------|--------|-----------|------|
| `role` | `"system" \| "user" \| "assistant" \| "tool"` | `"user" \| "assistant"` | Anthropic 系统提示单独传入 |
| `content` | `string \| ContentPart[]` | `string \| ContentBlock[]` | 两者都支持字符串或结构化内容 |
| `name` | 可选，标识发言者 | 不支持 | - |
| `tool_call_id` | role 为 `"tool"` 时必填 | 不适用 | - |

**OpenAI Message 类型**：

```typescript
type OpenAIMessage =
  | { role: "system"; content: string; name?: string }
  | { role: "user"; content: string | UserContentPart[]; name?: string }
  | { role: "assistant"; content: string | null; tool_calls?: ToolCall[]; refusal?: string }
  | { role: "tool"; content: string; tool_call_id: string };

type UserContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string; detail?: "auto" | "low" | "high" } };
```

**Anthropic Message 类型**：

```typescript
type AnthropicMessage = {
  role: "user" | "assistant";
  content: string | AnthropicContentBlock[];
};

type AnthropicContentBlock =
  | { type: "text"; text: string }
  | { type: "image"; source: ImageSource }
  | { type: "tool_use"; id: string; name: string; input: object }
  | { type: "tool_result"; tool_use_id: string; content: string | AnthropicContentBlock[]; is_error?: boolean };

type ImageSource =
  | { type: "base64"; media_type: "image/jpeg" | "image/png" | "image/gif" | "image/webp"; data: string }
  | { type: "url"; url: string };
```

---

### 3.2 Tool / Function 类型

**统一封装类型（两个 API 适用）**：

```typescript
// 通用工具定义（可转换为各平台格式）
interface UniversalTool {
  name: string;
  description: string;
  parameters: JSONSchema;
}

interface JSONSchema {
  type: "object";
  properties: Record<string, JSONSchemaProperty>;
  required?: string[];
  additionalProperties?: boolean;
}

interface JSONSchemaProperty {
  type: "string" | "number" | "integer" | "boolean" | "array" | "object" | "null";
  description?: string;
  enum?: (string | number)[];
  items?: JSONSchemaProperty;       // type 为 "array" 时使用
  properties?: Record<string, JSONSchemaProperty>; // type 为 "object" 时使用
  required?: string[];
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
  pattern?: string;                 // 正则表达式约束
}

// 工具调用结果（OpenAI）
interface OpenAIToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;    // JSON 字符串，需要 JSON.parse 解析
  };
}

// 工具调用结果（Anthropic）
interface AnthropicToolUse {
  type: "tool_use";
  id: string;
  name: string;
  input: object;          // 已自动解析的 JSON 对象
}
```

---

### 3.3 流式事件类型

**OpenAI 流式事件**：

```typescript
interface OpenAIStreamChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: "assistant";
      content?: string | null;
      tool_calls?: Array<{
        index: number;
        id?: string;
        type?: "function";
        function?: {
          name?: string;
          arguments?: string;   // 增量 JSON 字符串，需拼接后解析
        };
      }>;
    };
    finish_reason: string | null;
  }>;
  usage?: {                     // 需开启 stream_options: { include_usage: true }
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  } | null;
}
```

**Anthropic 流式事件**：

```typescript
type AnthropicStreamEvent =
  | { type: "message_start"; message: Partial<AnthropicMessageResponse> }
  | { type: "content_block_start"; index: number; content_block: ContentBlock }
  | { type: "content_block_delta"; index: number; delta: ContentBlockDelta }
  | { type: "content_block_stop"; index: number }
  | { type: "message_delta"; delta: { stop_reason: string; stop_sequence: string | null }; usage: { output_tokens: number } }
  | { type: "message_stop" }
  | { type: "ping" }
  | { type: "error"; error: { type: string; message: string } };

type ContentBlockDelta =
  | { type: "text_delta"; text: string }
  | { type: "input_json_delta"; partial_json: string };  // 工具调用参数的增量 JSON
```

---

## 4. 错误码参考

### 4.1 常见 HTTP 状态码

| HTTP 状态码 | 名称 | 说明 | 处理建议 |
|-------------|------|------|----------|
| `200` | OK | 请求成功 | 正常处理响应 |
| `400` | Bad Request | 请求格式错误或参数无效 | 检查请求体，修复参数错误 |
| `401` | Unauthorized | API Key 无效或缺失 | 检查 API Key 是否正确配置 |
| `403` | Forbidden | 无权访问该资源 | 检查账户权限或模型访问权限 |
| `404` | Not Found | 资源不存在 | 检查端点 URL 和模型 ID |
| `422` | Unprocessable Entity | 请求格式正确但内容无法处理 | 检查参数值是否在有效范围内 |
| `429` | Too Many Requests | 超出速率限制 | 实现指数退避重试策略 |
| `500` | Internal Server Error | API 服务器内部错误 | 等待后重试，可联系技术支持 |
| `502` | Bad Gateway | 网关错误 | 短暂等待后重试 |
| `503` | Service Unavailable | 服务暂时不可用（高负载或维护） | 实现重试策略，等待服务恢复 |
| `529` | Overloaded | 服务过载（Anthropic 特有） | 实现指数退避重试 |

---

### 4.2 API 特定错误

**OpenAI 错误格式**：

```typescript
interface OpenAIError {
  error: {
    message: string;   // 人类可读的错误描述
    type: string;      // 错误类型
    param: string | null;  // 导致错误的参数名
    code: string | null;   // 错误码
  };
}
```

**OpenAI 常见错误类型**：

| `type` | `code` | 说明 |
|--------|--------|------|
| `invalid_request_error` | `invalid_api_key` | API Key 无效 |
| `invalid_request_error` | `model_not_found` | 模型不存在或无权访问 |
| `invalid_request_error` | `context_length_exceeded` | 超出上下文长度限制 |
| `invalid_request_error` | `content_policy_violation` | 内容违反使用政策 |
| `rate_limit_error` | `rate_limit_exceeded` | 超出速率限制 |
| `insufficient_quota` | `insufficient_quota` | 账户余额不足 |
| `api_error` | `server_error` | 服务器内部错误 |

**Anthropic 错误格式**：

```typescript
interface AnthropicError {
  type: "error";
  error: {
    type: string;      // 错误类型
    message: string;   // 错误描述
  };
}
```

**Anthropic 常见错误类型**：

| `error.type` | HTTP 状态码 | 说明 |
|--------------|-------------|------|
| `authentication_error` | 401 | API Key 无效或缺失 |
| `permission_error` | 403 | 无权访问该资源 |
| `invalid_request_error` | 400 | 请求参数无效 |
| `not_found_error` | 404 | 资源不存在 |
| `rate_limit_error` | 429 | 超出速率限制 |
| `api_error` | 500 | API 服务器内部错误 |
| `overloaded_error` | 529 | API 服务过载 |

**错误处理最佳实践**：

```typescript
async function callWithRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  baseDelay = 1000
): Promise<T> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error: unknown) {
      const isRetryable =
        (error instanceof Error && "status" in error &&
          [429, 500, 502, 503, 529].includes((error as { status: number }).status));

      if (!isRetryable || attempt === maxRetries - 1) {
        throw error;
      }

      // 指数退避：1s, 2s, 4s...
      const delay = baseDelay * Math.pow(2, attempt);
      console.warn(`请求失败，${delay}ms 后重试（第 ${attempt + 1} 次）`);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw new Error("超出最大重试次数");
}
```

---

## 5. 速率限制与配额

### 5.1 Token 限制

**上下文窗口限制（模型输入 + 输出总和上限）**：

| 平台 | 模型 | 上下文窗口 | 最大输出 |
|------|------|-----------|----------|
| OpenAI | `gpt-4o` | 128,000 tokens | 16,384 tokens |
| OpenAI | `gpt-4o-mini` | 128,000 tokens | 16,384 tokens |
| OpenAI | `gpt-4-turbo` | 128,000 tokens | 4,096 tokens |
| OpenAI | `gpt-3.5-turbo` | 16,385 tokens | 4,096 tokens |
| Anthropic | `claude-opus-4-5` | 200,000 tokens | 8,096 tokens |
| Anthropic | `claude-sonnet-4-6` | 200,000 tokens | 8,096 tokens |
| Anthropic | `claude-haiku-3-5` | 200,000 tokens | 8,096 tokens |

**Token 计算规则**：
- 英文文本：约 1 token = 4 个字符，或 ~0.75 个词
- 中文文本：约 1 个汉字 = 1–2 个 token
- 代码：根据语言不同，token 密度有所差异

**Token 估算工具**：

```typescript
// 使用 tiktoken 库（OpenAI 官方工具）
import { encoding_for_model } from "tiktoken";

function countTokens(text: string, model = "gpt-4o"): number {
  const enc = encoding_for_model(model as Parameters<typeof encoding_for_model>[0]);
  const tokens = enc.encode(text);
  enc.free();
  return tokens.length;
}

// 估算对话所需 token 数
function estimateChatTokens(messages: { role: string; content: string }[]): number {
  let total = 3; // 每个回复固定开销
  for (const msg of messages) {
    total += 4; // 每条消息固定开销
    total += countTokens(msg.content);
    total += countTokens(msg.role);
  }
  return total;
}
```

---

### 5.2 请求限制

**OpenAI 速率限制（按 Tier 划分，以下为 Tier 1 示例）**：

| 模型 | RPM（每分钟请求数） | TPM（每分钟 token 数） | RPD（每日请求数） |
|------|---------------------|------------------------|------------------|
| `gpt-4o` | 500 | 30,000 | 10,000 |
| `gpt-4o-mini` | 500 | 200,000 | 10,000 |
| `gpt-3.5-turbo` | 3,500 | 200,000 | 10,000 |

**Anthropic 速率限制（以 Build 计划为例）**：

| 模型 | RPM | TPM | TPD（每日 token 数） |
|------|-----|-----|---------------------|
| `claude-opus-4-5` | 50 | 40,000 | 2,000,000 |
| `claude-sonnet-4-6` | 50 | 80,000 | 4,000,000 |
| `claude-haiku-3-5` | 50 | 100,000 | 5,000,000 |

> **注意**：以上数字仅供参考，实际限制会因账户等级、使用计划及平台政策变动而有所不同。请以各平台官方文档为准。

**响应头中的速率限制信息**：

```
# OpenAI 响应头
x-ratelimit-limit-requests: 500
x-ratelimit-limit-tokens: 30000
x-ratelimit-remaining-requests: 499
x-ratelimit-remaining-tokens: 29800
x-ratelimit-reset-requests: 120ms
x-ratelimit-reset-tokens: 0.5s

# Anthropic 响应头
anthropic-ratelimit-requests-limit: 50
anthropic-ratelimit-requests-remaining: 49
anthropic-ratelimit-requests-reset: 2024-01-01T00:01:00Z
anthropic-ratelimit-tokens-limit: 40000
anthropic-ratelimit-tokens-remaining: 39800
anthropic-ratelimit-tokens-reset: 2024-01-01T00:01:00Z
```

**解析速率限制响应头示例**：

```typescript
function parseRateLimitHeaders(headers: Headers): RateLimitInfo {
  // 适用于 OpenAI
  return {
    requestsLimit: parseInt(headers.get("x-ratelimit-limit-requests") ?? "0"),
    requestsRemaining: parseInt(headers.get("x-ratelimit-remaining-requests") ?? "0"),
    tokensLimit: parseInt(headers.get("x-ratelimit-limit-tokens") ?? "0"),
    tokensRemaining: parseInt(headers.get("x-ratelimit-remaining-tokens") ?? "0"),
    resetAfterMs: parseResetTime(headers.get("x-ratelimit-reset-tokens") ?? "0s"),
  };
}

function parseResetTime(resetStr: string): number {
  // 解析 "120ms" 或 "1.5s" 格式
  if (resetStr.endsWith("ms")) return parseFloat(resetStr);
  if (resetStr.endsWith("s")) return parseFloat(resetStr) * 1000;
  return 0;
}

interface RateLimitInfo {
  requestsLimit: number;
  requestsRemaining: number;
  tokensLimit: number;
  tokensRemaining: number;
  resetAfterMs: number;
}
```

**并发请求管理示例**：

```typescript
import PQueue from "p-queue";

// 限制并发请求数，避免触发速率限制
const queue = new PQueue({
  concurrency: 5,          // 最大并发数
  interval: 60_000,        // 时间窗口（1分钟）
  intervalCap: 50,         // 时间窗口内最大请求数
});

async function batchProcess<T>(
  items: string[],
  processItem: (item: string) => Promise<T>
): Promise<T[]> {
  const results = await Promise.all(
    items.map((item) => queue.add(() => processItem(item)))
  );
  return results as T[];
}
```

---

*附录B完。如需查阅其他附录，请参见目录页。*
