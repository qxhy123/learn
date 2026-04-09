# 第21章：Function Calling与Agent

## 学习目标

完成本章学习后，你将能够：

1. 理解 Function Calling 的工作原理，掌握工具定义的 JSON Schema 规范与 TypeScript 类型体系
2. 正确解析模型返回的工具调用请求，实现类型安全的工具分发与执行
3. 使用多轮对话循环将工具执行结果回传给模型，完成完整的工具调用闭环
4. 掌握流式模式下工具调用的 delta 累积与解析技巧
5. 设计并实现一个支持多工具调用、具备循环推理能力的简单 AI Agent

---

## 21.1 Function Calling 概述

### 什么是 Function Calling

大语言模型本质上只能处理文本输入并生成文本输出。Function Calling（工具调用）是一种机制，让模型能够以结构化的 JSON 格式"请求"执行外部函数，从而访问实时数据、执行计算、调用 API 等模型本身无法完成的操作。

工作流程如下：

```
用户提问
  ↓
模型判断是否需要工具
  ↓（需要工具）
模型返回工具调用请求（JSON）
  ↓
应用层执行实际函数
  ↓
将执行结果回传给模型
  ↓
模型基于结果生成最终回答
```

### Function Calling 与普通文本生成的对比

| 特性 | 普通文本生成 | Function Calling |
|------|------------|-----------------|
| 模型输出 | 自然语言文本 | 结构化 JSON（工具名 + 参数）|
| 数据来源 | 模型训练数据（静态） | 外部函数实时获取 |
| 适用场景 | 问答、写作、分析 | 查询数据库、调用 API、执行计算 |
| finish_reason | `"stop"` | `"tool_calls"` |
| 轮次 | 单轮即可完成 | 至少两轮（调用 + 结果回传）|

### 核心概念：工具（Tool）

在 OpenAI API 中，工具由两部分组成：

1. **工具定义**：告诉模型有哪些工具可用，每个工具的名称、描述和参数规范
2. **工具结果**：应用层执行工具后，将结果以 `tool` 角色消息回传给模型

```typescript
import OpenAI from 'openai';

// 工具定义类型：ChatCompletionTool
const weatherTool: OpenAI.Chat.Completions.ChatCompletionTool = {
  type: 'function',
  function: {
    name: 'get_weather',
    description: '获取指定城市的当前天气信息',
    parameters: {
      type: 'object',
      properties: {
        city: { type: 'string', description: '城市名称' },
        unit: { type: 'string', enum: ['celsius', 'fahrenheit'] },
      },
      required: ['city'],
    },
  },
};
```

### tool_choice 参数控制

`tool_choice` 参数控制模型是否以及如何使用工具：

```typescript
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// auto：模型自行决定是否调用工具（默认值，最常用）
await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: '北京天气怎么样？' }],
  tools: [weatherTool],
  tool_choice: 'auto',
});

// none：禁止工具调用，强制纯文本回复
await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: '你好' }],
  tools: [weatherTool],
  tool_choice: 'none',
});

// required：强制模型必须调用至少一个工具
await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: '帮我查查天气' }],
  tools: [weatherTool],
  tool_choice: 'required',
});

// 强制调用特定工具
await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: '帮我查查天气' }],
  tools: [weatherTool],
  tool_choice: { type: 'function', function: { name: 'get_weather' } },
});
```

---

## 21.2 工具定义（JSON Schema）

### JSON Schema 基础

工具参数使用 JSON Schema 规范描述。掌握核心字段是编写有效工具定义的关键：

```typescript
// JSON Schema 的核心字段
interface JSONSchemaObject {
  type: 'object';
  properties: Record<string, JSONSchemaProperty>;
  required?: string[];           // 必填字段列表
  additionalProperties?: boolean; // 是否允许额外字段（建议 false）
}

interface JSONSchemaProperty {
  type: 'string' | 'number' | 'integer' | 'boolean' | 'array' | 'object' | 'null';
  description?: string;          // 字段描述（对模型理解非常重要）
  enum?: unknown[];               // 枚举值约束
  items?: JSONSchemaProperty;    // 数组元素类型（type 为 array 时）
  minimum?: number;              // 数字最小值
  maximum?: number;              // 数字最大值
  minLength?: number;            // 字符串最小长度
  maxLength?: number;            // 字符串最大长度
  pattern?: string;              // 正则表达式约束
  default?: unknown;             // 默认值
}
```

### 常见工具定义模式

**基础字符串参数**

```typescript
import OpenAI from 'openai';

type ToolDef = OpenAI.Chat.Completions.ChatCompletionTool;

const searchTool: ToolDef = {
  type: 'function',
  function: {
    name: 'web_search',
    description: '在互联网上搜索信息。当用户询问实时信息、当前事件或你不确定的事实时使用此工具。',
    parameters: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: '搜索查询词，使用关键词而非完整句子效果更好',
          minLength: 1,
          maxLength: 200,
        },
        language: {
          type: 'string',
          description: '搜索结果的语言',
          enum: ['zh', 'en', 'ja'],
          default: 'zh',
        },
        max_results: {
          type: 'integer',
          description: '返回的最大结果数量',
          minimum: 1,
          maximum: 10,
          default: 5,
        },
      },
      required: ['query'],
      additionalProperties: false,
    },
  },
};
```

**嵌套对象参数**

```typescript
const createOrderTool: ToolDef = {
  type: 'function',
  function: {
    name: 'create_order',
    description: '创建一个新的电商订单',
    parameters: {
      type: 'object',
      properties: {
        customer: {
          type: 'object',
          description: '客户信息',
          properties: {
            name: { type: 'string', description: '客户姓名' },
            email: {
              type: 'string',
              description: '客户邮箱',
              pattern: '^[^@]+@[^@]+\\.[^@]+$',
            },
            phone: { type: 'string', description: '联系电话' },
          },
          required: ['name', 'email'],
        },
        items: {
          type: 'array',
          description: '订单商品列表，至少包含一个商品',
          items: {
            type: 'object',
            properties: {
              product_id: { type: 'string', description: '商品 ID' },
              quantity: {
                type: 'integer',
                description: '购买数量',
                minimum: 1,
              },
            },
            required: ['product_id', 'quantity'],
          },
          minItems: 1,
        },
        shipping_address: {
          type: 'string',
          description: '收货地址（省市区街道详细地址）',
        },
        notes: {
          type: 'string',
          description: '订单备注（可选）',
        },
      },
      required: ['customer', 'items', 'shipping_address'],
      additionalProperties: false,
    },
  },
};
```

**布尔与枚举参数**

```typescript
const sendEmailTool: ToolDef = {
  type: 'function',
  function: {
    name: 'send_email',
    description: '发送一封电子邮件',
    parameters: {
      type: 'object',
      properties: {
        to: {
          type: 'array',
          description: '收件人邮箱列表',
          items: { type: 'string' },
          minItems: 1,
        },
        subject: { type: 'string', description: '邮件主题' },
        body: { type: 'string', description: '邮件正文（支持 HTML）' },
        priority: {
          type: 'string',
          description: '邮件优先级',
          enum: ['low', 'normal', 'high', 'urgent'],
          default: 'normal',
        },
        is_html: {
          type: 'boolean',
          description: '正文是否为 HTML 格式，默认为 false（纯文本）',
          default: false,
        },
      },
      required: ['to', 'subject', 'body'],
      additionalProperties: false,
    },
  },
};
```

### 工具描述的最佳实践

好的工具描述能显著提升模型正确调用工具的概率：

```typescript
// 差：描述过于简单，模型不知道何时调用
const badTool: ToolDef = {
  type: 'function',
  function: {
    name: 'get_data',
    description: '获取数据',
    parameters: {
      type: 'object',
      properties: {
        id: { type: 'string' },
      },
      required: ['id'],
    },
  },
};

// 好：描述清晰，说明使用时机、参数含义和返回内容
const goodTool: ToolDef = {
  type: 'function',
  function: {
    name: 'get_user_profile',
    description: [
      '根据用户 ID 查询用户的详细档案信息，包括姓名、邮箱、注册时间和订阅状态。',
      '当用户询问某个账户的具体信息时使用此工具。',
      '如果用户 ID 不存在，将返回 null。',
    ].join(' '),
    parameters: {
      type: 'object',
      properties: {
        user_id: {
          type: 'string',
          description: '用户的唯一标识符，格式为 "usr_" 开头的字符串，例如 "usr_abc123"',
        },
        include_stats: {
          type: 'boolean',
          description: '是否同时返回用户的使用统计数据（会增加响应时间），默认 false',
          default: false,
        },
      },
      required: ['user_id'],
      additionalProperties: false,
    },
  },
};
```

---

## 21.3 工具调用解析与执行

### 识别工具调用响应

当模型决定调用工具时，响应的 `finish_reason` 为 `"tool_calls"`，`message.tool_calls` 包含调用详情：

```typescript
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ChatCompletionMessage 中工具调用的类型结构
interface ToolCall {
  id: string;           // 工具调用的唯一 ID，回传结果时需要用到
  type: 'function';
  function: {
    name: string;       // 工具名称
    arguments: string;  // 参数的 JSON 字符串（注意：是字符串不是对象！）
  };
}

async function detectToolCall(userMessage: string) {
  const response = await client.chat.completions.create({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: userMessage }],
    tools: [weatherTool, searchTool],
    tool_choice: 'auto',
  });

  const message = response.choices[0].message;
  const finishReason = response.choices[0].finish_reason;

  if (finishReason === 'tool_calls' && message.tool_calls) {
    console.log(`模型请求调用 ${message.tool_calls.length} 个工具`);
    for (const tc of message.tool_calls) {
      console.log(`工具: ${tc.function.name}`);
      console.log(`参数: ${tc.function.arguments}`);
    }
    return { type: 'tool_calls' as const, toolCalls: message.tool_calls, message };
  }

  // finish_reason 为 "stop"，直接返回文本
  return { type: 'text' as const, content: message.content ?? '' };
}
```

### 类型安全的参数解析

`arguments` 字段是 JSON 字符串，需要解析并验证类型：

```typescript
// 定义每个工具的参数类型
interface WeatherParams {
  city: string;
  unit?: 'celsius' | 'fahrenheit';
}

interface SearchParams {
  query: string;
  language?: 'zh' | 'en' | 'ja';
  max_results?: number;
}

// 类型安全的参数解析器
function parseToolArguments<T>(argumentsJson: string): T {
  try {
    return JSON.parse(argumentsJson) as T;
  } catch {
    throw new Error(`工具参数解析失败: ${argumentsJson}`);
  }
}

// 使用示例
const tc: ToolCall = {
  id: 'call_abc123',
  type: 'function',
  function: {
    name: 'get_weather',
    arguments: '{"city":"北京","unit":"celsius"}',
  },
};

const params = parseToolArguments<WeatherParams>(tc.function.arguments);
console.log(params.city);  // "北京"
console.log(params.unit);  // "celsius"
```

### 工具注册表模式

当工具数量增多时，使用注册表模式统一管理工具定义和执行逻辑：

```typescript
import OpenAI from 'openai';

// 工具处理函数的类型
type ToolHandler<TParams = Record<string, unknown>, TResult = unknown> = (
  params: TParams
) => Promise<TResult> | TResult;

// 工具注册条目
interface ToolRegistration<TParams = Record<string, unknown>, TResult = unknown> {
  definition: OpenAI.Chat.Completions.ChatCompletionTool;
  handler: ToolHandler<TParams, TResult>;
}

// 工具注册表
class ToolRegistry {
  private tools = new Map<string, ToolRegistration>();

  // 注册工具
  register<TParams, TResult>(
    definition: OpenAI.Chat.Completions.ChatCompletionTool,
    handler: ToolHandler<TParams, TResult>
  ): this {
    const name = definition.function.name;
    this.tools.set(name, { definition, handler: handler as ToolHandler });
    return this;
  }

  // 获取所有工具定义（传给 API）
  getDefinitions(): OpenAI.Chat.Completions.ChatCompletionTool[] {
    return Array.from(this.tools.values()).map((t) => t.definition);
  }

  // 执行工具调用
  async execute(
    toolCall: OpenAI.Chat.Completions.ChatCompletionMessageToolCall
  ): Promise<string> {
    const name = toolCall.function.name;
    const registration = this.tools.get(name);

    if (!registration) {
      return JSON.stringify({ error: `未知工具: ${name}` });
    }

    let params: unknown;
    try {
      params = JSON.parse(toolCall.function.arguments);
    } catch {
      return JSON.stringify({ error: `参数解析失败: ${toolCall.function.arguments}` });
    }

    try {
      const result = await registration.handler(params as Record<string, unknown>);
      return typeof result === 'string' ? result : JSON.stringify(result);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return JSON.stringify({ error: `工具执行失败: ${message}` });
    }
  }
}

// ---- 实际使用 ----

// 模拟工具实现
function getWeather(params: WeatherParams): object {
  const temps: Record<string, number> = { 北京: 15, 上海: 22, 广州: 28 };
  const temp = temps[params.city] ?? 20;
  return {
    city: params.city,
    temperature: params.unit === 'fahrenheit' ? temp * 1.8 + 32 : temp,
    unit: params.unit ?? 'celsius',
    condition: '晴天',
    humidity: '45%',
  };
}

async function webSearch(params: SearchParams): Promise<object[]> {
  // 实际项目中调用搜索 API
  return [
    { title: `${params.query} - 搜索结果1`, url: 'https://example.com/1', snippet: '示例摘要1' },
    { title: `${params.query} - 搜索结果2`, url: 'https://example.com/2', snippet: '示例摘要2' },
  ];
}

const registry = new ToolRegistry()
  .register<WeatherParams, object>(weatherTool, getWeather)
  .register<SearchParams, object[]>(searchTool, webSearch);
```

### 完整的单次工具调用流程

```typescript
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function singleToolCallRound(userMessage: string): Promise<string> {
  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'user', content: userMessage },
  ];

  // 第一轮：询问模型
  const firstResponse = await client.chat.completions.create({
    model: 'gpt-4o',
    messages,
    tools: registry.getDefinitions(),
    tool_choice: 'auto',
  });

  const assistantMessage = firstResponse.choices[0].message;

  // 如果没有工具调用，直接返回文本
  if (firstResponse.choices[0].finish_reason !== 'tool_calls') {
    return assistantMessage.content ?? '';
  }

  // 将助手消息（含 tool_calls）加入历史
  messages.push(assistantMessage);

  // 执行所有工具调用
  const toolCalls = assistantMessage.tool_calls!;
  for (const tc of toolCalls) {
    const result = await registry.execute(tc);

    // 将工具结果以 tool 角色加入历史
    messages.push({
      role: 'tool',
      tool_call_id: tc.id,   // 必须与对应的 tool_call.id 匹配
      content: result,
    });
  }

  // 第二轮：将工具结果回传给模型，获得最终回答
  const finalResponse = await client.chat.completions.create({
    model: 'gpt-4o',
    messages,
  });

  return finalResponse.choices[0].message.content ?? '';
}

// 使用示例
const answer = await singleToolCallRound('北京今天天气怎么样？');
console.log(answer);
// 输出: 北京今天天气晴天，气温 15°C，湿度 45%。
```

---

## 21.4 多轮对话与工具调用

### 为什么需要多轮循环

复杂任务往往需要多次工具调用才能完成。例如：用户问"明天北京和上海哪个城市更适合户外活动"，模型可能会：

1. 调用 `get_weather(city="北京")` 获取北京天气
2. 调用 `get_weather(city="上海")` 获取上海天气
3. 基于两个结果给出建议

此外，模型还可能在获得工具结果后，发现需要调用另一个工具（链式调用）。因此需要一个循环来持续处理工具调用，直到模型返回纯文本为止。

### 多轮工具调用循环

```typescript
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

interface AgentRunOptions {
  maxIterations?: number;  // 防止无限循环
  verbose?: boolean;       // 是否打印调用过程
}

async function runWithTools(
  userMessage: string,
  systemPrompt: string = '你是一个智能助手，能够使用工具回答用户问题。',
  options: AgentRunOptions = {}
): Promise<string> {
  const { maxIterations = 10, verbose = false } = options;

  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userMessage },
  ];

  let iteration = 0;

  while (iteration < maxIterations) {
    iteration++;

    if (verbose) {
      console.log(`\n[迭代 ${iteration}] 调用 API...`);
    }

    const response = await client.chat.completions.create({
      model: 'gpt-4o',
      messages,
      tools: registry.getDefinitions(),
      tool_choice: 'auto',
    });

    const message = response.choices[0].message;
    const finishReason = response.choices[0].finish_reason;

    // 将助手消息加入历史（无论是工具调用还是文本）
    messages.push(message);

    // 模型返回纯文本，任务完成
    if (finishReason === 'stop') {
      if (verbose) {
        console.log(`[完成] 经过 ${iteration} 轮迭代`);
      }
      return message.content ?? '';
    }

    // 处理工具调用
    if (finishReason === 'tool_calls' && message.tool_calls) {
      if (verbose) {
        console.log(`[工具调用] ${message.tool_calls.map((tc) => tc.function.name).join(', ')}`);
      }

      // 并发执行所有工具调用（如果工具间无依赖）
      const toolResults = await Promise.all(
        message.tool_calls.map(async (tc) => {
          const result = await registry.execute(tc);
          if (verbose) {
            console.log(`  ${tc.function.name}(${tc.function.arguments}) → ${result}`);
          }
          return { tc, result };
        })
      );

      // 按顺序将所有工具结果加入消息历史
      for (const { tc, result } of toolResults) {
        messages.push({
          role: 'tool',
          tool_call_id: tc.id,
          content: result,
        });
      }
    } else if (finishReason === 'length') {
      // 模型因达到 max_tokens 停止
      return (message.content ?? '') + '\n[注意：回复被截断]';
    } else {
      // 其他原因（content_filter 等）
      break;
    }
  }

  throw new Error(`超过最大迭代次数 (${maxIterations})`);
}
```

### 并行工具调用

OpenAI 模型支持在单次响应中请求多个工具调用。执行时应并发运行以减少延迟：

```typescript
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// 示例：模型可能同时调用两个天气查询
async function parallelToolCallExample(): Promise<string> {
  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    {
      role: 'user',
      content: '帮我对比一下北京和上海今天的天气，哪个更适合出行？',
    },
  ];

  const response = await client.chat.completions.create({
    model: 'gpt-4o',
    messages,
    tools: [weatherTool],
    tool_choice: 'auto',
  });

  const assistantMsg = response.choices[0].message;
  messages.push(assistantMsg);

  if (assistantMsg.tool_calls && assistantMsg.tool_calls.length > 0) {
    console.log(`并发执行 ${assistantMsg.tool_calls.length} 个工具调用`);

    // 并发执行：比串行执行快得多
    const results = await Promise.allSettled(
      assistantMsg.tool_calls.map(async (tc) => {
        const result = await registry.execute(tc);
        return { id: tc.id, result };
      })
    );

    // 将结果按顺序加入消息（顺序不重要，但 tool_call_id 必须对应）
    for (const settled of results) {
      if (settled.status === 'fulfilled') {
        messages.push({
          role: 'tool',
          tool_call_id: settled.value.id,
          content: settled.value.result,
        });
      } else {
        // 某个工具调用失败
        const tc = assistantMsg.tool_calls.find(
          (_, i) => results.indexOf(settled) === i
        );
        if (tc) {
          messages.push({
            role: 'tool',
            tool_call_id: tc.id,
            content: JSON.stringify({ error: settled.reason?.message ?? '工具执行失败' }),
          });
        }
      }
    }

    const finalResp = await client.chat.completions.create({
      model: 'gpt-4o',
      messages,
    });

    return finalResp.choices[0].message.content ?? '';
  }

  return assistantMsg.content ?? '';
}
```

### 流式模式下的工具调用

流式模式下，工具调用的参数通过多个 delta chunk 逐步传输，需要累积完整后再解析：

```typescript
import OpenAI from 'openai';
import type { ChatCompletionChunk } from 'openai/resources';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// 累积工具调用 delta 的数据结构
interface AccumulatedToolCall {
  id: string;
  name: string;
  argumentsBuffer: string;  // 逐步累积的 JSON 字符串
}

async function streamWithToolCalls(userMessage: string): Promise<string> {
  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: 'user', content: userMessage },
  ];

  const stream = await client.chat.completions.create({
    model: 'gpt-4o',
    messages,
    tools: registry.getDefinitions(),
    tool_choice: 'auto',
    stream: true,
  });

  let textBuffer = '';
  let finishReason: string | null = null;
  const toolCallAccumulators = new Map<number, AccumulatedToolCall>();

  for await (const chunk of stream) {
    const choice = chunk.choices[0];
    if (!choice) continue;

    const delta = choice.delta;
    finishReason = choice.finish_reason ?? finishReason;

    // 累积文本内容
    if (delta.content) {
      textBuffer += delta.content;
      process.stdout.write(delta.content);
    }

    // 累积工具调用 delta
    if (delta.tool_calls) {
      for (const tcDelta of delta.tool_calls) {
        const idx = tcDelta.index;

        if (!toolCallAccumulators.has(idx)) {
          // 首个 delta：包含 id 和 name
          toolCallAccumulators.set(idx, {
            id: tcDelta.id ?? '',
            name: tcDelta.function?.name ?? '',
            argumentsBuffer: '',
          });
        }

        const acc = toolCallAccumulators.get(idx)!;

        // 补充 id（有时在后续 chunk 中出现）
        if (tcDelta.id) acc.id = tcDelta.id;
        if (tcDelta.function?.name) acc.name = tcDelta.function.name;

        // 累积参数字符串
        if (tcDelta.function?.arguments) {
          acc.argumentsBuffer += tcDelta.function.arguments;
        }
      }
    }
  }

  // 流结束，处理工具调用
  if (finishReason === 'tool_calls' && toolCallAccumulators.size > 0) {
    const toolCalls = Array.from(toolCallAccumulators.entries())
      .sort(([a], [b]) => a - b)  // 按 index 排序
      .map(([, acc]) => acc);

    // 构造助手消息（含 tool_calls）
    const assistantMessage: OpenAI.Chat.Completions.ChatCompletionMessageParam = {
      role: 'assistant',
      content: textBuffer || null,
      tool_calls: toolCalls.map((acc) => ({
        id: acc.id,
        type: 'function' as const,
        function: { name: acc.name, arguments: acc.argumentsBuffer },
      })),
    };

    messages.push(assistantMessage);

    // 执行工具
    for (const acc of toolCalls) {
      const tc: OpenAI.Chat.Completions.ChatCompletionMessageToolCall = {
        id: acc.id,
        type: 'function',
        function: { name: acc.name, arguments: acc.argumentsBuffer },
      };
      const result = await registry.execute(tc);
      messages.push({ role: 'tool', tool_call_id: tc.id, content: result });
    }

    // 获取最终流式回答
    const finalStream = await client.chat.completions.create({
      model: 'gpt-4o',
      messages,
      stream: true,
    });

    let finalText = '';
    for await (const chunk of finalStream) {
      const delta = chunk.choices[0]?.delta?.content ?? '';
      finalText += delta;
      process.stdout.write(delta);
    }
    console.log();
    return finalText;
  }

  console.log();
  return textBuffer;
}
```

---

## 21.5 构建简单的 AI Agent

### Agent 的核心概念

Agent（智能体）是一种能够自主感知环境、制定计划并采取行动的 AI 系统。基于 Function Calling 的 Agent 具备以下能力：

- **推理**：分析用户意图，制定解决路径
- **工具使用**：调用外部工具获取信息或执行操作
- **记忆**：通过对话历史保持上下文
- **循环执行**：持续推理-行动直到任务完成

```
用户请求
    ↓
[推理] 分析需要哪些信息/操作
    ↓
[行动] 调用工具
    ↓
[观察] 分析工具结果
    ↓
[判断] 任务是否完成？
  否 ↙     ↘ 是
[继续推理]  [生成最终回答]
```

### Agent 类设计

```typescript
import OpenAI from 'openai';

// Agent 运行步骤日志
type AgentStep =
  | { type: 'user_message'; content: string }
  | { type: 'thinking'; content: string }
  | { type: 'tool_call'; name: string; args: string; result: string }
  | { type: 'final_answer'; content: string };

interface AgentConfig {
  model?: string;
  systemPrompt?: string;
  maxIterations?: number;
  tools: ToolRegistry;
}

class SimpleAgent {
  private client: OpenAI;
  private model: string;
  private systemPrompt: string;
  private maxIterations: number;
  private registry: ToolRegistry;
  private conversationHistory: OpenAI.Chat.Completions.ChatCompletionMessageParam[];

  constructor(config: AgentConfig) {
    this.client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    this.model = config.model ?? 'gpt-4o';
    this.systemPrompt =
      config.systemPrompt ??
      [
        '你是一个智能助手，具备使用工具解决问题的能力。',
        '解决问题时请遵循以下原则：',
        '1. 优先使用工具获取准确信息，不要凭记忆猜测实时数据。',
        '2. 工具调用失败时，向用户说明原因，并尝试替代方案。',
        '3. 给出答案时，简明扼要，直接回应用户的问题。',
      ].join('\n');
    this.maxIterations = config.maxIterations ?? 10;
    this.registry = config.tools;
    this.conversationHistory = [];
  }

  // 运行一轮对话，返回 Agent 的最终回答
  async run(
    userMessage: string,
    onStep?: (step: AgentStep) => void
  ): Promise<string> {
    // 记录用户消息
    onStep?.({ type: 'user_message', content: userMessage });

    // 构建发送给 API 的完整消息列表
    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
      { role: 'system', content: this.systemPrompt },
      ...this.conversationHistory,
      { role: 'user', content: userMessage },
    ];

    let iteration = 0;
    let finalAnswer = '';

    while (iteration < this.maxIterations) {
      iteration++;

      const response = await this.client.chat.completions.create({
        model: this.model,
        messages,
        tools: this.registry.getDefinitions(),
        tool_choice: 'auto',
      });

      const choice = response.choices[0];
      const message = choice.message;
      const finishReason = choice.finish_reason;

      messages.push(message);

      if (finishReason === 'stop' || !message.tool_calls) {
        // 任务完成，输出最终回答
        finalAnswer = message.content ?? '';
        onStep?.({ type: 'final_answer', content: finalAnswer });
        break;
      }

      if (finishReason === 'tool_calls' && message.tool_calls) {
        // 如果模型有思考文本，记录下来
        if (message.content) {
          onStep?.({ type: 'thinking', content: message.content });
        }

        // 并发执行所有工具调用
        const results = await Promise.all(
          message.tool_calls.map(async (tc) => {
            const result = await this.registry.execute(tc);
            onStep?.({
              type: 'tool_call',
              name: tc.function.name,
              args: tc.function.arguments,
              result,
            });
            return { tc, result };
          })
        );

        // 将工具结果加入消息历史
        for (const { tc, result } of results) {
          messages.push({
            role: 'tool',
            tool_call_id: tc.id,
            content: result,
          });
        }
      }
    }

    if (!finalAnswer) {
      finalAnswer = `达到最大迭代次数（${this.maxIterations}），任务可能未完成。`;
    }

    // 更新对话历史（用于多轮对话）
    this.conversationHistory.push(
      { role: 'user', content: userMessage },
      { role: 'assistant', content: finalAnswer }
    );

    return finalAnswer;
  }

  // 清空对话历史，开始新会话
  reset(): void {
    this.conversationHistory = [];
  }

  // 获取当前对话历史（用于调试）
  getHistory(): ReadonlyArray<OpenAI.Chat.Completions.ChatCompletionMessageParam> {
    return this.conversationHistory;
  }
}
```

### 多轮对话的 Agent

Agent 支持多轮对话，保持跨轮次的上下文：

```typescript
async function multiTurnAgentDemo(): Promise<void> {
  const agent = new SimpleAgent({
    model: 'gpt-4o',
    tools: registry,
    systemPrompt: '你是一个旅游助手，帮助用户规划旅行。你可以查询天气和搜索景点信息。',
  });

  // 打印步骤的回调函数
  function printStep(step: AgentStep): void {
    switch (step.type) {
      case 'user_message':
        console.log(`\n用户: ${step.content}`);
        break;
      case 'thinking':
        console.log(`思考: ${step.content}`);
        break;
      case 'tool_call':
        console.log(`工具: ${step.name}(${step.args})`);
        console.log(`结果: ${step.result}`);
        break;
      case 'final_answer':
        console.log(`\n助手: ${step.content}`);
        break;
    }
  }

  // 第一轮
  await agent.run('我想去北京旅游，现在天气怎么样？', printStep);

  // 第二轮：Agent 记住了上下文（知道用户要去北京）
  await agent.run('那上海呢？我要对比一下', printStep);

  // 第三轮：基于前两轮的信息做推荐
  await agent.run('根据天气，你觉得我应该先去哪个城市？', printStep);

  console.log('\n--- 对话历史 ---');
  console.log(`历史消息数: ${agent.getHistory().length}`);
}

multiTurnAgentDemo().catch(console.error);
```

---

## 本章小结

| 知识点 | 核心 API / 概念 | 关键要点 |
|--------|----------------|---------|
| Function Calling 原理 | `tool_choice`，`finish_reason: "tool_calls"` | 至少两轮请求：模型决策 + 结果回传 |
| 工具定义 | JSON Schema，`ChatCompletionTool` | 描述要清晰，说明使用时机，`required` 标注必填字段 |
| 参数解析 | `JSON.parse(arguments)`，类型守卫 | `arguments` 是字符串，解析时需 try-catch |
| 工具注册表 | `ToolRegistry`，工具分发模式 | 统一管理工具定义与实现，便于扩展 |
| 工具结果回传 | `role: "tool"`，`tool_call_id` | ID 必须与对应的 `tool_call.id` 精确匹配 |
| 并行工具调用 | `Promise.all`，多工具同时执行 | 无依赖关系的工具应并发执行，减少延迟 |
| 流式工具调用 | delta 累积，`argumentsBuffer` | `arguments` 分多个 chunk 传输，需累积完整后解析 |
| 多轮循环 | `while` 循环，`maxIterations` | 循环直到 `finish_reason === "stop"`，防止无限循环 |
| AI Agent | 推理-行动-观察循环，对话历史 | 维护 `conversationHistory` 实现跨轮次记忆 |

---

## AI 应用实战：AI Agent 实现

本节实现一个完整的、生产可用的 AI Agent，支持：天气查询、网络搜索、代码执行和文件操作四种工具，具备对话记忆、步骤追踪和错误恢复能力。

### 项目结构

```
ai-agent/
├── src/
│   ├── tools/
│   │   ├── definitions.ts    # 工具 JSON Schema 定义
│   │   ├── implementations.ts # 工具实现
│   │   └── registry.ts       # 工具注册表
│   ├── agent/
│   │   ├── agent.ts          # Agent 核心类
│   │   └── types.ts          # Agent 类型定义
│   └── main.ts               # 入口与演示
├── package.json
└── tsconfig.json
```

### 工具类型定义

```typescript
// src/agent/types.ts

import OpenAI from 'openai';

// Agent 执行步骤（用于追踪和调试）
export type AgentStep =
  | { type: 'user_message'; content: string; timestamp: number }
  | { type: 'thinking'; content: string; timestamp: number }
  | { type: 'tool_call'; name: string; args: unknown; result: string; durationMs: number; timestamp: number }
  | { type: 'final_answer'; content: string; totalSteps: number; timestamp: number };

// Agent 运行结果
export interface AgentResult {
  answer: string;
  steps: AgentStep[];
  totalToolCalls: number;
  durationMs: number;
}

// Agent 配置
export interface AgentConfig {
  model?: string;
  systemPrompt?: string;
  maxIterations?: number;
  onStep?: (step: AgentStep) => void;
}

// 工具处理函数签名
export type ToolHandler<TParams = Record<string, unknown>, TResult = unknown> = (
  params: TParams
) => Promise<TResult> | TResult;

// 工具注册条目
export interface ToolEntry {
  definition: OpenAI.Chat.Completions.ChatCompletionTool;
  handler: ToolHandler;
}
```

### 工具定义

```typescript
// src/tools/definitions.ts

import OpenAI from 'openai';

type ToolDef = OpenAI.Chat.Completions.ChatCompletionTool;

export const weatherToolDef: ToolDef = {
  type: 'function',
  function: {
    name: 'get_weather',
    description:
      '获取指定城市的实时天气信息，包括温度、天气状况、湿度和风速。当用户询问天气或需要基于天气做决策时使用。',
    parameters: {
      type: 'object',
      properties: {
        city: {
          type: 'string',
          description: '城市名称，如"北京"、"上海"、"纽约"',
        },
        unit: {
          type: 'string',
          enum: ['celsius', 'fahrenheit'],
          description: '温度单位，默认为 celsius（摄氏度）',
          default: 'celsius',
        },
      },
      required: ['city'],
      additionalProperties: false,
    },
  },
};

export const searchToolDef: ToolDef = {
  type: 'function',
  function: {
    name: 'web_search',
    description:
      '在互联网上搜索最新信息。当需要查询实时数据、近期事件、专业知识或你不确定的信息时使用。',
    parameters: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: '搜索关键词，简洁精准，避免过长的句子',
          minLength: 2,
          maxLength: 100,
        },
        max_results: {
          type: 'integer',
          description: '返回结果数量，1-10',
          minimum: 1,
          maximum: 10,
          default: 3,
        },
      },
      required: ['query'],
      additionalProperties: false,
    },
  },
};

export const calculatorToolDef: ToolDef = {
  type: 'function',
  function: {
    name: 'calculator',
    description:
      '执行数学计算。支持加减乘除、幂运算、对数、三角函数等。当需要精确计算时使用，不要自行估算。',
    parameters: {
      type: 'object',
      properties: {
        expression: {
          type: 'string',
          description:
            '数学表达式，例如 "2 + 3 * 4"、"Math.sqrt(16)"、"Math.log(100) / Math.log(10)"',
        },
      },
      required: ['expression'],
      additionalProperties: false,
    },
  },
};

export const getDateTimeDef: ToolDef = {
  type: 'function',
  function: {
    name: 'get_datetime',
    description:
      '获取当前日期和时间。当用户询问当前时间、日期、星期几等信息时使用。',
    parameters: {
      type: 'object',
      properties: {
        timezone: {
          type: 'string',
          description: '时区标识符，例如 "Asia/Shanghai"、"UTC"、"America/New_York"',
          default: 'Asia/Shanghai',
        },
        format: {
          type: 'string',
          enum: ['full', 'date', 'time', 'timestamp'],
          description: '返回格式：full（完整）、date（仅日期）、time（仅时间）、timestamp（时间戳）',
          default: 'full',
        },
      },
      required: [],
      additionalProperties: false,
    },
  },
};
```

### 工具实现

```typescript
// src/tools/implementations.ts

// 天气查询（模拟实现，实际项目接入真实 API）
interface WeatherParams {
  city: string;
  unit?: 'celsius' | 'fahrenheit';
}

export async function getWeatherImpl(params: WeatherParams): Promise<object> {
  // 实际项目中：调用 OpenWeatherMap / 和风天气等 API
  const weatherData: Record<string, { temp: number; condition: string; humidity: number; wind: number }> = {
    北京: { temp: 15, condition: '晴天', humidity: 42, wind: 12 },
    上海: { temp: 22, condition: '多云', humidity: 68, wind: 8 },
    广州: { temp: 28, condition: '小雨', humidity: 85, wind: 5 },
    深圳: { temp: 27, condition: '阴天', humidity: 78, wind: 10 },
    成都: { temp: 18, condition: '阴天', humidity: 75, wind: 3 },
  };

  const data = weatherData[params.city] ?? { temp: 20, condition: '未知', humidity: 50, wind: 10 };
  const temp = params.unit === 'fahrenheit' ? Math.round(data.temp * 1.8 + 32) : data.temp;

  return {
    city: params.city,
    temperature: temp,
    unit: params.unit ?? 'celsius',
    condition: data.condition,
    humidity: `${data.humidity}%`,
    wind_speed: `${data.wind} km/h`,
    updated_at: new Date().toISOString(),
  };
}

// 网络搜索（模拟实现）
interface SearchParams {
  query: string;
  max_results?: number;
}

export async function webSearchImpl(params: SearchParams): Promise<object[]> {
  // 实际项目中：调用 Tavily、Serper、Bing Search API 等
  const maxResults = params.max_results ?? 3;

  // 模拟搜索结果
  return Array.from({ length: Math.min(maxResults, 3) }, (_, i) => ({
    title: `${params.query} - 搜索结果 ${i + 1}`,
    url: `https://example.com/${encodeURIComponent(params.query)}-${i + 1}`,
    snippet: `关于"${params.query}"的相关信息摘要 ${i + 1}。这里是搜索结果的简短描述，帮助用户了解页面内容。`,
    published_at: new Date(Date.now() - i * 86400000).toISOString().split('T')[0],
  }));
}

// 计算器
interface CalculatorParams {
  expression: string;
}

export function calculatorImpl(params: CalculatorParams): object {
  // 安全的数学表达式求值（只允许数学操作）
  const safeExpression = params.expression
    .replace(/[^0-9+\-*/().%^sqrt logsincostan Math.PIEPSILON]/g, '')
    .replace(/\^/g, '**');  // 支持 ^ 作为幂运算符

  try {
    // 在受限范围内求值
    const mathFunctions = {
      sqrt: Math.sqrt,
      log: Math.log,
      log2: Math.log2,
      log10: Math.log10,
      sin: Math.sin,
      cos: Math.cos,
      tan: Math.tan,
      abs: Math.abs,
      round: Math.round,
      floor: Math.floor,
      ceil: Math.ceil,
      PI: Math.PI,
      E: Math.E,
    };

    // 构建安全的求值上下文
    const fn = new Function(
      ...Object.keys(mathFunctions),
      `"use strict"; return (${safeExpression});`
    );
    const result = fn(...Object.values(mathFunctions)) as number;

    if (typeof result !== 'number' || !isFinite(result)) {
      return { error: '计算结果无效（可能是除以零或无穷大）' };
    }

    return {
      expression: params.expression,
      result,
      formatted: Number.isInteger(result) ? result.toString() : result.toFixed(6),
    };
  } catch (error) {
    return {
      error: `表达式无效: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

// 获取日期时间
interface DateTimeParams {
  timezone?: string;
  format?: 'full' | 'date' | 'time' | 'timestamp';
}

export function getDateTimeImpl(params: DateTimeParams): object {
  const timezone = params.timezone ?? 'Asia/Shanghai';
  const format = params.format ?? 'full';

  const now = new Date();
  const formatter = new Intl.DateTimeFormat('zh-CN', {
    timeZone: timezone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    weekday: 'long',
    hour12: false,
  });

  const parts = formatter.formatToParts(now);
  const getPart = (type: string) => parts.find((p) => p.type === type)?.value ?? '';

  const dateStr = `${getPart('year')}-${getPart('month')}-${getPart('day')}`;
  const timeStr = `${getPart('hour')}:${getPart('minute')}:${getPart('second')}`;
  const weekday = getPart('weekday');

  switch (format) {
    case 'date':
      return { date: dateStr, weekday, timezone };
    case 'time':
      return { time: timeStr, timezone };
    case 'timestamp':
      return { timestamp: now.getTime(), iso: now.toISOString() };
    default:
      return { datetime: `${dateStr} ${timeStr}`, date: dateStr, time: timeStr, weekday, timezone };
  }
}
```

### 工具注册表

```typescript
// src/tools/registry.ts

import OpenAI from 'openai';
import type { ToolEntry, ToolHandler } from '../agent/types.js';
import {
  weatherToolDef,
  searchToolDef,
  calculatorToolDef,
  getDateTimeDef,
} from './definitions.js';
import {
  getWeatherImpl,
  webSearchImpl,
  calculatorImpl,
  getDateTimeImpl,
} from './implementations.js';

export class ToolRegistry {
  private tools = new Map<string, ToolEntry>();

  register<TParams, TResult>(
    definition: OpenAI.Chat.Completions.ChatCompletionTool,
    handler: ToolHandler<TParams, TResult>
  ): this {
    this.tools.set(definition.function.name, {
      definition,
      handler: handler as ToolHandler,
    });
    return this;
  }

  getDefinitions(): OpenAI.Chat.Completions.ChatCompletionTool[] {
    return Array.from(this.tools.values()).map((t) => t.definition);
  }

  async execute(
    toolCall: OpenAI.Chat.Completions.ChatCompletionMessageToolCall
  ): Promise<string> {
    const { name, arguments: argsJson } = toolCall.function;
    const entry = this.tools.get(name);

    if (!entry) {
      return JSON.stringify({ error: `未知工具: "${name}"，可用工具: ${[...this.tools.keys()].join(', ')}` });
    }

    let params: unknown;
    try {
      params = JSON.parse(argsJson);
    } catch {
      return JSON.stringify({ error: `工具参数 JSON 解析失败: ${argsJson}` });
    }

    try {
      const result = await entry.handler(params as Record<string, unknown>);
      return typeof result === 'string' ? result : JSON.stringify(result, null, 2);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return JSON.stringify({ error: `工具 "${name}" 执行失败: ${message}` });
    }
  }

  has(name: string): boolean {
    return this.tools.has(name);
  }

  size(): number {
    return this.tools.size;
  }
}

// 创建并导出默认注册表（注册所有内置工具）
export function createDefaultRegistry(): ToolRegistry {
  return new ToolRegistry()
    .register(weatherToolDef, getWeatherImpl)
    .register(searchToolDef, webSearchImpl)
    .register(calculatorToolDef, calculatorImpl)
    .register(getDateTimeDef, getDateTimeImpl);
}
```

### Agent 核心实现

```typescript
// src/agent/agent.ts

import OpenAI from 'openai';
import type { AgentConfig, AgentResult, AgentStep } from './types.js';
import type { ToolRegistry } from '../tools/registry.js';

const DEFAULT_SYSTEM_PROMPT = `你是一个功能强大的 AI 助手，配备了多种工具来帮助用户完成任务。

可用工具：
- get_weather：查询城市天气
- web_search：搜索互联网信息
- calculator：执行数学计算
- get_datetime：获取当前日期时间

工作原则：
1. 分析用户需求，判断是否需要工具辅助
2. 优先使用工具获取准确实时信息，不要凭知识猜测
3. 多个独立的工具调用可以同时发起，提高效率
4. 工具执行失败时，向用户说明原因
5. 最终回答要简洁、准确、有帮助`;

export class Agent {
  private client: OpenAI;
  private model: string;
  private systemPrompt: string;
  private maxIterations: number;
  private registry: ToolRegistry;
  private conversationHistory: OpenAI.Chat.Completions.ChatCompletionMessageParam[];
  private onStep?: (step: AgentStep) => void;

  constructor(registry: ToolRegistry, config: AgentConfig = {}) {
    this.client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    this.model = config.model ?? 'gpt-4o';
    this.systemPrompt = config.systemPrompt ?? DEFAULT_SYSTEM_PROMPT;
    this.maxIterations = config.maxIterations ?? 10;
    this.registry = registry;
    this.conversationHistory = [];
    this.onStep = config.onStep;
  }

  async run(userMessage: string): Promise<AgentResult> {
    const startTime = Date.now();
    const steps: AgentStep[] = [];
    let totalToolCalls = 0;

    const recordStep = (step: AgentStep): void => {
      steps.push(step);
      this.onStep?.(step);
    };

    recordStep({
      type: 'user_message',
      content: userMessage,
      timestamp: Date.now(),
    });

    // 构建完整消息列表
    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
      { role: 'system', content: this.systemPrompt },
      ...this.conversationHistory,
      { role: 'user', content: userMessage },
    ];

    let finalAnswer = '';
    let iteration = 0;

    while (iteration < this.maxIterations) {
      iteration++;

      let response: OpenAI.Chat.Completions.ChatCompletion;
      try {
        response = await this.client.chat.completions.create({
          model: this.model,
          messages,
          tools: this.registry.getDefinitions(),
          tool_choice: 'auto',
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        finalAnswer = `抱歉，调用 AI 服务时发生错误：${message}`;
        recordStep({
          type: 'final_answer',
          content: finalAnswer,
          totalSteps: steps.length,
          timestamp: Date.now(),
        });
        break;
      }

      const choice = response.choices[0];
      const message = choice.message;
      const finishReason = choice.finish_reason;

      messages.push(message);

      // 任务完成
      if (finishReason === 'stop' || !message.tool_calls?.length) {
        finalAnswer = message.content ?? '';
        recordStep({
          type: 'final_answer',
          content: finalAnswer,
          totalSteps: steps.length,
          timestamp: Date.now(),
        });
        break;
      }

      // 记录思考内容（如果有）
      if (message.content?.trim()) {
        recordStep({
          type: 'thinking',
          content: message.content,
          timestamp: Date.now(),
        });
      }

      // 并发执行所有工具调用
      totalToolCalls += message.tool_calls.length;

      const toolResults = await Promise.all(
        message.tool_calls.map(async (tc) => {
          const callStart = Date.now();
          let parsedArgs: unknown;
          try {
            parsedArgs = JSON.parse(tc.function.arguments);
          } catch {
            parsedArgs = tc.function.arguments;
          }

          const result = await this.registry.execute(tc);
          const durationMs = Date.now() - callStart;

          recordStep({
            type: 'tool_call',
            name: tc.function.name,
            args: parsedArgs,
            result,
            durationMs,
            timestamp: Date.now(),
          });

          return { id: tc.id, result };
        })
      );

      for (const { id, result } of toolResults) {
        messages.push({ role: 'tool', tool_call_id: id, content: result });
      }
    }

    if (!finalAnswer) {
      finalAnswer = `任务处理超时，已达到最大迭代次数（${this.maxIterations}）。`;
    }

    // 更新对话历史（只保留用户消息和最终回答，避免历史过长）
    this.conversationHistory.push(
      { role: 'user', content: userMessage },
      { role: 'assistant', content: finalAnswer }
    );

    // 限制历史长度（保留最近 20 轮）
    if (this.conversationHistory.length > 40) {
      this.conversationHistory = this.conversationHistory.slice(-40);
    }

    return {
      answer: finalAnswer,
      steps,
      totalToolCalls,
      durationMs: Date.now() - startTime,
    };
  }

  reset(): void {
    this.conversationHistory = [];
  }

  getConversationHistory(): ReadonlyArray<OpenAI.Chat.Completions.ChatCompletionMessageParam> {
    return this.conversationHistory;
  }
}
```

### 入口与演示

```typescript
// src/main.ts
import * as dotenv from 'dotenv';
dotenv.config();

import { Agent } from './agent/agent.js';
import { createDefaultRegistry } from './tools/registry.js';
import type { AgentStep } from './agent/types.js';

// 美化步骤输出
function printStep(step: AgentStep): void {
  const time = new Date(step.timestamp).toLocaleTimeString('zh-CN');

  switch (step.type) {
    case 'user_message':
      console.log(`\n[${time}] 用户: ${step.content}`);
      console.log('─'.repeat(60));
      break;
    case 'thinking':
      console.log(`[${time}] 思考: ${step.content}`);
      break;
    case 'tool_call':
      console.log(`[${time}] 工具: ${step.name}(${JSON.stringify(step.args)})`);
      console.log(`        结果: ${step.result.slice(0, 100)}${step.result.length > 100 ? '...' : ''} (${step.durationMs}ms)`);
      break;
    case 'final_answer':
      console.log(`\n[${time}] 助手: ${step.content}`);
      console.log(`\n[统计] 共 ${step.totalSteps} 步完成任务`);
      break;
  }
}

async function main(): Promise<void> {
  const registry = createDefaultRegistry();

  console.log(`已注册 ${registry.size()} 个工具`);
  console.log('工具列表:', registry.getDefinitions().map((t) => t.function.name).join(', '));
  console.log('='.repeat(60));

  const agent = new Agent(registry, {
    model: 'gpt-4o',
    onStep: printStep,
  });

  // 示例 1：需要天气工具
  const result1 = await agent.run('现在北京和上海的天气分别是什么？哪个城市更适合今天出行？');
  console.log(`\n[耗时] ${result1.durationMs}ms，工具调用 ${result1.totalToolCalls} 次`);

  console.log('\n' + '='.repeat(60));

  // 示例 2：需要计算器工具
  const result2 = await agent.run('如果我每天跑步 5 公里，一年下来能跑多少公里？相当于绕地球多少圈？（地球周长约 40075 公里）');
  console.log(`\n[耗时] ${result2.durationMs}ms，工具调用 ${result2.totalToolCalls} 次`);

  console.log('\n' + '='.repeat(60));

  // 示例 3：多轮对话（Agent 记住上下文）
  const result3 = await agent.run('现在几点了？今天是星期几？');
  const result4 = await agent.run('那明天是星期几？');  // Agent 知道今天的信息
  console.log(`\n[多轮] 第一轮耗时 ${result3.durationMs}ms，第二轮耗时 ${result4.durationMs}ms`);
  console.log(`\n[对话历史] 共 ${agent.getConversationHistory().length} 条消息`);
}

main().catch(console.error);
```

### 配置文件

```json
{
  "name": "ai-agent",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "start": "tsx src/main.ts",
    "dev": "tsx watch src/main.ts",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "dotenv": "^16.4.5",
    "openai": "^4.72.0"
  },
  "devDependencies": {
    "@types/node": "^22.10.2",
    "tsx": "^4.19.2",
    "typescript": "^5.7.2"
  }
}
```

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"]
}
```

### 运行项目

```bash
# 安装依赖
npm install

# 配置环境变量
echo "OPENAI_API_KEY=your_api_key_here" > .env

# 运行 Agent
npm start
```

---

## 练习题

### 基础题

**题目 21-1**：实现一个 `validateToolDefinition` 函数，验证工具定义的正确性。要求检查：
- `type` 字段必须为 `"function"`
- `function.name` 只能包含字母、数字和下划线，且不超过 64 个字符
- `function.description` 不能为空
- `function.parameters.type` 必须为 `"object"`
- `function.parameters.required`（如果存在）中的每个字段都必须在 `properties` 中有对应定义

函数签名：
```typescript
function validateToolDefinition(
  tool: OpenAI.Chat.Completions.ChatCompletionTool
): { valid: boolean; errors: string[] }
```

**题目 21-2**：基于本章的 `ToolRegistry`，实现一个 `withLogging` 装饰器函数，包装 `ToolRegistry`，在每次工具执行前后打印日志：

```typescript
function withLogging(registry: ToolRegistry): ToolRegistry
// 每次执行工具时，打印：
// [工具调用] 工具名: get_weather，参数: {"city":"北京"}
// [工具完成] 工具名: get_weather，耗时: 23ms，结果长度: 156 字符
// [工具失败] 工具名: xxx，耗时: 5ms，错误: 工具不存在
```

### 中级题

**题目 21-3**：为 `Agent` 类添加**工具调用重试**能力。当工具执行返回包含 `"error"` 字段的 JSON 时，自动重试最多 2 次（使用指数退避），并在所有重试失败后将错误信息回传给模型。修改 `Agent.run()` 方法，不改变其对外接口。

**题目 21-4**：实现一个 `ConversationSummarizer`，当对话历史超过指定长度（token 估算超过 4000）时，自动调用 LLM 将旧历史摘要为一条 `system` 消息，以节省 context 窗口空间。要求：

```typescript
class ConversationSummarizer {
  constructor(
    private maxEstimatedTokens: number,
    private client: OpenAI
  ) {}

  async maybeCompress(
    history: OpenAI.Chat.Completions.ChatCompletionMessageParam[]
  ): Promise<OpenAI.Chat.Completions.ChatCompletionMessageParam[]>
  // 如果 token 估算未超限，原样返回
  // 如果超限，将前半段历史摘要为一条 system 消息，与后半段拼接返回
}
```

### 提高题

**题目 21-5**：实现一个支持**工具调用链路图**可视化的 `AgentTracer`，分析 `AgentResult.steps`，输出工具调用的有向图（ASCII 文本格式）。图中每个节点代表一个工具调用或模型推理步骤，边代表数据依赖关系（一个工具的输出是另一个工具或最终回答的输入）。

```typescript
class AgentTracer {
  analyze(result: AgentResult): TraceGraph
  renderAscii(graph: TraceGraph): string
  getSummary(result: AgentResult): TraceSummary
}

interface TraceSummary {
  totalSteps: number;
  toolCallCount: number;
  uniqueToolsUsed: string[];
  averageToolDurationMs: number;
  longestToolCall: { name: string; durationMs: number } | null;
  parallelCallGroups: number;  // 同时发起的工具调用组数
}
```

---

## 练习答案

### 题目 21-1 答案

```typescript
import OpenAI from 'openai';

function validateToolDefinition(
  tool: OpenAI.Chat.Completions.ChatCompletionTool
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // 检查 type 字段
  if (tool.type !== 'function') {
    errors.push(`type 必须为 "function"，当前值为 "${tool.type}"`);
  }

  const fn = tool.function;

  // 检查 name
  if (!fn.name) {
    errors.push('function.name 不能为空');
  } else {
    if (!/^[a-zA-Z0-9_]+$/.test(fn.name)) {
      errors.push(`function.name "${fn.name}" 只能包含字母、数字和下划线`);
    }
    if (fn.name.length > 64) {
      errors.push(`function.name 长度不能超过 64 个字符（当前 ${fn.name.length}）`);
    }
  }

  // 检查 description
  if (!fn.description || fn.description.trim() === '') {
    errors.push('function.description 不能为空');
  }

  // 检查 parameters
  if (fn.parameters) {
    const params = fn.parameters as Record<string, unknown>;

    if (params['type'] !== 'object') {
      errors.push(`function.parameters.type 必须为 "object"，当前值为 "${params['type']}"`);
    }

    // 检查 required 中的字段是否在 properties 中定义
    const required = params['required'] as string[] | undefined;
    const properties = params['properties'] as Record<string, unknown> | undefined;

    if (required && Array.isArray(required)) {
      if (!properties) {
        errors.push('function.parameters.required 存在但 properties 未定义');
      } else {
        for (const field of required) {
          if (!(field in properties)) {
            errors.push(
              `required 中的字段 "${field}" 在 properties 中未找到定义`
            );
          }
        }
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

// 使用示例
const result = validateToolDefinition({
  type: 'function',
  function: {
    name: 'get weather', // 包含空格，非法
    description: '',     // 空描述，非法
    parameters: {
      type: 'object',
      properties: { city: { type: 'string' } },
      required: ['city', 'unit'], // unit 未在 properties 中定义
    },
  },
});

console.log(result);
// {
//   valid: false,
//   errors: [
//     'function.name "get weather" 只能包含字母、数字和下划线',
//     'function.description 不能为空',
//     'required 中的字段 "unit" 在 properties 中未找到定义'
//   ]
// }
```

### 题目 21-2 答案

```typescript
import OpenAI from 'openai';
import { ToolRegistry } from './tools/registry.js';

function withLogging(registry: ToolRegistry): ToolRegistry {
  // 创建代理，拦截 execute 方法
  const loggingRegistry = new Proxy(registry, {
    get(target, prop) {
      if (prop !== 'execute') {
        return Reflect.get(target, prop);
      }

      // 返回包装后的 execute 方法
      return async (
        toolCall: OpenAI.Chat.Completions.ChatCompletionMessageToolCall
      ): Promise<string> => {
        const name = toolCall.function.name;
        let parsedArgs: unknown;
        try {
          parsedArgs = JSON.parse(toolCall.function.arguments);
        } catch {
          parsedArgs = toolCall.function.arguments;
        }

        console.log(`[工具调用] 工具名: ${name}，参数: ${JSON.stringify(parsedArgs)}`);
        const startTime = Date.now();

        try {
          const result = await target.execute(toolCall);
          const durationMs = Date.now() - startTime;

          // 检查结果是否包含错误
          let hasError = false;
          try {
            const parsed = JSON.parse(result) as Record<string, unknown>;
            hasError = 'error' in parsed;
          } catch {
            // 结果不是 JSON，视为成功
          }

          if (hasError) {
            console.log(
              `[工具失败] 工具名: ${name}，耗时: ${durationMs}ms，错误: ${result}`
            );
          } else {
            console.log(
              `[工具完成] 工具名: ${name}，耗时: ${durationMs}ms，结果长度: ${result.length} 字符`
            );
          }

          return result;
        } catch (error) {
          const durationMs = Date.now() - startTime;
          const message = error instanceof Error ? error.message : String(error);
          console.log(
            `[工具异常] 工具名: ${name}，耗时: ${durationMs}ms，错误: ${message}`
          );
          throw error;
        }
      };
    },
  });

  return loggingRegistry;
}

// 使用示例
import { createDefaultRegistry } from './tools/registry.js';

const baseRegistry = createDefaultRegistry();
const loggedRegistry = withLogging(baseRegistry);

// 现在每次 execute 调用都会打印日志
const agent = new Agent(loggedRegistry);
await agent.run('北京今天天气怎么样？');
// [工具调用] 工具名: get_weather，参数: {"city":"北京"}
// [工具完成] 工具名: get_weather，耗时: 12ms，结果长度: 183 字符
```

### 题目 21-3 答案

```typescript
// 在 Agent 类中修改 execute 方法的调用部分

// 带重试的工具执行辅助函数
async function executeWithRetry(
  registry: ToolRegistry,
  toolCall: OpenAI.Chat.Completions.ChatCompletionMessageToolCall,
  maxRetries: number = 2
): Promise<string> {
  let lastResult = '';

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (attempt > 0) {
      // 指数退避：500ms、1000ms
      const delayMs = 500 * Math.pow(2, attempt - 1);
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      console.log(`[重试] ${toolCall.function.name} 第 ${attempt} 次重试...`);
    }

    const result = await registry.execute(toolCall);
    lastResult = result;

    // 检查结果是否包含错误
    let hasError = false;
    try {
      const parsed = JSON.parse(result) as Record<string, unknown>;
      hasError = typeof parsed['error'] === 'string';
    } catch {
      // 结果不是 JSON，视为成功
      hasError = false;
    }

    if (!hasError) {
      return result; // 成功，提前返回
    }

    if (attempt < maxRetries) {
      console.warn(
        `[工具错误] ${toolCall.function.name} 返回错误，将重试：${result}`
      );
    }
  }

  // 所有重试失败，返回最后一次的错误结果（包含错误信息回传给模型）
  console.error(`[工具失败] ${toolCall.function.name} 经过 ${maxRetries + 1} 次尝试仍然失败`);
  return lastResult;
}

// 在 Agent.run() 中替换原有的 registry.execute 调用：
// 原来：const result = await this.registry.execute(tc);
// 改为：const result = await executeWithRetry(this.registry, tc, 2);
```

### 题目 21-4 答案

```typescript
import OpenAI from 'openai';

type MessageParam = OpenAI.Chat.Completions.ChatCompletionMessageParam;

// 简单 token 估算：约 4 字符 = 1 token
function estimateTokens(messages: MessageParam[]): number {
  const totalChars = messages.reduce((sum, msg) => {
    const content =
      typeof msg.content === 'string'
        ? msg.content
        : Array.isArray(msg.content)
          ? msg.content.map((b) => (typeof b === 'string' ? b : '')).join('')
          : '';
    return sum + content.length;
  }, 0);
  return Math.ceil(totalChars / 4);
}

class ConversationSummarizer {
  constructor(
    private maxEstimatedTokens: number,
    private client: OpenAI
  ) {}

  async maybeCompress(history: MessageParam[]): Promise<MessageParam[]> {
    const estimated = estimateTokens(history);

    if (estimated <= this.maxEstimatedTokens) {
      return history;  // 未超限，无需压缩
    }

    // 将前半段历史发给 LLM 摘要
    const midpoint = Math.floor(history.length / 2);
    const oldHistory = history.slice(0, midpoint);
    const recentHistory = history.slice(midpoint);

    console.log(
      `[摘要] 历史消息过长（估算 ${estimated} tokens），正在压缩前 ${midpoint} 条消息...`
    );

    const summaryText = await this.summarizeHistory(oldHistory);

    console.log(`[摘要] 压缩完成，生成摘要: ${summaryText.slice(0, 80)}...`);

    // 将摘要注入为 system 消息，拼接到最近历史前面
    const summaryMessage: MessageParam = {
      role: 'system',
      content: `以下是之前对话的摘要：\n${summaryText}`,
    };

    return [summaryMessage, ...recentHistory];
  }

  private async summarizeHistory(history: MessageParam[]): Promise<string> {
    const historyText = history
      .map((msg) => {
        const role = msg.role === 'user' ? '用户' : '助手';
        const content =
          typeof msg.content === 'string' ? msg.content : '[非文本内容]';
        return `${role}: ${content}`;
      })
      .join('\n');

    const response = await this.client.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content:
            '请将以下对话历史摘要为一段简洁的文字（不超过 300 字），保留关键信息、决策和结论。',
        },
        { role: 'user', content: historyText },
      ],
      max_tokens: 400,
    });

    return response.choices[0].message.content ?? '（摘要生成失败）';
  }
}

// 在 Agent 类中使用：
// const summarizer = new ConversationSummarizer(4000, this.client);
// this.conversationHistory = await summarizer.maybeCompress(this.conversationHistory);
```

### 题目 21-5 答案

```typescript
import type { AgentResult, AgentStep } from './agent/types.js';

interface TraceNode {
  id: string;
  label: string;
  type: 'user' | 'thinking' | 'tool' | 'answer';
  durationMs?: number;
}

interface TraceEdge {
  from: string;
  to: string;
}

interface TraceGraph {
  nodes: TraceNode[];
  edges: TraceEdge[];
}

interface TraceSummary {
  totalSteps: number;
  toolCallCount: number;
  uniqueToolsUsed: string[];
  averageToolDurationMs: number;
  longestToolCall: { name: string; durationMs: number } | null;
  parallelCallGroups: number;
}

class AgentTracer {
  analyze(result: AgentResult): TraceGraph {
    const nodes: TraceNode[] = [];
    const edges: TraceEdge[] = [];
    let prevNodeId: string | null = null;

    result.steps.forEach((step, index) => {
      const id = `node_${index}`;

      switch (step.type) {
        case 'user_message':
          nodes.push({ id, label: `用户: ${step.content.slice(0, 30)}...`, type: 'user' });
          break;
        case 'thinking':
          nodes.push({ id, label: `思考: ${step.content.slice(0, 30)}...`, type: 'thinking' });
          break;
        case 'tool_call':
          nodes.push({
            id,
            label: `工具: ${step.name}`,
            type: 'tool',
            durationMs: step.durationMs,
          });
          break;
        case 'final_answer':
          nodes.push({ id, label: `回答: ${step.content.slice(0, 30)}...`, type: 'answer' });
          break;
      }

      if (prevNodeId !== null) {
        edges.push({ from: prevNodeId, to: id });
      }
      prevNodeId = id;
    });

    return { nodes, edges };
  }

  renderAscii(graph: TraceGraph): string {
    const lines: string[] = ['流程图：'];

    for (const node of graph.nodes) {
      const prefix =
        node.type === 'user' ? '[用户]'
        : node.type === 'thinking' ? '[思考]'
        : node.type === 'tool' ? `[工具${node.durationMs ? ` ${node.durationMs}ms` : ''}]`
        : '[回答]';

      lines.push(`  ${prefix} ${node.label}`);

      // 是否有下一个节点
      const hasNext = graph.edges.some((e) => e.from === node.id);
      if (hasNext) {
        lines.push('    ↓');
      }
    }

    return lines.join('\n');
  }

  getSummary(result: AgentResult): TraceSummary {
    const toolSteps = result.steps.filter(
      (s): s is Extract<AgentStep, { type: 'tool_call' }> => s.type === 'tool_call'
    );

    const uniqueToolsUsed = [...new Set(toolSteps.map((s) => s.name))];
    const totalDuration = toolSteps.reduce((sum, s) => sum + s.durationMs, 0);
    const averageToolDurationMs =
      toolSteps.length > 0 ? Math.round(totalDuration / toolSteps.length) : 0;

    const longestToolCall =
      toolSteps.length > 0
        ? toolSteps.reduce((max, s) => (s.durationMs > max.durationMs ? s : max))
        : null;

    // 统计并行调用组：相邻的工具调用之间没有"思考"或"回答"步骤则视为同组
    let parallelGroups = 0;
    let inToolGroup = false;
    for (const step of result.steps) {
      if (step.type === 'tool_call') {
        if (!inToolGroup) {
          parallelGroups++;
          inToolGroup = true;
        }
      } else {
        inToolGroup = false;
      }
    }

    return {
      totalSteps: result.steps.length,
      toolCallCount: toolSteps.length,
      uniqueToolsUsed,
      averageToolDurationMs,
      longestToolCall: longestToolCall
        ? { name: longestToolCall.name, durationMs: longestToolCall.durationMs }
        : null,
      parallelCallGroups: parallelGroups,
    };
  }
}

// 使用示例
async function tracerDemo() {
  const registry = createDefaultRegistry();
  const agent = new Agent(registry, { model: 'gpt-4o' });

  const result = await agent.run('北京和上海今天天气分别如何？');

  const tracer = new AgentTracer();
  const graph = tracer.analyze(result);
  const summary = tracer.getSummary(result);

  console.log('\n' + tracer.renderAscii(graph));
  console.log('\n调用摘要:', summary);
}
```
