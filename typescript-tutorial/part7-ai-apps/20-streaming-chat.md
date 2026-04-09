# 第20章：流式对话实现

## 学习目标

完成本章学习后，你将能够：

1. 理解流式 API 的工作原理，掌握 `stream: true` 参数的使用方式和底层机制
2. 正确解析 SSE delta 事件数据，提取增量文本内容并累积完整响应
3. 使用 TypeScript 实现流畅的打字机效果，处理速率控制与 UI 更新
4. 使用 `AbortController` 实现流式请求的中断与取消，优雅处理用户停止操作
5. 设计并管理流式对话的完整状态机，包括 loading、streaming、done、error 等状态

---

## 20.1 流式 API 调用（stream: true）

### 为什么需要流式调用

大语言模型生成文本是一个逐 token 产出的过程。非流式调用必须等待模型生成完整响应后才能返回，对于较长的回复可能需要等待数十秒，用户体验极差。流式调用让模型边生成边传输，用户可以立即看到开始的文字，感知响应速度大幅提升。

两种模式的对比：

| 特性 | 非流式（普通调用） | 流式（stream: true） |
|------|------------------|---------------------|
| 首字节时间 | 需等待全部生成完毕 | 极短，几乎立即开始 |
| 用户感知延迟 | 高 | 低 |
| 内存消耗 | 一次性加载全部 | 逐块处理 |
| 实现复杂度 | 低 | 中 |
| 适用场景 | 批处理、后台任务 | 聊天界面、实时展示 |

### OpenAI SDK 的流式调用

OpenAI Node.js SDK 提供两种流式调用方式：

**方式一：使用 `stream: true` 参数**

```typescript
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function streamWithRawParam(): Promise<void> {
  const stream = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    stream: true,
    messages: [
      { role: 'user', content: '用三句话介绍TypeScript' }
    ],
  });

  // stream 是 AsyncIterable<ChatCompletionChunk>
  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content ?? '';
    process.stdout.write(delta);
  }

  console.log('\n[流式传输完成]');
}

streamWithRawParam().catch(console.error);
```

**方式二：使用 SDK 的 `.stream()` 辅助方法（推荐）**

```typescript
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function streamWithHelper(): Promise<void> {
  // .stream() 返回 Stream 对象，提供更多辅助方法
  const stream = client.beta.chat.completions.stream({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: '用三句话介绍TypeScript' }],
  });

  // 监听文本增量事件
  stream.on('content', (delta: string) => {
    process.stdout.write(delta);
  });

  // 等待完整响应
  const finalMessage = await stream.finalMessage();
  console.log('\n\n完整消息:', finalMessage.choices[0].message.content);
}

streamWithHelper().catch(console.error);
```

### Anthropic SDK 的流式调用

```typescript
import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

async function anthropicStream(): Promise<void> {
  // 方式一：stream() 辅助方法
  const stream = await client.messages.stream({
    model: 'claude-3-5-haiku-20241022',
    max_tokens: 1024,
    messages: [{ role: 'user', content: '解释什么是TypeScript泛型' }],
  });

  for await (const event of stream) {
    if (
      event.type === 'content_block_delta' &&
      event.delta.type === 'text_delta'
    ) {
      process.stdout.write(event.delta.text);
    }
  }

  // 方式二：使用 .on() 事件监听
  const stream2 = client.messages.stream({
    model: 'claude-3-5-haiku-20241022',
    max_tokens: 1024,
    messages: [{ role: 'user', content: '你好' }],
  });

  stream2.on('text', (text: string) => {
    process.stdout.write(text);
  });

  await stream2.finalMessage();
}
```

### 流式响应的底层协议

流式 API 基于 HTTP 的 Server-Sent Events（SSE）协议。每个 chunk 格式如下：

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":" World"},"index":0}]}

data: [DONE]
```

SDK 已封装了底层 SSE 解析，开发者直接使用 `for await...of` 迭代 chunk 即可。

---

## 20.2 Delta 内容解析

### ChatCompletionChunk 的类型结构

理解 chunk 的 TypeScript 类型结构是正确解析的基础：

```typescript
import type { ChatCompletionChunk } from 'openai/resources';

// 核心类型结构（简化版）
interface ChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: 'assistant' | 'user' | 'system';
      content?: string | null;
      tool_calls?: Array<{
        index: number;
        id?: string;
        type?: 'function';
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
    };
    finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | null;
    logprobs: null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}
```

### 安全提取 delta 内容

由于 delta 中的字段大多是可选的，需要防御性地提取：

```typescript
import OpenAI from 'openai';
import type { ChatCompletionChunk } from 'openai/resources';

// 提取文本 delta（最常见场景）
function extractTextDelta(chunk: ChatCompletionChunk): string {
  return chunk.choices[0]?.delta?.content ?? '';
}

// 提取工具调用 delta（Function Calling 场景）
function extractToolCallDelta(chunk: ChatCompletionChunk) {
  const toolCalls = chunk.choices[0]?.delta?.tool_calls;
  if (!toolCalls || toolCalls.length === 0) return null;

  return toolCalls.map((tc) => ({
    index: tc.index,
    id: tc.id,
    name: tc.function?.name,
    args: tc.function?.arguments,
  }));
}

// 检查流是否结束
function isStreamDone(chunk: ChatCompletionChunk): boolean {
  const finishReason = chunk.choices[0]?.finish_reason;
  return finishReason !== null && finishReason !== undefined;
}

// 获取结束原因
type FinishReason = 'stop' | 'length' | 'tool_calls' | 'content_filter';

function getFinishReason(chunk: ChatCompletionChunk): FinishReason | null {
  return chunk.choices[0]?.finish_reason as FinishReason | null;
}
```

### 累积完整响应

实际应用中需要将所有 delta 累积成完整文本：

```typescript
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

interface StreamAccumulator {
  content: string;
  finishReason: string | null;
  promptTokens: number;
  completionTokens: number;
}

async function streamWithAccumulation(
  prompt: string
): Promise<StreamAccumulator> {
  const accumulator: StreamAccumulator = {
    content: '',
    finishReason: null,
    promptTokens: 0,
    completionTokens: 0,
  };

  const stream = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    stream: true,
    stream_options: { include_usage: true }, // 请求包含 token 用量
    messages: [{ role: 'user', content: prompt }],
  });

  for await (const chunk of stream) {
    // 累积文本内容
    const delta = chunk.choices[0]?.delta?.content;
    if (delta) {
      accumulator.content += delta;
    }

    // 记录结束原因
    const finishReason = chunk.choices[0]?.finish_reason;
    if (finishReason) {
      accumulator.finishReason = finishReason;
    }

    // 记录 token 用量（通常在最后一个 chunk）
    if (chunk.usage) {
      accumulator.promptTokens = chunk.usage.prompt_tokens;
      accumulator.completionTokens = chunk.usage.completion_tokens;
    }
  }

  return accumulator;
}

// 使用示例
async function main() {
  const result = await streamWithAccumulation('用两句话介绍TypeScript');
  console.log('完整内容:', result.content);
  console.log('结束原因:', result.finishReason);
  console.log(`Token 用量: ${result.promptTokens} + ${result.completionTokens}`);
}

main().catch(console.error);
```

### 处理多 choices 场景

当 `n > 1` 时，每个 chunk 可能包含多个 choice 的 delta：

```typescript
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function streamMultipleChoices(): Promise<string[]> {
  const n = 3;
  const contents: string[] = new Array(n).fill('');

  const stream = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    stream: true,
    n,
    messages: [{ role: 'user', content: '给我一个随机数字' }],
  });

  for await (const chunk of stream) {
    for (const choice of chunk.choices) {
      const delta = choice.delta?.content;
      if (delta) {
        contents[choice.index] += delta;
      }
    }
  }

  return contents;
}
```

---

## 20.3 打字机效果实现

### 基础打字机效果

打字机效果的核心思路是：控制文本显示速度，逐字符渲染，营造"正在输入"的视觉感受。

```typescript
// 基础打字机：逐字符延迟输出
async function typewriterEffect(
  text: string,
  delay: number = 30
): Promise<void> {
  for (const char of text) {
    process.stdout.write(char);
    await sleep(delay);
  }
  console.log();
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
```

### 与真实流式响应结合

实际场景中，流式响应的速率是不均匀的。我们需要一个缓冲队列来平滑显示速率：

```typescript
class TypewriterStream {
  private buffer: string = '';
  private isOutputting: boolean = false;
  private outputDelay: number;
  private onChar?: (char: string) => void;
  private onDone?: () => void;

  constructor(options: {
    outputDelay?: number;
    onChar?: (char: string) => void;
    onDone?: () => void;
  } = {}) {
    this.outputDelay = options.outputDelay ?? 20;
    this.onChar = options.onChar;
    this.onDone = options.onDone;
  }

  // 接收新的 delta 文本，加入缓冲队列
  push(text: string): void {
    this.buffer += text;
    if (!this.isOutputting) {
      this.startOutputting();
    }
  }

  // 通知流已结束
  end(): void {
    // 等待缓冲区清空后触发 onDone
    const checkDone = () => {
      if (this.buffer.length === 0 && !this.isOutputting) {
        this.onDone?.();
      } else {
        setTimeout(checkDone, this.outputDelay);
      }
    };
    checkDone();
  }

  private async startOutputting(): Promise<void> {
    this.isOutputting = true;

    while (this.buffer.length > 0) {
      const char = this.buffer[0];
      this.buffer = this.buffer.slice(1);

      // 触发字符输出回调
      this.onChar?.(char);

      await sleep(this.outputDelay);
    }

    this.isOutputting = false;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// 使用示例：Node.js 终端打字机效果
async function terminalTypewriter(prompt: string): Promise<void> {
  const client = new (await import('openai')).default({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const typewriter = new TypewriterStream({
    outputDelay: 25,
    onChar: (char) => process.stdout.write(char),
    onDone: () => console.log('\n\n[输出完成]'),
  });

  const stream = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    stream: true,
    messages: [{ role: 'user', content: prompt }],
  });

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content ?? '';
    if (delta) {
      typewriter.push(delta);
    }
  }

  typewriter.end();

  // 等待打字机完成
  await sleep(3000);
}
```

### React 组件中的打字机效果

在前端框架中，打字机效果通过状态更新驱动 DOM 渲染：

```typescript
import { useState, useRef, useCallback } from 'react';

interface UseTypewriterReturn {
  displayText: string;
  isTyping: boolean;
  push: (text: string) => void;
  reset: () => void;
}

// 自定义 Hook：流式打字机
function useTypewriter(delay: number = 20): UseTypewriterReturn {
  const [displayText, setDisplayText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const bufferRef = useRef<string>('');
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startTimer = useCallback(() => {
    if (timerRef.current) return;

    setIsTyping(true);
    timerRef.current = setInterval(() => {
      if (bufferRef.current.length > 0) {
        const char = bufferRef.current[0];
        bufferRef.current = bufferRef.current.slice(1);
        setDisplayText((prev) => prev + char);
      } else {
        // 缓冲区为空，暂停定时器
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
          setIsTyping(false);
        }
      }
    }, delay);
  }, [delay]);

  const push = useCallback(
    (text: string) => {
      bufferRef.current += text;
      startTimer();
    },
    [startTimer]
  );

  const reset = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    bufferRef.current = '';
    setDisplayText('');
    setIsTyping(false);
  }, []);

  return { displayText, isTyping, push, reset };
}

// 使用该 Hook 的组件示例
function StreamingMessage({ content }: { content: string }) {
  return (
    <div className="message">
      <span>{content}</span>
      <span className="cursor">|</span>
    </div>
  );
}
```

### 速率自适应策略

当模型响应极快（本地模型、缓存命中）时，固定延迟会导致用户等待过长。可以实现自适应速率：

```typescript
class AdaptiveTypewriter {
  private buffer: string = '';
  private baseDelay: number;
  private minDelay: number;
  private maxDelay: number;

  constructor(options: {
    baseDelay?: number;
    minDelay?: number;
    maxDelay?: number;
  } = {}) {
    this.baseDelay = options.baseDelay ?? 25;
    this.minDelay = options.minDelay ?? 5;
    this.maxDelay = options.maxDelay ?? 80;
  }

  // 根据缓冲区大小动态调整速度
  // 缓冲区越大，输出越快，避免积压
  private getCurrentDelay(): number {
    const bufferSize = this.buffer.length;

    if (bufferSize > 100) return this.minDelay;      // 积压严重，加速
    if (bufferSize > 50) return this.baseDelay / 2;  // 轻度积压，适当加速
    if (bufferSize < 5) return this.maxDelay;         // 缓冲区空，减速等待
    return this.baseDelay;                            // 正常速率
  }

  push(text: string): void {
    this.buffer += text;
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<string> {
    while (this.buffer.length > 0) {
      const char = this.buffer[0];
      this.buffer = this.buffer.slice(1);
      yield char;
      await new Promise((resolve) =>
        setTimeout(resolve, this.getCurrentDelay())
      );
    }
  }
}
```

---

## 20.4 流式中断与取消（AbortController）

### AbortController 基础

`AbortController` 是 Web 标准 API，用于取消异步操作。Node.js 16+ 和所有现代浏览器均原生支持：

```typescript
// 基础用法
const controller = new AbortController();
const { signal } = controller;

// 5 秒后自动取消
setTimeout(() => controller.abort('超时'), 5000);

// 在 fetch 中使用 signal
const response = await fetch('https://api.example.com/data', { signal });

// 手动取消
controller.abort('用户取消');
```

### 在 OpenAI SDK 中使用 AbortController

```typescript
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function cancellableStream(
  prompt: string,
  timeoutMs: number = 30000
): Promise<{ content: string; cancelled: boolean }> {
  const controller = new AbortController();

  // 设置超时自动取消
  const timeoutId = setTimeout(() => {
    controller.abort('请求超时');
  }, timeoutMs);

  let content = '';
  let cancelled = false;

  try {
    const stream = await client.chat.completions.create(
      {
        model: 'gpt-4o-mini',
        stream: true,
        messages: [{ role: 'user', content: prompt }],
      },
      { signal: controller.signal } // 传入 signal
    );

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content ?? '';
      content += delta;
      process.stdout.write(delta);
    }
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      cancelled = true;
      console.log('\n[流式传输已取消]');
    } else {
      throw error;
    }
  } finally {
    clearTimeout(timeoutId);
  }

  return { content, cancelled };
}
```

### 用户主动停止生成

在聊天应用中，用户点击"停止"按钮时需要取消正在进行的流式请求：

```typescript
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

class StreamingChatSession {
  private abortController: AbortController | null = null;

  // 检查是否有正在进行的流
  get isStreaming(): boolean {
    return this.abortController !== null;
  }

  // 停止当前流
  stop(): void {
    if (this.abortController) {
      this.abortController.abort('用户停止');
      this.abortController = null;
    }
  }

  async sendMessage(
    content: string,
    onDelta: (delta: string) => void,
    onComplete: (fullContent: string) => void,
    onError: (error: Error) => void
  ): Promise<void> {
    // 如果有进行中的流，先停止
    this.stop();

    this.abortController = new AbortController();
    let accumulated = '';

    try {
      const stream = await client.chat.completions.create(
        {
          model: 'gpt-4o-mini',
          stream: true,
          messages: [{ role: 'user', content }],
        },
        { signal: this.abortController.signal }
      );

      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content ?? '';
        if (delta) {
          accumulated += delta;
          onDelta(delta);
        }
      }

      onComplete(accumulated);
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          // 被取消不视为错误，使用已积累的部分内容
          onComplete(accumulated);
        } else {
          onError(error);
        }
      }
    } finally {
      this.abortController = null;
    }
  }
}
```

### 超时策略与重试

结合 `AbortController` 实现具备超时和重试能力的流式请求：

```typescript
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

interface StreamOptions {
  timeoutMs?: number;
  maxRetries?: number;
  retryDelayMs?: number;
}

async function robustStream(
  prompt: string,
  options: StreamOptions = {}
): Promise<string> {
  const {
    timeoutMs = 30000,
    maxRetries = 3,
    retryDelayMs = 1000,
  } = options;

  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort('超时'), timeoutMs);

    try {
      let content = '';

      const stream = await client.chat.completions.create(
        {
          model: 'gpt-4o-mini',
          stream: true,
          messages: [{ role: 'user', content: prompt }],
        },
        { signal: controller.signal }
      );

      for await (const chunk of stream) {
        content += chunk.choices[0]?.delta?.content ?? '';
      }

      clearTimeout(timeoutId);
      return content;
    } catch (error) {
      clearTimeout(timeoutId);
      lastError = error instanceof Error ? error : new Error(String(error));

      const isAbort = lastError.name === 'AbortError';
      const isRetryable =
        !isAbort &&
        (lastError.message.includes('timeout') ||
          lastError.message.includes('network') ||
          lastError.message.includes('500') ||
          lastError.message.includes('503'));

      if (!isRetryable || attempt === maxRetries) {
        throw lastError;
      }

      console.warn(`第 ${attempt} 次尝试失败，${retryDelayMs}ms 后重试...`);
      await new Promise((resolve) => setTimeout(resolve, retryDelayMs * attempt));
    }
  }

  throw lastError ?? new Error('未知错误');
}
```

---

## 20.5 流式进度与状态管理

### 流式对话的状态机

一个完整的流式对话包含多个状态，需要精确管理：

```typescript
// 流式消息的所有可能状态
type MessageStatus =
  | 'idle'       // 初始状态，未开始
  | 'loading'    // 已发送请求，等待首个 chunk
  | 'streaming'  // 正在接收 chunk
  | 'done'       // 流式传输完成
  | 'error'      // 发生错误
  | 'cancelled'; // 用户取消

interface StreamingMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  status: MessageStatus;
  error?: string;
  startedAt?: number;
  completedAt?: number;
  tokenCount?: number;
}

// 状态转换规则（有限状态机）
const validTransitions: Record<MessageStatus, MessageStatus[]> = {
  idle: ['loading'],
  loading: ['streaming', 'error', 'cancelled'],
  streaming: ['done', 'error', 'cancelled'],
  done: [],
  error: [],
  cancelled: [],
};

function canTransition(from: MessageStatus, to: MessageStatus): boolean {
  return validTransitions[from].includes(to);
}
```

### 使用 Zustand 管理对话状态

```typescript
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

interface ChatState {
  messages: StreamingMessage[];
  isLoading: boolean;
  addUserMessage: (content: string) => string;
  addAssistantMessage: () => string;
  updateMessageStatus: (id: string, status: MessageStatus) => void;
  appendMessageContent: (id: string, delta: string) => void;
  setMessageError: (id: string, error: string) => void;
  completeMessage: (id: string, tokenCount?: number) => void;
  clearMessages: () => void;
}

const useChatStore = create<ChatState>()(
  immer((set) => ({
    messages: [],
    isLoading: false,

    addUserMessage: (content: string): string => {
      const id = `msg_${Date.now()}_user`;
      set((state) => {
        state.messages.push({
          id,
          role: 'user',
          content,
          status: 'done',
          startedAt: Date.now(),
          completedAt: Date.now(),
        });
      });
      return id;
    },

    addAssistantMessage: (): string => {
      const id = `msg_${Date.now()}_assistant`;
      set((state) => {
        state.messages.push({
          id,
          role: 'assistant',
          content: '',
          status: 'loading',
          startedAt: Date.now(),
        });
        state.isLoading = true;
      });
      return id;
    },

    updateMessageStatus: (id: string, status: MessageStatus): void => {
      set((state) => {
        const msg = state.messages.find((m) => m.id === id);
        if (msg && canTransition(msg.status, status)) {
          msg.status = status;
        }
      });
    },

    appendMessageContent: (id: string, delta: string): void => {
      set((state) => {
        const msg = state.messages.find((m) => m.id === id);
        if (msg) {
          msg.content += delta;
          if (msg.status === 'loading') {
            msg.status = 'streaming';
          }
        }
      });
    },

    setMessageError: (id: string, error: string): void => {
      set((state) => {
        const msg = state.messages.find((m) => m.id === id);
        if (msg) {
          msg.status = 'error';
          msg.error = error;
        }
        state.isLoading = false;
      });
    },

    completeMessage: (id: string, tokenCount?: number): void => {
      set((state) => {
        const msg = state.messages.find((m) => m.id === id);
        if (msg) {
          msg.status = 'done';
          msg.completedAt = Date.now();
          if (tokenCount !== undefined) {
            msg.tokenCount = tokenCount;
          }
        }
        state.isLoading = false;
      });
    },

    clearMessages: (): void => {
      set((state) => {
        state.messages = [];
        state.isLoading = false;
      });
    },
  }))
);
```

### 流式进度指示器

提供实时的流式进度信息，提升用户体验：

```typescript
interface StreamProgress {
  charsReceived: number;
  wordsReceived: number;
  elapsedMs: number;
  charsPerSecond: number;
  estimatedTokens: number;
}

class StreamProgressTracker {
  private startTime: number;
  private content: string = '';
  private lastUpdateTime: number;
  private lastCharsCount: number = 0;

  constructor() {
    this.startTime = Date.now();
    this.lastUpdateTime = this.startTime;
  }

  update(delta: string): StreamProgress {
    this.content += delta;

    const now = Date.now();
    const elapsedMs = now - this.startTime;
    const intervalMs = now - this.lastUpdateTime;

    // 计算实时字符速率（基于最近更新间隔）
    const recentChars = this.content.length - this.lastCharsCount;
    const charsPerSecond =
      intervalMs > 0 ? (recentChars / intervalMs) * 1000 : 0;

    this.lastUpdateTime = now;
    this.lastCharsCount = this.content.length;

    return {
      charsReceived: this.content.length,
      wordsReceived: this.content.split(/\s+/).filter(Boolean).length,
      elapsedMs,
      charsPerSecond: Math.round(charsPerSecond),
      // 粗略估算：平均 4 个字符约 1 个 token
      estimatedTokens: Math.round(this.content.length / 4),
    };
  }

  getFinal(): StreamProgress & { totalMs: number } {
    const totalMs = Date.now() - this.startTime;
    const averageCharsPerSecond =
      totalMs > 0 ? (this.content.length / totalMs) * 1000 : 0;

    return {
      charsReceived: this.content.length,
      wordsReceived: this.content.split(/\s+/).filter(Boolean).length,
      elapsedMs: totalMs,
      totalMs,
      charsPerSecond: Math.round(averageCharsPerSecond),
      estimatedTokens: Math.round(this.content.length / 4),
    };
  }
}
```

### 流式错误处理与恢复

```typescript
import OpenAI from 'openai';

type StreamErrorType =
  | 'network'      // 网络中断
  | 'timeout'      // 请求超时
  | 'rate_limit'   // 频率限制
  | 'auth'         // 认证失败
  | 'server'       // 服务端错误
  | 'cancelled'    // 用户取消
  | 'unknown';

interface StreamError {
  type: StreamErrorType;
  message: string;
  retryable: boolean;
  retryAfterMs?: number;
}

function classifyStreamError(error: unknown): StreamError {
  if (error instanceof OpenAI.APIError) {
    if (error.status === 401) {
      return { type: 'auth', message: 'API 密钥无效', retryable: false };
    }
    if (error.status === 429) {
      const retryAfter = parseInt(
        error.headers?.['retry-after'] ?? '60',
        10
      );
      return {
        type: 'rate_limit',
        message: '请求频率超限',
        retryable: true,
        retryAfterMs: retryAfter * 1000,
      };
    }
    if (error.status >= 500) {
      return { type: 'server', message: '服务端错误', retryable: true };
    }
  }

  if (error instanceof Error) {
    if (error.name === 'AbortError') {
      return { type: 'cancelled', message: '请求已取消', retryable: false };
    }
    if (
      error.message.includes('network') ||
      error.message.includes('ECONNRESET') ||
      error.message.includes('ETIMEDOUT')
    ) {
      return { type: 'network', message: '网络连接中断', retryable: true };
    }
    if (error.message.includes('timeout')) {
      return { type: 'timeout', message: '请求超时', retryable: true };
    }
  }

  return {
    type: 'unknown',
    message: String(error),
    retryable: false,
  };
}
```

---

## 本章小结

| 知识点 | 核心 API / 概念 | 关键要点 |
|--------|----------------|---------|
| 流式调用 | `stream: true`，`for await...of` | 两种方式：raw 参数与 SDK 辅助方法 |
| Delta 解析 | `ChatCompletionChunk`，`delta.content` | 使用可选链防御性提取，注意 null 值 |
| 打字机效果 | `TypewriterStream`，`useTypewriter` | 缓冲队列平滑速率，自适应延迟控制 |
| 流式中断 | `AbortController`，`signal` | 传入 signal 参数，区分取消与真实错误 |
| 状态管理 | 有限状态机，Zustand + immer | 明确状态转换规则，防止非法状态跳转 |
| 进度追踪 | `StreamProgressTracker` | 实时字符速率，估算 token 用量 |
| 错误分类 | `StreamError`，错误类型枚举 | 区分可重试错误，合理设置重试策略 |

---

## AI 应用实战：实时聊天界面

本节实现一个完整的流式聊天应用，包含：Node.js 后端 API、前端状态管理、完整的错误处理和中断控制。

### 项目结构

```
streaming-chat/
├── src/
│   ├── server/
│   │   ├── index.ts          # Express 服务器
│   │   └── chat.ts           # 聊天路由处理器
│   ├── client/
│   │   ├── api.ts            # 前端 API 客户端
│   │   ├── store.ts          # 状态管理（Zustand）
│   │   ├── hooks.ts          # 自定义 Hooks
│   │   └── ChatApp.tsx       # 主组件
│   └── shared/
│       └── types.ts          # 共享类型定义
├── package.json
└── tsconfig.json
```

### 共享类型定义

```typescript
// src/shared/types.ts

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface StreamChunkPayload {
  delta: string;
  messageId: string;
}

export interface StreamDonePayload {
  messageId: string;
  totalTokens: number;
  finishReason: string;
}

export interface StreamErrorPayload {
  messageId: string;
  error: string;
  type: 'rate_limit' | 'auth' | 'server' | 'timeout' | 'unknown';
}

// SSE 事件类型
export type SSEEvent =
  | { type: 'chunk'; data: StreamChunkPayload }
  | { type: 'done'; data: StreamDonePayload }
  | { type: 'error'; data: StreamErrorPayload };
```

### 后端：流式 API 服务器

```typescript
// src/server/chat.ts
import { Router, Request, Response } from 'express';
import OpenAI from 'openai';
import type { ChatMessage, SSEEvent } from '../shared/types.js';

const router = Router();
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// 活跃请求映射：messageId -> AbortController
const activeRequests = new Map<string, AbortController>();

router.post('/chat/stream', async (req: Request, res: Response) => {
  const { messages, messageId } = req.body as {
    messages: ChatMessage[];
    messageId: string;
  };

  if (!messages || !messageId) {
    res.status(400).json({ error: '缺少必要参数' });
    return;
  }

  // 设置 SSE 响应头
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  // SSE 写入辅助函数
  const sendEvent = (event: SSEEvent): void => {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  };

  const controller = new AbortController();
  activeRequests.set(messageId, controller);

  // 客户端断开时取消请求
  req.on('close', () => {
    controller.abort('客户端断开连接');
    activeRequests.delete(messageId);
  });

  try {
    const stream = await client.chat.completions.create(
      {
        model: 'gpt-4o-mini',
        stream: true,
        stream_options: { include_usage: true },
        messages: messages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
      },
      { signal: controller.signal }
    );

    let totalTokens = 0;
    let finishReason = 'stop';

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content ?? '';

      if (delta) {
        sendEvent({
          type: 'chunk',
          data: { delta, messageId },
        });
      }

      const reason = chunk.choices[0]?.finish_reason;
      if (reason) {
        finishReason = reason;
      }

      if (chunk.usage) {
        totalTokens = chunk.usage.total_tokens;
      }
    }

    sendEvent({
      type: 'done',
      data: { messageId, totalTokens, finishReason },
    });
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      // 客户端主动取消，不发送错误事件
    } else if (error instanceof OpenAI.APIError) {
      const type =
        error.status === 429 ? 'rate_limit' :
        error.status === 401 ? 'auth' :
        error.status >= 500 ? 'server' : 'unknown';

      sendEvent({
        type: 'error',
        data: {
          messageId,
          error: error.message,
          type,
        },
      });
    } else {
      sendEvent({
        type: 'error',
        data: {
          messageId,
          error: '服务端内部错误',
          type: 'unknown',
        },
      });
    }
  } finally {
    activeRequests.delete(messageId);
    res.end();
  }
});

// 取消指定消息的流式请求
router.delete('/chat/stream/:messageId', (req: Request, res: Response) => {
  const { messageId } = req.params;
  const controller = activeRequests.get(messageId);

  if (controller) {
    controller.abort('用户停止生成');
    activeRequests.delete(messageId);
    res.json({ success: true });
  } else {
    res.status(404).json({ error: '未找到活跃请求' });
  }
});

export default router;
```

```typescript
// src/server/index.ts
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import chatRouter from './chat.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT ?? 3001;

app.use(cors({ origin: 'http://localhost:3000' }));
app.use(express.json());
app.use('/api', chatRouter);

app.listen(PORT, () => {
  console.log(`服务器运行在 http://localhost:${PORT}`);
});
```

### 前端：API 客户端

```typescript
// src/client/api.ts
import type { ChatMessage, SSEEvent } from '../shared/types.js';

const API_BASE = 'http://localhost:3001/api';

export interface StreamCallbacks {
  onDelta: (delta: string, messageId: string) => void;
  onDone: (messageId: string, totalTokens: number) => void;
  onError: (messageId: string, error: string) => void;
}

export async function sendChatMessage(
  messages: ChatMessage[],
  messageId: string,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, messageId }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('响应体不可读');

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      // 保留最后一个可能不完整的行
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;

        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr) as SSEEvent;

          switch (event.type) {
            case 'chunk':
              callbacks.onDelta(event.data.delta, event.data.messageId);
              break;
            case 'done':
              callbacks.onDone(event.data.messageId, event.data.totalTokens);
              break;
            case 'error':
              callbacks.onError(event.data.messageId, event.data.error);
              break;
          }
        } catch {
          console.warn('解析 SSE 事件失败:', jsonStr);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export async function cancelStream(messageId: string): Promise<void> {
  await fetch(`${API_BASE}/chat/stream/${messageId}`, {
    method: 'DELETE',
  });
}
```

### 前端：完整聊天 Hook

```typescript
// src/client/hooks.ts
import { useState, useRef, useCallback } from 'react';
import { sendChatMessage, cancelStream } from './api.js';
import type { ChatMessage } from '../shared/types.js';

export type MessageStatus = 'idle' | 'loading' | 'streaming' | 'done' | 'error';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  status: MessageStatus;
  error?: string;
  tokenCount?: number;
}

export interface UseChatReturn {
  messages: Message[];
  isLoading: boolean;
  sendMessage: (content: string) => Promise<void>;
  stopGeneration: () => void;
  clearHistory: () => void;
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const currentMessageIdRef = useRef<string | null>(null);

  const updateMessage = useCallback(
    (id: string, updater: (msg: Message) => Partial<Message>) => {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === id ? { ...msg, ...updater(msg) } : msg
        )
      );
    },
    []
  );

  const sendMessage = useCallback(
    async (content: string): Promise<void> => {
      if (isLoading) return;

      const userMessageId = `user_${Date.now()}`;
      const assistantMessageId = `assistant_${Date.now()}`;

      // 添加用户消息
      const userMsg: Message = {
        id: userMessageId,
        role: 'user',
        content,
        status: 'done',
      };

      // 添加助手占位消息
      const assistantMsg: Message = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        status: 'loading',
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsLoading(true);

      // 构建发送给 API 的消息历史
      const apiMessages: ChatMessage[] = [
        ...messages
          .filter((m) => m.status === 'done')
          .map((m) => ({ id: m.id, role: m.role, content: m.content })),
        { id: userMessageId, role: 'user' as const, content },
      ];

      abortControllerRef.current = new AbortController();
      currentMessageIdRef.current = assistantMessageId;

      try {
        await sendChatMessage(
          apiMessages,
          assistantMessageId,
          {
            onDelta: (delta, msgId) => {
              updateMessage(msgId, (msg) => ({
                content: msg.content + delta,
                status: 'streaming',
              }));
            },
            onDone: (msgId, totalTokens) => {
              updateMessage(msgId, () => ({
                status: 'done',
                tokenCount: totalTokens,
              }));
              setIsLoading(false);
            },
            onError: (msgId, error) => {
              updateMessage(msgId, () => ({
                status: 'error',
                error,
              }));
              setIsLoading(false);
            },
          },
          abortControllerRef.current.signal
        );
      } catch (error) {
        if (error instanceof Error && error.name !== 'AbortError') {
          updateMessage(assistantMessageId, () => ({
            status: 'error',
            error: error.message,
          }));
        } else {
          // AbortError：保留已接收的部分内容
          updateMessage(assistantMessageId, (msg) => ({
            status: msg.content ? 'done' : 'error',
            error: msg.content ? undefined : '请求已取消',
          }));
        }
        setIsLoading(false);
      } finally {
        abortControllerRef.current = null;
        currentMessageIdRef.current = null;
      }
    },
    [isLoading, messages, updateMessage]
  );

  const stopGeneration = useCallback((): void => {
    // 前端取消
    abortControllerRef.current?.abort('用户停止');

    // 同时通知后端取消
    if (currentMessageIdRef.current) {
      cancelStream(currentMessageIdRef.current).catch(console.error);
    }
  }, []);

  const clearHistory = useCallback((): void => {
    if (isLoading) {
      stopGeneration();
    }
    setMessages([]);
  }, [isLoading, stopGeneration]);

  return { messages, isLoading, sendMessage, stopGeneration, clearHistory };
}
```

### 前端：聊天界面组件

```typescript
// src/client/ChatApp.tsx
import { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { useChat, type Message } from './hooks.js';

// 消息气泡组件
function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';
  const isLoading = message.status === 'loading';
  const isStreaming = message.status === 'streaming';
  const isError = message.status === 'error';

  return (
    <div
      className={`message-bubble ${isUser ? 'user' : 'assistant'} ${isError ? 'error' : ''}`}
    >
      <div className="message-content">
        {message.content || (isLoading ? '' : '')}

        {/* 流式输入光标 */}
        {(isLoading || isStreaming) && (
          <span className="typing-cursor">▌</span>
        )}

        {/* Loading 动画（等待首个 chunk）*/}
        {isLoading && (
          <span className="loading-dots">
            <span>.</span><span>.</span><span>.</span>
          </span>
        )}
      </div>

      {/* 消息元信息 */}
      <div className="message-meta">
        {isError && (
          <span className="error-text">{message.error ?? '发送失败'}</span>
        )}
        {message.status === 'done' && message.tokenCount && !isUser && (
          <span className="token-count">~{message.tokenCount} tokens</span>
        )}
      </div>
    </div>
  );
}

// 主聊天应用
export function ChatApp() {
  const [input, setInput] = useState('');
  const { messages, isLoading, sendMessage, stopGeneration, clearHistory } =
    useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (): Promise<void> => {
    const content = input.trim();
    if (!content || isLoading) return;

    setInput('');
    inputRef.current?.focus();
    await sendMessage(content);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    // Enter 发送，Shift+Enter 换行
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-app">
      {/* 顶部工具栏 */}
      <div className="chat-header">
        <h1>AI 流式对话</h1>
        <button
          onClick={clearHistory}
          disabled={isLoading}
          className="btn-clear"
        >
          清空对话
        </button>
      </div>

      {/* 消息列表 */}
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>开始一段新的对话...</p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="input-area">
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入消息... (Enter 发送，Shift+Enter 换行)"
          disabled={isLoading}
          rows={3}
          className="message-input"
        />

        <div className="input-actions">
          {isLoading ? (
            <button onClick={stopGeneration} className="btn-stop">
              停止生成
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              className="btn-send"
            >
              发送
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
```

### 配置文件

```json
// package.json
{
  "name": "streaming-chat",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev:server": "tsx watch src/server/index.ts",
    "dev:client": "vite",
    "build": "tsc && vite build",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@anthropic-ai/sdk": "^0.32.0",
    "cors": "^2.8.5",
    "dotenv": "^16.4.5",
    "express": "^4.21.2",
    "openai": "^4.72.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "zustand": "^5.0.2"
  },
  "devDependencies": {
    "@types/cors": "^2.8.17",
    "@types/express": "^5.0.0",
    "@types/node": "^22.10.2",
    "@types/react": "^19.0.2",
    "@types/react-dom": "^19.0.2",
    "immer": "^10.1.1",
    "tsx": "^4.19.2",
    "typescript": "^5.7.2",
    "vite": "^6.0.5"
  }
}
```

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### 运行项目

```bash
# 安装依赖
npm install

# 配置环境变量
echo "OPENAI_API_KEY=your_api_key_here" > .env

# 启动后端（终端 1）
npm run dev:server

# 启动前端（终端 2）
npm run dev:client
```

---

## 练习题

### 基础题

**题目 20-1**：实现一个 `countStreamTokens` 函数，接受一个 `AsyncIterable<ChatCompletionChunk>` 流，返回 Promise，解析结果包含 `{ content: string; chunks: number; estimatedTokens: number }`。其中 `estimatedTokens` 使用每4字符约1 token的规则估算。

**题目 20-2**：实现一个 `withStreamTimeout` 高阶函数，包装任意返回 `AsyncIterable<T>` 的函数，添加超时能力。若超时发生，中断迭代并抛出 `TimeoutError`。函数签名如下：

```typescript
function withStreamTimeout<T>(
  fn: (...args: unknown[]) => AsyncIterable<T>,
  timeoutMs: number
): (...args: unknown[]) => AsyncIterable<T>
```

### 中级题

**题目 20-3**：实现一个 `StreamMerger` 类，能够并发运行多个流式请求（最多同时进行 N 个），并将所有流的输出合并为一个统一的异步迭代器。每个输出项包含来源标识：

```typescript
interface MergedChunk<T> {
  sourceIndex: number;
  value: T;
  timestamp: number;
}

class StreamMerger<T> {
  constructor(private maxConcurrent: number) {}
  async *merge(sources: Array<() => AsyncIterable<T>>): AsyncIterable<MergedChunk<T>> { /* ... */ }
}
```

**题目 20-4**：为本章实战项目的后端 `chat.ts` 路由添加请求速率限制功能。要求：
- 每个 IP 每分钟最多发起 10 次流式请求
- 超出限制时返回 HTTP 429 和适当的 `Retry-After` 响应头
- 使用 TypeScript 实现内存中的滑动窗口计数器（不依赖 Redis）

### 提高题

**题目 20-5**：实现一个 `ResumeableStream` 类，支持流式传输断点续传功能。当网络中断时，能够记录已接收的内容位置，在重连后从断点继续请求（通过在 system prompt 中注入已有内容，要求模型仅续写后续部分）。需处理以下情况：
- 最大重试次数限制（默认 3 次）
- 重试间隔指数退避
- 续写 prompt 的正确构造
- 最终拼接结果的一致性验证

---

## 练习答案

### 题目 20-1 答案

```typescript
import type { ChatCompletionChunk } from 'openai/resources';

interface StreamStats {
  content: string;
  chunks: number;
  estimatedTokens: number;
}

async function countStreamTokens(
  stream: AsyncIterable<ChatCompletionChunk>
): Promise<StreamStats> {
  let content = '';
  let chunks = 0;

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content ?? '';
    content += delta;
    chunks++;
  }

  return {
    content,
    chunks,
    estimatedTokens: Math.round(content.length / 4),
  };
}
```

### 题目 20-2 答案

```typescript
class TimeoutError extends Error {
  constructor(timeoutMs: number) {
    super(`流式传输超时（${timeoutMs}ms）`);
    this.name = 'TimeoutError';
  }
}

function withStreamTimeout<T>(
  fn: (...args: unknown[]) => AsyncIterable<T>,
  timeoutMs: number
): (...args: unknown[]) => AsyncIterable<T> {
  return function* wrappedFn(...args: unknown[]): AsyncIterable<T> {
    // 返回新的异步生成器
    return (async function* () {
      const source = fn(...args);
      const iterator = source[Symbol.asyncIterator]();

      while (true) {
        // 用 Promise.race 竞争：迭代器下一个值 vs 超时
        const timeoutPromise = new Promise<never>((_, reject) =>
          setTimeout(() => reject(new TimeoutError(timeoutMs)), timeoutMs)
        );

        const nextPromise = iterator.next();

        let result: IteratorResult<T>;
        try {
          result = await Promise.race([nextPromise, timeoutPromise]);
        } catch (error) {
          await iterator.return?.();
          throw error;
        }

        if (result.done) break;
        yield result.value;
      }
    })();
  } as (...args: unknown[]) => AsyncIterable<T>;
}
```

### 题目 20-3 答案

```typescript
interface MergedChunk<T> {
  sourceIndex: number;
  value: T;
  timestamp: number;
}

class StreamMerger<T> {
  constructor(private maxConcurrent: number) {}

  async *merge(
    sources: Array<() => AsyncIterable<T>>
  ): AsyncIterable<MergedChunk<T>> {
    // 使用共享队列收集所有流的输出
    const queue: Array<MergedChunk<T> | null> = [];
    let activeCount = 0;
    let resolveNext: (() => void) | null = null;

    const enqueue = (item: MergedChunk<T> | null): void => {
      queue.push(item);
      resolveNext?.();
      resolveNext = null;
    };

    // 处理单个源流
    const processSource = async (
      factory: () => AsyncIterable<T>,
      index: number
    ): Promise<void> => {
      try {
        for await (const value of factory()) {
          enqueue({ sourceIndex: index, value, timestamp: Date.now() });
        }
      } finally {
        activeCount--;
        if (activeCount === 0) {
          enqueue(null); // 所有流完成信号
        }
      }
    };

    // 按 maxConcurrent 批量启动
    const batches: Array<Array<() => AsyncIterable<T>>> = [];
    for (let i = 0; i < sources.length; i += this.maxConcurrent) {
      batches.push(sources.slice(i, i + this.maxConcurrent));
    }

    for (const batch of batches) {
      activeCount += batch.length;
      batch.forEach((factory, batchIndex) => {
        const globalIndex = batches.indexOf(batch) * this.maxConcurrent + batchIndex;
        processSource(factory, globalIndex);
      });
    }

    // 消费队列
    while (true) {
      if (queue.length > 0) {
        const item = queue.shift()!;
        if (item === null) break;
        yield item;
      } else {
        await new Promise<void>((resolve) => {
          resolveNext = resolve;
        });
      }
    }
  }
}
```

### 题目 20-4 答案

```typescript
// src/server/rateLimit.ts
import type { Request, Response, NextFunction } from 'express';

interface WindowEntry {
  count: number;
  windowStart: number;
}

const WINDOW_MS = 60 * 1000; // 1 分钟
const MAX_REQUESTS = 10;

class SlidingWindowRateLimiter {
  private windows = new Map<string, WindowEntry>();

  // 定期清理过期条目，防止内存泄漏
  constructor() {
    setInterval(() => this.cleanup(), WINDOW_MS * 2);
  }

  private cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.windows.entries()) {
      if (now - entry.windowStart > WINDOW_MS) {
        this.windows.delete(key);
      }
    }
  }

  check(ip: string): { allowed: boolean; remaining: number; retryAfterMs: number } {
    const now = Date.now();
    const entry = this.windows.get(ip);

    if (!entry || now - entry.windowStart > WINDOW_MS) {
      // 新窗口
      this.windows.set(ip, { count: 1, windowStart: now });
      return { allowed: true, remaining: MAX_REQUESTS - 1, retryAfterMs: 0 };
    }

    if (entry.count >= MAX_REQUESTS) {
      const retryAfterMs = WINDOW_MS - (now - entry.windowStart);
      return { allowed: false, remaining: 0, retryAfterMs };
    }

    entry.count++;
    return {
      allowed: true,
      remaining: MAX_REQUESTS - entry.count,
      retryAfterMs: 0,
    };
  }
}

const limiter = new SlidingWindowRateLimiter();

export function streamRateLimit(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const ip = req.ip ?? req.socket.remoteAddress ?? 'unknown';
  const result = limiter.check(ip);

  res.setHeader('X-RateLimit-Limit', MAX_REQUESTS);
  res.setHeader('X-RateLimit-Remaining', result.remaining);

  if (!result.allowed) {
    const retryAfterSec = Math.ceil(result.retryAfterMs / 1000);
    res.setHeader('Retry-After', retryAfterSec);
    res.status(429).json({
      error: '请求过于频繁，请稍后重试',
      retryAfter: retryAfterSec,
    });
    return;
  }

  next();
}
```

在 `chat.ts` 中引用：

```typescript
import { streamRateLimit } from './rateLimit.js';

// 在路由上应用速率限制中间件
router.post('/chat/stream', streamRateLimit, async (req, res) => {
  // ... 原有逻辑不变
});
```

### 题目 20-5 答案

```typescript
import OpenAI from 'openai';

interface ResumeableStreamOptions {
  maxRetries?: number;
  baseRetryDelayMs?: number;
  model?: string;
}

class ResumeableStream {
  private client: OpenAI;
  private maxRetries: number;
  private baseRetryDelayMs: number;
  private model: string;

  constructor(client: OpenAI, options: ResumeableStreamOptions = {}) {
    this.client = client;
    this.maxRetries = options.maxRetries ?? 3;
    this.baseRetryDelayMs = options.baseRetryDelayMs ?? 1000;
    this.model = options.model ?? 'gpt-4o-mini';
  }

  async fetchWithResume(
    userPrompt: string,
    systemPrompt?: string
  ): Promise<string> {
    let accumulated = '';
    let attempt = 0;

    while (attempt <= this.maxRetries) {
      try {
        const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

        if (systemPrompt) {
          messages.push({ role: 'system', content: systemPrompt });
        }

        // 如果有已积累的内容，构造续写 prompt
        if (accumulated.length > 0) {
          const resumeSystemPrompt = [
            systemPrompt ?? '',
            `\n\n[续写指令] 以下是你之前回复的开头部分，请直接从断点处继续，不要重复已有内容，不要添加任何前缀说明：`,
            `\n---\n${accumulated}\n---`,
          ].join('');

          messages.length = 0; // 清空
          messages.push({ role: 'system', content: resumeSystemPrompt });
          messages.push({ role: 'user', content: userPrompt });
        } else {
          messages.push({ role: 'user', content: userPrompt });
        }

        const stream = await this.client.chat.completions.create({
          model: this.model,
          stream: true,
          messages,
        });

        for await (const chunk of stream) {
          const delta = chunk.choices[0]?.delta?.content ?? '';
          accumulated += delta;
        }

        // 验证结果不为空
        if (!accumulated.trim()) {
          throw new Error('模型返回空内容');
        }

        return accumulated;
      } catch (error) {
        attempt++;

        const isNetworkError =
          error instanceof Error &&
          (error.message.includes('network') ||
            error.message.includes('ECONNRESET') ||
            error.message.includes('socket'));

        if (!isNetworkError || attempt > this.maxRetries) {
          throw error;
        }

        // 指数退避
        const delayMs = this.baseRetryDelayMs * Math.pow(2, attempt - 1);
        console.warn(
          `网络中断，第 ${attempt} 次重试（${delayMs}ms 后），已接收 ${accumulated.length} 字符`
        );
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }

    throw new Error('超过最大重试次数');
  }
}

// 使用示例
async function main() {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const resumeStream = new ResumeableStream(client, {
    maxRetries: 3,
    baseRetryDelayMs: 500,
  });

  const result = await resumeStream.fetchWithResume(
    '请详细介绍TypeScript的类型系统，包括基础类型、高级类型和工具类型。'
  );

  console.log('最终结果:', result);
}

main().catch(console.error);
```
