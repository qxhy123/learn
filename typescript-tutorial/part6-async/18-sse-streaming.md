# 第18章：Server-Sent Events与流式通信

## 学习目标

完成本章学习后，你将能够：

1. 理解 SSE（Server-Sent Events）协议的工作原理，掌握其与 WebSocket 的适用场景区别
2. 熟练使用浏览器原生 `EventSource` API 接收服务端推送事件
3. 掌握基于 `fetch` API 的 SSE 实现方式，突破 `EventSource` 的局限性
4. 使用 `TextDecoder` 和 `ReadableStream` 解析流式数据，处理分块传输
5. 实现完整的 LLM 流式响应客户端，能够解析 OpenAI 和 Anthropic 的流式输出格式

---

## 18.1 SSE协议介绍

### 什么是 Server-Sent Events

Server-Sent Events（SSE）是一种基于 HTTP 的服务器推送技术，允许服务器向客户端单向、持续地发送数据。它建立在普通 HTTP 连接之上，使用 `text/event-stream` 内容类型，通过长连接实现实时数据推送。

SSE 的核心特点：

- **单向通信**：只有服务器向客户端推送，客户端不能通过同一连接发送数据
- **基于 HTTP**：无需特殊协议，天然支持代理、负载均衡和 CDN
- **自动重连**：浏览器内置断线重连机制
- **轻量协议**：纯文本格式，易于调试和实现

### SSE 消息格式

SSE 使用一种简单的纯文本协议，每个事件由一个或多个字段组成，字段之间用换行符分隔，事件之间用空行（双换行）分隔：

```
data: 这是一条简单消息\n\n

id: 42\n
event: userMessage\n
data: {"userId": 1, "text": "Hello"}\n\n

: 这是一条注释，客户端会忽略\n\n

retry: 3000\n\n
```

字段说明：

| 字段 | 说明 | 示例 |
|------|------|------|
| `data` | 消息数据，可多行叠加 | `data: hello` |
| `id` | 事件 ID，用于断线续传 | `id: 42` |
| `event` | 自定义事件类型 | `event: update` |
| `retry` | 重连等待毫秒数 | `retry: 5000` |
| `:` | 注释行，保持连接心跳 | `: ping` |

多行 `data` 字段会被连接成一个字符串（中间加换行符）：

```
data: 第一行\n
data: 第二行\n\n

# 客户端收到："第一行\n第二行"
```

### SSE vs WebSocket vs 轮询

| 特性 | 短轮询 | 长轮询 | SSE | WebSocket |
|------|--------|--------|-----|-----------|
| 方向 | 单向 | 单向 | 单向 | 双向 |
| 实时性 | 低 | 中 | 高 | 高 |
| 服务器压力 | 高 | 中 | 低 | 低 |
| 协议 | HTTP | HTTP | HTTP | WS/WSS |
| 自动重连 | 手动 | 手动 | 内置 | 手动 |
| 代理支持 | 好 | 好 | 好 | 需配置 |
| 适用场景 | 低频更新 | 低频实时 | 单向推流 | 双向实时 |

**选择 SSE 的典型场景：**

- 实时日志推送
- 进度条更新
- 股票/行情数据推送
- LLM 流式文字输出（最重要的 AI 应用场景）
- 消息通知

---

## 18.2 EventSource API

### 基础用法

浏览器内置的 `EventSource` API 是使用 SSE 最简单的方式：

```typescript
// 建立 SSE 连接
const eventSource = new EventSource('/api/stream');

// 监听默认消息（event 字段缺省时触发）
eventSource.onmessage = (event: MessageEvent) => {
  console.log('收到消息:', event.data);
  console.log('消息ID:', event.lastEventId);
};

// 监听连接建立
eventSource.onopen = (event: Event) => {
  console.log('SSE 连接已建立', event);
};

// 监听错误
eventSource.onerror = (event: Event) => {
  console.error('SSE 连接错误', event);
  if (eventSource.readyState === EventSource.CLOSED) {
    console.log('连接已关闭');
  }
};

// 主动关闭连接
// eventSource.close();
```

### 监听自定义事件

当服务端发送带有 `event` 字段的消息时，需要用 `addEventListener` 监听：

```typescript
const eventSource = new EventSource('/api/events');

// 监听 "update" 类型事件
eventSource.addEventListener('update', (event: MessageEvent) => {
  const data = JSON.parse(event.data);
  console.log('更新事件:', data);
});

// 监听 "notification" 类型事件
eventSource.addEventListener('notification', (event: MessageEvent) => {
  const notification = JSON.parse(event.data) as {
    title: string;
    body: string;
    level: 'info' | 'warning' | 'error';
  };
  showNotification(notification);
});

// 监听 "close" 类型事件（服务端主动关闭信号）
eventSource.addEventListener('close', () => {
  console.log('服务端通知关闭连接');
  eventSource.close();
});

function showNotification(n: { title: string; body: string; level: string }) {
  console.log(`[${n.level.toUpperCase()}] ${n.title}: ${n.body}`);
}
```

### EventSource 的 readyState

```typescript
// readyState 的三种状态
const states: Record<number, string> = {
  [EventSource.CONNECTING]: 'CONNECTING (0) - 正在连接',
  [EventSource.OPEN]: 'OPEN (1) - 已连接',
  [EventSource.CLOSED]: 'CLOSED (2) - 已关闭',
};

function checkState(es: EventSource): void {
  console.log('当前状态:', states[es.readyState]);
}
```

### EventSource 的局限性

`EventSource` 有几个重要限制，在 AI 应用中往往需要绕过：

```typescript
// 1. 只支持 GET 请求，无法发送 POST body
// 这意味着无法在请求体中传入大量参数（如 prompt、模型配置等）
const es = new EventSource('/api/chat?prompt=hello'); // 参数只能放 URL

// 2. 无法设置自定义请求头
// 无法添加 Authorization: Bearer <token>
// new EventSource('/api/stream', { headers: {...} }); // 不支持！

// 3. 跨域请求需要服务端配置 CORS
const esCors = new EventSource('https://api.example.com/stream', {
  withCredentials: true, // 仅支持携带 Cookie，不支持自定义头
});
```

这些限制是我们在下一节介绍基于 `fetch` 的 SSE 实现的主要原因。

---

## 18.3 基于fetch的SSE实现

### 为什么需要 fetch 实现

在实际 AI 应用中，我们需要：

1. 使用 `POST` 请求发送复杂参数（模型、温度、系统提示等）
2. 携带 `Authorization` 头进行鉴权
3. 更灵活地控制请求生命周期

`fetch` API 配合 `ReadableStream` 可以完全实现 SSE 功能，同时没有上述限制。

### 基础的 fetch SSE 客户端

```typescript
interface SSEOptions {
  url: string;
  method?: 'GET' | 'POST';
  headers?: Record<string, string>;
  body?: unknown;
  onMessage: (data: string) => void;
  onError?: (error: Error) => void;
  onComplete?: () => void;
  signal?: AbortSignal;
}

async function fetchSSE(options: SSEOptions): Promise<void> {
  const {
    url,
    method = 'POST',
    headers = {},
    body,
    onMessage,
    onError,
    onComplete,
    signal,
  } = options;

  const response = await fetch(url, {
    method,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal,
  });

  if (!response.ok) {
    throw new Error(`HTTP 错误: ${response.status} ${response.statusText}`);
  }

  if (!response.body) {
    throw new Error('响应体为空，服务端不支持流式传输');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        // 处理缓冲区中剩余数据
        if (buffer.trim()) {
          parseSSEBuffer(buffer, onMessage);
        }
        onComplete?.();
        break;
      }

      // 将 Uint8Array 解码为字符串并追加到缓冲区
      buffer += decoder.decode(value, { stream: true });

      // 按空行分割处理完整事件
      const lines = buffer.split('\n\n');
      // 最后一个元素可能是不完整的事件，保留在缓冲区
      buffer = lines.pop() ?? '';

      for (const eventBlock of lines) {
        parseSSEBuffer(eventBlock, onMessage);
      }
    }
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      console.log('SSE 请求已取消');
      return;
    }
    onError?.(error instanceof Error ? error : new Error(String(error)));
    throw error;
  } finally {
    reader.releaseLock();
  }
}

// 解析单个 SSE 事件块
function parseSSEBuffer(block: string, onMessage: (data: string) => void): void {
  const lines = block.split('\n');
  const dataParts: string[] = [];

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      dataParts.push(line.slice(6)); // 移除 "data: " 前缀
    } else if (line.startsWith(':')) {
      // 注释行，忽略
    } else if (line.startsWith('id: ') || line.startsWith('event: ') || line.startsWith('retry: ')) {
      // 暂时忽略其他字段，按需扩展
    }
  }

  if (dataParts.length > 0) {
    onMessage(dataParts.join('\n'));
  }
}
```

### 完整的 SSE 解析器类

```typescript
type SSEEventHandler = (event: ParsedSSEEvent) => void;

interface ParsedSSEEvent {
  id?: string;
  event?: string;
  data: string;
  retry?: number;
}

class SSEParser {
  private buffer = '';
  private lastEventId = '';
  private handlers: Map<string, SSEEventHandler[]> = new Map();
  private defaultHandlers: SSEEventHandler[] = [];

  // 注册特定事件类型的处理器
  on(eventType: string, handler: SSEEventHandler): this {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, []);
    }
    this.handlers.get(eventType)!.push(handler);
    return this;
  }

  // 注册默认消息处理器（处理没有 event 字段的消息）
  onMessage(handler: SSEEventHandler): this {
    this.defaultHandlers.push(handler);
    return this;
  }

  // 处理新接收到的数据块
  feed(chunk: string): void {
    this.buffer += chunk;

    // 按双换行分割事件块
    const blocks = this.buffer.split('\n\n');
    this.buffer = blocks.pop() ?? '';

    for (const block of blocks) {
      if (block.trim()) {
        this.processBlock(block);
      }
    }
  }

  // 处理流结束时的剩余数据
  flush(): void {
    if (this.buffer.trim()) {
      this.processBlock(this.buffer);
      this.buffer = '';
    }
  }

  private processBlock(block: string): void {
    const event: ParsedSSEEvent = { data: '' };
    const dataParts: string[] = [];

    for (const line of block.split('\n')) {
      if (line === '') continue;

      if (line.startsWith(':')) {
        // 注释行，保持连接用，忽略
        continue;
      }

      const colonIndex = line.indexOf(':');
      if (colonIndex === -1) {
        // 只有字段名，没有值
        this.processField(line, '', event, dataParts);
      } else {
        const field = line.slice(0, colonIndex);
        // 值前面的单个空格按规范需要跳过
        const value = line.slice(colonIndex + 1).replace(/^ /, '');
        this.processField(field, value, event, dataParts);
      }
    }

    event.data = dataParts.join('\n');

    // 空 data 的事件不分发
    if (event.data === '') return;

    // 更新 lastEventId
    if (event.id !== undefined) {
      this.lastEventId = event.id;
    }

    // 分发事件
    this.dispatch(event);
  }

  private processField(
    field: string,
    value: string,
    event: ParsedSSEEvent,
    dataParts: string[]
  ): void {
    switch (field) {
      case 'data':
        dataParts.push(value);
        break;
      case 'id':
        if (!value.includes('\0')) {
          event.id = value;
        }
        break;
      case 'event':
        event.event = value;
        break;
      case 'retry':
        const retryMs = parseInt(value, 10);
        if (!isNaN(retryMs)) {
          event.retry = retryMs;
        }
        break;
    }
  }

  private dispatch(event: ParsedSSEEvent): void {
    const eventType = event.event ?? 'message';
    const handlers = this.handlers.get(eventType) ?? [];
    const isDefault = eventType === 'message';

    for (const handler of handlers) {
      handler(event);
    }

    if (isDefault) {
      for (const handler of this.defaultHandlers) {
        handler(event);
      }
    }
  }
}
```

---

## 18.4 流式数据解析

### TextDecoder 详解

`TextDecoder` 是处理流式二进制数据的关键工具。当数据分块到达时，多字节字符（如中文）可能被切断，`TextDecoder` 的流模式可以正确处理这种情况：

```typescript
// 非流模式：每次调用独立解码
const decoder1 = new TextDecoder('utf-8');
const chunk1 = new Uint8Array([0xe4, 0xb8]); // "中"字的前两个字节
const chunk2 = new Uint8Array([0xad]);       // "中"字的最后一个字节

// 错误：非流模式下截断的多字节字符会产生替换字符 "?"
console.log(decoder1.decode(chunk1)); // "?" - 不完整字符
console.log(decoder1.decode(chunk2)); // "?" - 孤立字节

// 流模式：{ stream: true } 告知解码器保留不完整的字节
const decoder2 = new TextDecoder('utf-8');
console.log(decoder2.decode(chunk1, { stream: true })); // "" - 暂存不完整字节
console.log(decoder2.decode(chunk2, { stream: true })); // "中" - 完整字符输出

// 流结束时调用不带 stream:true 的 decode 刷新剩余字节
const decoder3 = new TextDecoder('utf-8');
const incompleteChunk = new Uint8Array([0xe4, 0xb8]); // 不完整
decoder3.decode(incompleteChunk, { stream: true });
decoder3.decode(); // 刷新，输出替换字符（如果有剩余不完整字节）
```

### ReadableStream 基础

`ReadableStream` 是 Web Streams API 的核心，`fetch` 返回的 `response.body` 就是一个 `ReadableStream<Uint8Array>`：

```typescript
// ReadableStream 的三种消费方式

// 方式 1：使用 reader 手动读取（最底层）
async function consumeWithReader(stream: ReadableStream<Uint8Array>): Promise<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let result = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      result += decoder.decode(value, { stream: true });
    }
    result += decoder.decode(); // 刷新剩余字节
    return result;
  } finally {
    reader.releaseLock();
  }
}

// 方式 2：使用 pipeThrough 转换流（优雅的管道方式）
async function consumeWithPipe(stream: ReadableStream<Uint8Array>): Promise<void> {
  const textStream = stream
    .pipeThrough(new TextDecoderStream('utf-8'))  // 二进制 → 文本
    .pipeThrough(createSSELineStream());           // 文本 → 行

  const reader = textStream.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      console.log('行:', value);
    }
  } finally {
    reader.releaseLock();
  }
}

// 创建按行分割的 TransformStream
function createSSELineStream(): TransformStream<string, string> {
  let buffer = '';
  return new TransformStream<string, string>({
    transform(chunk, controller) {
      buffer += chunk;
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        controller.enqueue(line);
      }
    },
    flush(controller) {
      if (buffer) {
        controller.enqueue(buffer);
      }
    },
  });
}

// 方式 3：使用 for await...of 异步迭代（Node.js 18+/现代浏览器）
async function consumeWithAsyncIterable(response: Response): Promise<void> {
  if (!response.body) return;

  // Node.js 中 ReadableStream 支持 Symbol.asyncIterator
  for await (const chunk of response.body as unknown as AsyncIterable<Uint8Array>) {
    const text = new TextDecoder().decode(chunk);
    console.log(text);
  }
}
```

### 构建流式 SSE 解析管道

将以上知识结合，构建一个完整的流式 SSE 解析管道：

```typescript
// 将原始字节流转换为 SSE 事件流
function createSSETransform(): TransformStream<Uint8Array, ParsedSSEEvent> {
  const decoder = new TextDecoder('utf-8');
  const parser = new SSEParser();
  const eventQueue: ParsedSSEEvent[] = [];

  // 注册事件收集器
  parser.onMessage((event) => eventQueue.push(event));
  parser.on('update', (event) => eventQueue.push(event));
  parser.on('error', (event) => eventQueue.push(event));

  return new TransformStream<Uint8Array, ParsedSSEEvent>({
    transform(chunk, controller) {
      const text = decoder.decode(chunk, { stream: true });
      parser.feed(text);

      // 将收集到的事件推入输出流
      while (eventQueue.length > 0) {
        controller.enqueue(eventQueue.shift()!);
      }
    },
    flush(controller) {
      const remaining = decoder.decode(); // 刷新
      if (remaining) {
        parser.feed(remaining);
      }
      parser.flush();

      while (eventQueue.length > 0) {
        controller.enqueue(eventQueue.shift()!);
      }
    },
  });
}

// 使用示例：以流的方式处理 SSE 响应
async function streamSSEEvents(url: string, body: unknown): Promise<void> {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!response.body) throw new Error('No response body');

  const sseStream = response.body.pipeThrough(createSSETransform());
  const reader = sseStream.getReader();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // value 是已解析的 ParsedSSEEvent
      console.log(`事件类型: ${value.event ?? 'message'}`);
      console.log(`事件数据: ${value.data}`);
    }
  } finally {
    reader.releaseLock();
  }
}
```

---

## 18.5 SSE的错误处理与重连

### 错误分类与处理策略

```typescript
// 定义 SSE 错误类型
class SSEError extends Error {
  constructor(
    message: string,
    public readonly code: SSEErrorCode,
    public readonly statusCode?: number,
    public readonly retryable: boolean = false
  ) {
    super(message);
    this.name = 'SSEError';
  }
}

enum SSEErrorCode {
  NETWORK_ERROR = 'NETWORK_ERROR',       // 网络故障，可重试
  HTTP_ERROR = 'HTTP_ERROR',             // HTTP 错误，视状态码决定
  PARSE_ERROR = 'PARSE_ERROR',           // 数据解析错误，不可重试
  TIMEOUT = 'TIMEOUT',                   // 超时，可重试
  ABORTED = 'ABORTED',                   // 主动取消，不可重试
}

function classifyError(error: unknown, statusCode?: number): SSEError {
  if (error instanceof Error) {
    if (error.name === 'AbortError') {
      return new SSEError('请求被取消', SSEErrorCode.ABORTED, undefined, false);
    }
    if (error.name === 'TimeoutError' || error.message.includes('timeout')) {
      return new SSEError('请求超时', SSEErrorCode.TIMEOUT, undefined, true);
    }
    if (error.message.includes('network') || error.message.includes('fetch')) {
      return new SSEError('网络错误: ' + error.message, SSEErrorCode.NETWORK_ERROR, undefined, true);
    }
  }

  if (statusCode !== undefined) {
    const retryable = statusCode === 429 || statusCode >= 500;
    return new SSEError(
      `HTTP 错误 ${statusCode}`,
      SSEErrorCode.HTTP_ERROR,
      statusCode,
      retryable
    );
  }

  return new SSEError('未知错误', SSEErrorCode.NETWORK_ERROR, undefined, true);
}
```

### 指数退避重连

```typescript
interface RetryConfig {
  maxRetries: number;
  initialDelayMs: number;
  maxDelayMs: number;
  backoffFactor: number;
  jitter: boolean; // 添加随机抖动避免惊群效应
}

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 5,
  initialDelayMs: 1000,
  maxDelayMs: 30000,
  backoffFactor: 2,
  jitter: true,
};

function calculateDelay(attempt: number, config: RetryConfig): number {
  const baseDelay = Math.min(
    config.initialDelayMs * Math.pow(config.backoffFactor, attempt),
    config.maxDelayMs
  );

  if (config.jitter) {
    // 添加 ±25% 的随机抖动
    const jitterRange = baseDelay * 0.25;
    return baseDelay + (Math.random() * 2 - 1) * jitterRange;
  }

  return baseDelay;
}

async function fetchSSEWithRetry(
  options: SSEOptions,
  retryConfig: RetryConfig = DEFAULT_RETRY_CONFIG
): Promise<void> {
  let attempt = 0;
  let lastEventId: string | undefined;

  while (attempt <= retryConfig.maxRetries) {
    try {
      const headers: Record<string, string> = { ...options.headers };

      // 断线续传：发送最后收到的事件 ID
      if (lastEventId !== undefined) {
        headers['Last-Event-ID'] = lastEventId;
        console.log(`重连尝试 ${attempt}，从事件 ID ${lastEventId} 续传`);
      }

      await fetchSSE({
        ...options,
        headers,
        onMessage: (data) => {
          // 提取 ID 用于断线续传（需要扩展 SSE 解析器）
          options.onMessage(data);
        },
      });

      // 成功完成，退出重试循环
      return;
    } catch (error) {
      const sseError = error instanceof SSEError
        ? error
        : classifyError(error);

      // 不可重试的错误直接抛出
      if (!sseError.retryable || attempt >= retryConfig.maxRetries) {
        throw sseError;
      }

      const delay = calculateDelay(attempt, retryConfig);
      console.log(
        `SSE 连接失败 (${sseError.code})，${Math.round(delay)}ms 后重试 ` +
        `(${attempt + 1}/${retryConfig.maxRetries})`
      );

      await sleep(delay);
      attempt++;
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
```

### AbortController 超时控制

```typescript
interface SSEWithTimeoutOptions extends SSEOptions {
  timeoutMs?: number;
  idleTimeoutMs?: number; // 空闲超时：长时间未收到消息
}

async function fetchSSEWithTimeout(options: SSEWithTimeoutOptions): Promise<void> {
  const { timeoutMs = 30000, idleTimeoutMs = 60000, ...sseOptions } = options;

  const controller = new AbortController();
  let idleTimer: ReturnType<typeof setTimeout> | undefined;

  // 总超时
  const totalTimer = setTimeout(() => {
    controller.abort(new DOMException('请求超时', 'TimeoutError'));
  }, timeoutMs);

  // 重置空闲超时
  const resetIdleTimer = () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      controller.abort(new DOMException('空闲超时', 'TimeoutError'));
    }, idleTimeoutMs);
  };

  resetIdleTimer();

  try {
    await fetchSSE({
      ...sseOptions,
      signal: controller.signal,
      onMessage: (data) => {
        resetIdleTimer(); // 收到消息时重置空闲计时器
        sseOptions.onMessage(data);
      },
    });
  } finally {
    clearTimeout(totalTimer);
    if (idleTimer) clearTimeout(idleTimer);
  }
}
```

---

## 本章小结

| 知识点 | 核心内容 | 关键 API |
|--------|----------|----------|
| SSE 协议 | 文本格式、字段规范、事件结构 | `data`、`id`、`event`、`retry` 字段 |
| EventSource API | 浏览器原生支持，自动重连 | `EventSource`、`onmessage`、`addEventListener` |
| fetch + ReadableStream | 突破 EventSource 限制，支持 POST 和自定义头 | `response.body.getReader()`、`reader.read()` |
| TextDecoder | 流式解码，正确处理多字节字符 | `TextDecoder`、`{ stream: true }` |
| TransformStream | 流式管道处理，将字节流转换为结构化数据 | `pipeThrough()`、`TransformStream` |
| 错误处理 | 分类错误，指数退避重连，断线续传 | `AbortController`、`Last-Event-ID` 头 |

---

## AI应用实战：LLM流式响应实现

### 背景

现代大语言模型（LLM）的 API 通常支持流式响应，这允许客户端在模型生成文字的同时逐字展示，大幅提升用户体验。OpenAI、Anthropic 等主要 LLM 服务商均使用 SSE 协议传输流式响应。

本节将实现一个通用的 LLM 流式响应客户端，覆盖两种主流 API 格式。

### OpenAI 流式响应解析

```typescript
// OpenAI API 流式响应的数据结构
interface OpenAIChatChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: 'assistant';
      content?: string;
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
    finish_reason: 'stop' | 'length' | 'tool_calls' | null;
    logprobs: null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface OpenAIStreamOptions {
  apiKey: string;
  model?: string;
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
  temperature?: number;
  maxTokens?: number;
  signal?: AbortSignal;
}

interface StreamCallbacks {
  onToken: (token: string) => void;
  onComplete: (fullText: string, usage?: OpenAIChatChunk['usage']) => void;
  onError: (error: Error) => void;
}

async function streamOpenAI(
  options: OpenAIStreamOptions,
  callbacks: StreamCallbacks
): Promise<void> {
  const { apiKey, model = 'gpt-4o', messages, temperature = 0.7, maxTokens, signal } = options;

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages,
      temperature,
      max_tokens: maxTokens,
      stream: true,
    }),
    signal,
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`OpenAI API 错误 ${response.status}: ${errorBody}`);
  }

  if (!response.body) throw new Error('无响应体');

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let fullText = '';
  let finalUsage: OpenAIChatChunk['usage'] | undefined;

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();

        if (!trimmed || trimmed === '') continue;

        // SSE 注释行
        if (trimmed.startsWith(':')) continue;

        if (!trimmed.startsWith('data: ')) continue;

        const data = trimmed.slice(6); // 移除 "data: "

        // OpenAI 用 "[DONE]" 标记流结束
        if (data === '[DONE]') {
          callbacks.onComplete(fullText, finalUsage);
          return;
        }

        try {
          const chunk = JSON.parse(data) as OpenAIChatChunk;

          // 提取使用量（通常在最后一个 chunk）
          if (chunk.usage) {
            finalUsage = chunk.usage;
          }

          for (const choice of chunk.choices) {
            const token = choice.delta.content;
            if (token) {
              fullText += token;
              callbacks.onToken(token);
            }

            if (choice.finish_reason === 'stop') {
              callbacks.onComplete(fullText, finalUsage);
              return;
            }
          }
        } catch (parseError) {
          console.warn('解析 chunk 失败:', data, parseError);
        }
      }
    }

    callbacks.onComplete(fullText, finalUsage);
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      console.log('OpenAI 流被取消');
      return;
    }
    callbacks.onError(error instanceof Error ? error : new Error(String(error)));
    throw error;
  } finally {
    reader.releaseLock();
  }
}
```

### Anthropic 流式响应解析

```typescript
// Anthropic Messages API 的流式事件类型（messages-stream）
type AnthropicStreamEvent =
  | { type: 'message_start'; message: { id: string; model: string; usage: { input_tokens: number } } }
  | { type: 'content_block_start'; index: number; content_block: { type: 'text'; text: string } }
  | { type: 'content_block_delta'; index: number; delta: { type: 'text_delta'; text: string } }
  | { type: 'content_block_stop'; index: number }
  | { type: 'message_delta'; delta: { stop_reason: string }; usage: { output_tokens: number } }
  | { type: 'message_stop' }
  | { type: 'ping' }
  | { type: 'error'; error: { type: string; message: string } };

interface AnthropicStreamOptions {
  apiKey: string;
  model?: string;
  messages: Array<{ role: 'user' | 'assistant'; content: string }>;
  system?: string;
  maxTokens?: number;
  signal?: AbortSignal;
}

async function streamAnthropic(
  options: AnthropicStreamOptions,
  callbacks: StreamCallbacks
): Promise<void> {
  const {
    apiKey,
    model = 'claude-opus-4-5-20251101',
    messages,
    system,
    maxTokens = 1024,
    signal,
  } = options;

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model,
      messages,
      system,
      max_tokens: maxTokens,
      stream: true,
    }),
    signal,
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Anthropic API 错误 ${response.status}: ${errorBody}`);
  }

  if (!response.body) throw new Error('无响应体');

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let fullText = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      let currentEventType = '';

      for (const line of lines) {
        const trimmed = line.trim();

        if (!trimmed) continue;

        // Anthropic SSE 有 "event:" 字段指示事件类型
        if (trimmed.startsWith('event: ')) {
          currentEventType = trimmed.slice(7);
          continue;
        }

        if (!trimmed.startsWith('data: ')) continue;

        const data = trimmed.slice(6);

        try {
          const event = JSON.parse(data) as AnthropicStreamEvent;

          switch (event.type) {
            case 'content_block_delta':
              if (event.delta.type === 'text_delta') {
                fullText += event.delta.text;
                callbacks.onToken(event.delta.text);
              }
              break;

            case 'message_stop':
              callbacks.onComplete(fullText);
              return;

            case 'error':
              throw new Error(`Anthropic 流错误: ${event.error.message}`);

            case 'ping':
            case 'message_start':
            case 'content_block_start':
            case 'content_block_stop':
            case 'message_delta':
              // 这些事件类型无需特殊处理
              break;
          }
        } catch (parseError) {
          if (parseError instanceof Error && parseError.message.startsWith('Anthropic')) {
            throw parseError;
          }
          console.warn('解析 Anthropic chunk 失败:', data);
        }
      }
    }

    callbacks.onComplete(fullText);
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      return;
    }
    callbacks.onError(error instanceof Error ? error : new Error(String(error)));
    throw error;
  } finally {
    reader.releaseLock();
  }
}
```

### 统一的 LLM 流式客户端

```typescript
type LLMProvider = 'openai' | 'anthropic';

interface UnifiedLLMOptions {
  provider: LLMProvider;
  apiKey: string;
  model?: string;
  systemPrompt?: string;
  userMessage: string;
  temperature?: number;
  maxTokens?: number;
  timeoutMs?: number;
}

interface LLMStreamResult {
  fullText: string;
  tokensUsed?: number;
  durationMs: number;
}

async function streamLLM(
  options: UnifiedLLMOptions,
  onToken: (token: string) => void
): Promise<LLMStreamResult> {
  const { provider, apiKey, userMessage, systemPrompt, timeoutMs = 60000 } = options;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const startTime = Date.now();

  return new Promise<LLMStreamResult>((resolve, reject) => {
    const callbacks: StreamCallbacks = {
      onToken,
      onComplete: (fullText, usage) => {
        clearTimeout(timer);
        resolve({
          fullText,
          tokensUsed: usage?.total_tokens,
          durationMs: Date.now() - startTime,
        });
      },
      onError: (error) => {
        clearTimeout(timer);
        reject(error);
      },
    };

    const promise =
      provider === 'openai'
        ? streamOpenAI(
            {
              apiKey,
              model: options.model,
              messages: [
                ...(systemPrompt ? [{ role: 'system' as const, content: systemPrompt }] : []),
                { role: 'user', content: userMessage },
              ],
              temperature: options.temperature,
              maxTokens: options.maxTokens,
              signal: controller.signal,
            },
            callbacks
          )
        : streamAnthropic(
            {
              apiKey,
              model: options.model,
              messages: [{ role: 'user', content: userMessage }],
              system: systemPrompt,
              maxTokens: options.maxTokens,
              signal: controller.signal,
            },
            callbacks
          );

    promise.catch(reject);
  });
}

// 使用示例
async function main() {
  let outputText = '';

  process.stdout.write('AI: ');

  const result = await streamLLM(
    {
      provider: 'anthropic',
      apiKey: process.env.ANTHROPIC_API_KEY ?? '',
      systemPrompt: '你是一个简洁的助手，用中文回复。',
      userMessage: '用一句话解释什么是 TypeScript。',
      maxTokens: 200,
    },
    (token) => {
      process.stdout.write(token); // 实时打印 token
      outputText += token;
    }
  );

  console.log('\n');
  console.log(`完成！耗时 ${result.durationMs}ms，共 ${result.tokensUsed ?? '?'} tokens`);
}
```

### 在 React 中使用流式响应

```typescript
import { useState, useCallback, useRef } from 'react';

interface UseLLMStreamReturn {
  text: string;
  isStreaming: boolean;
  error: Error | null;
  startStream: (userMessage: string) => Promise<void>;
  stopStream: () => void;
  reset: () => void;
}

function useLLMStream(systemPrompt: string): UseLLMStreamReturn {
  const [text, setText] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const startStream = useCallback(
    async (userMessage: string) => {
      // 取消已有流
      abortControllerRef.current?.abort();

      const controller = new AbortController();
      abortControllerRef.current = controller;

      setText('');
      setError(null);
      setIsStreaming(true);

      try {
        await streamLLM(
          {
            provider: 'openai',
            apiKey: import.meta.env.VITE_OPENAI_API_KEY,
            systemPrompt,
            userMessage,
            maxTokens: 1000,
            timeoutMs: 30000,
          },
          (token) => {
            // 使用函数式更新避免闭包问题
            setText((prev) => prev + token);
          }
        );
      } catch (err) {
        if (err instanceof Error && err.name !== 'AbortError') {
          setError(err);
        }
      } finally {
        setIsStreaming(false);
      }
    },
    [systemPrompt]
  );

  const stopStream = useCallback(() => {
    abortControllerRef.current?.abort();
    setIsStreaming(false);
  }, []);

  const reset = useCallback(() => {
    abortControllerRef.current?.abort();
    setText('');
    setError(null);
    setIsStreaming(false);
  }, []);

  return { text, isStreaming, error, startStream, stopStream, reset };
}

// 组件使用示例
function ChatComponent() {
  const { text, isStreaming, error, startStream, stopStream } = useLLMStream(
    '你是一个友好的助手。'
  );
  const [input, setInput] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;
    await startStream(input);
    setInput('');
  };

  return (
    <div>
      <div className="response-area">
        {text}
        {isStreaming && <span className="cursor">|</span>}
        {error && <div className="error">错误: {error.message}</div>}
      </div>
      <form onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isStreaming}
          placeholder="输入消息..."
        />
        <button type="submit" disabled={isStreaming || !input.trim()}>
          发送
        </button>
        {isStreaming && (
          <button type="button" onClick={stopStream}>
            停止
          </button>
        )}
      </form>
    </div>
  );
}
```

---

## 练习题

### 基础题

**练习 18-1**：实现 `parseSSELine` 函数

编写一个函数，解析单行 SSE 文本，返回字段名和字段值：

```typescript
interface SSEField {
  field: 'data' | 'id' | 'event' | 'retry' | 'comment' | 'unknown';
  value: string;
}

function parseSSELine(line: string): SSEField | null {
  // 空行返回 null（事件分隔符）
  // ":" 开头为注释
  // "field: value" 格式
  // TODO: 实现此函数
}

// 测试用例
console.log(parseSSELine('data: hello'));         // { field: 'data', value: 'hello' }
console.log(parseSSELine('id: 42'));              // { field: 'id', value: '42' }
console.log(parseSSELine(': ping'));              // { field: 'comment', value: 'ping' }
console.log(parseSSELine(''));                    // null
console.log(parseSSELine('event: update'));       // { field: 'event', value: 'update' }
console.log(parseSSELine('data:no-space'));       // { field: 'data', value: 'no-space' }
```

**练习 18-2**：实现带超时的 `EventSource` 封装

```typescript
interface ManagedEventSource {
  close: () => void;
  getReadyState: () => number;
}

function createManagedEventSource(
  url: string,
  handlers: {
    onMessage: (data: string) => void;
    onError?: (event: Event) => void;
    onOpen?: () => void;
  },
  timeoutMs: number = 30000
): ManagedEventSource {
  // 实现：
  // 1. 创建 EventSource
  // 2. 设置超时计时器，超时后自动关闭
  // 3. 收到消息时重置超时计时器
  // 4. 返回控制对象
  // TODO: 实现此函数
}
```

### 中级题

**练习 18-3**：实现 SSE 事件录制与回放

```typescript
interface RecordedEvent {
  timestamp: number;
  data: string;
  event?: string;
}

class SSERecorder {
  private events: RecordedEvent[] = [];
  private startTime: number = Date.now();

  record(data: string, event?: string): void {
    // TODO: 记录事件及其相对时间戳
  }

  // 按原始时间间隔回放所有事件
  async replay(
    onEvent: (data: string, event?: string) => void,
    speedMultiplier: number = 1
  ): Promise<void> {
    // TODO: 实现回放，保持原始事件间隔（除以 speedMultiplier）
  }

  export(): RecordedEvent[] {
    return [...this.events];
  }
}
```

**练习 18-4**：实现支持多路合并的 SSE 客户端

```typescript
// 同时连接多个 SSE 源，合并成一个异步迭代器
async function* mergeSSEStreams(
  urls: string[],
  headers?: Record<string, string>
): AsyncGenerator<{ url: string; data: string }> {
  // TODO: 实现多路 SSE 合并
  // 提示：使用 Promise.race 或 ReadableStream.merge（如可用）
  // 任意一个流产生事件时立即 yield
  // 所有流结束时退出
}

// 使用示例
for await (const event of mergeSSEStreams(['/api/stream1', '/api/stream2'])) {
  console.log(`来自 ${event.url}: ${event.data}`);
}
```

### 高级题

**练习 18-5**：实现完整的 LLM 流式响应中间件

设计并实现一个 LLM 流式响应中间件，支持以下功能：

1. **Token 计数**：实时统计已生成的 token 数（以字符估算）
2. **速率限制**：限制每秒最多处理 N 个 token（用于动画效果）
3. **内容过滤**：检测并过滤特定关键词（出现时暂停并通知）
4. **分句缓冲**：等待完整句子（以 `。！？.!?` 结尾）再输出，避免破句

```typescript
interface StreamMiddlewareOptions {
  maxTokensPerSecond?: number;
  filteredWords?: string[];
  onFilteredWord?: (word: string, position: number) => void;
  bufferToSentence?: boolean;
}

function createStreamMiddleware(options: StreamMiddlewareOptions) {
  // 返回一个 (token: string) => Promise<void> 函数
  // 该函数在调用链中处理每个 token
  // TODO: 实现此中间件
}

// 使用示例
const middleware = createStreamMiddleware({
  maxTokensPerSecond: 20,
  filteredWords: ['敏感词'],
  onFilteredWord: (word, pos) => console.warn(`过滤词 "${word}" at ${pos}`),
  bufferToSentence: true,
});

await streamLLM(options, middleware);
```

---

## 练习答案

### 答案 18-1

```typescript
function parseSSELine(line: string): SSEField | null {
  // 空行是事件分隔符，返回 null
  if (line === '') return null;

  // 注释行
  if (line.startsWith(':')) {
    return { field: 'comment', value: line.slice(1).replace(/^ /, '') };
  }

  const colonIndex = line.indexOf(':');

  if (colonIndex === -1) {
    // 只有字段名，没有冒号，值为空字符串
    const fieldName = line as SSEField['field'];
    return { field: isKnownField(line) ? fieldName : 'unknown', value: '' };
  }

  const fieldName = line.slice(0, colonIndex);
  // 规范：冒号后紧跟的单个空格需要去掉
  const value = line.slice(colonIndex + 1).replace(/^ /, '');

  return {
    field: isKnownField(fieldName) ? (fieldName as SSEField['field']) : 'unknown',
    value,
  };
}

function isKnownField(name: string): boolean {
  return ['data', 'id', 'event', 'retry'].includes(name);
}
```

### 答案 18-2

```typescript
function createManagedEventSource(
  url: string,
  handlers: {
    onMessage: (data: string) => void;
    onError?: (event: Event) => void;
    onOpen?: () => void;
  },
  timeoutMs: number = 30000
): ManagedEventSource {
  const es = new EventSource(url);
  let timeoutId: ReturnType<typeof setTimeout>;

  const resetTimeout = () => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      console.warn(`EventSource 超时（${timeoutMs}ms），关闭连接`);
      es.close();
    }, timeoutMs);
  };

  es.onopen = () => {
    resetTimeout();
    handlers.onOpen?.();
  };

  es.onmessage = (event: MessageEvent) => {
    resetTimeout();
    handlers.onMessage(event.data);
  };

  es.onerror = (event: Event) => {
    handlers.onError?.(event);
    if (es.readyState === EventSource.CLOSED) {
      clearTimeout(timeoutId);
    }
  };

  resetTimeout();

  return {
    close: () => {
      clearTimeout(timeoutId);
      es.close();
    },
    getReadyState: () => es.readyState,
  };
}
```

### 答案 18-3

```typescript
class SSERecorder {
  private events: RecordedEvent[] = [];
  private startTime: number = Date.now();

  record(data: string, event?: string): void {
    this.events.push({
      timestamp: Date.now() - this.startTime,
      data,
      event,
    });
  }

  async replay(
    onEvent: (data: string, event?: string) => void,
    speedMultiplier: number = 1
  ): Promise<void> {
    if (this.events.length === 0) return;

    const replayStart = Date.now();

    for (const recorded of this.events) {
      const targetTime = recorded.timestamp / speedMultiplier;
      const elapsed = Date.now() - replayStart;
      const waitMs = targetTime - elapsed;

      if (waitMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, waitMs));
      }

      onEvent(recorded.data, recorded.event);
    }
  }

  export(): RecordedEvent[] {
    return [...this.events];
  }
}
```

### 答案 18-4

```typescript
async function* mergeSSEStreams(
  urls: string[],
  headers?: Record<string, string>
): AsyncGenerator<{ url: string; data: string }> {
  // 为每个 URL 创建一个事件队列和信号量
  type QueueItem = { url: string; data: string } | null; // null 表示该流结束

  const queues: QueueItem[][] = urls.map(() => []);
  const resolvers: Array<(() => void) | null> = urls.map(() => null);
  let activeStreams = urls.length;

  const notify = (index: number) => {
    const resolver = resolvers[index];
    if (resolver) {
      resolvers[index] = null;
      resolver();
    }
  };

  // 启动所有流的读取（不等待）
  const streamPromises = urls.map(async (url, index) => {
    try {
      const response = await fetch(url, {
        headers: { Accept: 'text/event-stream', ...headers },
      });

      if (!response.body) throw new Error(`${url}: 无响应体`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop() ?? '';

          for (const block of lines) {
            for (const line of block.split('\n')) {
              if (line.startsWith('data: ')) {
                queues[index].push({ url, data: line.slice(6) });
                notify(index);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      console.error(`${url} 流错误:`, error);
    } finally {
      queues[index].push(null); // 流结束信号
      activeStreams--;
      notify(index);
    }
  });

  // 等待任意队列有数据
  while (activeStreams > 0 || queues.some((q) => q.length > 0)) {
    // 找到有数据的队列
    let foundItem = false;
    for (let i = 0; i < queues.length; i++) {
      while (queues[i].length > 0) {
        const item = queues[i].shift()!;
        if (item === null) continue; // 跳过结束信号
        yield item;
        foundItem = true;
      }
    }

    if (!foundItem && activeStreams > 0) {
      // 等待任意流产生数据
      await new Promise<void>((resolve) => {
        for (let i = 0; i < resolvers.length; i++) {
          if (queues[i].length === 0) {
            resolvers[i] = resolve;
          }
        }
        // 已有数据时立即 resolve
        if (queues.some((q) => q.length > 0)) resolve();
      });
    }
  }
}
```

### 答案 18-5

```typescript
function createStreamMiddleware(options: StreamMiddlewareOptions) {
  const {
    maxTokensPerSecond,
    filteredWords = [],
    onFilteredWord,
    bufferToSentence = false,
  } = options;

  let tokenCount = 0;
  let windowStart = Date.now();
  let sentenceBuffer = '';
  let totalPosition = 0;
  const SENTENCE_ENDINGS = /[。！？.!?]/;

  const rateLimitedProcess = async (token: string, process: (t: string) => void) => {
    if (!maxTokensPerSecond) {
      process(token);
      return;
    }

    tokenCount++;
    const elapsed = Date.now() - windowStart;

    if (elapsed < 1000) {
      if (tokenCount > maxTokensPerSecond) {
        // 等待到下一个窗口
        const waitMs = 1000 - elapsed;
        await new Promise((r) => setTimeout(r, waitMs));
        tokenCount = 1;
        windowStart = Date.now();
      }
    } else {
      tokenCount = 1;
      windowStart = Date.now();
    }

    process(token);
  };

  return async (token: string): Promise<void> => {
    // 内容过滤检查
    for (const word of filteredWords) {
      const checkBuffer = sentenceBuffer + token;
      const wordIndex = checkBuffer.indexOf(word);
      if (wordIndex !== -1) {
        onFilteredWord?.(word, totalPosition + wordIndex);
        // 过滤：替换为星号
        token = token.replace(new RegExp(word, 'g'), '*'.repeat(word.length));
      }
    }

    totalPosition += token.length;

    if (bufferToSentence) {
      sentenceBuffer += token;

      // 检查是否有完整句子
      let lastEndIndex = -1;
      for (let i = 0; i < sentenceBuffer.length; i++) {
        if (SENTENCE_ENDINGS.test(sentenceBuffer[i])) {
          lastEndIndex = i;
        }
      }

      if (lastEndIndex !== -1) {
        const toOutput = sentenceBuffer.slice(0, lastEndIndex + 1);
        sentenceBuffer = sentenceBuffer.slice(lastEndIndex + 1);

        await rateLimitedProcess(toOutput, (t) => process.stdout.write(t));
      }
    } else {
      await rateLimitedProcess(token, (t) => process.stdout.write(t));
    }
  };
}
```

---

> **本章要点回顾**
>
> SSE 是 LLM 流式响应的核心传输机制。掌握基于 `fetch` 的 SSE 实现比依赖 `EventSource` 更灵活，特别是在需要 POST 请求和自定义鉴权头的 AI API 场景下。`TextDecoder` 的流模式和 `ReadableStream` 管道是正确处理分块数据的关键。在生产环境中，务必实现指数退避重连和超时控制，确保流式连接的稳定性。
