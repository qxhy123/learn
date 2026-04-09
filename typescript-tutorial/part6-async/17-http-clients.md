# 第17章：HTTP客户端与API调用

## 学习目标

完成本章学习后，你将能够：

1. 掌握 Fetch API 的 TypeScript 类型标注方式，安全地发起 HTTP 请求并处理响应
2. 定义完整的请求配置类型，包括请求头、查询参数、请求体等，实现类型安全的参数传递
3. 建立统一的响应处理与错误处理机制，区分网络错误、HTTP错误和业务逻辑错误
4. 实现请求拦截器与响应拦截器模式，在请求生命周期各阶段注入通用逻辑
5. 封装可复用的通用 HTTP 客户端类，并将其应用于 OpenAI API 的类型安全调用

---

## 17.1 Fetch API 与 TypeScript

### Fetch API 基础回顾

Fetch API 是现代浏览器和 Node.js（18+）内置的 HTTP 请求接口。TypeScript 已为其提供完整的类型定义，位于 `lib.dom.d.ts` 和 `@types/node` 中。

```typescript
// 最基础的 fetch 调用
async function getUser(id: number): Promise<void> {
  const response = await fetch(`https://api.example.com/users/${id}`);
  const data = await response.json();
  console.log(data);
}
```

上面的代码可以运行，但 `data` 的类型是 `any`，丧失了类型安全。下面我们逐步改进。

### 为响应数据标注类型

```typescript
// 定义响应数据结构
interface User {
  id: number;
  name: string;
  email: string;
  createdAt: string;
}

// 使用泛型标注 json() 的返回类型
async function getUser(id: number): Promise<User> {
  const response = await fetch(`https://api.example.com/users/${id}`);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  // response.json() 返回 Promise<any>，需要手动断言
  const user = (await response.json()) as User;
  return user;
}
```

### Response 对象的类型体系

TypeScript 中 `Response` 对象的关键类型成员：

```typescript
interface Response {
  // 状态信息
  readonly ok: boolean;           // status 在 200-299 范围内为 true
  readonly status: number;        // HTTP 状态码，如 200、404、500
  readonly statusText: string;    // 状态描述，如 "OK"、"Not Found"
  readonly url: string;           // 最终请求的 URL（含重定向）

  // 响应头
  readonly headers: Headers;

  // 响应体读取方法（每种只能调用一次）
  json(): Promise<any>;
  text(): Promise<string>;
  blob(): Promise<Blob>;
  arrayBuffer(): Promise<ArrayBuffer>;
  formData(): Promise<FormData>;

  // 流式读取
  readonly body: ReadableStream<Uint8Array> | null;
  readonly bodyUsed: boolean;

  // 克隆响应（允许多次读取 body）
  clone(): Response;
}
```

### 封装类型安全的 fetch 包装函数

```typescript
// 创建一个泛型 fetch 包装，自动完成类型转换
async function typedFetch<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(url, options);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Request failed: ${response.status} ${response.statusText} - ${errorText}`
    );
  }

  return response.json() as Promise<T>;
}

// 使用时自动获得类型推断
interface Post {
  id: number;
  title: string;
  body: string;
  userId: number;
}

// TypeScript 知道 post 的类型是 Post
const post = await typedFetch<Post>(
  'https://jsonplaceholder.typicode.com/posts/1'
);
console.log(post.title); // 类型安全，有自动补全
```

### Node.js 环境配置

在 Node.js 18+ 中，`fetch` 已内置。若使用旧版本，需额外配置：

```typescript
// tsconfig.json 中需要包含 DOM 类型库，或使用 node-fetch
// 方案一：tsconfig.json 添加 lib
{
  "compilerOptions": {
    "lib": ["ES2022", "DOM"]  // DOM 包含 fetch 类型
  }
}

// 方案二：使用 node-fetch（旧版 Node.js）
// npm install node-fetch @types/node-fetch
import fetch from 'node-fetch';

// 方案三：Node.js 18+ 直接使用，配置 tsconfig
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022"],
    "types": ["node"]  // @types/node 18+ 包含 fetch 类型
  }
}
```

---

## 17.2 请求配置与类型定义

### RequestInit 类型解析

TypeScript 内置的 `RequestInit` 接口定义了所有 fetch 请求选项：

```typescript
// 内置 RequestInit 的简化版本（实际定义更完整）
interface RequestInit {
  method?: string;            // "GET" | "POST" | "PUT" | "DELETE" | "PATCH" 等
  headers?: HeadersInit;      // 请求头
  body?: BodyInit | null;     // 请求体
  mode?: RequestMode;         // "cors" | "no-cors" | "same-origin"
  credentials?: RequestCredentials; // "include" | "omit" | "same-origin"
  cache?: RequestCache;       // 缓存策略
  redirect?: RequestRedirect; // "error" | "follow" | "manual"
  signal?: AbortSignal | null; // 用于取消请求
}
```

### 扩展请求配置类型

在实际项目中，我们通常在 `RequestInit` 基础上扩展业务相关的配置：

```typescript
// 扩展的请求配置类型
interface HttpRequestConfig extends Omit<RequestInit, 'body'> {
  // 基础配置
  url: string;
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH' | 'HEAD' | 'OPTIONS';

  // 请求数据
  params?: Record<string, string | number | boolean | undefined>;  // URL 查询参数
  data?: unknown;           // 请求体数据（会自动序列化为 JSON）

  // 超时控制
  timeout?: number;         // 毫秒，默认 30000

  // 响应类型
  responseType?: 'json' | 'text' | 'blob' | 'arrayBuffer';

  // 重试配置
  retry?: {
    count: number;          // 重试次数
    delay: number;          // 重试间隔（毫秒）
    retryOn?: number[];     // 触发重试的状态码，默认 [429, 500, 502, 503, 504]
  };
}

// 使用示例
const config: HttpRequestConfig = {
  url: '/api/users',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer token123',
  },
  data: {
    name: 'Alice',
    email: 'alice@example.com',
  },
  params: {
    version: 'v2',
    format: 'json',
  },
  timeout: 5000,
};
```

### 构建 URL 与查询参数

```typescript
// 将查询参数对象序列化为 URL 查询字符串
function buildUrl(
  baseUrl: string,
  params?: Record<string, string | number | boolean | undefined>
): string {
  if (!params) return baseUrl;

  // 过滤掉 undefined 值
  const filteredParams = Object.entries(params).filter(
    ([, value]) => value !== undefined
  ) as [string, string | number | boolean][];

  if (filteredParams.length === 0) return baseUrl;

  const queryString = new URLSearchParams(
    filteredParams.map(([key, value]) => [key, String(value)])
  ).toString();

  const separator = baseUrl.includes('?') ? '&' : '?';
  return `${baseUrl}${separator}${queryString}`;
}

// 测试
console.log(buildUrl('/api/users', { page: 1, size: 20, active: true }));
// 输出: /api/users?page=1&size=20&active=true

console.log(buildUrl('/api/users', { page: 1, name: undefined }));
// 输出: /api/users?page=1（undefined 被过滤）
```

### 请求头类型管理

```typescript
// 定义标准请求头类型
type ContentType =
  | 'application/json'
  | 'application/x-www-form-urlencoded'
  | 'multipart/form-data'
  | 'text/plain';

interface HttpHeaders {
  'Content-Type'?: ContentType;
  'Authorization'?: string;
  'Accept'?: string;
  'Accept-Language'?: string;
  'Cache-Control'?: string;
  'X-Request-ID'?: string;
  // 允许自定义头
  [key: string]: string | undefined;
}

// 合并请求头的工具函数
function mergeHeaders(
  ...headerObjects: (HttpHeaders | HeadersInit | undefined)[]
): Headers {
  const merged = new Headers();

  for (const headers of headerObjects) {
    if (!headers) continue;

    if (headers instanceof Headers) {
      headers.forEach((value, key) => merged.set(key, value));
    } else if (Array.isArray(headers)) {
      for (const [key, value] of headers) {
        merged.set(key, value);
      }
    } else {
      for (const [key, value] of Object.entries(headers)) {
        if (value !== undefined) {
          merged.set(key, value);
        }
      }
    }
  }

  return merged;
}
```

### 超时控制实现

```typescript
// 使用 AbortController 实现请求超时
function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs: number = 30000
): Promise<Response> {
  const controller = new AbortController();

  // 设置超时定时器
  const timeoutId = setTimeout(() => {
    controller.abort(new Error(`Request timeout after ${timeoutMs}ms`));
  }, timeoutMs);

  return fetch(url, {
    ...options,
    signal: controller.signal,
  }).finally(() => {
    clearTimeout(timeoutId);
  });
}

// 使用示例
try {
  const response = await fetchWithTimeout(
    'https://api.example.com/slow-endpoint',
    { method: 'GET' },
    5000  // 5秒超时
  );
  const data = await response.json();
} catch (error) {
  if (error instanceof Error && error.name === 'AbortError') {
    console.error('请求超时');
  }
}
```

---

## 17.3 响应处理与错误处理

### 统一响应结构

在前后端协作的项目中，通常约定统一的响应结构：

```typescript
// 定义通用 API 响应结构
interface ApiResponse<T = unknown> {
  code: number;           // 业务状态码，0 表示成功
  message: string;        // 状态描述
  data: T;                // 响应数据
  timestamp: number;      // 服务端时间戳
  requestId?: string;     // 请求 ID，用于链路追踪
}

// 分页响应结构
interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

// 使用示例
type UserListResponse = ApiResponse<PaginatedResponse<User>>;

async function getUserList(page: number): Promise<PaginatedResponse<User>> {
  const response = await typedFetch<UserListResponse>(
    `/api/users?page=${page}`
  );

  if (response.code !== 0) {
    throw new BusinessError(response.code, response.message);
  }

  return response.data;
}
```

### 错误类型体系

建立完整的错误类型体系，区分不同类型的失败：

```typescript
// 基础 HTTP 错误
class HttpError extends Error {
  constructor(
    public readonly status: number,
    public readonly statusText: string,
    public readonly url: string,
    public readonly responseBody?: string
  ) {
    super(`HTTP ${status} ${statusText}: ${url}`);
    this.name = 'HttpError';
  }
}

// 业务逻辑错误（HTTP 200 但业务失败）
class BusinessError extends Error {
  constructor(
    public readonly code: number,
    message: string,
    public readonly data?: unknown
  ) {
    super(message);
    this.name = 'BusinessError';
  }
}

// 网络错误（无法连接、DNS 失败等）
class NetworkError extends Error {
  constructor(
    message: string,
    public readonly cause?: Error
  ) {
    super(message);
    this.name = 'NetworkError';
  }
}

// 请求超时错误
class TimeoutError extends Error {
  constructor(public readonly timeoutMs: number) {
    super(`Request timed out after ${timeoutMs}ms`);
    this.name = 'TimeoutError';
  }
}

// 错误类型守卫
function isHttpError(error: unknown): error is HttpError {
  return error instanceof HttpError;
}

function isBusinessError(error: unknown): error is BusinessError {
  return error instanceof BusinessError;
}

function isNetworkError(error: unknown): error is NetworkError {
  return error instanceof NetworkError;
}
```

### 统一错误处理函数

```typescript
// 将 fetch 的原始错误转换为结构化错误
async function handleResponse<T>(
  response: Response,
  responseType: 'json' | 'text' = 'json'
): Promise<T> {
  if (!response.ok) {
    // 尝试读取响应体以获取错误详情
    let responseBody: string | undefined;
    try {
      responseBody = await response.text();
    } catch {
      // 忽略读取失败
    }

    throw new HttpError(
      response.status,
      response.statusText,
      response.url,
      responseBody
    );
  }

  try {
    if (responseType === 'text') {
      return (await response.text()) as T;
    }
    return (await response.json()) as T;
  } catch (error) {
    throw new Error(
      `Failed to parse response body: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

// 包装 fetch，捕获网络层错误
async function safeFetch(
  url: string,
  options?: RequestInit
): Promise<Response> {
  try {
    return await fetch(url, options);
  } catch (error) {
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new TimeoutError(
          (options?.signal as AbortSignal & { timeout?: number })?.timeout ?? 0
        );
      }
      throw new NetworkError(
        `Network request failed: ${error.message}`,
        error
      );
    }
    throw new NetworkError('Unknown network error');
  }
}
```

### 错误处理最佳实践

```typescript
// 完整的错误处理示例
async function fetchUserData(userId: number): Promise<User | null> {
  try {
    const response = await safeFetch(
      `https://api.example.com/users/${userId}`
    );
    return await handleResponse<User>(response);
  } catch (error) {
    if (isHttpError(error)) {
      switch (error.status) {
        case 401:
          // 未授权，跳转登录
          console.error('未授权，请重新登录');
          return null;
        case 403:
          // 无权限
          console.error('无权访问该资源');
          return null;
        case 404:
          // 资源不存在
          return null;
        case 429:
          // 限流
          console.error('请求过于频繁，请稍后重试');
          return null;
        default:
          console.error(`服务器错误: ${error.status}`);
          throw error;
      }
    }

    if (isNetworkError(error)) {
      console.error('网络连接失败，请检查网络');
      return null;
    }

    if (error instanceof TimeoutError) {
      console.error(`请求超时（${error.timeoutMs}ms）`);
      return null;
    }

    // 未知错误，向上抛出
    throw error;
  }
}
```

---

## 17.4 请求拦截与响应拦截

### 拦截器设计模式

拦截器（Interceptor）允许我们在请求发出前和响应返回后统一注入逻辑，类似于中间件模式。常见用途：

- **请求拦截**：自动注入认证 Token、添加追踪 ID、记录请求日志
- **响应拦截**：统一处理认证失败、刷新 Token、格式化响应数据

```typescript
// 拦截器函数类型定义
type RequestInterceptor = (
  config: HttpRequestConfig
) => HttpRequestConfig | Promise<HttpRequestConfig>;

type ResponseInterceptor<T = unknown> = (
  response: ApiResponse<T>
) => ApiResponse<T> | Promise<ApiResponse<T>>;

type ErrorInterceptor = (
  error: unknown
) => unknown | Promise<unknown>;

// 拦截器管理器
class InterceptorManager<T> {
  private interceptors: Array<{
    onFulfilled: (value: T) => T | Promise<T>;
    onRejected?: (error: unknown) => unknown;
  }> = [];

  use(
    onFulfilled: (value: T) => T | Promise<T>,
    onRejected?: (error: unknown) => unknown
  ): number {
    this.interceptors.push({ onFulfilled, onRejected });
    return this.interceptors.length - 1;
  }

  eject(id: number): void {
    this.interceptors.splice(id, 1);
  }

  getHandlers() {
    return [...this.interceptors];
  }
}
```

### 实现请求拦截链

```typescript
// 带拦截器的 HTTP 客户端（简化版）
class InterceptableHttpClient {
  private requestInterceptors = new InterceptorManager<HttpRequestConfig>();
  private responseInterceptors = new InterceptorManager<Response>();

  // 添加请求拦截器
  addRequestInterceptor(
    onFulfilled: (config: HttpRequestConfig) => HttpRequestConfig | Promise<HttpRequestConfig>,
    onRejected?: (error: unknown) => unknown
  ): number {
    return this.requestInterceptors.use(onFulfilled, onRejected);
  }

  // 添加响应拦截器
  addResponseInterceptor(
    onFulfilled: (response: Response) => Response | Promise<Response>,
    onRejected?: (error: unknown) => unknown
  ): number {
    return this.responseInterceptors.use(onFulfilled, onRejected);
  }

  async request<T>(config: HttpRequestConfig): Promise<T> {
    // 1. 执行请求拦截器链
    let processedConfig = { ...config };
    for (const interceptor of this.requestInterceptors.getHandlers()) {
      try {
        processedConfig = await interceptor.onFulfilled(processedConfig);
      } catch (error) {
        if (interceptor.onRejected) {
          processedConfig = await interceptor.onRejected(error) as HttpRequestConfig;
        } else {
          throw error;
        }
      }
    }

    // 2. 构建最终的 fetch 参数
    const url = buildUrl(processedConfig.url, processedConfig.params);
    const fetchOptions: RequestInit = {
      method: processedConfig.method ?? 'GET',
      headers: processedConfig.headers,
      body: processedConfig.data ? JSON.stringify(processedConfig.data) : undefined,
      signal: processedConfig.signal,
    };

    // 3. 发起请求
    let response = await fetch(url, fetchOptions);

    // 4. 执行响应拦截器链
    for (const interceptor of this.responseInterceptors.getHandlers()) {
      try {
        response = await interceptor.onFulfilled(response);
      } catch (error) {
        if (interceptor.onRejected) {
          response = await interceptor.onRejected(error) as Response;
        } else {
          throw error;
        }
      }
    }

    return handleResponse<T>(response);
  }
}
```

### 常用拦截器实现示例

```typescript
const client = new InterceptableHttpClient();

// 拦截器1：自动注入 Bearer Token
client.addRequestInterceptor((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    return {
      ...config,
      headers: {
        ...config.headers as Record<string, string>,
        'Authorization': `Bearer ${token}`,
      },
    };
  }
  return config;
});

// 拦截器2：添加请求追踪 ID
client.addRequestInterceptor((config) => ({
  ...config,
  headers: {
    ...config.headers as Record<string, string>,
    'X-Request-ID': crypto.randomUUID(),
    'X-Timestamp': Date.now().toString(),
  },
}));

// 拦截器3：请求日志
client.addRequestInterceptor((config) => {
  console.log(`[HTTP] ${config.method ?? 'GET'} ${config.url}`, {
    params: config.params,
    data: config.data,
  });
  return config;
});

// 响应拦截器：处理 401 自动刷新 Token
client.addResponseInterceptor(
  (response) => response,  // 成功直接透传
  async (error) => {       // 错误处理
    if (isHttpError(error) && error.status === 401) {
      try {
        // 尝试刷新 Token
        const newToken = await refreshAccessToken();
        localStorage.setItem('access_token', newToken);
        // 这里可以重新发起原始请求（需要保存原始 config）
        console.log('Token 已刷新，请重新发起请求');
      } catch {
        // 刷新失败，清除登录状态
        localStorage.removeItem('access_token');
        window.location.href = '/login';
      }
    }
    throw error;
  }
);

// 虚拟的 refreshAccessToken 实现示例
async function refreshAccessToken(): Promise<string> {
  const refreshToken = localStorage.getItem('refresh_token');
  if (!refreshToken) throw new Error('No refresh token');

  const response = await fetch('/api/auth/refresh', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refreshToken }),
  });

  if (!response.ok) throw new Error('Token refresh failed');
  const { accessToken } = await response.json() as { accessToken: string };
  return accessToken;
}
```

---

## 17.5 封装通用 HTTP 客户端

### 完整的 HttpClient 类

将前面所有内容整合为一个生产可用的 HTTP 客户端：

```typescript
// http-client.ts

// ── 类型定义 ──────────────────────────────────────────

export interface RequestConfig {
  baseURL?: string;
  url: string;
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH' | 'HEAD';
  headers?: Record<string, string>;
  params?: Record<string, string | number | boolean | undefined>;
  data?: unknown;
  timeout?: number;
  responseType?: 'json' | 'text' | 'blob';
}

export interface HttpClientConfig {
  baseURL?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

export class HttpClient {
  private baseURL: string;
  private defaultTimeout: number;
  private defaultHeaders: Record<string, string>;

  private requestInterceptors: Array<
    (config: RequestConfig) => RequestConfig | Promise<RequestConfig>
  > = [];
  private responseInterceptors: Array<
    (response: Response) => Response | Promise<Response>
  > = [];

  constructor(config: HttpClientConfig = {}) {
    this.baseURL = config.baseURL ?? '';
    this.defaultTimeout = config.timeout ?? 30000;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...config.headers,
    };
  }

  // ── 拦截器注册 ────────────────────────────────────────

  useRequestInterceptor(
    interceptor: (config: RequestConfig) => RequestConfig | Promise<RequestConfig>
  ): this {
    this.requestInterceptors.push(interceptor);
    return this;  // 支持链式调用
  }

  useResponseInterceptor(
    interceptor: (response: Response) => Response | Promise<Response>
  ): this {
    this.responseInterceptors.push(interceptor);
    return this;
  }

  // ── 核心请求方法 ──────────────────────────────────────

  async request<T>(config: RequestConfig): Promise<T> {
    // 合并默认配置
    let mergedConfig: RequestConfig = {
      ...config,
      baseURL: config.baseURL ?? this.baseURL,
      timeout: config.timeout ?? this.defaultTimeout,
      headers: {
        ...this.defaultHeaders,
        ...config.headers,
      },
    };

    // 执行请求拦截器
    for (const interceptor of this.requestInterceptors) {
      mergedConfig = await interceptor(mergedConfig);
    }

    // 构建完整 URL
    const fullUrl = mergedConfig.baseURL
      ? `${mergedConfig.baseURL.replace(/\/$/, '')}/${mergedConfig.url.replace(/^\//, '')}`
      : mergedConfig.url;

    const url = buildUrl(fullUrl, mergedConfig.params);

    // 超时控制
    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      mergedConfig.timeout
    );

    try {
      // 构建 fetch 参数
      const fetchOptions: RequestInit = {
        method: mergedConfig.method ?? 'GET',
        headers: mergedConfig.headers,
        signal: controller.signal,
      };

      // 仅在有请求体时设置 body
      if (mergedConfig.data !== undefined) {
        fetchOptions.body =
          typeof mergedConfig.data === 'string'
            ? mergedConfig.data
            : JSON.stringify(mergedConfig.data);
      }

      // 发起请求
      let response = await fetch(url, fetchOptions);

      // 执行响应拦截器
      for (const interceptor of this.responseInterceptors) {
        response = await interceptor(response);
      }

      // 处理 HTTP 错误
      if (!response.ok) {
        const body = await response.text().catch(() => '');
        throw new HttpError(response.status, response.statusText, url, body);
      }

      // 解析响应体
      const responseType = mergedConfig.responseType ?? 'json';
      if (responseType === 'text') return (await response.text()) as T;
      if (responseType === 'blob') return (await response.blob()) as T;
      return (await response.json()) as T;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new TimeoutError(mergedConfig.timeout!);
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // ── 便捷方法 ──────────────────────────────────────────

  get<T>(url: string, config?: Omit<RequestConfig, 'url' | 'method'>): Promise<T> {
    return this.request<T>({ ...config, url, method: 'GET' });
  }

  post<T>(url: string, data?: unknown, config?: Omit<RequestConfig, 'url' | 'method' | 'data'>): Promise<T> {
    return this.request<T>({ ...config, url, method: 'POST', data });
  }

  put<T>(url: string, data?: unknown, config?: Omit<RequestConfig, 'url' | 'method' | 'data'>): Promise<T> {
    return this.request<T>({ ...config, url, method: 'PUT', data });
  }

  patch<T>(url: string, data?: unknown, config?: Omit<RequestConfig, 'url' | 'method' | 'data'>): Promise<T> {
    return this.request<T>({ ...config, url, method: 'PATCH', data });
  }

  delete<T>(url: string, config?: Omit<RequestConfig, 'url' | 'method'>): Promise<T> {
    return this.request<T>({ ...config, url, method: 'DELETE' });
  }
}

// 工具函数（保持与上文一致）
function buildUrl(
  baseUrl: string,
  params?: Record<string, string | number | boolean | undefined>
): string {
  if (!params) return baseUrl;
  const entries = Object.entries(params).filter(([, v]) => v !== undefined) as [string, string | number | boolean][];
  if (entries.length === 0) return baseUrl;
  const qs = new URLSearchParams(entries.map(([k, v]) => [k, String(v)])).toString();
  return `${baseUrl}${baseUrl.includes('?') ? '&' : '?'}${qs}`;
}
```

### 使用示例

```typescript
// 创建业务专用客户端实例
const apiClient = new HttpClient({
  baseURL: 'https://api.example.com',
  timeout: 10000,
  headers: {
    'X-App-Version': '1.0.0',
  },
});

// 注册全局拦截器
apiClient
  .useRequestInterceptor((config) => ({
    ...config,
    headers: {
      ...config.headers,
      'Authorization': `Bearer ${getToken()}`,
    },
  }))
  .useResponseInterceptor((response) => {
    console.log(`[${response.status}] ${response.url}`);
    return response;
  });

// 类型化的 API 调用
interface CreateUserRequest {
  name: string;
  email: string;
  role: 'admin' | 'user';
}

interface CreateUserResponse {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'user';
  createdAt: string;
}

const newUser = await apiClient.post<CreateUserResponse>('/users', {
  name: 'Bob',
  email: 'bob@example.com',
  role: 'user',
} satisfies CreateUserRequest);

console.log(newUser.id);  // 类型安全

// 虚拟 getToken 函数
function getToken(): string {
  return localStorage.getItem('token') ?? '';
}
```

---

## 本章小结

| 知识点 | 核心内容 | 关键 API / 类型 |
|--------|----------|----------------|
| Fetch API 类型 | `Response`、`RequestInit` 的 TypeScript 类型体系 | `Response`、`RequestInit`、`Headers` |
| 响应数据类型化 | 使用泛型和类型断言为 `json()` 结果添加类型 | `response.json() as Promise<T>` |
| 请求配置扩展 | 在 `RequestInit` 基础上扩展业务配置类型 | `extends Omit<RequestInit, 'body'>` |
| URL 参数构建 | 类型安全地将对象序列化为查询字符串 | `URLSearchParams` |
| 超时控制 | 使用 `AbortController` 实现请求超时取消 | `AbortController`、`AbortSignal` |
| 错误类型体系 | 区分 `HttpError`、`NetworkError`、`TimeoutError` | 自定义 `Error` 子类 |
| 错误类型守卫 | 使用 `instanceof` 判断错误类型并分支处理 | `instanceof`、类型守卫函数 |
| 拦截器模式 | 请求/响应生命周期注入通用逻辑 | 函数数组、链式执行 |
| HTTP 客户端封装 | 整合配置、拦截器、便捷方法的完整客户端类 | 泛型方法、链式调用 |

---

## AI 应用实战：OpenAI API 封装

OpenAI 提供了官方 Node.js SDK，但了解如何手动封装类型安全的 OpenAI API 客户端，有助于理解底层机制，也便于在不引入 SDK 的轻量场景中使用。

### OpenAI API 类型定义

```typescript
// openai-types.ts

// ── 请求类型 ──────────────────────────────────────────

export type OpenAIModel =
  | 'gpt-4o'
  | 'gpt-4o-mini'
  | 'gpt-4-turbo'
  | 'gpt-3.5-turbo'
  | (string & {});  // 允许自定义模型名，同时保留字面量补全

export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

export interface ChatMessage {
  role: MessageRole;
  content: string | null;
  name?: string;
  tool_call_id?: string;
  tool_calls?: ToolCall[];
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;  // JSON 字符串
  };
}

export interface ChatCompletionRequest {
  model: OpenAIModel;
  messages: ChatMessage[];
  temperature?: number;           // 0-2，默认 1
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  stream?: boolean;
  stop?: string | string[];
  user?: string;                  // 用于滥用检测
  response_format?: {
    type: 'text' | 'json_object';
  };
  tools?: FunctionTool[];
  tool_choice?: 'none' | 'auto' | { type: 'function'; function: { name: string } };
}

export interface FunctionTool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: {
      type: 'object';
      properties: Record<string, unknown>;
      required?: string[];
    };
  };
}

// ── 响应类型 ──────────────────────────────────────────

export interface ChatCompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  system_fingerprint?: string;
}

export interface ChatCompletionChoice {
  index: number;
  message: ChatMessage;
  finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | null;
  logprobs: null;
}

// ── 流式响应类型 ──────────────────────────────────────

export interface ChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: ChatCompletionChunkChoice[];
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: {
    role?: MessageRole;
    content?: string | null;
    tool_calls?: Partial<ToolCall>[];
  };
  finish_reason: 'stop' | 'length' | 'tool_calls' | null;
}

// ── 错误类型 ──────────────────────────────────────────

export interface OpenAIError {
  error: {
    message: string;
    type: string;
    param: string | null;
    code: string | null;
  };
}
```

### OpenAI 客户端实现

```typescript
// openai-client.ts
import type {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  ChatMessage,
  OpenAIModel,
  OpenAIError,
} from './openai-types';

// OpenAI 专用错误
export class OpenAIApiError extends Error {
  constructor(
    public readonly type: string,
    message: string,
    public readonly code: string | null,
    public readonly httpStatus: number
  ) {
    super(message);
    this.name = 'OpenAIApiError';
  }
}

export interface OpenAIClientConfig {
  apiKey: string;
  baseURL?: string;
  organization?: string;
  timeout?: number;
  maxRetries?: number;
}

export class OpenAIClient {
  private readonly baseURL: string;
  private readonly headers: Record<string, string>;
  private readonly timeout: number;
  private readonly maxRetries: number;

  constructor(config: OpenAIClientConfig) {
    this.baseURL = config.baseURL ?? 'https://api.openai.com/v1';
    this.timeout = config.timeout ?? 60000;
    this.maxRetries = config.maxRetries ?? 2;
    this.headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${config.apiKey}`,
      ...(config.organization
        ? { 'OpenAI-Organization': config.organization }
        : {}),
    };
  }

  // ── 核心请求方法（带重试） ────────────────────────────

  private async fetchWithRetry(
    endpoint: string,
    body: unknown,
    attempt = 0
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      // 429 限流或 5xx 服务器错误时重试
      const shouldRetry =
        attempt < this.maxRetries &&
        (response.status === 429 || response.status >= 500);

      if (shouldRetry) {
        const retryAfter = response.headers.get('Retry-After');
        const delay = retryAfter
          ? parseInt(retryAfter, 10) * 1000
          : Math.pow(2, attempt) * 1000;  // 指数退避
        await new Promise((resolve) => setTimeout(resolve, delay));
        return this.fetchWithRetry(endpoint, body, attempt + 1);
      }

      return response;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // ── 解析错误响应 ──────────────────────────────────────

  private async parseError(response: Response): Promise<OpenAIApiError> {
    let errorData: OpenAIError;
    try {
      errorData = (await response.json()) as OpenAIError;
    } catch {
      throw new OpenAIApiError(
        'api_error',
        `HTTP ${response.status}: ${response.statusText}`,
        null,
        response.status
      );
    }

    return new OpenAIApiError(
      errorData.error.type,
      errorData.error.message,
      errorData.error.code,
      response.status
    );
  }

  // ── 普通聊天补全 ──────────────────────────────────────

  async chat(
    request: Omit<ChatCompletionRequest, 'stream'>
  ): Promise<ChatCompletionResponse> {
    const response = await this.fetchWithRetry('/chat/completions', {
      ...request,
      stream: false,
    });

    if (!response.ok) {
      throw await this.parseError(response);
    }

    return response.json() as Promise<ChatCompletionResponse>;
  }

  // ── 流式聊天补全 ──────────────────────────────────────

  async *chatStream(
    request: Omit<ChatCompletionRequest, 'stream'>
  ): AsyncGenerator<ChatCompletionChunk> {
    const response = await this.fetchWithRetry('/chat/completions', {
      ...request,
      stream: true,
    });

    if (!response.ok) {
      throw await this.parseError(response);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';  // 保留最后一个不完整行

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === 'data: [DONE]') continue;
          if (!trimmed.startsWith('data: ')) continue;

          try {
            const chunk = JSON.parse(trimmed.slice(6)) as ChatCompletionChunk;
            yield chunk;
          } catch {
            // 忽略解析失败的行
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // ── 便捷方法：简单对话 ────────────────────────────────

  async ask(
    prompt: string,
    options: {
      model?: OpenAIModel;
      systemPrompt?: string;
      temperature?: number;
    } = {}
  ): Promise<string> {
    const messages: ChatMessage[] = [];

    if (options.systemPrompt) {
      messages.push({ role: 'system', content: options.systemPrompt });
    }
    messages.push({ role: 'user', content: prompt });

    const response = await this.chat({
      model: options.model ?? 'gpt-4o-mini',
      messages,
      temperature: options.temperature ?? 0.7,
    });

    const content = response.choices[0]?.message.content;
    if (content === null || content === undefined) {
      throw new Error('Empty response from OpenAI');
    }

    return content;
  }

  // ── 便捷方法：流式对话（返回文本 AsyncGenerator） ─────

  async *askStream(
    prompt: string,
    options: {
      model?: OpenAIModel;
      systemPrompt?: string;
    } = {}
  ): AsyncGenerator<string> {
    const messages: ChatMessage[] = [];
    if (options.systemPrompt) {
      messages.push({ role: 'system', content: options.systemPrompt });
    }
    messages.push({ role: 'user', content: prompt });

    for await (const chunk of this.chatStream({
      model: options.model ?? 'gpt-4o-mini',
      messages,
    })) {
      const content = chunk.choices[0]?.delta.content;
      if (content) yield content;
    }
  }
}
```

### 使用 OpenAI 客户端

```typescript
// main.ts
import { OpenAIClient } from './openai-client';

const openai = new OpenAIClient({
  apiKey: process.env.OPENAI_API_KEY!,
  timeout: 30000,
  maxRetries: 2,
});

// 场景一：简单问答
async function simpleQA(): Promise<void> {
  const answer = await openai.ask(
    '用一句话解释什么是 TypeScript 的类型推断',
    {
      model: 'gpt-4o-mini',
      systemPrompt: '你是一位简洁的技术讲师，只给出简短直接的答案。',
      temperature: 0.3,
    }
  );
  console.log('回答:', answer);
}

// 场景二：流式输出（打字机效果）
async function streamingChat(): Promise<void> {
  process.stdout.write('AI: ');

  for await (const token of openai.askStream(
    '写一首关于 TypeScript 的五言绝句',
    { model: 'gpt-4o' }
  )) {
    process.stdout.write(token);
  }

  process.stdout.write('\n');
}

// 场景三：多轮对话
async function multiTurnChat(): Promise<void> {
  const history: import('./openai-types').ChatMessage[] = [
    {
      role: 'system',
      content: '你是一个 TypeScript 教学助手。',
    },
  ];

  const turns = [
    '什么是泛型？',
    '能给我一个实际的例子吗？',
    '这个例子中的 T 可以是任何类型吗？',
  ];

  for (const userInput of turns) {
    history.push({ role: 'user', content: userInput });

    const response = await openai.chat({
      model: 'gpt-4o-mini',
      messages: history,
      temperature: 0.7,
    });

    const assistantMessage = response.choices[0].message;
    history.push(assistantMessage);

    console.log(`用户: ${userInput}`);
    console.log(`AI: ${assistantMessage.content}\n`);
  }
}

// 运行示例
await simpleQA();
await streamingChat();
await multiTurnChat();
```

---

## 练习题

### 基础题

**练习 17-1**：实现带类型的 GET 请求包装函数

实现一个 `getJson<T>` 函数，要求：
- 接受 URL 字符串和可选的查询参数对象
- 查询参数中的 `undefined` 值应被过滤
- 如果 HTTP 状态码不是 2xx，抛出包含状态码信息的错误
- 函数返回类型应为 `Promise<T>`

```typescript
// 期望的使用方式
interface Product {
  id: number;
  name: string;
  price: number;
}

const product = await getJson<Product>('https://api.example.com/products/1');
const products = await getJson<Product[]>('https://api.example.com/products', {
  category: 'electronics',
  maxPrice: 1000,
  inStock: true,
});
```

---

**练习 17-2**：实现重试机制

为 HTTP 请求实现指数退避（Exponential Backoff）重试机制：

```typescript
interface RetryOptions {
  maxAttempts: number;     // 最大尝试次数（含首次）
  initialDelay: number;    // 首次重试延迟（毫秒）
  maxDelay: number;        // 最大延迟上限（毫秒）
  retryOn: number[];       // 触发重试的 HTTP 状态码
}

async function fetchWithRetry<T>(
  url: string,
  options?: RequestInit,
  retryOptions?: Partial<RetryOptions>
): Promise<T> {
  // 实现指数退避重试
  // 延迟计算：min(initialDelay * 2^attempt, maxDelay)
}
```

---

### 中级题

**练习 17-3**：实现请求去重

在 SPA 应用中，用户快速切换页面可能导致同一请求被发出多次。实现一个请求去重缓存，相同 URL + method + params 的请求在飞行中时不重复发起：

```typescript
class DeduplicatedHttpClient {
  private pendingRequests = new Map<string, Promise<unknown>>();

  async request<T>(config: RequestConfig): Promise<T> {
    // 生成请求的唯一键
    // 如果相同请求正在进行，返回已有的 Promise
    // 请求完成（成功或失败）后，从 Map 中移除
  }
}
```

---

**练习 17-4**：扩展 OpenAI 客户端支持 JSON Mode

为本章的 `OpenAIClient` 添加一个 `askJson<T>` 方法，该方法：
1. 自动设置 `response_format: { type: 'json_object' }`
2. 接受一个描述期望 JSON 结构的 `schema` 参数（字符串描述即可）
3. 将 OpenAI 返回的 JSON 字符串解析为类型 `T` 并返回
4. 如果解析失败，抛出包含原始响应的错误

```typescript
interface TranslationResult {
  original: string;
  translated: string;
  language: string;
}

const result = await openai.askJson<TranslationResult>(
  '将"Hello World"翻译成中文，以JSON格式返回',
  {
    schema: '{ original: string, translated: string, language: string }',
    model: 'gpt-4o-mini',
  }
);
console.log(result.translated); // "你好，世界"
```

---

### 提高题

**练习 17-5**：实现完整的请求队列与并发控制

实现一个 `RateLimitedClient`，支持以下特性：
1. **并发限制**：同时进行的请求数不超过 `maxConcurrent`
2. **速率限制**：每秒请求数不超过 `requestsPerSecond`
3. **队列管理**：超出限制的请求进入队列等待
4. **优先级支持**：高优先级请求可以插队
5. **统计信息**：提供 `getStats()` 方法返回当前队列长度和运行中请求数

```typescript
interface QueuedRequest<T> {
  config: RequestConfig;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
  priority: number;  // 数字越大优先级越高
  enqueueTime: number;
}

class RateLimitedClient extends HttpClient {
  constructor(
    httpConfig: HttpClientConfig,
    rateLimitConfig: {
      maxConcurrent: number;
      requestsPerSecond: number;
    }
  ) {
    // 实现
  }

  getStats(): {
    queued: number;
    running: number;
    completedTotal: number;
  } {
    // 实现
  }
}
```

---

## 练习答案

### 练习 17-1 答案

```typescript
async function getJson<T>(
  url: string,
  params?: Record<string, string | number | boolean | undefined>
): Promise<T> {
  // 构建含查询参数的 URL
  let finalUrl = url;
  if (params) {
    const entries = Object.entries(params).filter(
      ([, v]) => v !== undefined
    ) as [string, string | number | boolean][];

    if (entries.length > 0) {
      const qs = new URLSearchParams(
        entries.map(([k, v]) => [k, String(v)])
      ).toString();
      finalUrl = `${url}${url.includes('?') ? '&' : '?'}${qs}`;
    }
  }

  const response = await fetch(finalUrl, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });

  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new Error(
      `HTTP ${response.status} ${response.statusText}: ${body}`
    );
  }

  return response.json() as Promise<T>;
}
```

---

### 练习 17-2 答案

```typescript
const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  maxAttempts: 3,
  initialDelay: 500,
  maxDelay: 10000,
  retryOn: [429, 500, 502, 503, 504],
};

async function fetchWithRetry<T>(
  url: string,
  options?: RequestInit,
  retryOptions?: Partial<RetryOptions>
): Promise<T> {
  const opts: RetryOptions = { ...DEFAULT_RETRY_OPTIONS, ...retryOptions };

  for (let attempt = 0; attempt < opts.maxAttempts; attempt++) {
    const response = await fetch(url, options);

    // 成功响应，解析并返回
    if (response.ok) {
      return response.json() as Promise<T>;
    }

    // 判断是否需要重试
    const shouldRetry =
      attempt < opts.maxAttempts - 1 &&
      opts.retryOn.includes(response.status);

    if (!shouldRetry) {
      const body = await response.text().catch(() => '');
      throw new Error(`HTTP ${response.status}: ${body}`);
    }

    // 计算退避延迟：min(initialDelay * 2^attempt, maxDelay)
    const delay = Math.min(
      opts.initialDelay * Math.pow(2, attempt),
      opts.maxDelay
    );

    console.warn(
      `Request failed (${response.status}), retrying in ${delay}ms... ` +
      `(attempt ${attempt + 1}/${opts.maxAttempts})`
    );

    await new Promise<void>((resolve) => setTimeout(resolve, delay));
  }

  // 不应到达这里，但满足 TypeScript 类型检查
  throw new Error('Max retries exceeded');
}
```

---

### 练习 17-3 答案

```typescript
class DeduplicatedHttpClient {
  private pendingRequests = new Map<string, Promise<unknown>>();
  private client = new HttpClient();

  private buildKey(config: RequestConfig): string {
    return JSON.stringify({
      url: config.url,
      method: config.method ?? 'GET',
      params: config.params ?? {},
    });
  }

  async request<T>(config: RequestConfig): Promise<T> {
    const key = this.buildKey(config);

    // 如果相同请求正在进行，复用已有的 Promise
    const pending = this.pendingRequests.get(key);
    if (pending) {
      return pending as Promise<T>;
    }

    // 发起新请求
    const promise = this.client
      .request<T>(config)
      .finally(() => {
        // 无论成功或失败都从 Map 中移除
        this.pendingRequests.delete(key);
      });

    this.pendingRequests.set(key, promise as Promise<unknown>);
    return promise;
  }

  get<T>(url: string, params?: Record<string, string | number | boolean>): Promise<T> {
    return this.request<T>({ url, method: 'GET', params });
  }
}

// 测试去重效果
const deduped = new DeduplicatedHttpClient();

// 同时发出 3 次相同的请求，实际只会发起一次网络请求
const [r1, r2, r3] = await Promise.all([
  deduped.get('/api/config'),
  deduped.get('/api/config'),
  deduped.get('/api/config'),
]);
// r1 === r2 === r3（同一个 Promise 的结果）
```

---

### 练习 17-4 答案

```typescript
// 在 OpenAIClient 类中添加以下方法

async askJson<T>(
  prompt: string,
  options: {
    schema?: string;
    model?: OpenAIModel;
    systemPrompt?: string;
  } = {}
): Promise<T> {
  const systemContent = [
    options.systemPrompt ?? '你是一个有用的助手。',
    '请始终以有效的 JSON 格式返回你的回答。',
    options.schema ? `期望的 JSON 结构：${options.schema}` : '',
  ]
    .filter(Boolean)
    .join(' ');

  const response = await this.chat({
    model: options.model ?? 'gpt-4o-mini',
    messages: [
      { role: 'system', content: systemContent },
      { role: 'user', content: prompt },
    ],
    response_format: { type: 'json_object' },
    temperature: 0,  // JSON 模式下建议使用低温度
  });

  const rawContent = response.choices[0]?.message.content;
  if (!rawContent) {
    throw new Error('OpenAI returned empty content');
  }

  try {
    return JSON.parse(rawContent) as T;
  } catch (parseError) {
    throw new Error(
      `Failed to parse JSON response: ${rawContent}\n` +
      `Parse error: ${parseError instanceof Error ? parseError.message : String(parseError)}`
    );
  }
}
```

---

### 练习 17-5 答案

```typescript
class RateLimitedClient extends HttpClient {
  private maxConcurrent: number;
  private minInterval: number;    // 请求间最小间隔（毫秒）

  private running = 0;
  private completedTotal = 0;
  private lastRequestTime = 0;

  private queue: Array<QueuedRequest<unknown>> = [];

  constructor(
    httpConfig: HttpClientConfig,
    rateLimitConfig: {
      maxConcurrent: number;
      requestsPerSecond: number;
    }
  ) {
    super(httpConfig);
    this.maxConcurrent = rateLimitConfig.maxConcurrent;
    this.minInterval = 1000 / rateLimitConfig.requestsPerSecond;
  }

  override request<T>(config: RequestConfig, priority = 0): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const queued: QueuedRequest<T> = {
        config,
        resolve,
        reject,
        priority,
        enqueueTime: Date.now(),
      };

      // 按优先级插入队列（优先级高的排前面）
      const insertIdx = this.queue.findIndex(
        (item) => item.priority < priority
      );
      if (insertIdx === -1) {
        this.queue.push(queued as QueuedRequest<unknown>);
      } else {
        this.queue.splice(insertIdx, 0, queued as QueuedRequest<unknown>);
      }

      this.processQueue();
    });
  }

  private processQueue(): void {
    if (this.queue.length === 0 || this.running >= this.maxConcurrent) {
      return;
    }

    const now = Date.now();
    const timeSinceLastRequest = now - this.lastRequestTime;
    const waitTime = Math.max(0, this.minInterval - timeSinceLastRequest);

    if (waitTime > 0) {
      setTimeout(() => this.processQueue(), waitTime);
      return;
    }

    const next = this.queue.shift();
    if (!next) return;

    this.running++;
    this.lastRequestTime = Date.now();

    super
      .request(next.config)
      .then((result) => {
        next.resolve(result);
      })
      .catch((error: unknown) => {
        next.reject(error);
      })
      .finally(() => {
        this.running--;
        this.completedTotal++;
        // 继续处理队列中的下一个请求
        this.processQueue();
      });

    // 如果还有并发槽位，继续处理
    if (this.running < this.maxConcurrent) {
      this.processQueue();
    }
  }

  getStats(): { queued: number; running: number; completedTotal: number } {
    return {
      queued: this.queue.length,
      running: this.running,
      completedTotal: this.completedTotal,
    };
  }
}

// 使用示例
const rateLimited = new RateLimitedClient(
  { baseURL: 'https://api.example.com', timeout: 10000 },
  { maxConcurrent: 3, requestsPerSecond: 10 }
);

// 并发发出 20 个请求，会自动限速
const requests = Array.from({ length: 20 }, (_, i) =>
  rateLimited.request<{ id: number }>({ url: `/items/${i}` })
);

const results = await Promise.all(requests);
console.log('完成:', results.length);
console.log('统计:', rateLimited.getStats());
```
