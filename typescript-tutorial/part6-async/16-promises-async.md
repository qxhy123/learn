# 第16章：Promise与async/await

## 学习目标

完成本章学习后，你将能够：

1. 理解 Promise 的状态机制，掌握 `then`、`catch`、`finally` 的链式调用写法
2. 使用 `async/await` 语法编写清晰易读的异步代码，并理解其与 Promise 的关系
3. 正确处理异步错误，区分 `try/catch` 与 `.catch()` 的适用场景
4. 使用 `Promise.all`、`Promise.race`、`Promise.allSettled` 实现高效的并发控制
5. 掌握异步迭代器与 `for await...of` 语法，处理异步数据流

---

## 16.1 Promise 基础

### Promise 是什么

Promise 是 JavaScript/TypeScript 中处理异步操作的核心抽象。一个 Promise 代表一个**尚未完成但最终会有结果**的操作，它处于以下三种状态之一：

- **Pending（待定）**：初始状态，异步操作尚未完成
- **Fulfilled（已兑现）**：操作成功完成，持有结果值
- **Rejected（已拒绝）**：操作失败，持有错误原因

状态只能从 Pending 转变为 Fulfilled 或 Rejected，且**不可逆**。

```typescript
// 创建 Promise 的基本语法
const promise = new Promise<string>((resolve, reject) => {
  // 执行异步操作
  const success = true

  if (success) {
    resolve('操作成功')    // 转为 Fulfilled
  } else {
    reject(new Error('操作失败'))  // 转为 Rejected
  }
})
```

### 创建 Promise

```typescript
// 1. 基础 Promise：包装异步操作
function delay(ms: number): Promise<void> {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, ms)
  })
}

// 2. 立即兑现的 Promise
const resolved = Promise.resolve(42)
const resolvedStr = Promise.resolve<string>('hello')

// 3. 立即拒绝的 Promise
const rejected = Promise.reject<never>(new Error('Something went wrong'))

// 4. 包装回调风格的 API（Promise 化）
import * as fs from 'fs'

function readFileAsync(path: string): Promise<string> {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf-8', (err, data) => {
      if (err) {
        reject(err)
      } else {
        resolve(data)
      }
    })
  })
}

// 5. 带超时的 Promise
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  const timeout = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error(`操作超时（${ms}ms）`)), ms)
  })
  return Promise.race([promise, timeout])
}
```

### then、catch、finally

`.then()`、`.catch()`、`.finally()` 是 Promise 的三个核心方法，每个方法都返回一个新的 Promise，从而支持链式调用。

```typescript
// 模拟一个 API 请求
interface UserProfile {
  id: number
  name: string
  email: string
}

function fetchUser(id: number): Promise<UserProfile> {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (id <= 0) {
        reject(new Error(`无效的用户 ID: ${id}`))
      } else {
        resolve({ id, name: `用户${id}`, email: `user${id}@example.com` })
      }
    }, 300)
  })
}

// 基础链式调用
fetchUser(1)
  .then((user) => {
    console.log('获取用户成功:', user.name)
    return user.id  // 返回值传递给下一个 then
  })
  .then((userId) => {
    console.log('用户 ID:', userId)
  })
  .catch((err: Error) => {
    console.error('请求失败:', err.message)
  })
  .finally(() => {
    console.log('请求已完成（无论成功或失败）')
  })
```

**理解链式调用中的值传递：**

```typescript
// then 中返回普通值 → 下一个 then 收到该值
// then 中返回 Promise → 下一个 then 等待该 Promise 解决
// then 中抛出错误  → 跳到下一个 catch

function processUserPipeline(id: number): Promise<string> {
  return fetchUser(id)
    .then((user) => {
      if (!user.email.includes('@')) {
        throw new Error('邮箱格式无效')  // 触发 catch
      }
      return `${user.name} <${user.email}>`  // 传给下一个 then
    })
    .then((formatted) => {
      return formatted.toUpperCase()  // 继续变换
    })
    .catch((err: Error) => {
      // catch 也可以返回值，使链条从错误中恢复
      console.error('处理失败，使用默认值:', err.message)
      return 'UNKNOWN USER'
    })
    // finally 不影响链条的值
    .finally(() => {
      console.log(`processUserPipeline(${id}) 执行完毕`)
    })
}

processUserPipeline(1).then(console.log)   // 输出: USER1 <USER1@EXAMPLE.COM>
processUserPipeline(-1).then(console.log)  // 输出: UNKNOWN USER
```

### Promise 状态传播规则

理解链式调用中错误的传播方式是编写健壮代码的关键：

```typescript
// 规则一：错误会跳过所有 then，直到遇到 catch
Promise.reject(new Error('初始错误'))
  .then(() => console.log('这里不会执行'))  // 跳过
  .then(() => console.log('这里也不会'))    // 跳过
  .catch((err: Error) => {
    console.error('捕获到错误:', err.message)  // 执行
    return '已恢复'
  })
  .then((val) => console.log(val))  // 执行：输出 '已恢复'

// 规则二：catch 内如果再次抛出，错误继续传播
Promise.reject(new Error('原始错误'))
  .catch((err: Error) => {
    throw new Error(`包装后的错误: ${err.message}`)  // 再次抛出
  })
  .catch((err: Error) => {
    console.error(err.message)  // 捕获包装后的错误
  })

// 规则三：catch 返回值后，链条恢复正常
Promise.reject(new Error('可恢复错误'))
  .catch(() => '默认值')   // 返回值，链条恢复
  .then((val) => console.log(val))  // 执行：输出 '默认值'
```

---

## 16.2 async/await 语法

### 基本语法

`async/await` 是 Promise 的语法糖，让异步代码读起来像同步代码，大幅提升可读性。

```typescript
// async 函数总是返回 Promise
async function greet(name: string): Promise<string> {
  return `Hello, ${name}!`  // 等价于 Promise.resolve(`Hello, ${name}!`)
}

// await 只能在 async 函数内使用，暂停执行直到 Promise 解决
async function main(): Promise<void> {
  const message = await greet('TypeScript')
  console.log(message)  // Hello, TypeScript!
}

// async/await 与纯 Promise 写法对比
// Promise 写法
function fetchAndProcessPromise(id: number): Promise<string> {
  return fetchUser(id)
    .then((user) => `${user.name} (${user.email})`)
}

// async/await 写法（等价，但更清晰）
async function fetchAndProcess(id: number): Promise<string> {
  const user = await fetchUser(id)
  return `${user.name} (${user.email})`
}
```

### 顺序执行与并发执行

`await` 会暂停当前函数的执行，理解何时该顺序、何时该并发至关重要：

```typescript
// 顺序执行：总耗时 = 各操作时间之和（不推荐，除非有依赖关系）
async function sequential(): Promise<void> {
  console.time('sequential')

  const user = await fetchUser(1)           // 等待 300ms
  await delay(200)                           // 再等待 200ms
  console.log(user.name)

  console.timeEnd('sequential')             // ~500ms
}

// 并发执行：总耗时 = 最长操作时间（推荐，无依赖时使用）
async function concurrent(): Promise<void> {
  console.time('concurrent')

  const [user, _] = await Promise.all([
    fetchUser(1),  // 同时发起
    delay(200),    // 同时发起
  ])
  console.log(user.name)

  console.timeEnd('concurrent')  // ~300ms
}

// 注意：只有在操作之间没有依赖关系时才能并发
async function withDependency(): Promise<void> {
  // 必须顺序：第二步依赖第一步的结果
  const user = await fetchUser(1)
  const orders = await fetchOrdersByUser(user.id)  // 依赖 user.id
  console.log(orders)
}

// 虚构辅助函数
function fetchOrdersByUser(userId: number): Promise<string[]> {
  return Promise.resolve([`Order-${userId}-001`, `Order-${userId}-002`])
}
```

### await 在循环中的使用

```typescript
const userIds = [1, 2, 3, 4, 5]

// 错误用法：forEach 不支持 async/await（回调中的 await 不会被外层等待）
async function wrongWay(): Promise<void> {
  userIds.forEach(async (id) => {
    const user = await fetchUser(id)
    console.log(user.name)  // 顺序不保证，且外层 forEach 不等待
  })
  console.log('所有用户已处理')  // 这行会在 fetch 之前执行！
}

// 正确用法一：for...of（顺序执行）
async function sequentialLoop(): Promise<void> {
  for (const id of userIds) {
    const user = await fetchUser(id)  // 逐个等待
    console.log(user.name)
  }
  console.log('顺序处理完毕')  // 在所有 fetch 完成后执行
}

// 正确用法二：Promise.all + map（并发执行）
async function concurrentLoop(): Promise<void> {
  const users = await Promise.all(
    userIds.map((id) => fetchUser(id))  // 同时发起所有请求
  )
  users.forEach((user) => console.log(user.name))
  console.log('并发处理完毕')
}

// 正确用法三：批量并发（控制并发数）
async function batchedLoop(batchSize: number): Promise<void> {
  for (let i = 0; i < userIds.length; i += batchSize) {
    const batch = userIds.slice(i, i + batchSize)
    const users = await Promise.all(batch.map((id) => fetchUser(id)))
    users.forEach((user) => console.log(user.name))
    console.log(`批次 ${Math.floor(i / batchSize) + 1} 处理完毕`)
  }
}
```

### 顶层 await

TypeScript 4.0+ 支持在 ES 模块的顶层使用 `await`（无需包裹在 `async` 函数中）：

```typescript
// config.ts（ES 模块）
// tsconfig.json 需设置 "module": "ESNext" 且 "target": "ES2022"

const configPath = './config.json'
const configData = await readFileAsync(configPath)  // 顶层 await
const config = JSON.parse(configData) as Record<string, unknown>

export default config

// 常见用途：初始化数据库连接
// db.ts
// const db = await createDatabaseConnection(process.env.DATABASE_URL!)
// export { db }
```

---

## 16.3 错误处理

### try/catch 与 .catch() 的选择

```typescript
// 方式一：try/catch（async/await 风格，推荐）
async function fetchUserSafe(id: number): Promise<UserProfile | null> {
  try {
    const user = await fetchUser(id)
    return user
  } catch (err) {
    if (err instanceof Error) {
      console.error('获取用户失败:', err.message)
    }
    return null
  }
}

// 方式二：.catch()（Promise 链风格）
function fetchUserSafePromise(id: number): Promise<UserProfile | null> {
  return fetchUser(id).catch((err: Error) => {
    console.error('获取用户失败:', err.message)
    return null
  })
}

// 两种方式等价，选择取决于代码风格和上下文
```

### 自定义错误类型

在 TypeScript 中，定义具体的错误类型能提供更好的类型安全和错误区分能力：

```typescript
// 定义 API 错误层级
class AppError extends Error {
  constructor(
    message: string,
    public readonly code: string
  ) {
    super(message)
    this.name = 'AppError'
    // 修复 TypeScript 中继承 Error 的原型链问题
    Object.setPrototypeOf(this, new.target.prototype)
  }
}

class NetworkError extends AppError {
  constructor(
    message: string,
    public readonly statusCode: number
  ) {
    super(message, 'NETWORK_ERROR')
    this.name = 'NetworkError'
    Object.setPrototypeOf(this, new.target.prototype)
  }
}

class AuthError extends AppError {
  constructor(message: string) {
    super(message, 'AUTH_ERROR')
    this.name = 'AuthError'
    Object.setPrototypeOf(this, new.target.prototype)
  }
}

class RateLimitError extends AppError {
  constructor(
    public readonly retryAfterMs: number
  ) {
    super(`请求过于频繁，请 ${retryAfterMs / 1000} 秒后重试`, 'RATE_LIMIT')
    this.name = 'RateLimitError'
    Object.setPrototypeOf(this, new.target.prototype)
  }
}

// 根据错误类型采取不同处理策略
async function callAPI(endpoint: string): Promise<unknown> {
  try {
    const response = await fetch(endpoint)

    if (response.status === 401) {
      throw new AuthError('认证失败，请重新登录')
    }
    if (response.status === 429) {
      const retryAfter = Number(response.headers.get('Retry-After') ?? '60') * 1000
      throw new RateLimitError(retryAfter)
    }
    if (!response.ok) {
      throw new NetworkError(`请求失败: ${response.statusText}`, response.status)
    }

    return response.json()
  } catch (err) {
    if (err instanceof AuthError) {
      // 跳转到登录页
      console.error('[Auth]', err.message)
      throw err  // 重新抛出，让调用方处理
    }

    if (err instanceof RateLimitError) {
      // 等待后重试
      console.warn(`[RateLimit] 将在 ${err.retryAfterMs}ms 后重试`)
      await delay(err.retryAfterMs)
      return callAPI(endpoint)  // 重试
    }

    if (err instanceof NetworkError) {
      console.error(`[Network] HTTP ${err.statusCode}:`, err.message)
      throw err
    }

    // 未知错误
    throw new AppError(`未知错误: ${String(err)}`, 'UNKNOWN')
  }
}
```

### 错误边界模式

对于多步骤操作，可以使用 Result 类型（类似 Rust 的 Result）封装错误，避免异常打断控制流：

```typescript
// Result 类型：成功或失败
type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E }

// 辅助函数：将 Promise 转换为 Result
async function tryAsync<T>(
  promise: Promise<T>
): Promise<Result<T>> {
  try {
    const value = await promise
    return { ok: true, value }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error : new Error(String(error)) }
  }
}

// 使用 Result 类型，无需 try/catch
async function processWithResult(id: number): Promise<void> {
  const result = await tryAsync(fetchUser(id))

  if (!result.ok) {
    console.error('获取用户失败:', result.error.message)
    return
  }

  // TypeScript 知道 result.value 是 UserProfile
  const { value: user } = result
  console.log(`处理用户: ${user.name}`)
}

// 多步骤操作中的 Result 链
async function multiStepOperation(userId: number): Promise<Result<string>> {
  const userResult = await tryAsync(fetchUser(userId))
  if (!userResult.ok) return userResult

  const ordersResult = await tryAsync(fetchOrdersByUser(userResult.value.id))
  if (!ordersResult.ok) return ordersResult

  return {
    ok: true,
    value: `用户 ${userResult.value.name} 有 ${ordersResult.value.length} 个订单`,
  }
}
```

### 未捕获的 Promise 拒绝

```typescript
// Node.js 中处理未捕获的 Promise 拒绝
process.on('unhandledRejection', (reason: unknown, promise: Promise<unknown>) => {
  console.error('未处理的 Promise 拒绝:', reason)
  // 在生产环境中，可能需要上报错误并退出进程
  process.exit(1)
})

// 始终为 Promise 添加错误处理
const dangerousPromise = fetchUser(-1)  // 会失败
// dangerousPromise  ← 危险！没有 catch 会产生 UnhandledRejection

dangerousPromise.catch((err: Error) => {
  console.error('已处理:', err.message)  // 正确
})

// 或使用 void 操作符表明有意忽略（但不推荐）
void fetchUser(-1).catch((err: Error) => {
  console.error('已处理:', err.message)
})
```

---

## 16.4 并发控制

### Promise.all：等待全部完成

`Promise.all` 同时运行多个 Promise，**全部成功**时返回结果数组，**任一失败**时立即拒绝。

```typescript
interface ProductInfo {
  id: number
  name: string
  price: number
}

interface StockInfo {
  productId: number
  quantity: number
}

interface ReviewInfo {
  productId: number
  rating: number
  count: number
}

// 模拟 API 调用
function fetchProduct(id: number): Promise<ProductInfo> {
  return delay(100).then(() => ({ id, name: `产品${id}`, price: id * 100 }))
}

function fetchStock(productId: number): Promise<StockInfo> {
  return delay(150).then(() => ({ productId, quantity: productId * 10 }))
}

function fetchReviews(productId: number): Promise<ReviewInfo> {
  return delay(80).then(() => ({ productId, rating: 4.5, count: productId * 20 }))
}

// Promise.all：并发获取产品页面所需的所有数据
async function fetchProductPage(productId: number): Promise<{
  product: ProductInfo
  stock: StockInfo
  reviews: ReviewInfo
}> {
  // 三个请求同时发出，总耗时约 150ms（最慢的那个）
  const [product, stock, reviews] = await Promise.all([
    fetchProduct(productId),
    fetchStock(productId),
    fetchReviews(productId),
  ])

  return { product, stock, reviews }
}

// TypeScript 类型推断：results 的类型是 [ProductInfo, StockInfo, ReviewInfo]
async function typedPromiseAll(): Promise<void> {
  const results = await Promise.all([
    fetchProduct(1),    // Promise<ProductInfo>
    fetchStock(1),      // Promise<StockInfo>
    fetchReviews(1),    // Promise<ReviewInfo>
  ])
  // results: [ProductInfo, StockInfo, ReviewInfo]  ← 元组类型！

  const [product, stock, reviews] = results
  console.log(product.name, stock.quantity, reviews.rating)
}

// Promise.all 的失败快速机制
async function demonstrateFailFast(): Promise<void> {
  try {
    await Promise.all([
      fetchProduct(1),               // 成功
      Promise.reject(new Error('股票服务不可用')),  // 失败
      fetchReviews(1),               // 可能未完成就被放弃
    ])
  } catch (err) {
    console.error('有一个失败了:', (err as Error).message)
    // 注意：其他请求可能仍在进行，但结果被忽略
  }
}
```

### Promise.race：竞速，取最快的

`Promise.race` 返回**第一个**（最快的）Promise 的结果，无论成功还是失败。

```typescript
// 经典用途：实现请求超时
function fetchWithTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  timeoutMessage = '请求超时'
): Promise<T> {
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs)
  })

  return Promise.race([promise, timeoutPromise])
}

async function fetchUserWithTimeout(id: number): Promise<UserProfile | null> {
  try {
    return await fetchWithTimeout(fetchUser(id), 500, `获取用户 ${id} 超时`)
  } catch (err) {
    if ((err as Error).message.includes('超时')) {
      console.warn('请求超时，使用缓存或默认值')
      return null
    }
    throw err
  }
}

// 另一个用途：主备切换（两个端点竞速，取最快响应）
async function fetchFromFastestEndpoint(
  endpoints: string[]
): Promise<Response> {
  const requests = endpoints.map((url) => fetch(url))
  return Promise.race(requests)
}
```

### Promise.allSettled：等待全部结束（不管成功失败）

`Promise.allSettled` 等待**所有** Promise 结束，无论成功或失败，返回每个 Promise 的结果状态。

```typescript
type SettledResult<T> =
  | { status: 'fulfilled'; value: T }
  | { status: 'rejected'; reason: unknown }

// 批量操作时，不希望一个失败影响其他操作
async function batchFetchUsers(ids: number[]): Promise<{
  succeeded: UserProfile[]
  failed: Array<{ id: number; error: string }>
}> {
  const results = await Promise.allSettled(
    ids.map((id) => fetchUser(id))
  )

  const succeeded: UserProfile[] = []
  const failed: Array<{ id: number; error: string }> = []

  results.forEach((result, index) => {
    if (result.status === 'fulfilled') {
      succeeded.push(result.value)
    } else {
      failed.push({
        id: ids[index],
        error: result.reason instanceof Error
          ? result.reason.message
          : String(result.reason),
      })
    }
  })

  return { succeeded, failed }
}

async function demonstrateAllSettled(): Promise<void> {
  const ids = [1, -1, 2, -2, 3]  // -1 和 -2 会失败
  const { succeeded, failed } = await batchFetchUsers(ids)

  console.log(`成功: ${succeeded.length} 个`)
  console.log(`失败: ${failed.length} 个`)
  failed.forEach(({ id, error }) => {
    console.error(`  用户 ${id}: ${error}`)
  })
}
```

### Promise.any：取第一个成功的

`Promise.any` 返回**第一个成功**的 Promise 结果，只有**全部失败**时才拒绝（抛出 `AggregateError`）。

```typescript
// 用途：多个备用数据源，任一成功即可
async function fetchFromAnySource(
  sources: Array<() => Promise<string>>
): Promise<string> {
  try {
    // 取最先成功的那个
    return await Promise.any(sources.map((source) => source()))
  } catch (err) {
    // AggregateError: 所有 Promise 都失败了
    if (err instanceof AggregateError) {
      const messages = err.errors
        .map((e: unknown) => (e instanceof Error ? e.message : String(e)))
        .join('; ')
      throw new Error(`所有数据源均不可用: ${messages}`)
    }
    throw err
  }
}

// 实际使用：CDN 故障切换
async function loadResourceFromCDN(resourceName: string): Promise<string> {
  const cdnUrls = [
    `https://cdn1.example.com/${resourceName}`,
    `https://cdn2.example.com/${resourceName}`,
    `https://cdn3.example.com/${resourceName}`,
  ]

  return fetchFromAnySource(
    cdnUrls.map((url) => () =>
      fetch(url).then((res) => {
        if (!res.ok) throw new Error(`CDN ${url} 返回 ${res.status}`)
        return res.text()
      })
    )
  )
}
```

### 并发限制器

在真实应用中，同时发起过多请求可能超过服务端限制。需要实现一个**并发限制器**：

```typescript
// 并发池：最多同时运行 N 个 Promise
async function limitConcurrency<T>(
  tasks: Array<() => Promise<T>>,
  maxConcurrent: number
): Promise<T[]> {
  const results: T[] = new Array(tasks.length)
  let index = 0

  // Worker 函数：不断取任务执行
  async function worker(): Promise<void> {
    while (index < tasks.length) {
      const currentIndex = index++
      results[currentIndex] = await tasks[currentIndex]()
    }
  }

  // 启动 maxConcurrent 个 worker
  const workers = Array.from({ length: Math.min(maxConcurrent, tasks.length) }, worker)
  await Promise.all(workers)

  return results
}

// 使用：并发数限制为 3，处理 10 个用户
async function fetchAllUsersWithLimit(): Promise<void> {
  const ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  const tasks = ids.map((id) => () => fetchUser(id))

  console.time('limitConcurrency')
  const users = await limitConcurrency(tasks, 3)  // 最多同时 3 个
  console.timeEnd('limitConcurrency')

  console.log(`获取了 ${users.length} 个用户`)
}
```

### 四个并发方法对比

| 方法 | 返回时机 | 失败处理 | 典型用途 |
|------|----------|----------|----------|
| `Promise.all` | 全部成功时 | 任一失败立即拒绝 | 需要所有结果才能继续 |
| `Promise.race` | 第一个结束时 | 第一个失败则拒绝 | 超时控制、竞速 |
| `Promise.allSettled` | 全部结束时 | 收集所有结果（含失败） | 批量操作，容忍部分失败 |
| `Promise.any` | 第一个成功时 | 全部失败才拒绝 | 多备源降级、容错 |

---

## 16.5 异步迭代

### 异步迭代器协议

普通迭代器通过 `Symbol.iterator` 和 `.next()` 工作；异步迭代器通过 `Symbol.asyncIterator` 和返回 `Promise<IteratorResult>` 的 `.next()` 工作。

```typescript
// 实现异步迭代器：逐页分页获取数据
interface Page<T> {
  data: T[]
  nextCursor: string | null
}

// 模拟分页 API
function fetchPage(cursor: string | null): Promise<Page<UserProfile>> {
  return delay(100).then(() => {
    if (cursor === null) {
      return { data: [{ id: 1, name: '用户1', email: 'u1@example.com' }], nextCursor: 'page2' }
    } else if (cursor === 'page2') {
      return { data: [{ id: 2, name: '用户2', email: 'u2@example.com' }], nextCursor: 'page3' }
    } else if (cursor === 'page3') {
      return { data: [{ id: 3, name: '用户3', email: 'u3@example.com' }], nextCursor: null }
    }
    return { data: [], nextCursor: null }
  })
}

// 创建异步可迭代对象
function paginate<T>(
  fetcher: (cursor: string | null) => Promise<Page<T>>
): AsyncIterable<T> {
  return {
    [Symbol.asyncIterator](): AsyncIterator<T> {
      let cursor: string | null = null
      let buffer: T[] = []
      let bufferIndex = 0
      let done = false

      return {
        async next(): Promise<IteratorResult<T>> {
          // 缓冲区还有数据
          if (bufferIndex < buffer.length) {
            return { value: buffer[bufferIndex++], done: false }
          }

          // 已没有更多数据
          if (done) {
            return { value: undefined as unknown as T, done: true }
          }

          // 获取下一页
          const page = await fetcher(cursor)
          buffer = page.data
          bufferIndex = 0
          cursor = page.nextCursor
          done = page.nextCursor === null

          if (buffer.length === 0) {
            return { value: undefined as unknown as T, done: true }
          }

          return { value: buffer[bufferIndex++], done: false }
        },
      }
    },
  }
}
```

### for await...of

`for await...of` 是消费异步可迭代对象的最简洁方式：

```typescript
// 遍历所有用户（自动处理分页）
async function processAllUsers(): Promise<void> {
  const allUsers = paginate(fetchPage)

  for await (const user of allUsers) {
    // 每次迭代异步获取数据，但代码写起来像同步循环
    console.log(`处理用户: ${user.name}`)
  }

  console.log('所有用户处理完毕')
}

// for await...of 也可用于普通 Promise 数组
async function forAwaitOnPromises(): Promise<void> {
  const promises = [fetchUser(1), fetchUser(2), fetchUser(3)]

  // 注意：这是顺序等待，不是并发
  for await (const user of promises) {
    console.log(user.name)
  }
}
```

### 异步生成器

`async function*` 是创建异步可迭代对象的更简洁方式：

```typescript
// 异步生成器：更简洁的分页实现
async function* paginateWithGenerator<T>(
  fetcher: (cursor: string | null) => Promise<Page<T>>
): AsyncGenerator<T> {
  let cursor: string | null = null

  while (true) {
    const page = await fetcher(cursor)

    for (const item of page.data) {
      yield item  // 每次 yield 一个条目
    }

    if (page.nextCursor === null) break
    cursor = page.nextCursor
  }
}

// 异步生成器：模拟流式数据（如 LLM token 流）
async function* tokenStream(text: string): AsyncGenerator<string> {
  const tokens = text.split(' ')

  for (const token of tokens) {
    await delay(50)  // 模拟网络延迟
    yield token + ' '
  }
}

async function printTokenStream(): Promise<void> {
  const stream = tokenStream('Hello world this is a token stream')

  for await (const token of stream) {
    process.stdout.write(token)  // 逐 token 打印，不换行
  }
  console.log('\n流式输出完成')
}

// 异步生成器：重试逻辑
async function* withRetry<T>(
  operation: () => Promise<T>,
  maxRetries: number,
  delayMs: number
): AsyncGenerator<{ attempt: number; result?: T; error?: Error }> {
  for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
    try {
      const result = await operation()
      yield { attempt, result }
      return  // 成功后退出
    } catch (error) {
      yield { attempt, error: error instanceof Error ? error : new Error(String(error)) }

      if (attempt <= maxRetries) {
        await delay(delayMs * attempt)  // 指数退避
      }
    }
  }
}

// 使用重试生成器
async function fetchWithRetry(userId: number): Promise<UserProfile | null> {
  const retryGen = withRetry(() => fetchUser(userId), 3, 500)

  for await (const { attempt, result, error } of retryGen) {
    if (result) {
      console.log(`第 ${attempt} 次尝试成功`)
      return result
    }
    console.warn(`第 ${attempt} 次尝试失败: ${error?.message}`)
  }

  return null
}
```

---

## 本章小结

| 主题 | 概念/API | 核心要点 |
|------|----------|----------|
| Promise 基础 | `new Promise`、`resolve`、`reject` | 状态不可逆：Pending → Fulfilled/Rejected |
| 链式调用 | `.then()`、`.catch()`、`.finally()` | 错误向下传播，catch 可恢复链条 |
| async/await | `async function`、`await` | 语法糖，本质是 Promise；顺序 await 串行，需并发时用 Promise.all |
| 错误处理 | `try/catch`、自定义错误类 | 区分错误类型，使用 instanceof 分支处理 |
| 全部成功 | `Promise.all` | 任一失败立即拒绝，适合有强依赖的批量操作 |
| 竞速 | `Promise.race` | 第一个结束（成功或失败）决定结果，常用于超时控制 |
| 部分失败 | `Promise.allSettled` | 收集所有结果，适合可容忍部分失败的批量操作 |
| 第一成功 | `Promise.any` | 全部失败才拒绝，适合多备用源降级 |
| 异步迭代 | `for await...of`、`async function*` | 处理异步流、分页数据、token 流等场景 |

**最佳实践总结：**

1. 优先使用 `async/await`，仅在需要链式操作时使用 `.then()`
2. 无依赖的并发操作用 `Promise.all`，不要逐个 `await`
3. 批量操作且允许部分失败时，用 `Promise.allSettled` 而非 `Promise.all`
4. 定义具体的错误类型，用 `instanceof` 区分处理策略
5. 异步循环使用 `for...of` + `await` 或 `for await...of`，不要用 `forEach`

---

## AI 应用实战：并发 API 请求管理

在 AI 应用开发中，常常需要同时调用多个 LLM API（用于对比结果、投票决策或故障切换）。本节展示如何用 TypeScript 的并发控制优雅地管理这些请求。

### 场景设计

我们要构建一个**多模型聚合调用器**，它能够：

1. 并发调用多个 LLM（GPT-4、Claude、Gemini）
2. 取第一个成功的响应（降低延迟）
3. 在全部响应后进行投票或聚合
4. 处理超时、限流等错误

```typescript
// src/ai/types.ts

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface LLMResponse {
  model: string
  content: string
  tokensUsed: number
  latencyMs: number
}

export interface LLMConfig {
  name: string
  apiKey: string
  baseUrl: string
  model: string
  timeoutMs: number
}

// 调用结果（含错误信息）
export type CallResult<T> =
  | { status: 'fulfilled'; model: string; value: T; latencyMs: number }
  | { status: 'rejected'; model: string; error: Error; latencyMs: number }
```

### 单个 LLM 客户端

```typescript
// src/ai/llmClient.ts
import type { ChatMessage, LLMResponse, LLMConfig } from './types'

export class LLMClient {
  constructor(private readonly config: LLMConfig) {}

  get name(): string {
    return this.config.name
  }

  async chat(messages: ChatMessage[]): Promise<LLMResponse> {
    const start = Date.now()

    const controller = new AbortController()
    const timeoutId = setTimeout(
      () => controller.abort(),
      this.config.timeoutMs
    )

    try {
      const response = await fetch(`${this.config.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.config.apiKey}`,
        },
        body: JSON.stringify({
          model: this.config.model,
          messages,
        }),
        signal: controller.signal,
      })

      if (!response.ok) {
        const body = await response.json().catch(() => ({})) as Record<string, unknown>
        const errMsg = (body?.error as Record<string, unknown>)?.message as string | undefined
        throw new Error(errMsg ?? `HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json() as {
        choices: Array<{ message: { content: string } }>
        usage: { total_tokens: number }
      }

      return {
        model: this.config.name,
        content: data.choices[0]?.message.content ?? '',
        tokensUsed: data.usage.total_tokens,
        latencyMs: Date.now() - start,
      }
    } finally {
      clearTimeout(timeoutId)
    }
  }
}
```

### 多模型聚合调用器

```typescript
// src/ai/multiModelOrchestrator.ts
import type { ChatMessage, LLMResponse, CallResult } from './types'
import { LLMClient } from './llmClient'

export interface OrchestratorOptions {
  // 调用模式
  mode: 'fastest' | 'all' | 'vote'
  // 最小成功数（mode='all' 时生效）
  minSuccessCount?: number
  // 全局超时（毫秒）
  globalTimeoutMs?: number
  // 最大并发数
  maxConcurrent?: number
}

export interface OrchestratorResult {
  // 最终选用的响应
  primary: LLMResponse
  // 所有响应（含失败）
  all: Array<CallResult<LLMResponse>>
  // 统计信息
  stats: {
    successCount: number
    failureCount: number
    totalDurationMs: number
  }
}

export class MultiModelOrchestrator {
  private readonly clients: LLMClient[]

  constructor(clients: LLMClient[]) {
    this.clients = clients
  }

  async call(
    messages: ChatMessage[],
    options: OrchestratorOptions = { mode: 'fastest' }
  ): Promise<OrchestratorResult> {
    const start = Date.now()
    const { mode, globalTimeoutMs, maxConcurrent } = options

    // 为每个客户端调用包装一个带计时的 Promise
    const makeCall = async (client: LLMClient): Promise<CallResult<LLMResponse>> => {
      const callStart = Date.now()
      try {
        const value = await client.chat(messages)
        return {
          status: 'fulfilled',
          model: client.name,
          value,
          latencyMs: Date.now() - callStart,
        }
      } catch (err) {
        return {
          status: 'rejected',
          model: client.name,
          error: err instanceof Error ? err : new Error(String(err)),
          latencyMs: Date.now() - callStart,
        }
      }
    }

    // 构建任务列表（支持并发限制）
    const tasks = this.clients.map((client) => () => makeCall(client))

    // 应用全局超时
    const runTasks = async (): Promise<Array<CallResult<LLMResponse>>> => {
      if (maxConcurrent && maxConcurrent < this.clients.length) {
        return this.runWithConcurrencyLimit(tasks, maxConcurrent)
      }

      return Promise.allSettled(tasks.map((t) => t())).then((results) =>
        results.map((r) =>
          r.status === 'fulfilled'
            ? r.value
            : {
                status: 'rejected' as const,
                model: 'unknown',
                error: r.reason instanceof Error ? r.reason : new Error(String(r.reason)),
                latencyMs: 0,
              }
        )
      )
    }

    // 根据模式选择执行策略
    let allResults: Array<CallResult<LLMResponse>>

    if (mode === 'fastest') {
      allResults = await this.runFastest(tasks, globalTimeoutMs)
    } else {
      const withTimeout = globalTimeoutMs
        ? this.applyGlobalTimeout(runTasks(), globalTimeoutMs)
        : runTasks()
      allResults = await withTimeout
    }

    // 分析结果
    const successes = allResults.filter(
      (r): r is Extract<CallResult<LLMResponse>, { status: 'fulfilled' }> =>
        r.status === 'fulfilled'
    )
    const failures = allResults.filter((r) => r.status === 'rejected')

    if (successes.length === 0) {
      const errorMessages = failures
        .map((f) => `${f.model}: ${f.status === 'rejected' ? f.error.message : ''}`)
        .join('; ')
      throw new Error(`所有模型调用均失败: ${errorMessages}`)
    }

    // 选择主响应
    const primary = this.selectPrimary(successes, mode)

    return {
      primary: primary.value,
      all: allResults,
      stats: {
        successCount: successes.length,
        failureCount: failures.length,
        totalDurationMs: Date.now() - start,
      },
    }
  }

  // 竞速模式：取第一个成功的
  private async runFastest(
    tasks: Array<() => Promise<CallResult<LLMResponse>>>,
    timeoutMs?: number
  ): Promise<Array<CallResult<LLMResponse>>> {
    // 启动所有任务，但取第一个成功的响应
    const promises = tasks.map((t) => t())

    const firstSuccess = await Promise.any(
      promises.map(async (p) => {
        const result = await p
        if (result.status === 'rejected') {
          throw result.error
        }
        return result
      })
    ).catch(async () => {
      // 所有都失败了，等待全部完成并返回
      return null
    })

    // 收集其他结果（已经在进行，不额外等待）
    const allResults = await Promise.allSettled(promises).then((settled) =>
      settled.map((s) =>
        s.status === 'fulfilled'
          ? s.value
          : ({
              status: 'rejected' as const,
              model: 'unknown',
              error: s.reason instanceof Error ? s.reason : new Error(String(s.reason)),
              latencyMs: 0,
            } as CallResult<LLMResponse>)
      )
    )

    return firstSuccess ? allResults : allResults
  }

  // 并发限制执行
  private async runWithConcurrencyLimit(
    tasks: Array<() => Promise<CallResult<LLMResponse>>>,
    maxConcurrent: number
  ): Promise<Array<CallResult<LLMResponse>>> {
    const results: Array<CallResult<LLMResponse>> = new Array(tasks.length)
    let taskIndex = 0

    async function worker(): Promise<void> {
      while (taskIndex < tasks.length) {
        const currentIndex = taskIndex++
        results[currentIndex] = await tasks[currentIndex]()
      }
    }

    const workers = Array.from(
      { length: Math.min(maxConcurrent, tasks.length) },
      worker
    )
    await Promise.all(workers)

    return results
  }

  // 全局超时包装
  private applyGlobalTimeout<T>(
    promise: Promise<T>,
    timeoutMs: number
  ): Promise<T> {
    return Promise.race([
      promise,
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`全局超时 ${timeoutMs}ms`)), timeoutMs)
      ),
    ])
  }

  // 投票选择最佳响应（简单实现：取延迟最低的）
  private selectPrimary(
    successes: Array<Extract<CallResult<LLMResponse>, { status: 'fulfilled' }>>,
    _mode: OrchestratorOptions['mode']
  ): Extract<CallResult<LLMResponse>, { status: 'fulfilled' }> {
    // 按延迟排序，取最快的
    return successes.sort((a, b) => a.latencyMs - b.latencyMs)[0]
  }
}
```

### 实际使用示例

```typescript
// src/ai/example.ts
import { LLMClient } from './llmClient'
import { MultiModelOrchestrator } from './multiModelOrchestrator'
import type { ChatMessage } from './types'

// 配置多个 LLM 客户端
const orchestrator = new MultiModelOrchestrator([
  new LLMClient({
    name: 'gpt-4',
    apiKey: process.env.OPENAI_API_KEY!,
    baseUrl: 'https://api.openai.com/v1',
    model: 'gpt-4-turbo',
    timeoutMs: 30_000,
  }),
  new LLMClient({
    name: 'claude-3',
    apiKey: process.env.ANTHROPIC_API_KEY!,
    baseUrl: 'https://api.anthropic.com/v1',
    model: 'claude-3-5-sonnet-20241022',
    timeoutMs: 30_000,
  }),
  new LLMClient({
    name: 'gemini-pro',
    apiKey: process.env.GOOGLE_API_KEY!,
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai',
    model: 'gemini-pro',
    timeoutMs: 30_000,
  }),
])

const messages: ChatMessage[] = [
  { role: 'system', content: '你是一位专业的代码审查助手。' },
  { role: 'user', content: '请审查这段 TypeScript 代码并给出改进建议：\n\nconst x = (a,b) => a+b' },
]

// 模式一：竞速（取最快响应，适合对延迟敏感的场景）
async function fastestMode(): Promise<void> {
  console.log('=== 竞速模式 ===')
  const result = await orchestrator.call(messages, {
    mode: 'fastest',
    globalTimeoutMs: 10_000,
  })

  console.log(`最快响应来自: ${result.primary.model}（${result.primary.latencyMs}ms）`)
  console.log(`回答: ${result.primary.content.slice(0, 100)}...`)
  console.log(`成功: ${result.stats.successCount}，失败: ${result.stats.failureCount}`)
}

// 模式二：全量（等待所有响应，适合对比或审计）
async function allMode(): Promise<void> {
  console.log('\n=== 全量模式 ===')
  const result = await orchestrator.call(messages, {
    mode: 'all',
    globalTimeoutMs: 30_000,
    maxConcurrent: 2,  // 最多同时调用 2 个
  })

  console.log(`共收到 ${result.stats.successCount} 个响应，耗时 ${result.stats.totalDurationMs}ms`)

  result.all.forEach((r) => {
    if (r.status === 'fulfilled') {
      console.log(`\n[${r.model}] (${r.latencyMs}ms):`)
      console.log(r.value.content.slice(0, 150))
    } else {
      console.error(`\n[${r.model}] 失败: ${r.error.message}`)
    }
  })
}

// 模式三：带重试的调用
async function callWithRetry(
  messages: ChatMessage[],
  maxAttempts = 3
): Promise<string> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const result = await orchestrator.call(messages, { mode: 'fastest' })
      return result.primary.content
    } catch (err) {
      if (attempt === maxAttempts) throw err
      const waitMs = 1000 * Math.pow(2, attempt - 1)  // 指数退避
      console.warn(`第 ${attempt} 次调用失败，${waitMs}ms 后重试`)
      await new Promise<void>((resolve) => setTimeout(resolve, waitMs))
    }
  }
  throw new Error('不可达')  // TypeScript 需要，实际不会执行
}

// 主入口
async function main(): Promise<void> {
  try {
    await fastestMode()
    await allMode()
  } catch (err) {
    console.error('致命错误:', err instanceof Error ? err.message : err)
    process.exit(1)
  }
}

main()
```

### 关键设计要点

1. **Promise.allSettled 而非 Promise.all**：多模型调用允许部分失败，使用 `allSettled` 收集所有结果
2. **AbortController 实现超时**：每个客户端内部使用 `AbortController` 中止超时请求，释放网络资源
3. **类型安全的 Result 联合**：`CallResult<T>` 使用判别联合，TypeScript 能在处理时正确收窄类型
4. **并发限制保护服务端**：`maxConcurrent` 参数防止同时发起过多请求导致限流
5. **全局超时兜底**：`globalTimeoutMs` 作为最终保障，防止整个操作无限阻塞

---

## 练习题

### 基础练习

**练习 1**：实现一个 `delay` 工具函数，并用它编写一个 `countdown` 函数，每秒打印一次倒计时，最终打印"发射！"。

```typescript
// 要求：
// 1. delay(ms: number): Promise<void> —— 等待指定毫秒
// 2. countdown(from: number): Promise<void> —— 从 from 倒数到 0

// 示例输出：
// 3...
// 2...
// 1...
// 发射！
```

**练习 2**：将以下回调风格的函数改写为返回 Promise 的函数，并使用 `async/await` 调用它：

```typescript
// 原始回调风格
function readConfig(
  path: string,
  callback: (err: Error | null, config: Record<string, string> | null) => void
): void {
  setTimeout(() => {
    if (path === '') {
      callback(new Error('路径不能为空'), null)
    } else {
      callback(null, { host: 'localhost', port: '8080' })
    }
  }, 100)
}

// 要求：
// 1. 实现 readConfigAsync(path: string): Promise<Record<string, string>>
// 2. 用 async/await 调用它，正确处理成功和失败情况
```

---

### 中级练习

**练习 3**：实现一个 `retry` 函数，在异步操作失败时自动重试，支持指数退避：

```typescript
// 接口要求：
async function retry<T>(
  operation: () => Promise<T>,
  options: {
    maxAttempts: number
    initialDelayMs: number
    shouldRetry?: (error: Error) => boolean  // 可选，默认重试所有错误
  }
): Promise<T> {
  // 实现：
  // 1. 最多尝试 maxAttempts 次
  // 2. 每次失败后延迟 initialDelayMs * 2^(attempt-1) 毫秒
  // 3. 如果提供 shouldRetry 且返回 false，立即抛出不重试
  // 4. 所有尝试失败后，抛出最后一次的错误
}
```

**练习 4**：用 `for await...of` 和异步生成器实现一个流式文本处理管道：

```typescript
// 要求实现以下管道：
// 1. asyncTextSource(text: string): AsyncGenerator<string>
//    —— 将文本按单词拆分，每 50ms yield 一个单词
// 2. transformUpperCase(source: AsyncIterable<string>): AsyncGenerator<string>
//    —— 将每个 token 转为大写
// 3. collectToString(source: AsyncIterable<string>): Promise<string>
//    —— 收集所有 token 并拼接为字符串

// 最终使用：
// const result = await collectToString(
//   transformUpperCase(asyncTextSource('hello world foo bar'))
// )
// console.log(result)  // 'HELLO WORLD FOO BAR '
```

---

### 高级练习

**练习 5**：实现一个**异步事件总线**，支持发布/订阅模式，且订阅者可以是异步函数：

```typescript
// 要求：
// 1. EventBus<T extends Record<string, unknown>> 类
// 2. on(event: keyof T, handler: (payload: T[keyof T]) => Promise<void> | void): () => void
//    —— 注册事件处理器，返回取消订阅函数
// 3. emit(event: keyof T, payload: T[keyof T]): Promise<void>
//    —— 触发事件，等待所有处理器完成（使用 Promise.allSettled）
// 4. once(event: keyof T, handler: ...): void
//    —— 只处理一次，之后自动取消订阅

// 示例：
interface AppEvents {
  userCreated: { id: number; name: string }
  orderPlaced: { orderId: string; amount: number }
}

const bus = new EventBus<AppEvents>()

const unsubscribe = bus.on('userCreated', async (user) => {
  await delay(100)
  console.log(`欢迎 ${user.name}！`)
})

bus.once('orderPlaced', (order) => {
  console.log(`订单 ${order.orderId} 已接收`)
})

await bus.emit('userCreated', { id: 1, name: 'Alice' })
await bus.emit('orderPlaced', { orderId: 'ORD-001', amount: 299 })
await bus.emit('orderPlaced', { orderId: 'ORD-002', amount: 599 })  // once 不再触发

unsubscribe()  // 取消订阅
```

---

## 练习答案

### 答案 1：delay 与 countdown

```typescript
function delay(ms: number): Promise<void> {
  return new Promise<void>((resolve) => setTimeout(resolve, ms))
}

async function countdown(from: number): Promise<void> {
  for (let i = from; i > 0; i--) {
    console.log(`${i}...`)
    await delay(1000)
  }
  console.log('发射！')
}

// 调用
await countdown(3)
```

---

### 答案 2：Promise 化回调函数

```typescript
function readConfigAsync(path: string): Promise<Record<string, string>> {
  return new Promise((resolve, reject) => {
    readConfig(path, (err, config) => {
      if (err) {
        reject(err)
      } else {
        // 非空断言：err 为 null 时 config 一定不为 null
        resolve(config!)
      }
    })
  })
}

// 使用 async/await 调用
async function loadConfig(): Promise<void> {
  try {
    const config = await readConfigAsync('./app.config')
    console.log('配置加载成功:', config)
  } catch (err) {
    if (err instanceof Error) {
      console.error('配置加载失败:', err.message)
    }
  }
}

await loadConfig()
// 失败情况：
try {
  await readConfigAsync('')
} catch (err) {
  console.error('空路径错误:', (err as Error).message)  // 路径不能为空
}
```

---

### 答案 3：retry 函数

```typescript
async function retry<T>(
  operation: () => Promise<T>,
  options: {
    maxAttempts: number
    initialDelayMs: number
    shouldRetry?: (error: Error) => boolean
  }
): Promise<T> {
  const { maxAttempts, initialDelayMs, shouldRetry } = options
  let lastError: Error | null = null

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await operation()
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err))

      // 检查是否应该重试
      if (shouldRetry && !shouldRetry(lastError)) {
        throw lastError
      }

      // 最后一次尝试不需要等待
      if (attempt < maxAttempts) {
        const waitMs = initialDelayMs * Math.pow(2, attempt - 1)
        console.warn(`第 ${attempt}/${maxAttempts} 次失败，${waitMs}ms 后重试: ${lastError.message}`)
        await new Promise<void>((resolve) => setTimeout(resolve, waitMs))
      }
    }
  }

  throw lastError ?? new Error('操作失败（未知原因）')
}

// 测试
let callCount = 0
const result = await retry(
  async () => {
    callCount++
    if (callCount < 3) throw new Error(`第 ${callCount} 次失败`)
    return `第 ${callCount} 次成功`
  },
  { maxAttempts: 5, initialDelayMs: 100 }
)
console.log(result)  // 第 3 次成功

// 测试不重试某类错误
await retry(
  async () => { throw new Error('AUTH_FAILED') },
  {
    maxAttempts: 5,
    initialDelayMs: 100,
    shouldRetry: (err) => !err.message.includes('AUTH'),  // 认证错误不重试
  }
).catch((err: Error) => console.error('不重试，立即失败:', err.message))
```

---

### 答案 4：流式处理管道

```typescript
async function* asyncTextSource(text: string): AsyncGenerator<string> {
  const words = text.split(' ')
  for (const word of words) {
    await new Promise<void>((resolve) => setTimeout(resolve, 50))
    yield word + ' '
  }
}

async function* transformUpperCase(
  source: AsyncIterable<string>
): AsyncGenerator<string> {
  for await (const token of source) {
    yield token.toUpperCase()
  }
}

async function collectToString(source: AsyncIterable<string>): Promise<string> {
  let result = ''
  for await (const token of source) {
    result += token
  }
  return result
}

// 组合使用
const result = await collectToString(
  transformUpperCase(asyncTextSource('hello world foo bar'))
)
console.log(result)  // 'HELLO WORLD FOO BAR '

// 也可以拆步骤来看：
const source = asyncTextSource('hello world foo bar')
const upper = transformUpperCase(source)
const final = await collectToString(upper)
console.log(final)   // 'HELLO WORLD FOO BAR '
```

---

### 答案 5：异步事件总线

```typescript
type Handler<T> = (payload: T) => Promise<void> | void

class EventBus<T extends Record<string, unknown>> {
  private readonly handlers = new Map<
    keyof T,
    Set<Handler<T[keyof T]>>
  >()

  on<K extends keyof T>(
    event: K,
    handler: Handler<T[K]>
  ): () => void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set())
    }

    // 类型安全地存储处理器
    const handlerSet = this.handlers.get(event)!
    const wrappedHandler = handler as Handler<T[keyof T]>
    handlerSet.add(wrappedHandler)

    // 返回取消订阅函数
    return () => {
      handlerSet.delete(wrappedHandler)
    }
  }

  once<K extends keyof T>(
    event: K,
    handler: Handler<T[K]>
  ): void {
    // 包装：触发一次后自动取消
    const unsubscribe = this.on(event, async (payload) => {
      unsubscribe()
      await handler(payload as T[K])
    })
  }

  async emit<K extends keyof T>(
    event: K,
    payload: T[K]
  ): Promise<void> {
    const handlerSet = this.handlers.get(event)
    if (!handlerSet || handlerSet.size === 0) return

    // 并发调用所有处理器，收集错误但不中断
    const results = await Promise.allSettled(
      [...handlerSet].map((handler) =>
        Promise.resolve(handler(payload as T[keyof T]))
      )
    )

    // 报告失败的处理器
    results.forEach((result, index) => {
      if (result.status === 'rejected') {
        console.error(
          `事件 '${String(event)}' 的第 ${index + 1} 个处理器失败:`,
          result.reason
        )
      }
    })
  }
}

// 验证
interface AppEvents {
  userCreated: { id: number; name: string }
  orderPlaced: { orderId: string; amount: number }
}

const bus = new EventBus<AppEvents>()

const unsubscribe = bus.on('userCreated', async (user) => {
  await new Promise<void>((resolve) => setTimeout(resolve, 100))
  console.log(`欢迎 ${user.name}！`)
})

bus.once('orderPlaced', (order) => {
  console.log(`订单 ${order.orderId} 已接收（只触发一次）`)
})

await bus.emit('userCreated', { id: 1, name: 'Alice' })
// 输出: 欢迎 Alice！

await bus.emit('orderPlaced', { orderId: 'ORD-001', amount: 299 })
// 输出: 订单 ORD-001 已接收（只触发一次）

await bus.emit('orderPlaced', { orderId: 'ORD-002', amount: 599 })
// 无输出（once 已取消）

unsubscribe()

await bus.emit('userCreated', { id: 2, name: 'Bob' })
// 无输出（已取消订阅）
```

---

*本章完 — 下一章将介绍 HTTP 客户端与 API 调用，深入 fetch、axios 的使用及请求拦截器设计。*
