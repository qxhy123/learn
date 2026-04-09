# 第15章：测试与调试

## 学习目标

完成本章学习后，你将能够：

1. 配置并使用 Vitest 测试框架，理解其与 Jest 的差异和优势
2. 编写规范的单元测试，掌握 `describe`、`it`、`expect` 的使用方式
3. 使用 `tsd` 和 `expect-type` 对 TypeScript 类型进行静态测试
4. 理解并实践 Mock 与 Stub 技术，隔离外部依赖进行测试
5. 掌握 VS Code 调试配置和 sourcemap 技术，高效定位代码问题

---

## 15.1 Vitest 测试框架配置

### 为什么选择 Vitest

Vitest 是专为 Vite 生态设计的现代测试框架，与 Jest API 高度兼容，但具备以下优势：

- **原生 ESM 支持**：无需额外配置即可测试 ES 模块
- **TypeScript 支持**：无需 `ts-jest` 或 Babel 转译
- **极速运行**：基于 Vite 的按需编译，冷启动更快
- **统一配置**：与 `vite.config.ts` 共享配置
- **内置 UI**：提供可视化测试界面

### 安装与基础配置

```bash
# 初始化项目
mkdir ts-testing-demo && cd ts-testing-demo
npm init -y

# 安装依赖
npm install -D vitest @vitest/ui typescript vite
npm install -D @types/node

# 初始化 TypeScript 配置
npx tsc --init
```

`tsconfig.json` 配置：

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
    "rootDir": "./src",
    "types": ["vitest/globals"]
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

`vite.config.ts` 配置：

```typescript
import { defineConfig } from 'vite'

export default defineConfig({
  test: {
    // 全局注入 describe、it、expect，无需手动导入
    globals: true,
    // 测试环境：node（默认）、jsdom、happy-dom
    environment: 'node',
    // 覆盖率配置
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.ts'],
      exclude: ['src/**/*.test.ts', 'src/**/*.spec.ts'],
    },
    // 测试文件匹配模式
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    // 全局测试超时（毫秒）
    testTimeout: 10000,
    // 并发运行
    pool: 'threads',
    poolOptions: {
      threads: {
        maxThreads: 4,
      },
    },
  },
})
```

`package.json` 脚本配置：

```json
{
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest run --coverage",
    "typecheck": "tsc --noEmit"
  }
}
```

### 项目目录结构

```
ts-testing-demo/
├── src/
│   ├── utils/
│   │   ├── math.ts
│   │   └── math.test.ts
│   ├── services/
│   │   ├── api.ts
│   │   └── api.test.ts
│   └── types/
│       └── index.ts
├── vite.config.ts
├── tsconfig.json
└── package.json
```

### 运行测试

```bash
# 单次运行
npm test

# 监听模式（文件变更自动重跑）
npm run test:watch

# 可视化 UI
npm run test:ui

# 生成覆盖率报告
npm run test:coverage

# 运行特定文件
npx vitest run src/utils/math.test.ts

# 运行匹配关键字的测试
npx vitest run -t "addition"
```

---

## 15.2 单元测试编写

### 基础测试结构

Vitest 的 API 与 Jest 几乎完全相同，核心概念：

- `describe`：测试套件，用于分组相关测试
- `it` / `test`：单个测试用例
- `expect`：断言函数
- `beforeAll` / `afterAll`：套件级别的前置/后置钩子
- `beforeEach` / `afterEach`：用例级别的前置/后置钩子

```typescript
// src/utils/math.ts
export function add(a: number, b: number): number {
  return a + b
}

export function divide(a: number, b: number): number {
  if (b === 0) {
    throw new Error('Division by zero')
  }
  return a / b
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

export async function fetchSquare(n: number): Promise<number> {
  // 模拟异步操作
  return new Promise((resolve) => {
    setTimeout(() => resolve(n * n), 100)
  })
}
```

```typescript
// src/utils/math.test.ts
import { describe, it, expect, beforeAll, afterEach } from 'vitest'
import { add, divide, clamp, fetchSquare } from './math'

describe('数学工具函数', () => {
  // 测试套件级别的钩子
  beforeAll(() => {
    console.log('开始运行数学工具测试')
  })

  afterEach(() => {
    // 每个测试后清理（如果需要）
  })

  // --- add 函数测试 ---
  describe('add()', () => {
    it('应正确相加两个正整数', () => {
      expect(add(1, 2)).toBe(3)
      expect(add(10, 20)).toBe(30)
    })

    it('应处理负数', () => {
      expect(add(-1, -2)).toBe(-3)
      expect(add(-5, 5)).toBe(0)
    })

    it('应处理浮点数', () => {
      // 浮点数比较使用 toBeCloseTo
      expect(add(0.1, 0.2)).toBeCloseTo(0.3)
    })
  })

  // --- divide 函数测试 ---
  describe('divide()', () => {
    it('应正确除两个数', () => {
      expect(divide(10, 2)).toBe(5)
      expect(divide(9, 3)).toBe(3)
    })

    it('应在除以零时抛出错误', () => {
      expect(() => divide(10, 0)).toThrow('Division by zero')
      expect(() => divide(10, 0)).toThrow(Error)
    })
  })

  // --- clamp 函数测试 ---
  describe('clamp()', () => {
    it.each([
      // [value, min, max, expected]
      [5, 0, 10, 5],   // 在范围内
      [-5, 0, 10, 0],  // 小于最小值
      [15, 0, 10, 10], // 大于最大值
      [0, 0, 10, 0],   // 等于最小值边界
      [10, 0, 10, 10], // 等于最大值边界
    ])('clamp(%i, %i, %i) 应返回 %i', (value, min, max, expected) => {
      expect(clamp(value, min, max)).toBe(expected)
    })
  })

  // --- 异步测试 ---
  describe('fetchSquare()', () => {
    it('应返回数字的平方', async () => {
      const result = await fetchSquare(5)
      expect(result).toBe(25)
    })

    it('应在超时内完成', async () => {
      const start = Date.now()
      await fetchSquare(3)
      const elapsed = Date.now() - start
      expect(elapsed).toBeLessThan(500)
    })
  })
})
```

### 常用断言（Matchers）

```typescript
// src/utils/assertions-demo.test.ts
import { describe, it, expect } from 'vitest'

describe('Vitest 断言示例', () => {
  it('基本相等断言', () => {
    expect(1 + 1).toBe(2)                    // 严格相等（===）
    expect({ a: 1 }).toEqual({ a: 1 })       // 深度相等
    expect({ a: 1 }).not.toBe({ a: 1 })      // 不是同一引用
  })

  it('真值/假值断言', () => {
    expect(true).toBeTruthy()
    expect(false).toBeFalsy()
    expect(null).toBeNull()
    expect(undefined).toBeUndefined()
    expect(0).toBeDefined()                   // 已定义（非 undefined）
  })

  it('数值断言', () => {
    expect(5).toBeGreaterThan(4)
    expect(5).toBeGreaterThanOrEqual(5)
    expect(3).toBeLessThan(4)
    expect(3.14159).toBeCloseTo(3.14, 2)     // 精度到小数点后 2 位
  })

  it('字符串断言', () => {
    expect('hello world').toContain('world')
    expect('hello').toMatch(/^hell/)
    expect('typescript').toHaveLength(10)
  })

  it('数组/对象断言', () => {
    expect([1, 2, 3]).toContain(2)
    expect([1, 2, 3]).toHaveLength(3)
    expect({ name: 'Alice', age: 30 }).toMatchObject({ name: 'Alice' })
    expect([1, 2, 3]).toEqual(expect.arrayContaining([1, 3]))
  })

  it('异常断言', () => {
    const throwFn = () => { throw new TypeError('bad input') }
    expect(throwFn).toThrow()
    expect(throwFn).toThrow(TypeError)
    expect(throwFn).toThrow('bad input')
  })

  it('快照测试', () => {
    const config = {
      host: 'localhost',
      port: 8080,
      debug: false,
    }
    expect(config).toMatchInlineSnapshot(`
      {
        "debug": false,
        "host": "localhost",
        "port": 8080,
      }
    `)
  })
})
```

### 测试生命周期与上下文共享

```typescript
// src/services/database.test.ts
import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest'

// 模拟数据库连接
class MockDatabase {
  private records: Map<string, unknown> = new Map()
  connected = false

  connect() { this.connected = true }
  disconnect() { this.connected = false }
  set(key: string, value: unknown) { this.records.set(key, value) }
  get(key: string) { return this.records.get(key) }
  clear() { this.records.clear() }
}

describe('数据库服务', () => {
  let db: MockDatabase

  // 整个套件只执行一次
  beforeAll(() => {
    db = new MockDatabase()
    db.connect()
  })

  afterAll(() => {
    db.disconnect()
  })

  // 每个测试前重置数据
  beforeEach(() => {
    db.clear()
  })

  it('应存储并检索数据', () => {
    db.set('user:1', { name: 'Alice' })
    expect(db.get('user:1')).toEqual({ name: 'Alice' })
  })

  it('未找到键时应返回 undefined', () => {
    expect(db.get('nonexistent')).toBeUndefined()
  })

  it('应在 beforeEach 后状态为空', () => {
    // 前一个测试设置的数据已被清除
    expect(db.get('user:1')).toBeUndefined()
  })
})
```

---

## 15.3 类型测试

TypeScript 的类型系统是核心资产。除了运行时测试，我们还需要验证类型推断是否符合预期。

### 使用 expect-type 进行内联类型测试

```bash
npm install -D expect-type
```

```typescript
// src/types/index.ts
export type User = {
  id: number
  name: string
  email: string
  role: 'admin' | 'user' | 'guest'
  createdAt: Date
}

export type PartialUser = Partial<User>
export type UserKeys = keyof User
export type ReadonlyUser = Readonly<User>

// 工具类型：提取函数返回类型
export type AsyncReturnType<T extends (...args: unknown[]) => Promise<unknown>> =
  T extends (...args: unknown[]) => Promise<infer R> ? R : never

// 工具类型：深度只读
export type DeepReadonly<T> = {
  readonly [K in keyof T]: T[K] extends object ? DeepReadonly<T[K]> : T[K]
}
```

```typescript
// src/types/index.test.ts
import { describe, it } from 'vitest'
import { expectTypeOf } from 'expect-type'
import type { User, PartialUser, UserKeys, AsyncReturnType } from './index'

describe('类型测试', () => {
  it('User 类型应包含正确字段', () => {
    expectTypeOf<User>().toHaveProperty('id')
    expectTypeOf<User>().toHaveProperty('name')
    expectTypeOf<User['id']>().toBeNumber()
    expectTypeOf<User['name']>().toBeString()
    expectTypeOf<User['role']>().toEqualTypeOf<'admin' | 'user' | 'guest'>()
  })

  it('PartialUser 应使所有字段可选', () => {
    expectTypeOf<PartialUser['id']>().toEqualTypeOf<number | undefined>()
    expectTypeOf<PartialUser['name']>().toEqualTypeOf<string | undefined>()
  })

  it('UserKeys 应为 User 的键联合类型', () => {
    expectTypeOf<UserKeys>().toEqualTypeOf<
      'id' | 'name' | 'email' | 'role' | 'createdAt'
    >()
  })

  it('AsyncReturnType 应正确提取 Promise 返回类型', () => {
    type FetchUser = () => Promise<User>
    expectTypeOf<AsyncReturnType<FetchUser>>().toEqualTypeOf<User>()
  })

  it('函数参数类型测试', () => {
    function greet(user: User): string {
      return `Hello, ${user.name}`
    }

    expectTypeOf(greet).parameter(0).toEqualTypeOf<User>()
    expectTypeOf(greet).returns.toBeString()
  })

  it('泛型函数类型测试', () => {
    function identity<T>(value: T): T {
      return value
    }

    expectTypeOf(identity<string>).toEqualTypeOf<(value: string) => string>()
    expectTypeOf(identity<number>).toEqualTypeOf<(value: number) => number>()
  })
})
```

### 使用 tsd 进行独立类型测试文件

```bash
npm install -D tsd
```

`package.json` 中添加 tsd 配置：

```json
{
  "scripts": {
    "test:types": "tsd"
  },
  "tsd": {
    "directory": "src/types"
  }
}
```

```typescript
// src/types/index.test-d.ts
import { expectType, expectError, expectAssignable } from 'tsd'
import type { User, DeepReadonly } from './index'

// 验证合法赋值
const user: User = {
  id: 1,
  name: 'Alice',
  email: 'alice@example.com',
  role: 'admin',
  createdAt: new Date(),
}

expectType<number>(user.id)
expectType<string>(user.name)
expectType<'admin' | 'user' | 'guest'>(user.role)

// 验证非法赋值会产生错误
expectError<User>({
  id: 'not-a-number',  // id 应为 number
  name: 'Bob',
  email: 'bob@example.com',
  role: 'superadmin',  // 非法 role
  createdAt: new Date(),
})

// 验证类型可赋值性
expectAssignable<Partial<User>>(user)

// 验证 DeepReadonly
type ReadonlyConfig = DeepReadonly<{ db: { host: string; port: number } }>
const config: ReadonlyConfig = { db: { host: 'localhost', port: 5432 } }

// 这应该产生类型错误
expectError(() => {
  // @ts-expect-error 只读属性不可修改
  config.db.host = 'remotehost'
})
```

---

## 15.4 Mock 与 Stub

### Vitest 内置 Mock 功能

Mock 是测试中的核心技术，用于替换真实依赖（如网络请求、数据库、外部 API）。

```typescript
// src/services/userService.ts
import type { User } from '../types'

// 模拟外部依赖
export interface HttpClient {
  get<T>(url: string): Promise<T>
  post<T>(url: string, data: unknown): Promise<T>
}

export class UserService {
  constructor(private readonly http: HttpClient) {}

  async getUser(id: number): Promise<User> {
    return this.http.get<User>(`/api/users/${id}`)
  }

  async createUser(data: Omit<User, 'id' | 'createdAt'>): Promise<User> {
    return this.http.post<User>('/api/users', {
      ...data,
      createdAt: new Date(),
    })
  }

  async getUsersByRole(role: User['role']): Promise<User[]> {
    const users = await this.http.get<User[]>('/api/users')
    return users.filter((u) => u.role === role)
  }
}
```

```typescript
// src/services/userService.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { UserService } from './userService'
import type { HttpClient, User } from '../types'

// 创建类型安全的 Mock 对象
function createMockHttpClient(): HttpClient {
  return {
    get: vi.fn(),
    post: vi.fn(),
  }
}

const mockUser: User = {
  id: 1,
  name: 'Alice',
  email: 'alice@example.com',
  role: 'admin',
  createdAt: new Date('2024-01-01'),
}

describe('UserService', () => {
  let mockHttp: HttpClient
  let userService: UserService

  beforeEach(() => {
    // 每次测试前重置 Mock 状态
    mockHttp = createMockHttpClient()
    userService = new UserService(mockHttp)
  })

  describe('getUser()', () => {
    it('应调用正确的 API 端点并返回用户', async () => {
      // Arrange：设置 Mock 返回值
      vi.mocked(mockHttp.get).mockResolvedValueOnce(mockUser)

      // Act：执行操作
      const result = await userService.getUser(1)

      // Assert：验证结果和调用
      expect(result).toEqual(mockUser)
      expect(mockHttp.get).toHaveBeenCalledOnce()
      expect(mockHttp.get).toHaveBeenCalledWith('/api/users/1')
    })

    it('应在 API 失败时传播错误', async () => {
      vi.mocked(mockHttp.get).mockRejectedValueOnce(new Error('Network error'))

      await expect(userService.getUser(1)).rejects.toThrow('Network error')
    })
  })

  describe('getUsersByRole()', () => {
    it('应按角色过滤用户', async () => {
      const users: User[] = [
        { ...mockUser, id: 1, role: 'admin' },
        { ...mockUser, id: 2, role: 'user' },
        { ...mockUser, id: 3, role: 'admin' },
      ]

      vi.mocked(mockHttp.get).mockResolvedValueOnce(users)

      const admins = await userService.getUsersByRole('admin')

      expect(admins).toHaveLength(2)
      expect(admins.every((u) => u.role === 'admin')).toBe(true)
    })
  })
})
```

### Mock 模块（vi.mock）

```typescript
// src/utils/logger.ts
export const logger = {
  info: (message: string, data?: unknown) => {
    console.log(`[INFO] ${message}`, data)
  },
  error: (message: string, error?: unknown) => {
    console.error(`[ERROR] ${message}`, error)
  },
}
```

```typescript
// src/services/orderService.ts
import { logger } from '../utils/logger'

export async function processOrder(orderId: string): Promise<{ success: boolean }> {
  logger.info('Processing order', { orderId })

  try {
    // 模拟处理逻辑
    if (!orderId.startsWith('ORD-')) {
      throw new Error(`Invalid order ID: ${orderId}`)
    }
    logger.info('Order processed successfully', { orderId })
    return { success: true }
  } catch (error) {
    logger.error('Order processing failed', error)
    return { success: false }
  }
}
```

```typescript
// src/services/orderService.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock 整个模块（在任何导入之前调用）
vi.mock('../utils/logger', () => ({
  logger: {
    info: vi.fn(),
    error: vi.fn(),
  },
}))

import { processOrder } from './orderService'
import { logger } from '../utils/logger'

describe('processOrder()', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('应成功处理合法订单并记录日志', async () => {
    const result = await processOrder('ORD-12345')

    expect(result.success).toBe(true)
    expect(logger.info).toHaveBeenCalledTimes(2)
    expect(logger.info).toHaveBeenCalledWith('Processing order', { orderId: 'ORD-12345' })
    expect(logger.error).not.toHaveBeenCalled()
  })

  it('应处理无效订单并记录错误日志', async () => {
    const result = await processOrder('INVALID-123')

    expect(result.success).toBe(false)
    expect(logger.error).toHaveBeenCalledOnce()
    expect(logger.info).toHaveBeenCalledTimes(1) // 只有开始的 info
  })
})
```

### 使用 vi.spyOn 监听真实方法

```typescript
// src/utils/timer.test.ts
import { describe, it, expect, vi, afterEach } from 'vitest'

describe('定时器测试', () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('应使用虚拟定时器测试延迟逻辑', async () => {
    vi.useFakeTimers()

    let called = false
    setTimeout(() => {
      called = false
    }, 1000)

    // 推进虚拟时钟
    await vi.advanceTimersByTimeAsync(1000)

    expect(called).toBe(false)
  })

  it('应监听 Date.now()', () => {
    const mockDate = new Date('2024-06-15T10:00:00Z')
    vi.setSystemTime(mockDate)

    const spy = vi.spyOn(Date, 'now')

    Date.now()

    expect(spy).toHaveBeenCalledOnce()
    expect(Date.now()).toBe(mockDate.getTime())
  })

  it('应监听对象方法', () => {
    const calc = {
      multiply(a: number, b: number): number {
        return a * b
      },
    }

    const spy = vi.spyOn(calc, 'multiply')

    calc.multiply(3, 4)
    calc.multiply(5, 6)

    expect(spy).toHaveBeenCalledTimes(2)
    expect(spy).toHaveBeenNthCalledWith(1, 3, 4)
    expect(spy).toHaveBeenNthCalledWith(2, 5, 6)
    expect(spy).toHaveReturnedWith(12)
    expect(spy).toHaveReturnedWith(30)
  })
})
```

---

## 15.5 调试技巧

### VS Code 调试配置

在项目根目录创建 `.vscode/launch.json`：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Vitest Tests",
      "type": "node",
      "request": "launch",
      "program": "${workspaceFolder}/node_modules/vitest/vitest.mjs",
      "args": ["run", "--reporter=verbose", "--no-coverage"],
      "console": "integratedTerminal",
      "internalConsoleOptions": "neverOpen",
      "smartStep": true,
      "sourceMaps": true,
      "cwd": "${workspaceFolder}"
    },
    {
      "name": "Debug Current Test File",
      "type": "node",
      "request": "launch",
      "program": "${workspaceFolder}/node_modules/vitest/vitest.mjs",
      "args": ["run", "${relativeFile}", "--reporter=verbose"],
      "console": "integratedTerminal",
      "internalConsoleOptions": "neverOpen",
      "smartStep": true,
      "sourceMaps": true,
      "cwd": "${workspaceFolder}"
    },
    {
      "name": "Debug TypeScript File",
      "type": "node",
      "request": "launch",
      "runtimeArgs": [
        "--loader",
        "ts-node/esm"
      ],
      "args": ["${file}"],
      "console": "integratedTerminal",
      "skipFiles": ["<node_internals>/**"],
      "sourceMaps": true
    }
  ]
}
```

### Sourcemap 配置

Sourcemap 将编译后的 JavaScript 映射回原始 TypeScript，让调试器显示 TypeScript 源码。

`tsconfig.json` 中开启：

```json
{
  "compilerOptions": {
    "sourceMap": true,
    "inlineSources": true,
    "sourceRoot": "/"
  }
}
```

`vite.config.ts` 中开启：

```typescript
import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    sourcemap: true,  // 'inline' | 'hidden' | true
  },
  test: {
    // 测试中也开启 sourcemap
    coverage: {
      sourcemap: true,
    },
  },
})
```

### 常用调试技巧

**技巧一：使用 `console.log` 替代方案**

```typescript
// 在测试中输出结构化信息（不影响测试结果）
import { describe, it, expect } from 'vitest'

describe('调试技巧', () => {
  it('使用 structuredClone 打印深拷贝', () => {
    const obj = { a: { b: { c: 42 } } }

    // 避免引用问题，打印当前快照
    console.dir(structuredClone(obj), { depth: null })

    expect(obj.a.b.c).toBe(42)
  })
})
```

**技巧二：使用 `--inspect` 标志进行 Node 调试**

```bash
# 在调试模式下运行测试（监听 9229 端口）
node --inspect-brk node_modules/.bin/vitest run src/services/userService.test.ts
```

然后在 Chrome 浏览器打开 `chrome://inspect`，点击 "inspect" 连接调试器。

**技巧三：隔离单个测试**

```typescript
// 使用 .only 只运行特定测试（调试完记得删除）
it.only('这个测试有问题', () => {
  // ...
})

// 使用 .skip 跳过暂时无法修复的测试
it.skip('这个测试待修复', () => {
  // ...
})

// 使用 .todo 标记待实现的测试
it.todo('待实现：用户权限验证测试')
```

**技巧四：测试中的类型断言调试**

```typescript
// src/debug/typecheck-demo.ts
function processValue(value: string | number | null): string {
  if (value === null) {
    return 'null'
  }

  if (typeof value === 'number') {
    // 这里 value 被 TypeScript 缩窄为 number
    return value.toFixed(2)
  }

  // 这里 value 被缩窄为 string
  return value.toUpperCase()
}

// 验证类型缩窄在运行时符合预期
import { describe, it, expect } from 'vitest'

describe('processValue 类型缩窄', () => {
  it('应处理所有类型', () => {
    expect(processValue(null)).toBe('null')
    expect(processValue(3.14159)).toBe('3.14')
    expect(processValue('hello')).toBe('HELLO')
  })
})
```

**技巧五：使用 `vi.fn()` 追踪调用链**

```typescript
// 追踪复杂调用链
const callTracker = {
  calls: [] as string[],
  track(name: string) {
    this.calls.push(name)
  },
}

// 在测试后分析调用顺序
console.log('调用顺序:', callTracker.calls)
```

---

## 本章小结

| 主题 | 工具/技术 | 核心用途 |
|------|-----------|----------|
| 测试框架 | Vitest | 运行单元测试、集成测试，生成覆盖率报告 |
| 测试结构 | `describe` / `it` / `expect` | 组织和编写测试用例 |
| 类型测试 | `expect-type` / `tsd` | 验证 TypeScript 类型推断是否正确 |
| Mock 替换 | `vi.mock()` | 替换整个模块，隔离外部依赖 |
| 函数监听 | `vi.spyOn()` | 监听真实对象方法的调用情况 |
| 虚拟定时器 | `vi.useFakeTimers()` | 控制时间流逝，测试定时逻辑 |
| VS Code 调试 | `launch.json` | 在 IDE 中设置断点，单步调试测试 |
| Sourcemap | `tsconfig.json` + `vite.config.ts` | 将调试信息映射回 TypeScript 源码 |
| 测试隔离 | `it.only` / `it.skip` | 临时聚焦或跳过特定测试 |

**最佳实践总结：**

1. 遵循 **AAA 模式**（Arrange - Act - Assert），每个测试只验证一件事
2. 测试函数名清晰描述预期行为，如"应在输入无效时抛出错误"
3. 优先测试**边界条件**：空值、零、最大值、并发等
4. Mock 只替换**你不控制的依赖**（网络、数据库、文件系统）
5. 保持测试**独立性**：使用 `beforeEach` 重置状态，避免测试间依赖

---

## AI 应用实战：API Mock 与集成测试

本节演示如何在不调用真实 LLM API 的情况下测试 AI 应用。这在 CI/CD 环境中尤为重要，可避免不必要的 API 费用和网络依赖。

### 场景：测试 LLM 驱动的文本分析服务

```typescript
// src/ai/types.ts
export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface LLMResponse {
  id: string
  choices: Array<{
    message: ChatMessage
    finish_reason: 'stop' | 'length' | 'content_filter'
  }>
  usage: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

export interface LLMClient {
  chat(messages: ChatMessage[]): Promise<LLMResponse>
}

export interface AnalysisResult {
  sentiment: 'positive' | 'negative' | 'neutral'
  keywords: string[]
  summary: string
  confidence: number
}
```

```typescript
// src/ai/textAnalyzer.ts
import type { LLMClient, ChatMessage, AnalysisResult } from './types'

export class TextAnalyzer {
  private readonly systemPrompt = `你是一个专业的文本分析助手。
分析用户提供的文本，返回 JSON 格式的结果，包含以下字段：
- sentiment: 情感倾向（positive/negative/neutral）
- keywords: 关键词数组（最多5个）
- summary: 一句话摘要
- confidence: 置信度（0-1）

只返回纯 JSON，不要有其他文字。`

  constructor(private readonly llm: LLMClient) {}

  async analyze(text: string): Promise<AnalysisResult> {
    const messages: ChatMessage[] = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: text },
    ]

    const response = await this.llm.chat(messages)
    const content = response.choices[0]?.message.content

    if (!content) {
      throw new Error('LLM returned empty response')
    }

    try {
      return JSON.parse(content) as AnalysisResult
    } catch {
      throw new Error(`Failed to parse LLM response: ${content}`)
    }
  }

  async batchAnalyze(texts: string[]): Promise<AnalysisResult[]> {
    return Promise.all(texts.map((text) => this.analyze(text)))
  }
}
```

```typescript
// src/ai/textAnalyzer.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { TextAnalyzer } from './textAnalyzer'
import type { LLMClient, LLMResponse, AnalysisResult } from './types'

// =============================================
// Mock LLM 响应构建器（测试工厂函数）
// =============================================
function createMockLLMResponse(content: string, tokensUsed = 100): LLMResponse {
  return {
    id: `mock-${Date.now()}`,
    choices: [
      {
        message: {
          role: 'assistant',
          content,
        },
        finish_reason: 'stop',
      },
    ],
    usage: {
      prompt_tokens: 50,
      completion_tokens: tokensUsed - 50,
      total_tokens: tokensUsed,
    },
  }
}

function createMockAnalysisResult(overrides: Partial<AnalysisResult> = {}): AnalysisResult {
  return {
    sentiment: 'positive',
    keywords: ['测试', '分析', 'AI'],
    summary: '这是一段正面的测试文本',
    confidence: 0.92,
    ...overrides,
  }
}

// =============================================
// 创建 Mock LLM 客户端
// =============================================
function createMockLLMClient(): LLMClient {
  return {
    chat: vi.fn(),
  }
}

describe('TextAnalyzer', () => {
  let mockLLM: LLMClient
  let analyzer: TextAnalyzer

  beforeEach(() => {
    mockLLM = createMockLLMClient()
    analyzer = new TextAnalyzer(mockLLM)
    vi.clearAllMocks()
  })

  // -------------------------------------------
  // 正常情况测试
  // -------------------------------------------
  describe('analyze() - 正常情况', () => {
    it('应调用 LLM 并解析返回的 JSON', async () => {
      const mockResult = createMockAnalysisResult()
      vi.mocked(mockLLM.chat).mockResolvedValueOnce(
        createMockLLMResponse(JSON.stringify(mockResult))
      )

      const result = await analyzer.analyze('这是一段很棒的产品评价！')

      expect(result).toEqual(mockResult)
      expect(result.sentiment).toBe('positive')
      expect(result.confidence).toBeGreaterThan(0.9)
    })

    it('应在调用时传递正确的消息结构', async () => {
      const mockResult = createMockAnalysisResult()
      vi.mocked(mockLLM.chat).mockResolvedValueOnce(
        createMockLLMResponse(JSON.stringify(mockResult))
      )

      const inputText = '分析这段文字'
      await analyzer.analyze(inputText)

      expect(mockLLM.chat).toHaveBeenCalledOnce()
      const calledMessages = vi.mocked(mockLLM.chat).mock.calls[0][0]

      // 验证消息结构
      expect(calledMessages).toHaveLength(2)
      expect(calledMessages[0].role).toBe('system')
      expect(calledMessages[1].role).toBe('user')
      expect(calledMessages[1].content).toBe(inputText)
    })

    it('应正确处理负面情感分析', async () => {
      const negativeResult = createMockAnalysisResult({
        sentiment: 'negative',
        keywords: ['问题', '故障', '失望'],
        summary: '这是一段负面的用户反馈',
        confidence: 0.88,
      })

      vi.mocked(mockLLM.chat).mockResolvedValueOnce(
        createMockLLMResponse(JSON.stringify(negativeResult))
      )

      const result = await analyzer.analyze('产品质量很差，完全无法使用')
      expect(result.sentiment).toBe('negative')
      expect(result.keywords).toContain('问题')
    })
  })

  // -------------------------------------------
  // 错误处理测试
  // -------------------------------------------
  describe('analyze() - 错误处理', () => {
    it('应在 LLM 返回空响应时抛出错误', async () => {
      vi.mocked(mockLLM.chat).mockResolvedValueOnce(
        createMockLLMResponse('')
      )

      await expect(analyzer.analyze('测试')).rejects.toThrow(
        'LLM returned empty response'
      )
    })

    it('应在 LLM 返回无效 JSON 时抛出解析错误', async () => {
      vi.mocked(mockLLM.chat).mockResolvedValueOnce(
        createMockLLMResponse('这不是 JSON 格式的响应')
      )

      await expect(analyzer.analyze('测试')).rejects.toThrow(
        'Failed to parse LLM response'
      )
    })

    it('应在 LLM API 调用失败时传播网络错误', async () => {
      vi.mocked(mockLLM.chat).mockRejectedValueOnce(
        new Error('Rate limit exceeded')
      )

      await expect(analyzer.analyze('测试')).rejects.toThrow(
        'Rate limit exceeded'
      )
    })
  })

  // -------------------------------------------
  // 批量处理测试
  // -------------------------------------------
  describe('batchAnalyze()', () => {
    it('应并发分析多段文本', async () => {
      const results = [
        createMockAnalysisResult({ sentiment: 'positive' }),
        createMockAnalysisResult({ sentiment: 'negative' }),
        createMockAnalysisResult({ sentiment: 'neutral' }),
      ]

      // 依次返回不同结果
      results.forEach((result) => {
        vi.mocked(mockLLM.chat).mockResolvedValueOnce(
          createMockLLMResponse(JSON.stringify(result))
        )
      })

      const texts = ['正面文本', '负面文本', '中性文本']
      const batchResults = await analyzer.batchAnalyze(texts)

      expect(batchResults).toHaveLength(3)
      expect(batchResults[0].sentiment).toBe('positive')
      expect(batchResults[1].sentiment).toBe('negative')
      expect(batchResults[2].sentiment).toBe('neutral')
      expect(mockLLM.chat).toHaveBeenCalledTimes(3)
    })

    it('应在任意一个分析失败时拒绝整个批量操作', async () => {
      vi.mocked(mockLLM.chat)
        .mockResolvedValueOnce(
          createMockLLMResponse(JSON.stringify(createMockAnalysisResult()))
        )
        .mockRejectedValueOnce(new Error('API Error'))

      await expect(
        analyzer.batchAnalyze(['文本一', '文本二'])
      ).rejects.toThrow('API Error')
    })
  })
})
```

### 集成测试：使用 MSW 拦截 HTTP 请求

对于直接调用 HTTP API 的服务，可以使用 `msw`（Mock Service Worker）拦截请求：

```bash
npm install -D msw
```

```typescript
// src/ai/openaiClient.ts
import type { LLMClient, ChatMessage, LLMResponse } from './types'

export class OpenAIClient implements LLMClient {
  constructor(
    private readonly apiKey: string,
    private readonly model: string = 'gpt-4'
  ) {}

  async chat(messages: ChatMessage[]): Promise<LLMResponse> {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        messages,
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(`OpenAI API error: ${error.error?.message ?? response.statusText}`)
    }

    return response.json() as Promise<LLMResponse>
  }
}
```

```typescript
// src/ai/openaiClient.test.ts
import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { OpenAIClient } from './openaiClient'
import type { LLMResponse } from './types'

// 创建 MSW 服务器
const server = setupServer()

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('OpenAIClient（MSW 集成测试）', () => {
  const client = new OpenAIClient('test-api-key', 'gpt-4')

  it('应成功调用 API 并返回响应', async () => {
    const mockResponse: LLMResponse = {
      id: 'chatcmpl-test123',
      choices: [
        {
          message: { role: 'assistant', content: '{"sentiment":"positive"}' },
          finish_reason: 'stop',
        },
      ],
      usage: { prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 },
    }

    // 注册请求处理器
    server.use(
      http.post('https://api.openai.com/v1/chat/completions', () => {
        return HttpResponse.json(mockResponse)
      })
    )

    const result = await client.chat([
      { role: 'user', content: '分析这段文字' },
    ])

    expect(result.id).toBe('chatcmpl-test123')
    expect(result.choices[0].message.content).toBe('{"sentiment":"positive"}')
  })

  it('应在 API 返回 429 时抛出限流错误', async () => {
    server.use(
      http.post('https://api.openai.com/v1/chat/completions', () => {
        return HttpResponse.json(
          { error: { message: 'Rate limit exceeded', type: 'rate_limit_error' } },
          { status: 429 }
        )
      })
    )

    await expect(
      client.chat([{ role: 'user', content: '测试' }])
    ).rejects.toThrow('Rate limit exceeded')
  })

  it('应在 API 返回 401 时抛出认证错误', async () => {
    server.use(
      http.post('https://api.openai.com/v1/chat/completions', () => {
        return HttpResponse.json(
          { error: { message: 'Invalid API key', type: 'auth_error' } },
          { status: 401 }
        )
      })
    )

    await expect(
      client.chat([{ role: 'user', content: '测试' }])
    ).rejects.toThrow('Invalid API key')
  })
})
```

---

## 练习题

### 基础练习

**练习 1**：为以下 `Stack` 类编写完整的单元测试，覆盖所有方法和边界条件：

```typescript
// src/utils/stack.ts
export class Stack<T> {
  private items: T[] = []

  push(item: T): void {
    this.items.push(item)
  }

  pop(): T {
    if (this.isEmpty()) {
      throw new Error('Stack is empty')
    }
    return this.items.pop()!
  }

  peek(): T {
    if (this.isEmpty()) {
      throw new Error('Stack is empty')
    }
    return this.items[this.items.length - 1]
  }

  isEmpty(): boolean {
    return this.items.length === 0
  }

  get size(): number {
    return this.items.length
  }
}
```

要求：编写至少 8 个测试用例，使用 `describe` 按方法分组，覆盖空栈操作、泛型类型等边界情况。

---

**练习 2**：配置 Vitest 并编写类型测试，验证以下工具类型的行为：

```typescript
// 要测试的类型
type NonNullable<T> = T extends null | undefined ? never : T
type ReturnType<T extends (...args: unknown[]) => unknown> =
  T extends (...args: unknown[]) => infer R ? R : never
type Parameters<T extends (...args: unknown[]) => unknown> =
  T extends (...args: infer P) => unknown ? P : never
```

要求：使用 `expect-type` 编写类型测试，验证每个工具类型在各种输入下的输出。

---

### 中级练习

**练习 3**：为以下缓存服务编写测试，要求使用 `vi.useFakeTimers()` 测试 TTL（生存时间）过期逻辑：

```typescript
// src/services/cache.ts
export class TTLCache<K, V> {
  private store = new Map<K, { value: V; expiresAt: number }>()

  set(key: K, value: V, ttlMs: number): void {
    this.store.set(key, {
      value,
      expiresAt: Date.now() + ttlMs,
    })
  }

  get(key: K): V | undefined {
    const entry = this.store.get(key)
    if (!entry) return undefined
    if (Date.now() > entry.expiresAt) {
      this.store.delete(key)
      return undefined
    }
    return entry.value
  }

  has(key: K): boolean {
    return this.get(key) !== undefined
  }

  delete(key: K): boolean {
    return this.store.delete(key)
  }

  get size(): number {
    // 只计算未过期的条目
    let count = 0
    for (const [key] of this.store) {
      if (this.has(key)) count++
    }
    return count
  }
}
```

要求：测试正常存取、TTL 过期（使用虚拟定时器）、删除操作、大小统计等场景。

---

**练习 4**：为以下 AI 聊天历史管理器编写完整测试，包括 Mock LLM 调用和错误处理：

```typescript
// src/ai/chatHistory.ts
import type { LLMClient, ChatMessage, LLMResponse } from './types'

export class ChatSession {
  private history: ChatMessage[] = []
  private readonly maxHistoryLength: number

  constructor(
    private readonly llm: LLMClient,
    private readonly systemPrompt: string,
    maxHistoryLength = 20
  ) {
    this.maxHistoryLength = maxHistoryLength
  }

  async sendMessage(userMessage: string): Promise<string> {
    this.history.push({ role: 'user', content: userMessage })

    // 超出历史长度时，保留系统提示并截断早期消息
    if (this.history.length > this.maxHistoryLength) {
      this.history = this.history.slice(-this.maxHistoryLength)
    }

    const messages: ChatMessage[] = [
      { role: 'system', content: this.systemPrompt },
      ...this.history,
    ]

    const response = await this.llm.chat(messages)
    const assistantMessage = response.choices[0]?.message.content ?? ''

    this.history.push({ role: 'assistant', content: assistantMessage })
    return assistantMessage
  }

  getHistory(): ReadonlyArray<ChatMessage> {
    return this.history
  }

  clearHistory(): void {
    this.history = []
  }
}
```

---

### 高级练习

**练习 5**：实现并测试一个带有重试机制的 LLM 客户端包装器。要求：

1. 实现 `RetryableLLMClient` 类，当 API 调用失败时自动重试
2. 支持配置最大重试次数和退避延迟（指数退避）
3. 对于某些错误类型（如 401 认证失败），不应重试
4. 编写完整测试，使用 `vi.useFakeTimers()` 验证退避延迟逻辑
5. 验证重试次数耗尽后正确抛出最后一个错误

```typescript
// 接口定义（供实现参考）
interface RetryConfig {
  maxRetries: number
  initialDelayMs: number
  maxDelayMs: number
  retryableStatusCodes: number[]
}

class RetryableLLMClient implements LLMClient {
  constructor(
    private readonly inner: LLMClient,
    private readonly config: RetryConfig
  ) {}

  async chat(messages: ChatMessage[]): Promise<LLMResponse> {
    // 实现指数退避重试逻辑
  }
}
```

---

## 练习答案

### 答案 1：Stack 单元测试

```typescript
// src/utils/stack.test.ts
import { describe, it, expect, beforeEach } from 'vitest'
import { Stack } from './stack'

describe('Stack<T>', () => {
  let stack: Stack<number>

  beforeEach(() => {
    stack = new Stack<number>()
  })

  describe('初始状态', () => {
    it('新建栈应为空', () => {
      expect(stack.isEmpty()).toBe(true)
      expect(stack.size).toBe(0)
    })
  })

  describe('push()', () => {
    it('应添加元素并更新大小', () => {
      stack.push(1)
      expect(stack.size).toBe(1)
      expect(stack.isEmpty()).toBe(false)

      stack.push(2)
      expect(stack.size).toBe(2)
    })

    it('应支持泛型类型', () => {
      const strStack = new Stack<string>()
      strStack.push('hello')
      strStack.push('world')
      expect(strStack.size).toBe(2)
      expect(strStack.peek()).toBe('world')
    })
  })

  describe('pop()', () => {
    it('应按 LIFO 顺序弹出元素', () => {
      stack.push(1)
      stack.push(2)
      stack.push(3)

      expect(stack.pop()).toBe(3)
      expect(stack.pop()).toBe(2)
      expect(stack.pop()).toBe(1)
    })

    it('弹出后应减少大小', () => {
      stack.push(10)
      stack.pop()
      expect(stack.size).toBe(0)
      expect(stack.isEmpty()).toBe(true)
    })

    it('在空栈上调用时应抛出错误', () => {
      expect(() => stack.pop()).toThrow('Stack is empty')
    })
  })

  describe('peek()', () => {
    it('应返回顶部元素但不移除', () => {
      stack.push(42)
      expect(stack.peek()).toBe(42)
      expect(stack.size).toBe(1) // 大小不变
    })

    it('在空栈上调用时应抛出错误', () => {
      expect(() => stack.peek()).toThrow('Stack is empty')
    })
  })

  describe('复合操作', () => {
    it('应正确处理多次 push 和 pop', () => {
      stack.push(1)
      stack.push(2)
      expect(stack.pop()).toBe(2)
      stack.push(3)
      expect(stack.pop()).toBe(3)
      expect(stack.pop()).toBe(1)
      expect(stack.isEmpty()).toBe(true)
    })
  })
})
```

### 答案 2：类型测试

```typescript
// src/types/utility.test.ts
import { describe, it } from 'vitest'
import { expectTypeOf } from 'expect-type'

type MyNonNullable<T> = T extends null | undefined ? never : T
type MyReturnType<T extends (...args: unknown[]) => unknown> =
  T extends (...args: unknown[]) => infer R ? R : never
type MyParameters<T extends (...args: unknown[]) => unknown> =
  T extends (...args: infer P) => unknown ? P : never

describe('工具类型测试', () => {
  it('NonNullable 应移除 null 和 undefined', () => {
    expectTypeOf<MyNonNullable<string | null>>().toEqualTypeOf<string>()
    expectTypeOf<MyNonNullable<number | undefined>>().toEqualTypeOf<number>()
    expectTypeOf<MyNonNullable<string | null | undefined>>().toEqualTypeOf<string>()
    expectTypeOf<MyNonNullable<string>>().toEqualTypeOf<string>()
  })

  it('ReturnType 应提取函数返回类型', () => {
    type F1 = () => string
    type F2 = (x: number) => boolean
    type F3 = () => Promise<number[]>

    expectTypeOf<MyReturnType<F1>>().toEqualTypeOf<string>()
    expectTypeOf<MyReturnType<F2>>().toEqualTypeOf<boolean>()
    expectTypeOf<MyReturnType<F3>>().toEqualTypeOf<Promise<number[]>>()
  })

  it('Parameters 应提取函数参数类型', () => {
    type F1 = (a: string, b: number) => void
    type F2 = () => void
    type F3 = (items: string[]) => boolean

    expectTypeOf<MyParameters<F1>>().toEqualTypeOf<[string, number]>()
    expectTypeOf<MyParameters<F2>>().toEqualTypeOf<[]>()
    expectTypeOf<MyParameters<F3>>().toEqualTypeOf<[string[]]>()
  })
})
```

### 答案 3：TTLCache 虚拟定时器测试

```typescript
// src/services/cache.test.ts
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { TTLCache } from './cache'

describe('TTLCache', () => {
  let cache: TTLCache<string, number>

  beforeEach(() => {
    vi.useFakeTimers()
    cache = new TTLCache<string, number>()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('应存储并返回未过期的值', () => {
    cache.set('key1', 100, 5000)
    expect(cache.get('key1')).toBe(100)
    expect(cache.has('key1')).toBe(true)
  })

  it('应在 TTL 过期后返回 undefined', () => {
    cache.set('key1', 100, 1000)

    vi.advanceTimersByTime(999)
    expect(cache.get('key1')).toBe(100) // 未过期

    vi.advanceTimersByTime(1)
    expect(cache.get('key1')).toBeUndefined() // 已过期
  })

  it('过期后 has() 应返回 false', () => {
    cache.set('temp', 42, 500)
    vi.advanceTimersByTime(501)
    expect(cache.has('temp')).toBe(false)
  })

  it('size 应只计算未过期的条目', () => {
    cache.set('a', 1, 1000)
    cache.set('b', 2, 2000)
    cache.set('c', 3, 3000)

    expect(cache.size).toBe(3)

    vi.advanceTimersByTime(1500)
    expect(cache.size).toBe(2) // 'a' 已过期

    vi.advanceTimersByTime(1000)
    expect(cache.size).toBe(1) // 'b' 已过期
  })

  it('delete() 应移除指定键', () => {
    cache.set('key1', 1, 5000)
    expect(cache.delete('key1')).toBe(true)
    expect(cache.get('key1')).toBeUndefined()
    expect(cache.delete('nonexistent')).toBe(false)
  })
})
```

### 答案 5：RetryableLLMClient 实现与测试

```typescript
// src/ai/retryClient.ts
import type { LLMClient, ChatMessage, LLMResponse } from './types'

export interface RetryConfig {
  maxRetries: number
  initialDelayMs: number
  maxDelayMs: number
  retryableStatusCodes: number[]
}

export class APIError extends Error {
  constructor(
    message: string,
    public readonly statusCode: number
  ) {
    super(message)
    this.name = 'APIError'
  }
}

export class RetryableLLMClient implements LLMClient {
  constructor(
    private readonly inner: LLMClient,
    private readonly config: RetryConfig
  ) {}

  private async delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms))
  }

  private getBackoffDelay(attempt: number): number {
    const delay = this.config.initialDelayMs * Math.pow(2, attempt)
    return Math.min(delay, this.config.maxDelayMs)
  }

  async chat(messages: ChatMessage[]): Promise<LLMResponse> {
    let lastError: Error | null = null

    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      try {
        return await this.inner.chat(messages)
      } catch (error) {
        lastError = error as Error

        // 非重试错误（如认证失败）直接抛出
        if (error instanceof APIError) {
          if (!this.config.retryableStatusCodes.includes(error.statusCode)) {
            throw error
          }
        }

        // 最后一次尝试后不再等待
        if (attempt < this.config.maxRetries) {
          const waitTime = this.getBackoffDelay(attempt)
          await this.delay(waitTime)
        }
      }
    }

    throw lastError
  }
}
```

```typescript
// src/ai/retryClient.test.ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { RetryableLLMClient, APIError } from './retryClient'
import type { LLMClient, LLMResponse } from './types'

const mockResponse: LLMResponse = {
  id: 'test-id',
  choices: [{ message: { role: 'assistant', content: 'OK' }, finish_reason: 'stop' }],
  usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
}

describe('RetryableLLMClient', () => {
  let mockInner: LLMClient
  let client: RetryableLLMClient

  beforeEach(() => {
    vi.useFakeTimers()
    mockInner = { chat: vi.fn() }
    client = new RetryableLLMClient(mockInner, {
      maxRetries: 3,
      initialDelayMs: 1000,
      maxDelayMs: 8000,
      retryableStatusCodes: [429, 500, 503],
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('成功时不应重试', async () => {
    vi.mocked(mockInner.chat).mockResolvedValueOnce(mockResponse)
    const result = await client.chat([{ role: 'user', content: '测试' }])
    expect(result).toEqual(mockResponse)
    expect(mockInner.chat).toHaveBeenCalledOnce()
  })

  it('应对可重试错误进行指数退避重试', async () => {
    vi.mocked(mockInner.chat)
      .mockRejectedValueOnce(new APIError('Rate limited', 429))
      .mockRejectedValueOnce(new APIError('Server error', 500))
      .mockResolvedValueOnce(mockResponse)

    const chatPromise = client.chat([{ role: 'user', content: '测试' }])

    // 第一次重试：等待 1000ms
    await vi.advanceTimersByTimeAsync(1000)
    // 第二次重试：等待 2000ms
    await vi.advanceTimersByTimeAsync(2000)

    const result = await chatPromise
    expect(result).toEqual(mockResponse)
    expect(mockInner.chat).toHaveBeenCalledTimes(3)
  })

  it('应对不可重试的 401 错误立即失败', async () => {
    vi.mocked(mockInner.chat).mockRejectedValueOnce(
      new APIError('Unauthorized', 401)
    )

    await expect(
      client.chat([{ role: 'user', content: '测试' }])
    ).rejects.toThrow('Unauthorized')

    expect(mockInner.chat).toHaveBeenCalledOnce()
  })

  it('重试次数耗尽后应抛出最后一个错误', async () => {
    const lastError = new APIError('Service unavailable', 503)
    vi.mocked(mockInner.chat).mockRejectedValue(lastError)

    const chatPromise = client.chat([{ role: 'user', content: '测试' }])

    // 推进所有退避延迟（1000 + 2000 + 4000 = 7000ms）
    await vi.advanceTimersByTimeAsync(7000)

    await expect(chatPromise).rejects.toThrow('Service unavailable')
    expect(mockInner.chat).toHaveBeenCalledTimes(4) // 1 次初始 + 3 次重试
  })
})
```

---

*本章完 — 下一章将介绍 TypeScript 项目的构建优化与 CI/CD 配置。*
