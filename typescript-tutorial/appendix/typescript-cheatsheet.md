# 附录A：TypeScript速查表

> 本附录是全教程的快速参考手册，涵盖类型系统、函数、类与接口、工具类型、高级类型及编译配置。每个知识点以表格或代码片段呈现，方便查阅。

---

## A.1 基本类型速查

### A.1.1 原始类型

| 类型        | 说明                          | 示例                                      |
|-------------|-------------------------------|-------------------------------------------|
| `string`    | 文本字符串                    | `let name: string = "Claude"`             |
| `number`    | 整数与浮点数（64位）          | `let temp: number = 0.7`                  |
| `boolean`   | 布尔值                        | `let stream: boolean = true`              |
| `bigint`    | 任意精度整数                  | `let big: bigint = 9007199254740993n`     |
| `symbol`    | 唯一标识符                    | `let id: symbol = Symbol("id")`           |
| `null`      | 空值（需开启 strictNullChecks）| `let val: null = null`                    |
| `undefined` | 未赋值                        | `let u: undefined = undefined`            |
| `any`       | 关闭类型检查（慎用）          | `let x: any = "anything"`                 |
| `unknown`   | 类型安全的 any，需先收窄      | `let raw: unknown = fetch(...)`           |
| `never`     | 永不产生值（穷举检查）        | `function fail(): never { throw ... }`   |
| `void`      | 无返回值的函数                | `function log(): void { console.log() }` |
| `object`    | 非原始类型的对象              | `let o: object = { key: "val" }`          |

```typescript
// 字面量类型：将值本身作为类型
type Model = "gpt-4o" | "claude-sonnet-4-6" | "gemini-pro";
type Status = 200 | 400 | 401 | 403 | 500;
type Direction = "up" | "down" | "left" | "right";

// 模板字面量类型
type EventName = `on${Capitalize<string>}`;  // "onClick" | "onHover" | ...
type ApiPath = `/api/${string}`;             // "/api/chat" | "/api/models" | ...
```

### A.1.2 数组与元组

```typescript
// 数组：两种等价写法
let models: string[] = ["gpt-4o", "claude-sonnet-4-6"];
let scores: Array<number> = [0.7, 0.9, 1.0];

// 只读数组：防止修改
const endpoints: ReadonlyArray<string> = ["/chat", "/models"];
const flags: readonly boolean[] = [true, false];

// 元组：固定长度与类型的数组
type ApiResult = [number, string];           // [状态码, 消息]
const result: ApiResult = [200, "OK"];

// 带标签的元组（TypeScript 4.0+）
type Range = [start: number, end: number];
const r: Range = [0, 100];

// 可选元组成员
type Config = [string, number, boolean?];
const c1: Config = ["model", 1024];         // 第三个可省
const c2: Config = ["model", 1024, true];

// 剩余元组成员
type StringsAndNumber = [...string[], number];
const sn: StringsAndNumber = ["a", "b", 42];
```

### A.1.3 枚举

```typescript
// 数字枚举（默认从 0 递增）
enum Direction {
  Up,       // 0
  Down,     // 1
  Left,     // 2
  Right,    // 3
}

// 指定起始值
enum HttpStatus {
  OK = 200,
  Created = 201,
  BadRequest = 400,
  Unauthorized = 401,
  NotFound = 404,
}

// 字符串枚举（推荐：调试时可读性更好）
enum LogLevel {
  Debug = "DEBUG",
  Info  = "INFO",
  Warn  = "WARN",
  Error = "ERROR",
}

// const 枚举（编译后内联为字面量，无运行时对象）
const enum Role {
  User      = "user",
  Assistant = "assistant",
  System    = "system",
}
const r: Role = Role.User;  // 编译为 const r = "user"

// 枚举成员作为类型
function setLevel(level: LogLevel): void { /* ... */ }
setLevel(LogLevel.Info);    // OK
setLevel("INFO");           // 错误：不能将 string 赋值给 LogLevel
```

---

## A.2 类型操作速查

### A.2.1 联合与交叉

```typescript
// 联合类型 (|)：A 或 B
type StringOrNumber = string | number;
type Nullable<T> = T | null;
type ApiResponse = SuccessResponse | ErrorResponse;

// 交叉类型 (&)：A 且 B（合并所有属性）
type WithTimestamp<T> = T & { createdAt: Date; updatedAt: Date };
type AdminUser = User & { permissions: string[] };

// 可辨识联合（每个成员含唯一字面量字段）
type Shape =
  | { kind: "circle";    radius: number }
  | { kind: "rect";      width: number; height: number }
  | { kind: "triangle";  base: number;  height: number };

function area(s: Shape): number {
  switch (s.kind) {
    case "circle":   return Math.PI * s.radius ** 2;
    case "rect":     return s.width * s.height;
    case "triangle": return (s.base * s.height) / 2;
  }
}
```

### A.2.2 类型守卫

| 守卫方式            | 语法                             | 适用场景                   |
|---------------------|----------------------------------|----------------------------|
| `typeof`            | `typeof x === "string"`          | 原始类型判断               |
| `instanceof`        | `x instanceof Date`              | 类实例判断                 |
| `in` 操作符         | `"prop" in obj`                  | 对象属性存在性判断         |
| 字面量相等          | `x.kind === "circle"`            | 可辨识联合收窄             |
| 自定义类型谓词      | `function isX(v): v is X`        | 复杂运行时判断             |
| 断言函数            | `function assert(v): asserts v`  | 先决条件断言               |

```typescript
// typeof 守卫
function format(value: string | number): string {
  if (typeof value === "string") return value.toUpperCase();
  return value.toFixed(2);
}

// 自定义类型谓词
interface StreamChunk { delta: string; finishReason?: string }
function isStreamChunk(val: unknown): val is StreamChunk {
  return (
    typeof val === "object" && val !== null &&
    "delta" in val && typeof (val as any).delta === "string"
  );
}

// 断言函数（TypeScript 3.7+）
function assertDefined<T>(val: T | null | undefined): asserts val is T {
  if (val == null) throw new Error("Expected defined value");
}
```

### A.2.3 类型断言

```typescript
// as 语法（推荐）
const input = document.getElementById("prompt") as HTMLInputElement;

// 尖括号语法（JSX 文件中不可用）
const input2 = <HTMLInputElement>document.getElementById("prompt");

// 双重断言（应极力避免；仅用于过渡期迁移）
const val = (unknownValue as unknown) as TargetType;

// const 断言：将字面量收窄为最窄类型
const config = {
  model: "claude-sonnet-4-6",
  temperature: 0.7,
} as const;
// config.model 的类型是 "claude-sonnet-4-6"，而非 string

// 满足运算符（TypeScript 4.9+）：校验类型而不改变推断结果
const palette = {
  red:   [255, 0, 0],
  green: "#00ff00",
} satisfies Record<string, string | number[]>;
// palette.red 仍推断为 number[]，而非 string | number[]
```

---

## A.3 函数类型速查

### A.3.1 函数签名

```typescript
// 具名函数
function add(a: number, b: number): number {
  return a + b;
}

// 箭头函数
const multiply = (a: number, b: number): number => a * b;

// 可选参数（必须在必选参数之后）
function greet(name: string, title?: string): string {
  return title ? `${title} ${name}` : name;
}

// 默认参数
function request(url: string, method: string = "GET"): Promise<Response> {
  return fetch(url, { method });
}

// 剩余参数
function sum(...nums: number[]): number {
  return nums.reduce((a, b) => a + b, 0);
}

// 函数类型别名
type Handler = (event: string, data: unknown) => void;
type AsyncFetcher<T> = (url: string) => Promise<T>;

// 调用签名（在接口中）
interface Formatter {
  (value: number, precision: number): string;
  defaultPrecision: number;     // 函数也可以有属性
}
```

### A.3.2 泛型函数

```typescript
// 基本泛型函数
function identity<T>(value: T): T {
  return value;
}

// 多类型参数
function pair<A, B>(a: A, b: B): [A, B] {
  return [a, b];
}

// 泛型约束（extends）
function getLength<T extends { length: number }>(val: T): number {
  return val.length;
}

// 约束参数关系（keyof）
function getProp<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

// 条件默认值
async function fetchJSON<T = unknown>(url: string): Promise<T> {
  const res = await fetch(url);
  return res.json() as T;
}

// 泛型箭头函数（TSX 文件中需加逗号避免歧义）
const wrap = <T,>(val: T): { data: T } => ({ data: val });
```

### A.3.3 重载

```typescript
// 函数重载：多个签名 + 一个实现
function parseInput(input: string): string[];
function parseInput(input: string[]): string;
function parseInput(input: string | string[]): string | string[] {
  if (typeof input === "string") return input.split(",");
  return input.join(",");
}

// 方法重载（类中）
class Formatter {
  format(value: number): string;
  format(value: Date): string;
  format(value: number | Date): string {
    if (typeof value === "number") return value.toFixed(2);
    return value.toISOString();
  }
}
```

---

## A.4 类与接口速查

### A.4.1 类定义

```typescript
class LLMClient {
  // 实例属性
  readonly baseURL: string;
  private apiKey: string;
  protected model: string;
  maxRetries: number = 3;

  // 构造函数（参数属性简写）
  constructor(
    private readonly config: ClientConfig,  // 自动声明为私有只读属性
    apiKey: string
  ) {
    this.baseURL = config.baseURL;
    this.apiKey = apiKey;
    this.model = config.model;
  }

  // 实例方法
  async chat(prompt: string): Promise<string> {
    const res = await fetch(`${this.baseURL}/chat`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
      body: JSON.stringify({ model: this.model, prompt }),
    });
    return res.text();
  }

  // 静态方法
  static create(config: ClientConfig, apiKey: string): LLMClient {
    return new LLMClient(config, apiKey);
  }

  // getter / setter
  get modelName(): string { return this.model; }
  set modelName(name: string) { this.model = name; }
}

// 继承
class StreamingClient extends LLMClient {
  async stream(prompt: string): AsyncGenerator<string> {
    // ...
  }
}
```

### A.4.2 接口

```typescript
// 基本接口
interface User {
  id: number;
  name: string;
  email?: string;      // 可选属性
  readonly token: string;  // 只读属性
}

// 函数类型接口
interface Comparator<T> {
  (a: T, b: T): number;
}

// 索引签名
interface StringMap {
  [key: string]: string;
}

// 接口继承（可多继承）
interface AdminUser extends User {
  role: "admin";
  permissions: string[];
}

interface AuditedUser extends User, Auditable {
  lastAudit: Date;
}

// 接口合并（声明合并）
interface Window {
  myAnalytics: () => void;  // 扩展全局 Window 类型
}

// 接口 vs 类型别名快速对比
// interface：可声明合并、更适合描述对象结构
// type：支持联合/交叉/映射等复杂类型运算
```

### A.4.3 访问修饰符

| 修饰符      | 可访问范围                          | 编译后行为                |
|-------------|-------------------------------------|---------------------------|
| `public`    | 任何地方（默认）                    | 无变化                    |
| `private`   | 仅当前类内部（TypeScript 编译期）   | 无变化（JS 中仍可访问）   |
| `#field`    | 仅当前类内部（JS 运行时原生私有）   | 真正的私有字段            |
| `protected` | 当前类及子类                        | 无变化                    |
| `readonly`  | 只能在声明或构造函数中赋值          | 无变化                    |
| `static`    | 通过类名访问，不依赖实例            | 无变化                    |
| `abstract`  | 必须在子类中实现，不能直接实例化    | 仅类型层面                |
| `override`  | 显式声明覆盖父类方法（TS 4.3+）     | 无变化                    |

```typescript
class SecureClient {
  #secret: string;           // JS 原生私有（运行时隔离）
  private tsPrivate: string; // TS 私有（编译期检查）

  constructor(secret: string) {
    this.#secret = secret;
    this.tsPrivate = secret;
  }
}
```

---

## A.5 工具类型速查

### A.5.1 Partial / Required / Readonly

| 工具类型        | 效果                           | 典型场景                     |
|-----------------|--------------------------------|------------------------------|
| `Partial<T>`    | 所有属性变为可选               | 配置更新、PATCH 请求体       |
| `Required<T>`   | 所有属性变为必选               | 校验完整性、POST 请求体      |
| `Readonly<T>`   | 所有属性变为只读               | 不可变配置、常量对象         |

```typescript
interface ModelConfig {
  model: string;
  temperature: number;
  maxTokens: number;
  stream: boolean;
}

// Partial：只传需要更新的字段
function updateConfig(base: ModelConfig, patch: Partial<ModelConfig>): ModelConfig {
  return { ...base, ...patch };
}

// Required：确保所有字段齐全
function validateConfig(c: Required<ModelConfig>): void { /* ... */ }

// Readonly：防止意外修改
const DEFAULT_CONFIG: Readonly<ModelConfig> = {
  model: "claude-sonnet-4-6",
  temperature: 0.7,
  maxTokens: 2048,
  stream: false,
};
```

### A.5.2 Pick / Omit

| 工具类型          | 效果                             | 典型场景                        |
|-------------------|----------------------------------|---------------------------------|
| `Pick<T, K>`      | 从 T 中只保留 K 指定的属性      | 提取子集、DTO 定义              |
| `Omit<T, K>`      | 从 T 中排除 K 指定的属性        | 隐藏敏感字段、去掉 id 字段      |

```typescript
interface UserRecord {
  id: number;
  name: string;
  email: string;
  passwordHash: string;
  createdAt: Date;
}

// Pick：只取公开字段
type PublicUser = Pick<UserRecord, "id" | "name" | "email">;

// Omit：排除敏感字段
type SafeUser = Omit<UserRecord, "passwordHash">;

// 常用组合：创建时去掉自动生成的字段
type CreateUserDTO = Omit<UserRecord, "id" | "createdAt">;
```

### A.5.3 Record / Extract / Exclude

| 工具类型              | 效果                                         | 典型场景                    |
|-----------------------|----------------------------------------------|-----------------------------|
| `Record<K, V>`        | 构造键为 K、值为 V 的对象类型               | 字典、映射表                |
| `Extract<T, U>`       | 从 T 中提取可赋值给 U 的成员               | 过滤联合类型               |
| `Exclude<T, U>`       | 从 T 中排除可赋值给 U 的成员               | 反向过滤联合类型           |
| `NonNullable<T>`      | 从 T 中排除 null 和 undefined              | 确保非空                   |

```typescript
// Record：模型能力映射表
type ModelCapability = "chat" | "vision" | "code" | "embedding";
const supportedModels: Record<string, ModelCapability[]> = {
  "gpt-4o":            ["chat", "vision", "code"],
  "claude-sonnet-4-6": ["chat", "vision", "code"],
  "text-embedding-3-small": ["embedding"],
};

// Extract：只保留字符串成员
type StringKeys = Extract<"a" | 1 | "b" | true, string>;  // "a" | "b"

// Exclude：去掉 null/undefined
type CleanId = Exclude<string | number | null | undefined, null | undefined>;
// 等价于 NonNullable<string | number | null | undefined>
```

### A.5.4 ReturnType / Parameters / InstanceType

| 工具类型                  | 效果                                    | 典型场景                         |
|---------------------------|-----------------------------------------|----------------------------------|
| `ReturnType<T>`           | 提取函数返回值的类型                   | 避免重复定义返回类型             |
| `Parameters<T>`           | 提取函数参数类型（元组）               | 包装函数、中间件                 |
| `ConstructorParameters<T>`| 提取构造函数参数类型                   | 工厂函数                         |
| `InstanceType<T>`         | 提取类构造出的实例类型                 | 泛型工厂                         |
| `Awaited<T>`              | 解包 Promise 的内部类型（TS 4.5+）     | 异步函数返回类型                 |

```typescript
async function fetchUser(id: number) {
  return { id, name: "Alice", email: "alice@example.com" };
}

// 从函数推断类型，避免重复定义
type FetchUserReturn = Awaited<ReturnType<typeof fetchUser>>;
// { id: number; name: string; email: string }

type FetchUserParams = Parameters<typeof fetchUser>;
// [id: number]

// 用于高阶函数
function withLogging<T extends (...args: any[]) => any>(fn: T) {
  return (...args: Parameters<T>): ReturnType<T> => {
    console.log("calling with", args);
    return fn(...args);
  };
}
```

---

## A.6 高级类型速查

### A.6.1 映射类型

```typescript
// 基础映射（遍历键，转换值）
type Stringify<T> = {
  [K in keyof T]: string;
};

// 添加修饰符
type DeepReadonly<T> = {
  readonly [K in keyof T]: T[K] extends object ? DeepReadonly<T[K]> : T[K];
};

// 移除修饰符（- 前缀）
type Mutable<T> = {
  -readonly [K in keyof T]: T[K];
};
type AllRequired<T> = {
  [K in keyof T]-?: T[K];
};

// 键重映射（as，TypeScript 4.1+）
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K];
};
// Getters<{ name: string }> → { getName: () => string }

// 过滤键
type OnlyStrings<T> = {
  [K in keyof T as T[K] extends string ? K : never]: T[K];
};
```

### A.6.2 条件类型

```typescript
// 基本条件类型
type IsString<T> = T extends string ? true : false;
type IsString_Test = IsString<"hello">;  // true
type IsString_Test2 = IsString<42>;      // false

// 联合类型分配律（裸类型参数会被分配）
type ToArray<T> = T extends any ? T[] : never;
type Arr = ToArray<string | number>;  // string[] | number[]

// 阻止分配（用元组包裹）
type ToArrayUnion<T> = [T] extends [any] ? T[] : never;
type ArrUnion = ToArrayUnion<string | number>;  // (string | number)[]

// 内置条件类型实现参考
type MyNonNullable<T> = T extends null | undefined ? never : T;
type MyReturnType<T> = T extends (...args: any[]) => infer R ? R : never;
type MyPartial<T> = { [K in keyof T]?: T[K] };

// 递归条件类型（TypeScript 4.1+）
type DeepPartial<T> = T extends object
  ? { [K in keyof T]?: DeepPartial<T[K]> }
  : T;
```

### A.6.3 infer 关键字

```typescript
// infer：在条件类型中推断并捕获类型变量

// 提取函数返回值
type ReturnT<T> = T extends (...args: any[]) => infer R ? R : never;

// 提取 Promise 内部类型
type Unwrap<T> = T extends Promise<infer U> ? Unwrap<U> : T;
type Test = Unwrap<Promise<Promise<string>>>;  // string

// 提取数组元素类型
type ElementType<T> = T extends (infer E)[] ? E : never;
type Elem = ElementType<string[]>;  // string

// 提取函数第一个参数
type FirstParam<T> = T extends (first: infer F, ...rest: any[]) => any ? F : never;

// 提取构造函数实例类型
type CtorInstance<T> = T extends new (...args: any[]) => infer I ? I : never;

// 多处使用同一 infer 变量（协变位置取联合，逆变位置取交叉）
type UnionToIntersection<U> =
  (U extends any ? (x: U) => void : never) extends (x: infer I) => void ? I : never;
```

---

## A.7 常用 tsconfig 选项速查

### A.7.1 核心编译选项

| 选项                         | 类型      | 推荐值            | 说明                                        |
|------------------------------|-----------|-------------------|---------------------------------------------|
| `target`                     | string    | `"ES2022"`        | 编译目标 JS 版本                            |
| `module`                     | string    | `"NodeNext"`      | 模块系统（Node 18+ 用 NodeNext）            |
| `moduleResolution`           | string    | `"NodeNext"`      | 模块解析算法（与 module 保持一致）          |
| `lib`                        | string[]  | `["ES2022"]`      | 内置类型声明库                              |
| `outDir`                     | string    | `"./dist"`        | 编译输出目录                                |
| `rootDir`                    | string    | `"./src"`         | 源码根目录                                  |
| `declaration`                | boolean   | `true`            | 同时生成 `.d.ts` 声明文件                   |
| `declarationMap`             | boolean   | `true`            | 生成声明文件的 source map                   |
| `sourceMap`                  | boolean   | `true`            | 生成 `.js.map` 文件                         |
| `removeComments`             | boolean   | `false`           | 是否移除注释                                |
| `noEmit`                     | boolean   | `true`            | 仅做类型检查，不输出文件（配合 bundler）    |
| `esModuleInterop`            | boolean   | `true`            | 允许默认导入 CommonJS 模块                  |
| `allowSyntheticDefaultImports`| boolean  | `true`            | 允许没有默认导出的模块使用默认导入          |
| `resolveJsonModule`          | boolean   | `true`            | 允许导入 `.json` 文件                       |
| `allowJs`                    | boolean   | `false`           | 允许编译 `.js` 文件（JS 迁移期使用）        |
| `checkJs`                    | boolean   | `false`           | 对 JS 文件也进行类型检查                    |

### A.7.2 严格模式选项

| 选项                       | 说明                                                    |
|----------------------------|---------------------------------------------------------|
| `strict`                   | 总开关，启用以下所有严格选项（**强烈推荐**）            |
| `strictNullChecks`         | null/undefined 不能赋值给其他类型                      |
| `strictFunctionTypes`      | 函数参数使用逆变而非双变检查                            |
| `strictBindCallApply`      | bind/call/apply 的参数类型严格检查                     |
| `strictPropertyInitialization` | 类属性必须在构造函数中初始化                        |
| `noImplicitAny`            | 禁止隐式 any                                           |
| `noImplicitThis`           | 禁止 this 隐式为 any                                   |
| `alwaysStrict`             | 每个文件顶部插入 `"use strict"`                        |
| `useUnknownInCatchVariables` | catch 变量类型为 unknown（TS 4.4+）                  |

### A.7.3 额外检查选项

| 选项                           | 说明                                        |
|--------------------------------|---------------------------------------------|
| `noUnusedLocals`               | 报告未使用的局部变量                        |
| `noUnusedParameters`           | 报告未使用的函数参数                        |
| `noImplicitReturns`            | 函数所有路径必须显式返回                    |
| `noFallthroughCasesInSwitch`   | switch 语句禁止贯穿（fallthrough）          |
| `noUncheckedIndexedAccess`     | 索引访问结果类型包含 undefined（TS 4.1+）   |
| `exactOptionalPropertyTypes`   | 可选属性不允许显式赋 undefined（TS 4.4+）   |
| `noPropertyAccessFromIndexSignature` | 强制用括号访问索引签名属性            |

### A.7.4 路径别名配置

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*":          ["src/*"],
      "@types/*":     ["src/types/*"],
      "@utils/*":     ["src/utils/*"],
      "@services/*":  ["src/services/*"]
    }
  }
}
```

```typescript
// 配置后可使用别名导入，无需相对路径
import { fetchChat } from "@services/llm";
import type { ModelConfig } from "@types/config";
```

### A.7.5 典型项目配置模板

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "esModuleInterop": true,
    "resolveJsonModule": true,
    "skipLibCheck": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

---

## A.8 快速决策指南

### 何时用 interface vs type

| 场景                           | 推荐        | 原因                                  |
|--------------------------------|-------------|---------------------------------------|
| 描述对象/类的结构              | `interface` | 可声明合并，可被 class implements     |
| 联合类型、交叉类型             | `type`      | interface 不支持                      |
| 映射类型、条件类型             | `type`      | interface 不支持                      |
| 扩展第三方库类型               | `interface` | 利用声明合并机制                      |
| 函数类型                       | `type`      | 语法更简洁                            |
| 元组类型                       | `type`      | interface 不支持                      |

### 何时用 any vs unknown

| 场景                           | 推荐      | 原因                              |
|--------------------------------|-----------|-----------------------------------|
| 确实不关心类型（极少）         | `any`     | 完全关闭类型检查                  |
| 来自外部的不确定数据           | `unknown` | 必须先收窄才能使用，更安全        |
| 需要与任意类型兼容             | `unknown` | 赋值给 unknown 永远合法           |
| 临时迁移 JS 代码               | `any`     | 逐步添加类型时的过渡手段          |

### 常见错误速查

| 错误信息                                                  | 原因                          | 解决方案                               |
|-----------------------------------------------------------|-------------------------------|----------------------------------------|
| `Object is possibly 'null'`                               | 未做 null 检查                | 用 `?.`、`??` 或 `if` 守卫             |
| `Property 'x' does not exist on type 'Y'`                 | 访问不存在的属性              | 检查类型定义或用类型断言               |
| `Type 'X' is not assignable to type 'Y'`                  | 类型不兼容                    | 检查类型，必要时用 Pick/Omit 调整      |
| `Argument of type 'X' is not assignable to parameter 'Y'` | 参数类型不匹配                | 确认函数签名，或收窄参数类型           |
| `Cannot find module 'X'`                                  | 模块路径错误或缺少声明文件    | 检查路径别名配置或安装 `@types/x`      |
| `Expression of type 'string' can't be used to index type` | 用 string 索引有明确键的对象  | 用 `keyof` 约束或改用 `Record<string>` |
| `This expression is not callable`                         | 对非函数类型调用              | 检查类型定义，确认是否为函数类型       |

---

*本附录配合教程各章使用效果最佳。建议在实际编写代码时将本页作为侧边参考，遇到语法疑问时快速定位对应小节。*
