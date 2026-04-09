# 附录C：练习答案汇总

本附录为全教程 20 章练习题提供索引、题目摘要与解题关键点，帮助读者在查阅完整答案前先独立思考、自我检验。

---

## 使用说明

### 练习难度说明

每章练习题按难度递进排列：

| 难度标识 | 说明 | 建议用时 |
|----------|------|----------|
| 基础题 | 考查本章核心语法，照着示例能完成 | 10–20 分钟 |
| 进阶题 / 中级题 | 需要综合运用本章及前几章知识 | 20–40 分钟 |
| 挑战题 / 高级题 | 接近真实项目场景，鼓励查阅文档 | 40–90 分钟 |

### 如何有效使用练习题

1. **先独立作答**：至少花 15 分钟自行思考，不要立刻翻答案。
2. **利用编译器反馈**：TypeScript 的报错信息本身就是提示，善用 `tsc --noEmit` 检查。
3. **对比而非抄写**：看完答案后，对照自己的实现找差异，理解为何官方答案更优。
4. **变体练习**：完成答案后，尝试修改需求（例如加一个字段、改变返回类型），看代码如何演化。
5. **跳过再返回**：卡住超过 30 分钟可先跳过，学完后续章节再回头，往往豁然开朗。

### 常见错误提示

- **忘记类型注解导致推断为 `any`**：开启 `strict: true` 可在编译阶段发现此类问题。
- **混淆 `interface` 与 `type`**：两者在大多数场景可互换，但 `interface` 支持声明合并，`type` 支持联合/交叉等复杂组合。
- **泛型约束写错方向**：`T extends U` 表示 T 是 U 的子类型，不是 U 继承 T。
- **`async` 函数返回值类型**：返回 `T` 的 `async` 函数实际类型是 `Promise<T>`，标注时注意区分。
- **枚举的反向映射陷阱**：数字枚举有反向映射，字符串枚举没有，混用时容易出错。
- **`unknown` 与 `any` 的区别**：`unknown` 在使用前必须收窄类型，比 `any` 安全，应优先使用。

---

## 第一部分：TypeScript 基础

### 第1章：环境搭建与基本语法

> 章节链接：[第1章](../part1-basics/01-environment-syntax.md)

本章练习围绕开发环境验证、变量声明与作用域、以及 AI 参数配置管理展开。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 1.1 | 环境验证 | 基础 | 基本类型注解（`string`、`number`、`boolean`）、函数签名 |
| 1.2 | 变量作用域 | 基础 | `let` 块级作用域、变量遮蔽（shadowing）、`const` 不可重赋值 |
| 1.3 | AI 参数验证器 | 进阶 | 函数实现、多条件校验、返回对象类型 `{ valid: boolean; errors: string[] }` |
| 1.4 | Token 费用计算器 | 进阶 | `const` 常量定义、数值运算、函数组合 |
| 1.5 | 环境配置加载器 | 挑战 | `process.env` 读取、默认值合并、抛出错误、综合函数设计 |

#### 关键解题提示

- **练习 1.1**：直接在 `___` 处填入 `string`、`number`、`boolean`；函数返回值是 `number`，可让编译器自动推断或显式标注。
- **练习 1.2**：`if` 块内的 `let x` 与函数内的 `let x` 是独立变量；`const z` 在块外不可访问，注释掉的那行会产生编译错误（不是运行时错误）。输出顺序：(A) 40，(B) 30，(D) 10。
- **练习 1.3**：三个验证条件独立检查，都不满足时 `valid` 为 `false`；用 `errors.push(...)` 累积错误信息，最后 `return { valid: errors.length === 0, errors }`。
- **练习 1.4**：关键是将模型名作为条件分支选择定价常量，注意精度问题可用 `.toFixed(6)` 格式化。
- **练习 1.5**：`loadConfig()` 的核心是 `process.env.XXX ?? defaultValue`；`validateConfig()` 返回 `string[]`；`getConfigOrThrow()` 调用前两者，若 `errors.length > 0` 则 `throw new Error(errors.join('\n'))`。

---

### 第2章：数据类型详解

> 章节链接：[第2章](../part1-basics/02-data-types.md)

本章练习覆盖类型注解补全、枚举设计、元组应用、`unknown` 类型收窄，以及综合类型系统设计。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 1 | 类型注解补全 | 基础 | 原始类型、`string[]`、`null` 联合类型 |
| 2 | 枚举设计 | 基础 | 字符串枚举、语义化枚举值 |
| 3 | 元组应用 | 中级 | 命名元组、可选元组元素、元组处理函数 |
| 4 | 类型收窄与 unknown | 中级 | `unknown` 转为具体类型、类型守卫、辨识联合 |
| 5 | 综合类型系统设计 | 高级 | 泛型函数、`const enum`、`never` 穷尽检查、建造者模式 |

#### 关键解题提示

- **练习 1**：`lastError` 的正确类型是 `string | null`（不是 `null`），因为它后续可能被赋予字符串；`getModelById` 的 `models` 参数需要接口类型 `ModelInfo[]`。
- **练习 2**：使用字符串枚举（`enum SessionStatus { Active = "active" }`）而非数字枚举，值要有实际语义，便于序列化到 JSON。
- **练习 3**：命名元组语法 `type ApiCallRecord = [requestId: string, timestamp: number, ...]`；可选元素放在末尾并加 `?`；`calculateSuccessRate` 用 `filter(r => r[3]).length / records.length`。
- **练习 4**：先用 `typeof data === 'object' && data !== null` 排除非对象，再检查 `'type' in data`，最后检查 `data.type` 的具体字符串值。
- **练习 5**：`never` 穷尽检查的标准写法是在 `switch` 的 `default` 分支中 `const _exhaustive: never = provider`，当新增 provider 未处理时编译报错。

---

### 第3章：控制流与函数

> 章节链接：[第3章](../part1-basics/03-control-flow-functions.md)

本章练习考查条件判断、循环遍历、泛型函数设计、函数重载，以及并发重试控制。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 题目 1 | classifyNumber | 基础 | `if/else` 链、函数类型注解、返回字符串字面量 |
| 题目 2 | countWords | 基础 | `for...of`、`Map<string, number>`、字符串处理 |
| 题目 3 | 泛型 pipe 函数 | 进阶 | 泛型、剩余参数、函数数组遍历（`reduce`） |
| 题目 4 | formatOutput 重载 | 进阶 | 函数重载签名、实现签名与重载签名的关系 |
| 题目 5 | withConcurrentRetry | 挑战 | 并发控制、`Promise.allSettled`、结果顺序保持 |

#### 关键解题提示

- **题目 1**：按从小到大顺序判断：先 `< 0`，再 `=== 0`，再 `<= 100`，最后 `else`；返回类型可标注为字符串字面量联合。
- **题目 2**：用 `Map` 的 `get` + `set` 模式：`map.set(word, (map.get(word) ?? 0) + 1)`；先用正则去标点再 `split(/\s+/)`。
- **题目 3**：`pipe` 的核心是 `fns.reduce((acc, fn) => fn(acc), initial)`；泛型 `T` 约束所有函数输入输出类型一致。
- **题目 4**：重载签名须在实现签名之前声明；实现签名参数类型用联合类型兼容所有重载，返回类型用联合类型，内部用 `typeof` 判断分支。
- **题目 5**：维护一个 `active` 计数器和结果数组；用 `while (queue.length || active > 0)` 循环；`await` 一个 Promise，完成后 `active--` 并继续取任务。

---

## 第二部分：类型系统进阶

### 第4章：联合类型与交叉类型

> 章节链接：[第4章](../part2-types/04-union-intersection.md)

本章练习覆盖可辨识联合、字面量类型、交叉类型工具、以及类型安全的命令模式。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 4-1 | Shape 联合类型与面积计算 | 基础 | 可辨识联合（判别式 `type`）、`switch` 穷举 |
| 4-2 | HttpMethod 字面量类型 | 基础 | 字符串字面量联合、接口设计、条件分支 |
| 4-3 | 任务队列可辨识联合 | 中级 | 三态联合、类型收窄、状态机设计 |
| 4-4 | 交叉类型工具类型 | 中级 | 交叉类型、泛型、`Prettify` 类型 |
| 4-5 | 命令模式类型系统 | 高级 | 命令联合类型、类型安全 dispatch、撤销/重做栈 |

#### 关键解题提示

- **4-1**：每个形状接口必须有 `type` 字段作为判别式（`type: "circle"` 等），`switch (shape.type)` 后编译器自动收窄类型，漏掉一个分支会提示 `never` 错误。
- **4-2**：`buildFetchOptions` 内部检查 `method === "GET" || method === "DELETE"` 来决定是否附加 `body`；TypeScript 能推断此后 `body` 属性存在。
- **4-3**：状态转换函数如 `transitionToRunning(task: Task & { status: 'queued' })` 可精确限定输入状态；返回值类型是转换后的状态类型。
- **4-4**：`Prettify<T>` 的经典写法是 `{ [K in keyof T]: T[K] } & {}`，将交叉类型展开为单一对象类型，IDE 提示更友好。
- **4-5**：命令类型定义为可辨识联合，`dispatch` 函数用 `switch` 分发；撤销栈保存反向命令，`redo` 栈在撤销时填充。

---

### 第5章：类型守卫与类型收窄

> 章节链接：[第5章](../part2-types/05-type-guards.md)

本章练习覆盖 `typeof`/`instanceof` 守卫、自定义类型谓词、穷举检查，以及流式事件路由器。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 5-1 | typeof 守卫 | 基础 | `typeof` 收窄、函数重载与守卫结合 |
| 5-2 | instanceof 区分错误 | 基础 | `instanceof`、自定义错误类继承 |
| 5-3 | 自定义类型守卫 | 中级 | `is` 类型谓词、运行时结构验证 |
| 5-4 | 判别联合与穷举检查 | 中级 | `never` 穷举、`assertNever` 函数 |
| 5-5 | 流式事件路由器 | 高级 | 泛型事件映射、类型安全回调注册 |

#### 关键解题提示

- **5-1**：`typeof` 守卫后的分支中，TypeScript 会自动将类型收窄到对应的原始类型；对 `null` 的检查要用 `value === null`（`typeof null === "object"`）。
- **5-2**：自定义错误类需 `class NetworkError extends Error { constructor(...) { super(...); this.name = 'NetworkError'; } }`；`instanceof` 检查需要是类而非接口。
- **5-3**：类型谓词函数签名 `function isXxx(val: unknown): val is XxxType`；内部用属性存在检查 `typeof val === 'object' && val !== null && 'field' in val`。
- **5-4**：穷举检查标准模式：`function assertNever(x: never): never { throw new Error('Unhandled case: ' + x) }`，在 `switch` 的 `default` 分支调用。
- **5-5**：事件映射类型 `EventMap` 将事件名映射到数据类型；`on<K extends keyof EventMap>(event: K, handler: (data: EventMap[K]) => void)` 保证回调类型与事件匹配。

---

### 第6章：泛型编程

> 章节链接：[第6章](../part2-types/06-generics.md)

本章练习覆盖泛型工具函数、泛型类、泛型约束事件系统、泛型管道，以及带重试缓存的请求装饰器。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 6-1 | 泛型工具函数 | 基础 | 泛型参数、类型推断、`keyof T`、`T[K]` |
| 6-2 | 泛型 Queue 类 | 基础 | 泛型类、`Array<T>` 内部存储、队列方法 |
| 6-3 | 泛型事件系统 | 中级 | 泛型约束 `T extends Record<string, unknown>`、映射类型 |
| 6-4 | 泛型管道 | 中级 | 泛型函数链、元组类型、条件类型推断 |
| 6-5 | 带重试缓存的请求装饰器 | 高级 | 泛型装饰函数、`Map` 缓存、指数退避、组合模式 |

#### 关键解题提示

- **6-1**：`pick<T, K extends keyof T>(obj: T, keys: K[]): Pick<T, K>` 用 `keys.reduce` 构建新对象；`groupBy<T, K extends string>(arr: T[], fn: (item: T) => K)` 返回 `Record<K, T[]>`。
- **6-2**：`Queue<T>` 内部维护 `private items: T[] = []`；`dequeue()` 返回 `T | undefined`；`peek()` 访问 `items[0]` 不移除。
- **6-3**：事件系统约束 `T extends Record<string, unknown>` 使事件数据必须是对象；监听器类型 `(data: T[K]) => void` 保证类型安全。
- **6-4**：管道类型的难点在于推断相邻步骤的输入输出类型一致性；用元组类型和条件类型逐步推断或用重载枚举常见长度。
- **6-5**：装饰器接受原函数，返回同签名新函数；缓存键用参数序列化；重试用递归或循环加 `await new Promise(resolve => setTimeout(resolve, delay))`。

---

## 第三部分：面向对象编程

### 第7章：类与接口

> 章节链接：[第7章](../part3-oop/07-classes-interfaces.md)

本章练习覆盖类的属性与方法、接口实现、类继承、接口继承，以及混入模式。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 7-1 | ModelProfile 类 | 基础 | 类属性、构造函数、访问修饰符、getter/setter |
| 7-2 | 实现接口并验证 | 基础 | `implements`、接口方法签名、类型兼容性 |
| 7-3 | 类继承与 super | 中级 | `extends`、`super()` 调用、方法重写 |
| 7-4 | 接口继承设计 | 中级 | 接口 `extends`、多接口继承、设计层次结构 |
| 7-5 | 混入模式实战 | 高级 | Mixin 函数、交叉类型、`Object.assign`、类型安全混入 |

#### 关键解题提示

- **7-1**：`private` 字段配合 `get`/`set` 访问器控制读写；`readonly` 属性只能在构造函数中赋值；类方法中 `this` 类型可用 `this: typeof this` 泛型约束实现链式调用。
- **7-2**：实现接口时每个方法必须类型兼容（参数类型可以更宽松，返回类型可以更具体）；缺少方法会得到编译错误提示，逐一补全即可。
- **7-3**：子类构造函数必须先调用 `super()`；重写方法时用 `override` 关键字（TS 4.3+）更安全；`super.method()` 调用父类方法。
- **7-4**：接口继承 `interface C extends A, B {}` 合并所有父接口的属性；若父接口有相同属性名但类型不兼容则报错。
- **7-5**：Mixin 函数签名 `function Serializable<T extends Constructor>(Base: T)`；内部 `return class extends Base { ... }` 添加新方法；多个 Mixin 叠加用函数组合。

---

### 第8章：抽象类与装饰器

> 章节链接：[第8章](../part3-oop/08-abstract-decorators.md)

本章练习覆盖抽象类设计、方法装饰器、属性装饰器、缓存装饰器，以及 RAG 系统框架设计。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 8.1 | 抽象类 DataProcessor | 基础 | `abstract` 类和方法、子类实现、模板方法模式 |
| 8.2 | @Deprecated 装饰器 | 基础 | 方法装饰器签名、`descriptor.value` 替换 |
| 8.3 | @Clamp 属性装饰器 | 中级 | 属性装饰器、`Object.defineProperty`、getter/setter 拦截 |
| 8.4 | @Cache(ttlMs) 装饰器 | 中级 | 装饰器工厂、`Map` 缓存、参数序列化为缓存键、TTL 过期 |
| 8.5 | RAG 系统框架 | 高级 | 抽象类 + 装饰器组合、泛型约束、完整系统设计 |

#### 关键解题提示

- **8.1**：`abstract process(data: unknown): unknown` 强制子类实现；基类可提供具体方法（如 `validate`）供子类复用；子类用 `override` 关键字标记实现。
- **8.2**：装饰器工厂 `Deprecated(message)` 返回装饰器函数；内部用 `const original = descriptor.value`，替换为包装函数，包装函数中先 `console.warn`，再 `return original.apply(this, args)`。
- **8.3**：属性装饰器无法直接访问实例值，需用 `Object.defineProperty` 定义带拦截的 getter/setter；在 setter 内做 `Math.min(max, Math.max(min, value))`。
- **8.4**：缓存键 `JSON.stringify(args)` 序列化参数；检查 `cache.get(key)` 是否存在且未过期（`Date.now() - entry.timestamp < ttlMs`）；命中则直接返回，否则调用原方法并写入缓存。
- **8.5**：`abstract class BaseRetriever` 定义 `abstract retrieve(query: string): Promise<Document[]>`；`abstract class BaseGenerator` 定义生成方法；用 `@Cache` 装饰检索方法降低重复请求。

---

### 第9章：模块系统

> 章节链接：[第9章](../part3-oop/09-modules.md)

本章练习覆盖模块导出、模块解析错误修复、动态导入插件系统、声明文件编写，以及多环境配置模块。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 1 | string-utils 模块 | 基础 | 命名导出、默认导出、`export { }` 语法 |
| 2 | 模块解析错误修复 | 基础 | 相对路径、`.js` 扩展名（ESM）、循环依赖识别 |
| 3 | PluginLoader 动态导入 | 中级 | `import()`、`Promise`、插件接口约束 |
| 4 | 第三方库声明文件 | 中级 | `namespace`、`declare module`、类型补充 |
| 5 | 多环境配置模块 | 高级 | 条件导出、`import.meta.env`、环境感知模块设计 |

#### 关键解题提示

- **练习 1**：`export function xxx` 是命名导出；`export default` 每个模块只有一个；`export type { }` 导出纯类型（有助于 `isolatedModules` 模式）。
- **练习 2**：ESM 模式下相对导入必须带 `.js` 扩展名（即使源文件是 `.ts`）；检查 `tsconfig.json` 的 `moduleResolution` 是否为 `node16` 或 `bundler`。
- **练习 3**：`const plugin = await import(path)` 返回模块对象；用接口 `interface Plugin { name: string; execute(): void }` 约束插件形状；动态导入失败需 `try/catch` 处理。
- **练习 4**：`declare namespace LibName { function method(arg: string): void }` 声明全局命名空间；`declare module "lib-name" { export function ... }` 为 npm 包提供类型。
- **练习 5**：`const config = process.env.NODE_ENV === 'production' ? prodConfig : devConfig` 运行时分支；静态分析友好的做法是单独文件导出各环境配置，入口文件按环境选择。

---

## 第四部分：高级类型特性

### 第10章：映射类型与条件类型

> 章节链接：[第10章](../part4-advanced/10-mapped-conditional.md)

本章练习覆盖映射类型变换、条件类型工具、按值过滤属性、`infer` 提取类型，以及插件系统类型框架。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 10-1 | Nullable 与 NonNullableProperties | 基础 | 映射类型 `{ [K in keyof T]: T[K] \| null }`、类型逆操作 |
| 10-2 | IsArray 与 Unwrap | 基础 | 条件类型 `T extends any[] ? ... : ...`、分发特性 |
| 10-3 | PickByValue | 中级 | 映射类型 + 条件类型组合、`as` 键重映射 |
| 10-4 | UnpackPromiseArray 与 infer | 中级 | `infer` 关键字、递归条件类型、`Promise<infer U>` |
| 10-5 | 插件系统类型框架 | 高级 | 泛型插件注册表、条件类型分发、类型安全扩展点 |

#### 关键解题提示

- **10-1**：`type Nullable<T> = { [K in keyof T]: T[K] | null }`；逆操作用 `{ [K in keyof T]: NonNullable<T[K]> }`（利用内置 `NonNullable`）。
- **10-2**：`IsArray<T> = T extends any[] ? true : false`；`Unwrap<T> = T extends (infer U)[] ? U : T`；注意泛型分发特性，传入联合类型时会分别计算。
- **10-3**：`type PickByValue<T, V> = { [K in keyof T as T[K] extends V ? K : never]: T[K] }`；`as` 子句用于过滤键，结果为 `never` 时键被删除。
- **10-4**：`type UnpackPromiseArray<T> = T extends Promise<infer U>[] ? U : never`；若要处理嵌套 Promise，用递归条件类型 `T extends Promise<infer U> ? UnpackPromise<U> : T`。
- **10-5**：插件注册表类型 `type PluginRegistry<T extends Record<string, PluginDef>> = { [K in keyof T]: InstanceType<T[K]['class']> }`；`getPlugin<K extends keyof T>(name: K): PluginRegistry<T>[K]` 保证返回具体类型。

---

### 第11章：实用工具类型

> 章节链接：[第11章](../part4-advanced/11-utility-types.md)

本章练习覆盖 `Partial`/`Required`/`Readonly` 的实际应用、`Pick`/`Omit`/`Record` 的类型操作、类型安全存储、函数注册表，以及 AI Pipeline 类型系统。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 11-1 | ModelConfigManager | 基础 | `Partial<T>` 用于更新、`Required<T>` 用于验证、`Readonly<T>` 冻结配置 |
| 11-2 | Pick/Omit/Record 操作 | 基础 | `Pick` 选取字段、`Omit` 排除字段、`Record` 创建映射 |
| 11-3 | createTypedStorage | 中级 | 泛型工厂函数、JSON 序列化/反序列化、`ReturnType` |
| 11-4 | 函数注册表 | 中级 | `Parameters<F>`、`ReturnType<F>`、`Record` 动态映射 |
| 11-5 | 类型安全 AI Pipeline | 高级 | 泛型管道步骤链、`Awaited<T>`、条件类型推断中间类型 |

#### 关键解题提示

- **11-1**：`update(patch: Partial<ModelConfig>)` 用 `{ ...this.config, ...patch }` 合并；`Required<ModelConfig>` 确保所有字段在验证时存在；`Object.freeze` 配合 `Readonly<T>` 实现运行时冻结。
- **11-2**：`Pick<User, 'id' | 'name'>` 创建只含两字段的类型；`Omit<User, 'password'>` 排除敏感字段；`Record<Status, StatusInfo>` 创建枚举到信息的映射。
- **11-3**：`createTypedStorage<T>()` 返回带 `get`/`set`/`remove` 方法的对象；`set` 调用 `JSON.stringify`，`get` 调用 `JSON.parse`；用泛型确保存取类型一致。
- **11-4**：注册表类型 `type Registry = Record<string, (...args: any[]) => any>`；`call<K extends keyof Registry>(name: K, ...args: Parameters<Registry[K]>): ReturnType<Registry[K]>` 保证类型安全调用。
- **11-5**：Pipeline 步骤类型 `type Step<In, Out> = (input: In) => Out | Promise<Out>`；用元组类型链式推断每一步的输入输出类型，最终类型为最后一步的 `Awaited<Out>`。

---

### 第12章：声明文件与类型定义

> 章节链接：[第12章](../part4-advanced/12-declaration-files.md)

本章练习覆盖全局注入对象的声明、`tsconfig types` 配置、第三方 SDK 声明、模块扩充（augmentation），以及泛型提示词模板引擎声明。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 12-1 | 全局注入对象声明 | 基础 | `declare const`、`declare namespace`、全局 `.d.ts` |
| 12-2 | tsconfig types 配置 | 基础 | `types` 白名单、`typeRoots`、为何限制全局类型 |
| 12-3 | vecdb-js SDK 声明 | 中级 | `declare module`、接口定义、方法重载声明 |
| 12-4 | Express Request 扩充 | 中级 | 模块扩充 `declare module 'express'`、接口合并 |
| 12-5 | 泛型提示词模板引擎 | 高级 | 泛型声明文件、模板变量类型推断、避免 `any` |

#### 关键解题提示

- **12-1**：在 `global.d.ts` 中 `declare const myLib: { version: string; ... }`；若使用命名空间则 `declare namespace MyLib { ... }`；确保文件不含 `import`/`export`（否则变为模块而非全局声明）。
- **12-2**：`"types": ["node", "jest"]` 限制只加载这两个包的全局类型；防止其他 `@types` 包污染全局命名空间（如同时安装 `@types/jquery` 引入 `$` 全局变量）。
- **12-3**：`declare module "vecdb-js" { export class VectorStore { constructor(options: StoreOptions); insert(vec: number[], metadata: object): Promise<string>; search(vec: number[], k: number): Promise<SearchResult[]>; } }`
- **12-4**：在 `src/types/express.d.ts` 中 `import { Request } from 'express'; declare module 'express-serve-static-core' { interface Request { user?: UserInfo; requestId: string; } }`（注意扩充的是内部模块）。
- **12-5**：泛型模板声明 `function render<T extends Record<string, string>>(template: string, vars: T): string`；用模板字面量类型 `type ExtractVars<T extends string>` 提取变量名可实现更精确的类型推断。

---

## 第五部分：工程化实践

### 第13章：tsconfig 配置详解

> 章节链接：[第13章](../part5-tooling/13-tsconfig.md)

本章练习覆盖 `tsconfig.json` 编写、常见错误诊断、monorepo 项目引用配置、泛型配置加载器，以及配置验证工具设计。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 13-1 | 完整 tsconfig 编写 | 基础 | `target`、`module`、`strict`、`paths`、`outDir` |
| 13-2 | 错误配置诊断 | 基础 | 识别不良配置、理解各选项含义与副作用 |
| 13-3 | monorepo 项目引用 | 中级 | `composite`、`references`、增量构建 |
| 13-4 | 泛型配置加载器 | 中级 | `z.infer`（Zod）或手写验证、泛型约束 |
| 13-5 | 配置验证工具 | 高级 | AST 解析 `tsconfig`、规则引擎设计、构建时检查 |

#### 关键解题提示

- **13-1**：路径别名 `"paths": { "@utils/*": ["src/utils/*"] }` 需配合打包工具的别名配置才能在运行时生效；`"declaration": true` 与 `"declarationMap": true` 配合使用。
- **13-2**：常见不良配置包括：`"noImplicitAny": false` 削弱类型安全、`"skipLibCheck": true` 可能掩盖真实错误、`"target": "ES5"` 与 `"module": "ESNext"` 不匹配。
- **13-3**：被引用包需 `"composite": true` 且必须有 `"declaration": true`；根目录 `tsconfig.json` 的 `references` 数组包含所有子包路径；`tsc --build` 按依赖顺序增量编译。
- **13-4**：`loadTypedConfig<T>(schema: Schema<T>, path: string): T` 读取 JSON 文件后用 schema 验证；Zod 用 `schema.parse(raw)`，自定义验证用类型守卫。
- **13-5**：读取 `tsconfig.json` 可用 `ts.readConfigFile` 或 JSON 解析；规则引擎遍历编译选项，对每条规则调用检查函数，收集违规信息输出。

---

### 第14章：构建工具集成

> 章节链接：[第14章](../part5-tooling/14-build-tools.md)

本章练习覆盖 tsconfig 目标配置、运行 TS 文件三种方式对比、esbuild 构建脚本、Vite 代码分割，以及 monorepo 构建配置。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 14-1 | tsconfig 完整配置 | 基础 | Node 18 目标、strict 全开、路径别名、source map |
| 14-2 | 三种运行方式对比 | 基础 | `tsc + node`、`ts-node`、`tsx watch` 的场景差异 |
| 14-3 | esbuild 构建脚本 | 中级 | `esbuild.build()`、bundle/minify/sourcemap 选项 |
| 14-4 | Vite 代码分割 | 中级 | `React.lazy`、`import()`、`manualChunks` 配置 |
| 14-5 | monorepo 构建配置 | 高级 | `composite` 项目引用、`workspace` 依赖、增量构建脚本 |

#### 关键解题提示

- **14-1**：`"module": "Node16"` 需配合 `package.json` 的 `"type": "module"`；路径别名 `@utils/*` 在 esbuild/Vite 中需单独配置 `alias`，tsc 的 `paths` 不影响运行时。
- **14-2**：`tsc + node` 适合生产构建；`ts-node` 适合脚本和 REPL；`tsx watch` 适合开发时热重载，不做类型检查速度最快。
- **14-3**：`external: ['node:*', ...Object.keys(pkg.dependencies)]` 排除所有外部依赖；`define: { 'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV) }` 注入环境变量。
- **14-4**：`const AIChart = React.lazy(() => import('./AIChartDashboard'))`；`vite.config.ts` 中 `build.rollupOptions.output.manualChunks = (id) => id.includes('@anthropic-ai') ? 'ai-sdk' : undefined`。
- **14-5**：`shared/tsconfig.json` 设 `"composite": true`；`backend/tsconfig.json` 设 `"references": [{ "path": "../shared" }]`；根目录脚本 `"build:all": "tsc --build packages/backend"`（会自动构建依赖）。

---

### 第15章：测试与调试

> 章节链接：[第15章](../part5-tooling/15-testing-debugging.md)

本章练习覆盖单元测试编写、类型测试、TTL 缓存测试（假时钟）、AI 聊天历史 Mock 测试，以及带重试的 LLM 客户端测试。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 1 | Stack 类单元测试 | 基础 | `describe`/`it`/`expect`、边界条件、`toThrow` |
| 2 | 工具类型测试 | 基础 | `expectTypeOf`、`Expect`/`Equal` 类型断言 |
| 3 | TTL 缓存测试 | 中级 | `vi.useFakeTimers()`、`vi.advanceTimersByTime()`、TTL 过期逻辑 |
| 4 | AI 聊天历史 Mock 测试 | 中级 | `vi.fn()`、`vi.mock()`、错误分支测试 |
| 5 | 带重试的 LLM 客户端 | 高级 | 实现 + 测试、重试计数断言、指数退避验证 |

#### 关键解题提示

- **练习 1**：测试 `pop()` 空栈时应 `expect(() => stack.pop()).toThrow()`；用 `beforeEach` 重置 `stack = new Stack()` 避免测试间状态污染。
- **练习 2**：`expectTypeOf(fn).returns.toEqualTypeOf<string>()`；`type Test = Expect<Equal<Partial<User>, { name?: string; age?: number }>>` 在类型层面断言。
- **练习 3**：`vi.useFakeTimers()` 在 `beforeEach` 中调用，`vi.useRealTimers()` 在 `afterEach` 中恢复；`vi.advanceTimersByTime(ttl + 1)` 推进时钟使缓存过期。
- **练习 4**：`vi.mock('../llm-client')` 自动 Mock 模块；用 `mockResolvedValue` 设置成功响应，`mockRejectedValue` 设置失败；`expect(mockFn).toHaveBeenCalledWith(...)` 验证调用参数。
- **练习 5**：实现重试时用 `for` 循环而非递归（避免深调用栈）；测试中验证 `mockFn` 被调用次数等于重试次数 + 1（首次调用）；用 Fake Timers 验证退避延迟。

---

## 第六部分：异步编程与网络通信

### 第16章：Promise 与 async/await

> 章节链接：[第16章](../part6-async/16-promises-async.md)

本章练习覆盖 `delay`/倒计时实现、Promise 化回调、带指数退避的 `retry`、异步生成器管道，以及异步事件总线。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 1 | delay 与 countdown | 基础 | `new Promise(setTimeout)`、`async/await`、`for` 循环 |
| 2 | 回调转 Promise | 基础 | Promise 构造函数、`resolve`/`reject`、`async/await` 调用 |
| 3 | retry 与指数退避 | 中级 | 递归 `async` 函数、`Math.pow(2, attempt)`、错误类型判断 |
| 4 | 异步生成器管道 | 中级 | `async function*`、`for await...of`、流式处理 |
| 5 | 异步事件总线 | 高级 | `Map<string, Set<AsyncHandler>>`、`Promise.all` 并发通知 |

#### 关键解题提示

- **练习 1**：`const delay = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))`；`countdown` 函数用 `for` 循环配合 `await delay(1000)`。
- **练习 2**：`new Promise<T>((resolve, reject) => { callbackFn(arg, (err, result) => err ? reject(err) : resolve(result!)) })`；注意 `resolve` 类型要与回调结果类型匹配。
- **练习 3**：`retry` 接受 `fn`、`maxAttempts`、`backoffMs`；失败时 `await delay(backoffMs * Math.pow(2, attempt))`；最后一次失败直接 `throw`。
- **练习 4**：`async function* pipeline(source: AsyncIterable<string>)` 用 `for await...of source` 遍历，`yield` 转换后的值；多个 `async generator` 可用 `yield*` 组合。
- **练习 5**：`publish(event, data)` 调用 `Promise.all([...handlers].map(h => h(data)))` 并行触发所有订阅者；订阅者返回 `Promise<void>` 需要 `await`。

---

### 第17章：HTTP 客户端与 API 调用

> 章节链接：[第17章](../part6-async/17-http-clients.md)

本章练习覆盖类型安全 GET 请求封装、请求重试机制、请求去重、OpenAI JSON Mode 扩展，以及请求队列与并发控制。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 17-1 | 类型安全 GET 请求封装 | 基础 | 泛型 `fetch` 封装、`Response.json()`、错误处理 |
| 17-2 | 重试机制 | 基础 | 循环重试、指数退避、可重试状态码判断 |
| 17-3 | 请求去重 | 中级 | `Map<string, Promise<T>>` 缓存进行中的请求 |
| 17-4 | OpenAI JSON Mode 扩展 | 中级 | `response_format`、`JSON.parse` 结果、泛型返回 |
| 17-5 | 请求队列与并发控制 | 高级 | 优先队列、`Promise` 并发槽、背压控制 |

#### 关键解题提示

- **17-1**：`async function get<T>(url: string, options?: RequestInit): Promise<T>` 内部 `const res = await fetch(url, options); if (!res.ok) throw new HttpError(res.status)`；`return res.json() as Promise<T>`。
- **17-2**：重试判断 `isRetryable(status: number)` 返回 `status >= 500 || status === 429`；退避用 `await sleep(delay * 2 ** attempt)`；达到最大次数后抛出最后一次错误。
- **17-3**：去重键用 `url + JSON.stringify(params)`；检查 `inflightMap.get(key)` 是否存在，存在则直接返回同一 Promise；完成后（`.finally`）从 Map 中删除。
- **17-4**：请求体加 `response_format: { type: 'json_object' }`；响应 `message.content` 是 JSON 字符串，需 `JSON.parse`；用泛型 `<T>` 标注解析结果类型。
- **17-5**：用 `Map<Priority, Array<QueueItem>>` 实现优先队列；活跃请求计数器 `active < maxConcurrent` 时出队执行；每个任务完成后触发下一个。

---

### 第18章：Server-Sent Events 与流式通信

> 章节链接：[第18章](../part6-async/18-sse-streaming.md)

本章练习覆盖 SSE 行解析、带超时的 EventSource 封装、事件录制回放、多路 SSE 合并，以及 LLM 流式响应中间件。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 18-1 | parseSSELine 函数 | 基础 | SSE 协议格式、`data:`/`event:`/`id:` 解析 |
| 18-2 | 带超时的 EventSource 封装 | 基础 | `AbortController`、`setTimeout` + `abort()`、清理逻辑 |
| 18-3 | SSE 事件录制与回放 | 中级 | 事件时间戳记录、`setTimeout` 序列回放、类型安全 |
| 18-4 | 多路合并 SSE 客户端 | 中级 | `Promise.race`/并发读取、合并异步迭代器 |
| 18-5 | LLM 流式响应中间件 | 高级 | 中间件链、流式 Transform、背压处理 |

#### 关键解题提示

- **18-1**：SSE 行格式 `field: value`；`data:` 累积到 `buffer`，遇到空行 `\n\n` 派发事件；`event:` 设置事件类型；忽略以 `:` 开头的注释行。
- **18-2**：`const controller = new AbortController()`；`const timer = setTimeout(() => controller.abort(), timeoutMs)`；在 `abort` 事件或连接关闭时 `clearTimeout(timer)`。
- **18-3**：录制时记录 `{ event, timestamp: Date.now() }`；回放时计算相对延迟 `events[i].timestamp - events[0].timestamp`，用 `setTimeout` 按顺序派发。
- **18-4**：为每个 SSE 源创建异步迭代器，用共享队列 + `Promise` 解析器合并；每个源推送到同一队列，主循环从队列消费。
- **18-5**：中间件函数签名 `type StreamMiddleware = (chunk: string, next: (chunk: string) => void) => void`；链式组合用 `reduceRight` 包裹；支持异步中间件需将 `next` 改为返回 `Promise`。

---

## 第七部分：AI 应用开发实战

### 第19章：LLM API 集成

> 章节链接：[第19章](../part7-ai-apps/19-llm-api-integration.md)

本章练习覆盖品牌类型约束、流式响应收集器、带回退的提供商链、Token 用量追踪器，以及带缓存的 LLM 客户端。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 1 | 品牌类型 Temperature | 基础 | Branded Type、工厂函数、运行时范围检查 |
| 2 | 流式响应收集器 | 基础 | `AsyncIterable<T>`、`for await...of`、累积文本 |
| 3 | 带回退的提供商链 | 进阶 | 高阶函数、`try/catch` 回退、错误日志 |
| 4 | Token 用量追踪器 | 进阶 | 装饰器模式（非语法糖）、`Record` 统计、包装类设计 |
| 5 | 带缓存的 LLM 客户端 | 挑战 | SHA-256 哈希键、TTL 过期、`crypto` 模块、缓存统计 |

#### 关键解题提示

- **练习 1**：`type Temperature = number & { readonly __brand: 'Temperature' }`；工厂函数 `createTemperature` 验证范围后用 `as Temperature` 类型断言；品牌字段 `__brand` 仅存在于类型层面，运行时不占用内存。
- **练习 2**：`for await (const event of iterable)` 遍历流；根据事件类型提取 `delta` 字符串累积到 `content`；流结束（`event.type === 'done'`）时返回结果对象。
- **练习 3**：`withFallback` 返回实现 `LLMProvider` 接口的对象；内部先 `await primary.complete(req)`，`catch` 中 `console.warn` 并 `await fallback.complete(req)`。
- **练习 4**：`UsageTracker` 用 `private stats = new Map<string, UsageStat>()` 存储；每次调用后 `stats.get(model)` 累加，不存在则初始化；`getReport()` 将 Map 转为 `Record`。
- **练习 5**：缓存键 `createHash('sha256').update(JSON.stringify({ messages, model, temperature, maxTokens })).digest('hex')`；`CacheEntry` 包含 `{ value, expiresAt: Date.now() + ttlMs }`；命中时检查 `entry.expiresAt > Date.now()`。

---

### 第20章：流式对话实现

> 章节链接：[第20章](../part7-ai-apps/20-streaming-chat.md)

本章练习覆盖流式 Token 统计、带超时的流高阶函数、多流合并器、速率限制，以及断点续传流式请求。

#### 练习概览

| 编号 | 题目 | 难度 | 考查重点 |
|------|------|------|----------|
| 20-1 | countStreamTokens | 基础 | `AsyncIterable` 遍历、字符/Token 估算、累积统计 |
| 20-2 | withStreamTimeout | 基础 | 高阶函数、`AbortController`、超时中断异步迭代 |
| 20-3 | StreamMerger 类 | 中级 | 并发异步迭代、共享队列、来源标识 |
| 20-4 | 速率限制中间件 | 中级 | 滑动窗口计数器、`Map<IP, number[]>`、HTTP 429 响应 |
| 20-5 | ResumeableStream | 高级 | 断点续传、指数退避重试、续写 prompt 构造 |

#### 关键解题提示

- **20-1**：遍历时 `chunks++`，累积 `content += delta`；最后 `estimatedTokens = Math.ceil(content.length / 4)`；返回类型 `Promise<{ content: string; chunks: number; estimatedTokens: number }>`。
- **20-2**：内部创建 `AbortController`，`setTimeout` 触发 `abort()`；在 `for await...of` 循环中检查 `signal.aborted`，是则 `throw new TimeoutError()`；迭代完成后 `clearTimeout`。
- **20-3**：为每个源创建 Promise，推送到共享优先队列；用 `{ resolve, reject }` 对实现等待机制；主循环 `yield` 队列中的 `MergedChunk`，同时启动新任务填充并发槽。
- **20-4**：滑动窗口 `Map<string, number[]>` 存每个 IP 的请求时间戳数组；每次请求前过滤掉 60 秒之前的时间戳，若剩余数量 `>= 10` 则返回 429 并设 `Retry-After` 头。
- **20-5**：`ResumeableStream` 维护 `accumulated` 字符串；失败时将 `accumulated` 注入 system prompt 要求续写；重试间隔 `baseDelay * 2 ** attempt`；最终拼接结果检查无重叠或缺失。

---

## 章节速查索引

| 章节 | 标题 | 部分 | 核心技能 |
|------|------|------|----------|
| 第1章 | 环境搭建与基本语法 | 基础 | 类型注解、变量声明 |
| 第2章 | 数据类型详解 | 基础 | 枚举、元组、unknown |
| 第3章 | 控制流与函数 | 基础 | 函数重载、泛型、并发 |
| 第4章 | 联合类型与交叉类型 | 类型系统 | 可辨识联合、字面量类型 |
| 第5章 | 类型守卫与类型收窄 | 类型系统 | 类型谓词、穷举检查 |
| 第6章 | 泛型编程 | 类型系统 | 泛型函数、泛型类、约束 |
| 第7章 | 类与接口 | OOP | 继承、实现、混入 |
| 第8章 | 抽象类与装饰器 | OOP | 抽象类、装饰器工厂 |
| 第9章 | 模块系统 | OOP | ESM、动态导入 |
| 第10章 | 映射类型与条件类型 | 高级类型 | 映射、条件、infer |
| 第11章 | 实用工具类型 | 高级类型 | Partial、Pick、Record |
| 第12章 | 声明文件与类型定义 | 高级类型 | .d.ts、模块扩充 |
| 第13章 | tsconfig 配置详解 | 工程化 | 编译选项、项目引用 |
| 第14章 | 构建工具集成 | 工程化 | esbuild、Vite、monorepo |
| 第15章 | 测试与调试 | 工程化 | Vitest、Mock、类型测试 |
| 第16章 | Promise 与 async/await | 异步 | 异步模式、并发控制 |
| 第17章 | HTTP 客户端与 API 调用 | 异步 | fetch 封装、重试、去重 |
| 第18章 | SSE 与流式通信 | 异步 | SSE 解析、流合并 |
| 第19章 | LLM API 集成 | AI 实战 | SDK 封装、品牌类型 |
| 第20章 | 流式对话实现 | AI 实战 | 流合并、速率限制、续传 |

---

*第21章（Function Calling 与 Agent）、第22章（Node.js 后端开发）、第23章（React 前端集成）、第24章（完整项目实战）的练习答案将在对应章节完成后补充至本附录。*
