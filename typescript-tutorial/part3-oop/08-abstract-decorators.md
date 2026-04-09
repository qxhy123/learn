# 第8章：抽象类与装饰器

## 学习目标

学完本章后，你将能够：

1. 理解抽象类和抽象方法的概念，并在设计类层次结构时正确运用
2. 掌握 TypeScript 装饰器的启用方式及基本语法
3. 编写类装饰器和方法装饰器，为类和方法添加横切关注点
4. 使用属性装饰器和参数装饰器实现数据验证与元数据注入
5. 组合多个装饰器，构建可复用的装饰器工厂函数

---

## 8.1 抽象类

### 什么是抽象类

抽象类是不能直接实例化的类，它作为其他类的基类存在。抽象类可以包含抽象方法（只有声明、没有实现）和具体方法（有完整实现）。

抽象类用于定义一组相关类的公共接口和部分实现，强制子类提供特定方法的具体实现。

```typescript
abstract class Shape {
  // 抽象方法：子类必须实现
  abstract area(): number;
  abstract perimeter(): number;

  // 具体方法：子类可以直接继承
  describe(): string {
    return `面积为 ${this.area().toFixed(2)}，周长为 ${this.perimeter().toFixed(2)}`;
  }
}

// 错误：不能直接实例化抽象类
// const s = new Shape(); // TS2511: Cannot create an instance of an abstract class.

class Circle extends Shape {
  constructor(private radius: number) {
    super();
  }

  area(): number {
    return Math.PI * this.radius ** 2;
  }

  perimeter(): number {
    return 2 * Math.PI * this.radius;
  }
}

class Rectangle extends Shape {
  constructor(private width: number, private height: number) {
    super();
  }

  area(): number {
    return this.width * this.height;
  }

  perimeter(): number {
    return 2 * (this.width + this.height);
  }
}

const circle = new Circle(5);
const rect = new Rectangle(4, 6);

console.log(circle.describe()); // 面积为 78.54，周长为 31.42
console.log(rect.describe());   // 面积为 24.00，周长为 20.00
```

### 抽象类与接口的区别

| 特性           | 抽象类                       | 接口                         |
|--------------|----------------------------|----------------------------|
| 实例化         | 不能直接实例化                 | 不能直接实例化                 |
| 方法实现        | 可以有具体实现                 | 只有声明（TypeScript 3.x 前）  |
| 构造函数        | 可以有                       | 不能有                       |
| 成员访问修饰符    | 可以使用 public/protected/private | 默认 public                |
| 继承/实现数量    | 一个类只能继承一个抽象类          | 一个类可以实现多个接口           |
| 包含状态        | 可以有实例属性                 | 不持有实例状态                 |

```typescript
// 接口：定义"能做什么"
interface Serializable {
  serialize(): string;
  deserialize(data: string): void;
}

interface Loggable {
  log(message: string): void;
}

// 抽象类：定义"是什么"，同时提供部分实现
abstract class BaseModel implements Serializable, Loggable {
  protected createdAt: Date = new Date();
  protected updatedAt: Date = new Date();

  // 具体实现：通用的日志方法
  log(message: string): void {
    console.log(`[${this.constructor.name}] ${message}`);
  }

  // 具体实现：通用的序列化基础
  serialize(): string {
    return JSON.stringify(this.toJSON());
  }

  // 抽象方法：子类必须定义如何转为 JSON
  abstract toJSON(): Record<string, unknown>;

  // 抽象方法：子类必须定义如何从 JSON 恢复
  abstract deserialize(data: string): void;
}

class UserModel extends BaseModel {
  constructor(public name: string, public email: string) {
    super();
  }

  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      email: this.email,
      createdAt: this.createdAt.toISOString(),
    };
  }

  deserialize(data: string): void {
    const parsed = JSON.parse(data);
    this.name = parsed.name;
    this.email = parsed.email;
  }
}

const user = new UserModel("张三", "zhangsan@example.com");
user.log("用户已创建");           // [UserModel] 用户已创建
console.log(user.serialize());   // {"name":"张三","email":"zhangsan@example.com",...}
```

### 抽象类在 AI 应用中的应用

在构建 AI 应用时，抽象类非常适合定义不同 AI 模型提供商的统一接口：

```typescript
// 定义 AI 模型的抽象基类
abstract class AIModel {
  constructor(
    protected modelName: string,
    protected apiKey: string
  ) {}

  // 抽象方法：不同提供商有不同的实现
  abstract complete(prompt: string): Promise<string>;
  abstract embed(text: string): Promise<number[]>;

  // 具体方法：通用的 token 估算（简化版）
  estimateTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }

  // 具体方法：带重试的请求包装
  protected async withRetry<T>(
    fn: () => Promise<T>,
    maxRetries = 3
  ): Promise<T> {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        if (attempt === maxRetries) throw error;
        const delay = 1000 * attempt;
        console.log(`第 ${attempt} 次失败，${delay}ms 后重试...`);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
    throw new Error("不可达的代码");
  }
}

// OpenAI 实现
class OpenAIModel extends AIModel {
  constructor(apiKey: string) {
    super("gpt-4o", apiKey);
  }

  async complete(prompt: string): Promise<string> {
    return this.withRetry(async () => {
      // 实际调用 OpenAI API（此处为示意）
      console.log(`[OpenAI] 调用 ${this.modelName}，估算 token: ${this.estimateTokens(prompt)}`);
      return `OpenAI 回复: ${prompt.slice(0, 20)}...`;
    });
  }

  async embed(text: string): Promise<number[]> {
    return this.withRetry(async () => {
      console.log(`[OpenAI] 生成嵌入向量`);
      return new Array(1536).fill(0).map(() => Math.random());
    });
  }
}

// Anthropic 实现
class AnthropicModel extends AIModel {
  constructor(apiKey: string) {
    super("claude-opus-4-5", apiKey);
  }

  async complete(prompt: string): Promise<string> {
    return this.withRetry(async () => {
      console.log(`[Anthropic] 调用 ${this.modelName}`);
      return `Anthropic 回复: ${prompt.slice(0, 20)}...`;
    });
  }

  async embed(text: string): Promise<number[]> {
    return this.withRetry(async () => {
      console.log(`[Anthropic] 生成嵌入向量`);
      return new Array(1024).fill(0).map(() => Math.random());
    });
  }
}

// 使用统一接口
async function processWithModel(model: AIModel, prompt: string): Promise<void> {
  const response = await model.complete(prompt);
  console.log(`回复: ${response}`);
  console.log(`预估 token: ${model.estimateTokens(prompt)}`);
}
```

---

## 8.2 装饰器基础

### 启用装饰器

装饰器是 TypeScript 的实验性特性（Stage 3 提案），需要在 `tsconfig.json` 中启用：

```json
{
  "compilerOptions": {
    "experimentalDecorators": true,
    "emitDecoratorMetadata": true,
    "target": "ES2020",
    "module": "commonjs"
  }
}
```

也可以在命令行中启用：

```bash
tsc --experimentalDecorators --emitDecoratorMetadata
```

### 装饰器的本质

装饰器本质上是一个函数，它接收被装饰目标的元数据，并可以修改或增强目标的行为。

```typescript
// 最简单的装饰器：一个函数
function simpleDecorator(target: Function) {
  console.log(`正在装饰: ${target.name}`);
}

@simpleDecorator
class MyClass {
  greet() {
    console.log("Hello!");
  }
}

// 输出: 正在装饰: MyClass
```

装饰器使用 `@` 符号应用，放在类、方法、属性或参数声明的前面。

### 装饰器的执行时机

装饰器在**类定义时**（不是实例化时）执行：

```typescript
function log(message: string) {
  return function (target: any) {
    console.log(message);
  };
}

console.log("开始定义类");

@log("类装饰器执行")
class Demo {
  @log("属性装饰器执行")
  name: string = "";

  @log("方法装饰器执行")
  greet() {}
}

console.log("类定义完成");

// 输出顺序：
// 开始定义类
// 属性装饰器执行
// 方法装饰器执行
// 类装饰器执行
// 类定义完成
```

---

## 8.3 类装饰器与方法装饰器

### 类装饰器

类装饰器接收类的构造函数作为参数，可以返回一个新的构造函数来替换原始类。

**签名：**

```typescript
type ClassDecorator = (target: Function) => void | Function;
```

**示例：添加时间戳**

```typescript
function Timestamped<T extends { new(...args: any[]): {} }>(constructor: T) {
  return class extends constructor {
    createdAt = new Date();
    updatedAt = new Date();

    touch() {
      (this as any).updatedAt = new Date();
    }
  };
}

@Timestamped
class User {
  constructor(public name: string) {}
}

const user = new User("李四");
console.log((user as any).createdAt); // 当前时间
console.log((user as any).updatedAt); // 当前时间
(user as any).touch();
console.log((user as any).updatedAt); // 更新后的时间
```

**示例：单例装饰器**

```typescript
function Singleton<T extends { new(...args: any[]): {} }>(constructor: T) {
  let instance: InstanceType<T> | null = null;

  return new Proxy(constructor, {
    construct(target, args) {
      if (!instance) {
        instance = Reflect.construct(target, args) as InstanceType<T>;
      }
      return instance;
    },
  });
}

@Singleton
class DatabaseConnection {
  private connectionId: string;

  constructor(url: string) {
    this.connectionId = Math.random().toString(36).slice(2);
    console.log(`创建连接 ${this.connectionId} -> ${url}`);
  }

  query(sql: string): string {
    return `[${this.connectionId}] 执行: ${sql}`;
  }
}

const db1 = new DatabaseConnection("postgres://localhost/mydb");
const db2 = new DatabaseConnection("postgres://localhost/mydb");
console.log(db1 === db2); // true，同一个实例
```

### 方法装饰器

方法装饰器接收三个参数：目标对象、方法名、属性描述符。

**签名：**

```typescript
type MethodDecorator = (
  target: Object,
  propertyKey: string | symbol,
  descriptor: PropertyDescriptor
) => PropertyDescriptor | void;
```

**示例：执行时间测量**

```typescript
function MeasureTime(
  target: Object,
  propertyKey: string,
  descriptor: PropertyDescriptor
): PropertyDescriptor {
  const originalMethod = descriptor.value;

  descriptor.value = async function (...args: any[]) {
    const start = performance.now();
    const result = await originalMethod.apply(this, args);
    const end = performance.now();
    console.log(`[${propertyKey}] 耗时: ${(end - start).toFixed(2)}ms`);
    return result;
  };

  return descriptor;
}

class DataService {
  @MeasureTime
  async fetchUsers(): Promise<string[]> {
    // 模拟网络请求
    await new Promise((resolve) => setTimeout(resolve, 200));
    return ["张三", "李四", "王五"];
  }
}

const service = new DataService();
service.fetchUsers().then(users => {
  console.log(users);
  // [fetchUsers] 耗时: 203.14ms
  // ['张三', '李四', '王五']
});
```

**示例：方法缓存（Memoize）**

```typescript
function Memoize(
  target: Object,
  propertyKey: string,
  descriptor: PropertyDescriptor
): PropertyDescriptor {
  const originalMethod = descriptor.value;
  const cache = new Map<string, any>();

  descriptor.value = function (...args: any[]) {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      console.log(`[${propertyKey}] 命中缓存，key: ${key}`);
      return cache.get(key);
    }
    const result = originalMethod.apply(this, args);
    cache.set(key, result);
    return result;
  };

  return descriptor;
}

class MathService {
  @Memoize
  fibonacci(n: number): number {
    if (n <= 1) return n;
    return this.fibonacci(n - 1) + this.fibonacci(n - 2);
  }
}

const math = new MathService();
console.log(math.fibonacci(10)); // 55（计算）
console.log(math.fibonacci(10)); // 55（缓存）
console.log(math.fibonacci(8));  // 21（缓存，因为递归时已计算）
```

**示例：访问控制装饰器**

```typescript
type Role = "admin" | "user" | "guest";

function RequireRole(role: Role) {
  return function (
    target: Object,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ): PropertyDescriptor {
    const originalMethod = descriptor.value;

    descriptor.value = function (...args: any[]) {
      const currentRole: Role = (this as any).currentRole ?? "guest";
      const roleHierarchy: Role[] = ["guest", "user", "admin"];

      if (roleHierarchy.indexOf(currentRole) < roleHierarchy.indexOf(role)) {
        throw new Error(`权限不足：需要 ${role}，当前为 ${currentRole}`);
      }

      return originalMethod.apply(this, args);
    };

    return descriptor;
  };
}

class AdminService {
  currentRole: Role = "user";

  @RequireRole("user")
  viewProfile(): string {
    return "查看个人资料";
  }

  @RequireRole("admin")
  deleteUser(userId: string): string {
    return `删除用户 ${userId}`;
  }
}

const adminService = new AdminService();
console.log(adminService.viewProfile()); // 查看个人资料
try {
  adminService.deleteUser("123"); // 抛出错误
} catch (e) {
  console.error((e as Error).message); // 权限不足：需要 admin，当前为 user
}

adminService.currentRole = "admin";
console.log(adminService.deleteUser("123")); // 删除用户 123
```

---

## 8.4 属性装饰器与参数装饰器

### 属性装饰器

属性装饰器接收两个参数：目标对象和属性名。注意：属性装饰器**不接收**属性描述符。

**签名：**

```typescript
type PropertyDecorator = (
  target: Object,
  propertyKey: string | symbol
) => void;
```

**示例：属性验证**

```typescript
// 使用 Reflect.metadata 需要安装 reflect-metadata 包
// 此处用简单的 Map 模拟元数据存储
const validationRules = new Map<string, Map<string, any>>();

function Min(minValue: number) {
  return function (target: Object, propertyKey: string): void {
    const className = target.constructor.name;
    if (!validationRules.has(className)) {
      validationRules.set(className, new Map());
    }
    validationRules.get(className)!.set(propertyKey, { min: minValue });
  };
}

function Max(maxValue: number) {
  return function (target: Object, propertyKey: string): void {
    const className = target.constructor.name;
    if (!validationRules.has(className)) {
      validationRules.set(className, new Map());
    }
    const existing = validationRules.get(className)!.get(propertyKey) ?? {};
    validationRules.get(className)!.set(propertyKey, { ...existing, max: maxValue });
  };
}

function validate(instance: object): string[] {
  const className = instance.constructor.name;
  const rules = validationRules.get(className);
  const errors: string[] = [];

  if (!rules) return errors;

  for (const [prop, rule] of rules) {
    const value = (instance as any)[prop];
    if (rule.min !== undefined && value < rule.min) {
      errors.push(`${prop} 不能小于 ${rule.min}，当前值: ${value}`);
    }
    if (rule.max !== undefined && value > rule.max) {
      errors.push(`${prop} 不能大于 ${rule.max}，当前值: ${value}`);
    }
  }

  return errors;
}

class AIConfig {
  @Min(1)
  @Max(32000)
  maxTokens: number;

  @Min(0)
  @Max(2)
  temperature: number;

  constructor(maxTokens: number, temperature: number) {
    this.maxTokens = maxTokens;
    this.temperature = temperature;
  }
}

const config = new AIConfig(500, 0.7);
console.log(validate(config)); // []（无错误）

const badConfig = new AIConfig(-100, 3.5);
console.log(validate(badConfig));
// ['maxTokens 不能小于 1，当前值: -100', 'temperature 不能大于 2，当前值: 3.5']
```

**示例：Observable 属性**

```typescript
type Observer<T> = (newValue: T, oldValue: T) => void;

const observers = new WeakMap<object, Map<string, Observer<any>[]>>();

function Observable(target: Object, propertyKey: string): void {
  let value: any;

  Object.defineProperty(target, propertyKey, {
    get() {
      return value;
    },
    set(newValue: any) {
      const oldValue = value;
      value = newValue;

      const instanceObservers = observers.get(this);
      if (instanceObservers) {
        const propObservers = instanceObservers.get(propertyKey) ?? [];
        propObservers.forEach((observer) => observer(newValue, oldValue));
      }
    },
    enumerable: true,
    configurable: true,
  });
}

function addObserver<T>(
  instance: object,
  propertyKey: string,
  observer: Observer<T>
): void {
  if (!observers.has(instance)) {
    observers.set(instance, new Map());
  }
  const map = observers.get(instance)!;
  if (!map.has(propertyKey)) {
    map.set(propertyKey, []);
  }
  map.get(propertyKey)!.push(observer);
}

class ModelConfig {
  @Observable
  model: string = "gpt-4o";

  @Observable
  temperature: number = 0.7;
}

const modelConfig = new ModelConfig();
addObserver(modelConfig, "model", (newVal, oldVal) => {
  console.log(`模型从 ${oldVal} 切换为 ${newVal}`);
});
addObserver(modelConfig, "temperature", (newVal, oldVal) => {
  console.log(`温度从 ${oldVal} 调整为 ${newVal}`);
});

modelConfig.model = "claude-opus-4-5";     // 模型从 gpt-4o 切换为 claude-opus-4-5
modelConfig.temperature = 0.9;           // 温度从 0.7 调整为 0.9
```

### 参数装饰器

参数装饰器接收三个参数：目标对象、方法名、参数索引。

**签名：**

```typescript
type ParameterDecorator = (
  target: Object,
  propertyKey: string | symbol,
  parameterIndex: number
) => void;
```

**示例：参数验证**

```typescript
const requiredParams = new Map<string, Map<string, number[]>>();

function Required(target: Object, propertyKey: string, parameterIndex: number): void {
  const className = target.constructor.name;
  if (!requiredParams.has(className)) {
    requiredParams.set(className, new Map());
  }
  const methodParams = requiredParams.get(className)!;
  if (!methodParams.has(propertyKey)) {
    methodParams.set(propertyKey, []);
  }
  methodParams.get(propertyKey)!.push(parameterIndex);
}

function ValidateParams(
  target: Object,
  propertyKey: string,
  descriptor: PropertyDescriptor
): PropertyDescriptor {
  const originalMethod = descriptor.value;

  descriptor.value = function (...args: any[]) {
    const className = (target as any).constructor.name;
    const methodRequired = requiredParams.get(className)?.get(propertyKey) ?? [];

    for (const index of methodRequired) {
      if (args[index] === undefined || args[index] === null || args[index] === "") {
        throw new Error(`参数 ${index} 是必填项`);
      }
    }

    return originalMethod.apply(this, args);
  };

  return descriptor;
}

class PromptService {
  @ValidateParams
  buildPrompt(
    @Required systemPrompt: string,
    @Required userMessage: string,
    context?: string
  ): string {
    const parts = [
      `System: ${systemPrompt}`,
      context ? `Context: ${context}` : null,
      `User: ${userMessage}`,
    ].filter(Boolean);
    return parts.join("\n");
  }
}

const promptService = new PromptService();

try {
  promptService.buildPrompt("", "你好"); // 抛出错误
} catch (e) {
  console.error((e as Error).message); // 参数 0 是必填项
}

const prompt = promptService.buildPrompt(
  "你是一个 AI 助手",
  "解释一下什么是装饰器",
  "TypeScript 教程第8章"
);
console.log(prompt);
// System: 你是一个 AI 助手
// Context: TypeScript 教程第8章
// User: 解释一下什么是装饰器
```

---

## 8.5 装饰器工厂与装饰器组合

### 装饰器工厂

装饰器工厂是一个返回装饰器的函数，它允许装饰器接受配置参数。

```typescript
// 不带参数的装饰器
function SimpleLog(target: Object, key: string, desc: PropertyDescriptor) {
  const original = desc.value;
  desc.value = function (...args: any[]) {
    console.log(`调用 ${key}`);
    return original.apply(this, args);
  };
  return desc;
}

// 带参数的装饰器工厂
function Log(options: { level?: string; prefix?: string } = {}) {
  const { level = "INFO", prefix = "" } = options;

  return function (target: Object, key: string, desc: PropertyDescriptor): PropertyDescriptor {
    const original = desc.value;

    desc.value = async function (...args: any[]) {
      const label = prefix ? `[${prefix}]` : "";
      console.log(`[${level}]${label} 调用 ${key}，参数:`, args);
      try {
        const result = await original.apply(this, args);
        console.log(`[${level}]${label} ${key} 返回:`, result);
        return result;
      } catch (error) {
        console.error(`[ERROR]${label} ${key} 失败:`, error);
        throw error;
      }
    };

    return desc;
  };
}

class AIService {
  @Log({ level: "DEBUG", prefix: "AI" })
  async generateText(prompt: string, maxTokens: number): Promise<string> {
    // 模拟 AI 调用
    await new Promise((resolve) => setTimeout(resolve, 100));
    return `生成的文本（${maxTokens} tokens）`;
  }

  @Log({ level: "INFO" })
  async analyzeImage(imageUrl: string): Promise<string> {
    await new Promise((resolve) => setTimeout(resolve, 150));
    return "图像分析结果";
  }
}
```

### 装饰器组合

多个装饰器可以叠加使用。执行顺序如下：
- **从下到上**：先执行靠近方法的装饰器（求值顺序从上到下，但调用顺序从下到上）

```typescript
function First() {
  console.log("First 工厂求值");
  return function (target: Object, key: string, desc: PropertyDescriptor) {
    console.log("First 装饰器调用");
    return desc;
  };
}

function Second() {
  console.log("Second 工厂求值");
  return function (target: Object, key: string, desc: PropertyDescriptor) {
    console.log("Second 装饰器调用");
    return desc;
  };
}

class OrderDemo {
  @First()
  @Second()
  method() {}
}

// 输出：
// First 工厂求值
// Second 工厂求值
// Second 装饰器调用
// First 装饰器调用
```

### 实用装饰器组合：限流 + 重试 + 日志

```typescript
// 限流装饰器：控制调用频率
function RateLimit(callsPerSecond: number) {
  const queue: Array<() => void> = [];
  let lastCallTime = 0;

  return function (target: Object, key: string, desc: PropertyDescriptor): PropertyDescriptor {
    const original = desc.value;

    desc.value = function (...args: any[]) {
      return new Promise((resolve, reject) => {
        const now = Date.now();
        const minInterval = 1000 / callsPerSecond;
        const delay = Math.max(0, lastCallTime + minInterval - now);

        setTimeout(async () => {
          lastCallTime = Date.now();
          try {
            const result = await original.apply(this, args);
            resolve(result);
          } catch (error) {
            reject(error);
          }
        }, delay);
      });
    };

    return desc;
  };
}

// 重试装饰器
function Retry(maxAttempts: number, delayMs: number = 1000) {
  return function (target: Object, key: string, desc: PropertyDescriptor): PropertyDescriptor {
    const original = desc.value;

    desc.value = async function (...args: any[]) {
      let lastError: Error | null = null;

      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          return await original.apply(this, args);
        } catch (error) {
          lastError = error as Error;
          if (attempt < maxAttempts) {
            console.warn(`[Retry] ${key} 第 ${attempt} 次失败，${delayMs}ms 后重试`);
            await new Promise((resolve) => setTimeout(resolve, delayMs));
          }
        }
      }

      throw lastError;
    };

    return desc;
  };
}

// 超时装饰器
function Timeout(ms: number) {
  return function (target: Object, key: string, desc: PropertyDescriptor): PropertyDescriptor {
    const original = desc.value;

    desc.value = function (...args: any[]) {
      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`${key} 超时（${ms}ms）`)), ms)
      );
      return Promise.race([original.apply(this, args), timeoutPromise]);
    };

    return desc;
  };
}

class RobustAIClient {
  @Log({ level: "INFO", prefix: "AI" })
  @Retry(3, 500)
  @RateLimit(2)
  @Timeout(5000)
  async callAPI(prompt: string): Promise<string> {
    // 模拟可能失败的 API 调用
    if (Math.random() < 0.3) {
      throw new Error("API 暂时不可用");
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
    return `AI 回复: ${prompt.slice(0, 30)}`;
  }
}

// 装饰器执行顺序（从外到内包装，从内到外执行）：
// 1. Timeout 最先检查超时
// 2. RateLimit 控制调用频率
// 3. Retry 处理重试逻辑
// 4. Log 记录调用信息
```

---

## 本章小结

| 概念            | 关键点                                                         |
|---------------|--------------------------------------------------------------|
| 抽象类          | 用 `abstract` 关键字声明，不能直接实例化，可包含抽象方法和具体方法           |
| 抽象方法         | 只有声明没有实现，子类必须重写，用 `abstract` 修饰                       |
| 抽象类 vs 接口   | 抽象类可有实现和状态，接口只定义形状；一个类只能继承一个抽象类但可实现多个接口     |
| 启用装饰器        | `tsconfig.json` 中设置 `"experimentalDecorators": true`        |
| 类装饰器         | 接收构造函数，可返回新类替换原类，常用于混入（Mixin）、单例等模式               |
| 方法装饰器        | 接收 target、key、descriptor，修改 descriptor.value 替换方法实现     |
| 属性装饰器        | 接收 target、key，通常配合 defineProperty 实现响应式或验证逻辑           |
| 参数装饰器        | 接收 target、key、index，通常配合方法装饰器实现参数验证                   |
| 装饰器工厂        | 返回装饰器的函数，使装饰器可接受配置参数                                   |
| 装饰器执行顺序     | 工厂从上到下求值，装饰器从下到上调用；类装饰器最后执行                        |

---

## AI 应用实战：请求日志装饰器

在 AI 应用中，对外部 API 的每次调用都需要记录详细日志，以便调试、监控和计费。下面展示如何用装饰器实现一个完整的 API 请求日志系统。

```typescript
// ============================================================
// 请求日志装饰器系统
// ============================================================

// 日志条目的数据结构
interface RequestLog {
  id: string;
  timestamp: string;
  service: string;
  method: string;
  args: unknown[];
  result?: unknown;
  error?: string;
  durationMs: number;
  success: boolean;
}

// 日志存储（实际项目中可以写入数据库或日志服务）
const requestLogs: RequestLog[] = [];

// 生成唯一请求 ID
function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

// 核心：API 日志装饰器工厂
function ApiLogger(options: {
  service?: string;
  sensitiveParams?: number[]; // 需要脱敏的参数索引
  maxLogLength?: number;
} = {}) {
  const {
    service = "unknown",
    sensitiveParams = [],
    maxLogLength = 200,
  } = options;

  return function (
    target: Object,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ): PropertyDescriptor {
    const originalMethod = descriptor.value;

    descriptor.value = async function (...args: any[]) {
      const requestId = generateRequestId();
      const startTime = Date.now();

      // 对敏感参数进行脱敏处理
      const sanitizedArgs = args.map((arg, index) => {
        if (sensitiveParams.includes(index)) {
          return typeof arg === "string"
            ? `${arg.slice(0, 4)}${"*".repeat(Math.max(0, arg.length - 4))}`
            : "***";
        }
        return arg;
      });

      console.log(`[${requestId}] 开始请求 ${service}.${propertyKey}`);
      console.log(`[${requestId}] 参数:`, JSON.stringify(sanitizedArgs).slice(0, maxLogLength));

      let result: unknown;
      let success = false;
      let errorMessage: string | undefined;

      try {
        result = await originalMethod.apply(this, args);
        success = true;

        // 对结果进行截断，避免日志过大
        const resultPreview = JSON.stringify(result).slice(0, maxLogLength);
        console.log(`[${requestId}] 成功，结果预览: ${resultPreview}`);

        return result;
      } catch (error) {
        errorMessage = (error as Error).message;
        console.error(`[${requestId}] 失败: ${errorMessage}`);
        throw error;
      } finally {
        const durationMs = Date.now() - startTime;

        // 记录结构化日志
        const log: RequestLog = {
          id: requestId,
          timestamp: new Date().toISOString(),
          service,
          method: propertyKey,
          args: sanitizedArgs,
          result: success ? result : undefined,
          error: errorMessage,
          durationMs,
          success,
        };

        requestLogs.push(log);
        console.log(`[${requestId}] 耗时: ${durationMs}ms`);
      }
    };

    return descriptor;
  };
}

// 统计报告装饰器：在类上收集所有方法的统计信息
function TrackStats<T extends { new(...args: any[]): {} }>(constructor: T) {
  return class extends constructor {
    getStats(): Record<string, { calls: number; avgDurationMs: number; errorRate: number }> {
      const stats: Record<string, { calls: number; totalMs: number; errors: number }> = {};

      for (const log of requestLogs) {
        if (log.service !== constructor.name) continue;
        if (!stats[log.method]) {
          stats[log.method] = { calls: 0, totalMs: 0, errors: 0 };
        }
        stats[log.method].calls++;
        stats[log.method].totalMs += log.durationMs;
        if (!log.success) stats[log.method].errors++;
      }

      return Object.fromEntries(
        Object.entries(stats).map(([method, data]) => [
          method,
          {
            calls: data.calls,
            avgDurationMs: Math.round(data.totalMs / data.calls),
            errorRate: data.errors / data.calls,
          },
        ])
      );
    }
  };
}

// ============================================================
// 应用装饰器的 AI 客户端
// ============================================================

@TrackStats
class OpenAIClient {
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  @ApiLogger({
    service: "OpenAIClient",
    sensitiveParams: [0], // 第一个参数（apiKey）脱敏
  })
  async createChatCompletion(
    prompt: string,
    model: string = "gpt-4o",
    maxTokens: number = 1000
  ): Promise<{ content: string; tokensUsed: number }> {
    // 模拟 API 调用
    await new Promise((resolve) => setTimeout(resolve, 150 + Math.random() * 100));

    if (Math.random() < 0.1) {
      throw new Error("Rate limit exceeded");
    }

    return {
      content: `[模拟回复] ${prompt.slice(0, 50)}...`,
      tokensUsed: Math.floor(maxTokens * 0.6),
    };
  }

  @ApiLogger({
    service: "OpenAIClient",
    maxLogLength: 100,
  })
  async createEmbedding(
    text: string
  ): Promise<{ embedding: number[]; model: string }> {
    await new Promise((resolve) => setTimeout(resolve, 50 + Math.random() * 50));
    return {
      embedding: new Array(1536).fill(0).map(() => Math.random() - 0.5),
      model: "text-embedding-3-small",
    };
  }
}

// ============================================================
// 使用示例
// ============================================================

async function demonstrateApiLogger(): Promise<void> {
  const client = new OpenAIClient("sk-xxxxxxxxxxxxxxxx");

  console.log("=".repeat(50));
  console.log("演示 API 请求日志装饰器");
  console.log("=".repeat(50));

  // 发起多次请求
  const prompts = [
    "解释量子纠缠的原理",
    "用 Python 写一个快速排序",
    "TypeScript 装饰器有哪些类型",
  ];

  for (const prompt of prompts) {
    try {
      await client.createChatCompletion(prompt);
    } catch (error) {
      console.warn("请求失败（已记录）");
    }
  }

  await client.createEmbedding("机器学习中的向量嵌入");

  // 输出统计报告
  console.log("\n" + "=".repeat(50));
  console.log("API 调用统计报告");
  console.log("=".repeat(50));
  const stats = (client as any).getStats();
  for (const [method, data] of Object.entries(stats)) {
    console.log(`\n方法: ${method}`);
    console.log(`  总调用次数: ${(data as any).calls}`);
    console.log(`  平均耗时: ${(data as any).avgDurationMs}ms`);
    console.log(`  错误率: ${((data as any).errorRate * 100).toFixed(1)}%`);
  }

  // 输出所有日志
  console.log("\n全部日志条目数:", requestLogs.length);
  console.log("最近一条日志:", JSON.stringify(requestLogs[requestLogs.length - 1], null, 2));
}

demonstrateApiLogger().catch(console.error);
```

**运行输出示例：**

```
==================================================
演示 API 请求日志装饰器
==================================================
[req_1712345678_abc1234] 开始请求 OpenAIClient.createChatCompletion
[req_1712345678_abc1234] 参数: ["解释量子纠缠的原理","gpt-4o",1000]
[req_1712345678_abc1234] 成功，结果预览: {"content":"[模拟回复] 解释量子纠缠的原理...","tokensUsed":600}
[req_1712345678_abc1234] 耗时: 213ms
...
==================================================
API 调用统计报告
==================================================

方法: createChatCompletion
  总调用次数: 3
  平均耗时: 198ms
  错误率: 0.0%

方法: createEmbedding
  总调用次数: 1
  平均耗时: 72ms
  错误率: 0.0%
```

---

## 练习题

### 基础题

**练习 8.1** 实现一个抽象类 `DataProcessor`，定义以下抽象方法：
- `parse(raw: string): unknown` - 解析原始数据
- `validate(data: unknown): boolean` - 验证数据合法性
- `transform(data: unknown): string` - 转换数据为输出格式

在基类中实现一个具体方法 `process(raw: string): string`，它按顺序调用上述三个方法，如果验证失败则抛出错误。

然后实现两个子类：
- `JsonProcessor` - 处理 JSON 数据
- `CsvProcessor` - 处理简单的单行 CSV 数据（格式如 `"a,b,c"`）

**练习 8.2** 编写一个方法装饰器 `@Deprecated(message: string)`，当被装饰的方法被调用时，在控制台输出一条警告信息（使用 `console.warn`），内容格式为：`[Deprecated] 方法名: 自定义消息`，但方法仍然正常执行。

---

### 进阶题

**练习 8.3** 实现一个属性装饰器 `@Clamp(min: number, max: number)`，使得当给属性赋值时，如果值超出 `[min, max]` 范围，自动将值夹紧到边界值（而不是抛出错误）。例如，对 `@Clamp(0, 100) score` 赋值 `150` 时，实际存储的是 `100`。

**练习 8.4** 实现一个装饰器工厂 `@Cache(ttlMs: number)`，为异步方法添加带过期时间的缓存。当在 TTL 时间内用相同参数再次调用方法时，直接返回缓存结果；超过 TTL 后重新调用原方法。

---

### 综合题

**练习 8.5** 结合本章知识，设计一个用于 AI RAG（检索增强生成）系统的基础框架：

1. 定义抽象类 `VectorStore`，包含：
   - 抽象方法 `insert(id: string, vector: number[], metadata: object): Promise<void>`
   - 抽象方法 `search(query: number[], topK: number): Promise<Array<{ id: string; score: number; metadata: object }>>`
   - 具体方法 `cosineSimilarity(a: number[], b: number[]): number`（实现余弦相似度计算）

2. 实现一个 `InMemoryVectorStore` 子类，使用内存数组存储向量数据

3. 为 `insert` 和 `search` 方法添加 `@MeasureTime` 装饰器

4. 为类添加 `@Singleton` 装饰器，确保整个应用只有一个 VectorStore 实例

---

## 练习答案

### 练习 8.1 答案

```typescript
abstract class DataProcessor {
  abstract parse(raw: string): unknown;
  abstract validate(data: unknown): boolean;
  abstract transform(data: unknown): string;

  process(raw: string): string {
    const parsed = this.parse(raw);
    if (!this.validate(parsed)) {
      throw new Error("数据验证失败");
    }
    return this.transform(parsed);
  }
}

class JsonProcessor extends DataProcessor {
  parse(raw: string): unknown {
    return JSON.parse(raw);
  }

  validate(data: unknown): boolean {
    return data !== null && typeof data === "object";
  }

  transform(data: unknown): string {
    return JSON.stringify(data, null, 2);
  }
}

class CsvProcessor extends DataProcessor {
  parse(raw: string): string[] {
    return raw.split(",").map((s) => s.trim());
  }

  validate(data: unknown): boolean {
    return Array.isArray(data) && data.length > 0;
  }

  transform(data: unknown): string {
    const arr = data as string[];
    return arr.map((item, i) => `${i + 1}. ${item}`).join("\n");
  }
}

const json = new JsonProcessor();
console.log(json.process('{"name":"张三","age":25}'));
// {
//   "name": "张三",
//   "age": 25
// }

const csv = new CsvProcessor();
console.log(csv.process("苹果, 香蕉, 橙子"));
// 1. 苹果
// 2. 香蕉
// 3. 橙子
```

### 练习 8.2 答案

```typescript
function Deprecated(message: string) {
  return function (
    target: Object,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ): PropertyDescriptor {
    const original = descriptor.value;

    descriptor.value = function (...args: any[]) {
      console.warn(`[Deprecated] ${propertyKey}: ${message}`);
      return original.apply(this, args);
    };

    return descriptor;
  };
}

class OldApiClient {
  @Deprecated("请使用 newFetch 方法替代")
  oldFetch(url: string): string {
    return `获取: ${url}`;
  }

  newFetch(url: string): string {
    return `新版获取: ${url}`;
  }
}

const client = new OldApiClient();
console.log(client.oldFetch("https://api.example.com/data"));
// [Deprecated] oldFetch: 请使用 newFetch 方法替代
// 获取: https://api.example.com/data
```

### 练习 8.3 答案

```typescript
function Clamp(min: number, max: number) {
  return function (target: Object, propertyKey: string): void {
    let value: number;

    Object.defineProperty(target, propertyKey, {
      get() {
        return value;
      },
      set(newValue: number) {
        value = Math.min(max, Math.max(min, newValue));
      },
      enumerable: true,
      configurable: true,
    });
  };
}

class PlayerStats {
  @Clamp(0, 100)
  health: number = 100;

  @Clamp(0, 9999)
  score: number = 0;
}

const player = new PlayerStats();
player.health = 150;
console.log(player.health); // 100（自动夹紧）

player.health = -50;
console.log(player.health); // 0（自动夹紧）

player.score = 10000;
console.log(player.score); // 9999（自动夹紧）
```

### 练习 8.4 答案

```typescript
function Cache(ttlMs: number) {
  return function (
    target: Object,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ): PropertyDescriptor {
    const original = descriptor.value;
    const cache = new Map<string, { value: unknown; expiresAt: number }>();

    descriptor.value = async function (...args: any[]) {
      const key = JSON.stringify(args);
      const now = Date.now();
      const cached = cache.get(key);

      if (cached && now < cached.expiresAt) {
        console.log(`[Cache] ${propertyKey} 命中缓存`);
        return cached.value;
      }

      const result = await original.apply(this, args);
      cache.set(key, { value: result, expiresAt: now + ttlMs });
      console.log(`[Cache] ${propertyKey} 已缓存，TTL: ${ttlMs}ms`);
      return result;
    };

    return descriptor;
  };
}

class SearchService {
  @Cache(5000) // 缓存 5 秒
  async searchDocuments(query: string, limit: number): Promise<string[]> {
    console.log(`[SearchService] 执行搜索: ${query}`);
    await new Promise((resolve) => setTimeout(resolve, 100));
    return [`文档1: ${query}`, `文档2: ${query}`].slice(0, limit);
  }
}

async function testCache() {
  const search = new SearchService();
  await search.searchDocuments("TypeScript", 2); // 执行搜索
  await search.searchDocuments("TypeScript", 2); // 命中缓存
  await search.searchDocuments("Python", 2);     // 执行搜索（不同参数）
}

testCache();
```

### 练习 8.5 答案

```typescript
function MeasureTime(
  target: Object,
  propertyKey: string,
  descriptor: PropertyDescriptor
): PropertyDescriptor {
  const original = descriptor.value;
  descriptor.value = async function (...args: any[]) {
    const start = performance.now();
    const result = await original.apply(this, args);
    console.log(`[${propertyKey}] 耗时: ${(performance.now() - start).toFixed(2)}ms`);
    return result;
  };
  return descriptor;
}

function Singleton<T extends { new(...args: any[]): {} }>(constructor: T) {
  let instance: InstanceType<T> | null = null;
  return new Proxy(constructor, {
    construct(target, args) {
      if (!instance) {
        instance = Reflect.construct(target, args) as InstanceType<T>;
      }
      return instance;
    },
  });
}

interface SearchResult {
  id: string;
  score: number;
  metadata: object;
}

abstract class VectorStore {
  abstract insert(id: string, vector: number[], metadata: object): Promise<void>;
  abstract search(query: number[], topK: number): Promise<SearchResult[]>;

  cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) throw new Error("向量维度不匹配");

    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] ** 2;
      normB += b[i] ** 2;
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
  }
}

@Singleton
class InMemoryVectorStore extends VectorStore {
  private store: Array<{ id: string; vector: number[]; metadata: object }> = [];

  @MeasureTime
  async insert(id: string, vector: number[], metadata: object): Promise<void> {
    this.store.push({ id, vector, metadata });
    console.log(`[VectorStore] 已插入: ${id}`);
  }

  @MeasureTime
  async search(query: number[], topK: number): Promise<SearchResult[]> {
    return this.store
      .map((item) => ({
        id: item.id,
        score: this.cosineSimilarity(query, item.vector),
        metadata: item.metadata,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
}

async function testVectorStore() {
  const store1 = new InMemoryVectorStore();
  const store2 = new InMemoryVectorStore();
  console.log("同一实例:", store1 === store2); // true

  await store1.insert("doc1", [0.1, 0.2, 0.3], { title: "TypeScript 入门" });
  await store1.insert("doc2", [0.4, 0.5, 0.6], { title: "装饰器模式" });
  await store1.insert("doc3", [0.1, 0.3, 0.2], { title: "抽象类应用" });

  const results = await store1.search([0.1, 0.25, 0.25], 2);
  console.log("搜索结果:");
  results.forEach((r) => {
    console.log(`  ${r.id} (score: ${r.score.toFixed(4)}):`, r.metadata);
  });
}

testVectorStore();
```
