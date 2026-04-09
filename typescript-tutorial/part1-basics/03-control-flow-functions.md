# 第3章：控制流与函数

> **本章定位**：掌握TypeScript中控制流与函数的完整体系，为构建AI应用的核心逻辑打下坚实基础。

---

## 学习目标

完成本章学习后，你将能够：

1. 熟练使用 `if/else`、`switch` 和三元运算符实现条件分支，并理解TypeScript的类型收窄机制
2. 掌握各类循环语句（`for`、`for...of`、`for...in`、`while`）的适用场景与性能差异
3. 区分函数声明、函数表达式和箭头函数的行为差异，正确选择使用场景
4. 为函数参数和返回值添加精确的类型注解，包括可选参数、默认参数和剩余参数
5. 理解函数重载与 `this` 类型，编写类型安全的面向对象代码

---

## 3.1 条件语句

### 3.1.1 if / else 与类型收窄

TypeScript 的 `if/else` 不只是执行分支，还能通过**类型收窄（Type Narrowing）**让编译器在不同分支中获得更精确的类型信息。

```typescript
// 基础 if/else
function describeScore(score: number): string {
  if (score >= 90) {
    return "优秀";
  } else if (score >= 75) {
    return "良好";
  } else if (score >= 60) {
    return "及格";
  } else {
    return "不及格";
  }
}

console.log(describeScore(88)); // 良好
```

**联合类型中的类型收窄**：

```typescript
// AI场景：处理不同类型的模型输出
type ModelOutput =
  | { type: "text"; content: string }
  | { type: "embedding"; vector: number[] }
  | { type: "error"; message: string; code: number };

function handleModelOutput(output: ModelOutput): void {
  if (output.type === "text") {
    // 此处 output 被收窄为 { type: "text"; content: string }
    console.log("文本输出:", output.content.trim());
  } else if (output.type === "embedding") {
    // 此处 output 被收窄为 { type: "embedding"; vector: number[] }
    const norm = Math.sqrt(output.vector.reduce((sum, v) => sum + v * v, 0));
    console.log(`嵌入向量，维度: ${output.vector.length}，模长: ${norm.toFixed(4)}`);
  } else {
    // 此处 output 被收窄为 { type: "error"; message: string; code: number }
    console.error(`模型错误 [${output.code}]: ${output.message}`);
  }
}
```

**typeof 守卫**：

```typescript
function processInput(input: string | number | boolean): string {
  if (typeof input === "string") {
    return input.toUpperCase();          // input: string
  } else if (typeof input === "number") {
    return input.toFixed(2);             // input: number
  } else {
    return input ? "true" : "false";     // input: boolean
  }
}
```

**instanceof 守卫**：

```typescript
class NetworkError extends Error {
  constructor(public statusCode: number, message: string) {
    super(message);
    this.name = "NetworkError";
  }
}

class ParseError extends Error {
  constructor(public position: number, message: string) {
    super(message);
    this.name = "ParseError";
  }
}

function handleError(err: Error): void {
  if (err instanceof NetworkError) {
    console.error(`网络错误 HTTP ${err.statusCode}: ${err.message}`);
  } else if (err instanceof ParseError) {
    console.error(`解析错误（位置 ${err.position}）: ${err.message}`);
  } else {
    console.error(`未知错误: ${err.message}`);
  }
}
```

### 3.1.2 switch 语句

`switch` 适合处理多个**离散值**的分支，比多个 `if/else if` 更清晰：

```typescript
type HttpMethod = "GET" | "POST" | "PUT" | "DELETE" | "PATCH";

function describeMethod(method: HttpMethod): string {
  switch (method) {
    case "GET":
      return "获取资源";
    case "POST":
      return "创建资源";
    case "PUT":
      return "完整替换资源";
    case "PATCH":
      return "部分更新资源";
    case "DELETE":
      return "删除资源";
    // TypeScript 会检查是否穷尽了所有 HttpMethod 的可能值
    // 如果漏了某个 case，编译时不会报错，但可以用穷尽性检查技巧来保障
  }
}
```

**穷尽性检查（Exhaustiveness Check）**：

```typescript
// 利用 never 类型做编译时穷尽检查
function assertNever(value: never): never {
  throw new Error(`未处理的值: ${JSON.stringify(value)}`);
}

type AIProvider = "openai" | "anthropic" | "google";

function getApiEndpoint(provider: AIProvider): string {
  switch (provider) {
    case "openai":
      return "https://api.openai.com/v1";
    case "anthropic":
      return "https://api.anthropic.com/v1";
    case "google":
      return "https://generativelanguage.googleapis.com/v1beta";
    default:
      // 如果 AIProvider 增加了新成员但 switch 没有处理，这里会报编译错误
      return assertNever(provider);
  }
}
```

### 3.1.3 三元运算符

三元运算符适合**简单的单行条件赋值**：

```typescript
// 基础用法
const status = (score: number) => score >= 60 ? "通过" : "未通过";

// AI场景：根据置信度决定是否采用模型预测结果
function applyPrediction(confidence: number, prediction: string): string {
  return confidence >= 0.85
    ? `采用预测: ${prediction}`
    : `置信度不足 (${(confidence * 100).toFixed(1)}%)，需要人工审核`;
}

// 嵌套三元（谨慎使用，超过两层建议改用 if/else）
const grade = (score: number): string =>
  score >= 90 ? "A" :
  score >= 75 ? "B" :
  score >= 60 ? "C" : "D";
```

> **最佳实践**：三元运算符嵌套超过两层时，可读性急剧下降，应改用 `if/else` 或 `switch`。

---

## 3.2 循环语句

### 3.2.1 for 循环

经典的索引式 `for` 循环，适合需要**精确控制索引**的场景：

```typescript
// 基础 for 循环
for (let i = 0; i < 5; i++) {
  console.log(i); // 0 1 2 3 4
}

// AI场景：批量处理训练数据，按批次（batch）划分
function processBatches<T>(data: T[], batchSize: number): T[][] {
  const batches: T[][] = [];
  for (let i = 0; i < data.length; i += batchSize) {
    batches.push(data.slice(i, i + batchSize));
  }
  return batches;
}

const samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const batches = processBatches(samples, 3);
// [[1,2,3], [4,5,6], [7,8,9], [10]]
```

### 3.2.2 for...of 循环

`for...of` 遍历**可迭代对象**（数组、字符串、Map、Set 等），语法更简洁，也是处理数组最推荐的方式：

```typescript
// 遍历数组
const temperatures = [36.5, 37.2, 38.1, 36.8];
for (const temp of temperatures) {
  console.log(temp > 37.5 ? `发烧: ${temp}°C` : `正常: ${temp}°C`);
}

// 遍历字符串
for (const char of "TypeScript") {
  process.stdout.write(char + " ");
}

// 遍历 Map
const modelCapabilities = new Map<string, string[]>([
  ["gpt-4", ["text", "vision", "function_calling"]],
  ["claude-3", ["text", "vision", "long_context"]],
]);

for (const [model, caps] of modelCapabilities) {
  console.log(`${model}: ${caps.join(", ")}`);
}

// 结合解构赋值遍历对象数组
interface TrainingSample {
  id: number;
  input: string;
  label: string;
}

const dataset: TrainingSample[] = [
  { id: 1, input: "今天天气真好", label: "positive" },
  { id: 2, input: "这个产品太差了", label: "negative" },
  { id: 3, input: "一般般吧", label: "neutral" },
];

for (const { id, input, label } of dataset) {
  console.log(`[${id}] "${input}" -> ${label}`);
}
```

### 3.2.3 for...in 循环

`for...in` 遍历对象的**可枚举属性键**，主要用于遍历普通对象：

```typescript
// 遍历对象属性
const hyperparams: Record<string, number> = {
  learningRate: 0.001,
  batchSize: 32,
  epochs: 100,
  dropout: 0.2,
};

for (const key in hyperparams) {
  console.log(`${key}: ${hyperparams[key]}`);
}

// 注意：for...in 会遍历原型链上的属性，推荐配合 hasOwnProperty 使用
for (const key in hyperparams) {
  if (Object.prototype.hasOwnProperty.call(hyperparams, key)) {
    console.log(`${key} = ${hyperparams[key]}`);
  }
}
```

> **提示**：遍历数组时优先使用 `for...of`，`for...in` 遍历数组会得到字符串类型的索引，容易出错。

### 3.2.4 while 与 do...while

适合**循环次数不确定**、依赖条件的场景：

```typescript
// while：先检查条件
function waitForCondition(maxAttempts: number): boolean {
  let attempts = 0;
  while (attempts < maxAttempts) {
    // 模拟检查某个条件是否满足
    const isReady = Math.random() > 0.7;
    if (isReady) {
      console.log(`第 ${attempts + 1} 次尝试成功`);
      return true;
    }
    attempts++;
    console.log(`第 ${attempts} 次尝试失败，继续...`);
  }
  return false;
}

// do...while：至少执行一次
function promptUser(): string {
  let input: string;
  do {
    // 模拟获取用户输入
    input = Math.random() > 0.5 ? "valid_input" : "";
    console.log(input ? `接收到输入: ${input}` : "输入为空，请重试");
  } while (!input);
  return input;
}
```

### 3.2.5 break 与 continue

```typescript
// break：提前退出循环
function findFirstPositive(numbers: number[]): number | undefined {
  for (const n of numbers) {
    if (n > 0) {
      return n; // 函数中直接 return 等效于 break
    }
  }
  return undefined;
}

// continue：跳过当前迭代
function filterAndProcess(items: string[]): string[] {
  const results: string[] = [];
  for (const item of items) {
    if (item.trim() === "") continue; // 跳过空字符串
    if (item.startsWith("#")) continue; // 跳过注释行
    results.push(item.toUpperCase());
  }
  return results;
}

// 带标签的 break（用于跳出嵌套循环）
function findPair(matrix: number[][], target: number): [number, number] | null {
  outer: for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      if (matrix[i][j] === target) {
        console.log(`找到目标 ${target} 在位置 [${i}, ${j}]`);
        break outer; // 同时跳出两层循环
      }
    }
  }
  return null;
}
```

---

## 3.3 函数声明与表达式

### 3.3.1 函数声明（Function Declaration）

```typescript
// 函数声明会被提升（hoisting），可以在声明前调用
console.log(add(1, 2)); // 3（正常工作，因为声明被提升）

function add(a: number, b: number): number {
  return a + b;
}

// AI场景：计算余弦相似度
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) {
    throw new Error(`向量维度不匹配: ${vecA.length} vs ${vecB.length}`);
  }
  const dot = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}
```

### 3.3.2 函数表达式（Function Expression）

```typescript
// 函数表达式不会被提升，必须先定义再使用
const multiply = function(a: number, b: number): number {
  return a * b;
};

// 可以赋值给变量，作为参数传递，或存入数据结构
const operations: Record<string, (a: number, b: number) => number> = {
  add: function(a, b) { return a + b; },
  subtract: function(a, b) { return a - b; },
  multiply: function(a, b) { return a * b; },
};

console.log(operations["add"](10, 5));      // 15
console.log(operations["multiply"](4, 7));   // 28
```

### 3.3.3 箭头函数（Arrow Function）

箭头函数是函数表达式的简写，有两个关键特性：**更简洁的语法**和**不绑定自己的 `this`**。

```typescript
// 语法演进
const double1 = function(x: number): number { return x * 2; };
const double2 = (x: number): number => { return x * 2; };
const double3 = (x: number): number => x * 2;  // 单行可省略 return 和花括号

// 无参数
const getTimestamp = (): number => Date.now();

// 单参数（可以省略括号，但建议保留）
const square = (x: number): number => x * x;

// 返回对象字面量时需要用括号包裹，避免与函数体的花括号混淆
const makePoint = (x: number, y: number) => ({ x, y });
console.log(makePoint(3, 4)); // { x: 3, y: 4 }

// AI场景：数组管道处理
const rawScores = [0.92, 0.45, 0.78, 0.31, 0.88];

const highConfidencePredictions = rawScores
  .filter((score) => score >= 0.75)           // 过滤低置信度
  .map((score) => Math.round(score * 100))     // 转换为百分比整数
  .sort((a, b) => b - a);                      // 降序排列

console.log(highConfidencePredictions); // [92, 88, 78]
```

### 3.3.4 三种函数形式的对比

| 特性 | 函数声明 | 函数表达式 | 箭头函数 |
|------|---------|-----------|---------|
| 提升（Hoisting） | 是 | 否 | 否 |
| `this` 绑定 | 动态绑定 | 动态绑定 | 词法绑定 |
| `arguments` 对象 | 有 | 有 | 无 |
| 可用作构造函数 | 是 | 是 | 否 |
| 适合场景 | 模块级工具函数 | 回调、赋值 | 回调、方法简写 |

```typescript
// this 绑定差异演示
class Timer {
  private count = 0;

  startWithRegular(): void {
    // 普通函数：this 是动态绑定的，在 setTimeout 回调中 this 会丢失
    setTimeout(function() {
      this.count++; // 错误！此处 this 不是 Timer 实例
    }, 1000);
  }

  startWithArrow(): void {
    // 箭头函数：this 词法绑定，始终是 Timer 实例
    setTimeout(() => {
      this.count++; // 正确！this 是 Timer 实例
      console.log(this.count);
    }, 1000);
  }
}
```

---

## 3.4 函数类型

### 3.4.1 参数类型与返回类型

```typescript
// 明确的参数类型和返回类型
function tokenize(text: string, separator: string): string[] {
  return text.split(separator).map((t) => t.trim()).filter((t) => t.length > 0);
}

// 返回 void：函数没有有意义的返回值
function logMessage(level: "info" | "warn" | "error", message: string): void {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] [${level.toUpperCase()}] ${message}`);
}

// 返回 never：函数永远不会正常返回（抛出异常或无限循环）
function throwApiError(statusCode: number, message: string): never {
  throw new Error(`API Error ${statusCode}: ${message}`);
}

// 函数类型注解
type Transformer<T, U> = (input: T, index: number) => U;
type Predicate<T> = (item: T) => boolean;
type Comparator<T> = (a: T, b: T) => number;

// 使用函数类型
function transformArray<T, U>(arr: T[], fn: Transformer<T, U>): U[] {
  return arr.map(fn);
}

const numbers = [1, 2, 3, 4, 5];
const doubled = transformArray(numbers, (n, _i) => n * 2);
console.log(doubled); // [2, 4, 6, 8, 10]
```

### 3.4.2 可选参数（Optional Parameters）

用 `?` 标记可选参数，可选参数的类型自动变为 `T | undefined`：

```typescript
// 可选参数必须放在必选参数之后
function createPrompt(
  instruction: string,
  context?: string,
  examples?: string[]
): string {
  let prompt = instruction;
  if (context) {
    prompt += `\n\n背景信息：${context}`;
  }
  if (examples && examples.length > 0) {
    prompt += `\n\n示例：\n${examples.map((e, i) => `${i + 1}. ${e}`).join("\n")}`;
  }
  return prompt;
}

console.log(createPrompt("请总结以下内容"));
console.log(createPrompt("请翻译以下内容", "将中文翻译为英文"));
console.log(createPrompt("请分类", "情感分析", ["积极", "消极", "中性"]));
```

### 3.4.3 默认参数（Default Parameters）

```typescript
// 默认参数：调用时省略或传 undefined 时使用默认值
function callChatAPI(
  prompt: string,
  model: string = "gpt-4o",
  temperature: number = 0.7,
  maxTokens: number = 1024
): object {
  return { prompt, model, temperature, maxTokens };
}

callChatAPI("你好");                          // model="gpt-4o", temp=0.7, max=1024
callChatAPI("你好", "claude-3-5-sonnet");     // temp=0.7, max=1024
callChatAPI("你好", undefined, 0.0);          // model="gpt-4o"（undefined 触发默认值）
callChatAPI("你好", "gpt-4o", 0.9, 2048);

// 默认参数可以引用前面的参数
function createEmbeddingRequest(
  text: string,
  model: string = "text-embedding-3-small",
  dimensions: number = model.includes("large") ? 3072 : 1536
): object {
  return { text, model, dimensions };
}
```

### 3.4.4 剩余参数（Rest Parameters）

```typescript
// 剩余参数收集任意数量的参数为数组，必须是最后一个参数
function mergeTexts(separator: string, ...texts: string[]): string {
  return texts.join(separator);
}

console.log(mergeTexts(" | ", "段落一", "段落二", "段落三"));
// "段落一 | 段落二 | 段落三"

// AI场景：批量发送消息到对话历史
interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

function buildConversation(
  systemPrompt: string,
  ...exchanges: [string, string][]  // [用户消息, 助手回复] 的元组数组
): Message[] {
  const messages: Message[] = [
    { role: "system", content: systemPrompt }
  ];
  for (const [userMsg, assistantMsg] of exchanges) {
    messages.push({ role: "user", content: userMsg });
    messages.push({ role: "assistant", content: assistantMsg });
  }
  return messages;
}

const history = buildConversation(
  "你是一个有帮助的助手",
  ["你好", "你好！有什么我可以帮助你的？"],
  ["TypeScript好学吗？", "TypeScript上手很快，特别是有JavaScript基础的话。"]
);
```

### 3.4.5 函数类型别名与接口

```typescript
// 类型别名定义函数类型
type ApiCallFn = (
  endpoint: string,
  options?: RequestInit
) => Promise<Response>;

// 接口定义函数类型（适合需要扩展的场景）
interface Logger {
  (message: string, level?: "info" | "warn" | "error"): void;
  prefix: string;   // 接口可以同时定义属性
}

// 泛型函数类型
type MapFn = <T, U>(arr: T[], fn: (item: T, index: number) => U) => U[];
```

---

## 3.5 函数重载与this类型

### 3.5.1 函数重载（Function Overloading）

TypeScript 的函数重载允许同一个函数名对应**多个不同的调用签名**，从而实现更精确的类型推断：

```typescript
// 重载签名（只有类型声明，没有实现）
function parseInput(input: string): string[];
function parseInput(input: number): number[];
function parseInput(input: string[]): string;
// 实现签名（必须兼容所有重载签名）
function parseInput(input: string | number | string[]): string[] | number[] | string {
  if (typeof input === "string") {
    return input.split(",").map((s) => s.trim());
  } else if (typeof input === "number") {
    return Array.from({ length: input }, (_, i) => i);
  } else {
    return input.join(", ");
  }
}

const words = parseInput("apple, banana, cherry"); // 类型: string[]
const indices = parseInput(5);                      // 类型: number[]
const joined = parseInput(["a", "b", "c"]);         // 类型: string

console.log(words);   // ["apple", "banana", "cherry"]
console.log(indices); // [0, 1, 2, 3, 4]
console.log(joined);  // "a, b, c"
```

**AI场景中的重载实例**：

```typescript
// 根据参数类型返回不同的嵌入向量处理结果
interface EmbeddingResult {
  vector: number[];
  model: string;
}

// 单文本输入返回单个结果
function embed(text: string): Promise<EmbeddingResult>;
// 批量文本输入返回结果数组
function embed(texts: string[]): Promise<EmbeddingResult[]>;
// 实现
async function embed(
  input: string | string[]
): Promise<EmbeddingResult | EmbeddingResult[]> {
  const texts = Array.isArray(input) ? input : [input];
  // 模拟 API 调用
  const results = texts.map((text) => ({
    vector: Array.from({ length: 1536 }, () => Math.random() - 0.5),
    model: "text-embedding-3-small",
  }));
  return Array.isArray(input) ? results : results[0];
}

// 调用时类型正确
const single = await embed("Hello world");   // EmbeddingResult
const batch = await embed(["Hi", "Bye"]);    // EmbeddingResult[]
```

### 3.5.2 方法重载

```typescript
class VectorStore {
  private data: Map<string, number[]> = new Map();

  // 重载：单个插入
  add(id: string, vector: number[]): void;
  // 重载：批量插入
  add(entries: Array<{ id: string; vector: number[] }>): void;
  // 实现
  add(
    idOrEntries: string | Array<{ id: string; vector: number[] }>,
    vector?: number[]
  ): void {
    if (typeof idOrEntries === "string") {
      this.data.set(idOrEntries, vector!);
    } else {
      for (const { id, vector: vec } of idOrEntries) {
        this.data.set(id, vec);
      }
    }
  }

  get size(): number {
    return this.data.size;
  }
}

const store = new VectorStore();
store.add("doc-1", [0.1, 0.2, 0.3]);
store.add([
  { id: "doc-2", vector: [0.4, 0.5, 0.6] },
  { id: "doc-3", vector: [0.7, 0.8, 0.9] },
]);
console.log(store.size); // 3
```

### 3.5.3 this 类型

TypeScript 允许将 `this` 作为**第一个参数**来显式声明，这个参数只用于类型检查，不会出现在运行时：

```typescript
interface UIElement {
  addClickListener(onClick: (this: void, e: Event) => void): void;
}

// 明确 this 不应被使用的回调
function handleClick(this: void, event: Event): void {
  // 如果尝试访问 this.anything，TypeScript 会报错
  console.log("Clicked:", event.type);
}
```

**流式链式调用中的 this 类型**：

```typescript
class QueryBuilder {
  private filters: string[] = [];
  private limitValue = 10;
  private offsetValue = 0;

  where(condition: string): this {
    this.filters.push(condition);
    return this; // 返回 this 使子类也能正确链式调用
  }

  limit(n: number): this {
    this.limitValue = n;
    return this;
  }

  offset(n: number): this {
    this.offsetValue = n;
    return this;
  }

  build(): string {
    const whereClause = this.filters.length > 0
      ? ` WHERE ${this.filters.join(" AND ")}`
      : "";
    return `SELECT * FROM documents${whereClause} LIMIT ${this.limitValue} OFFSET ${this.offsetValue}`;
  }
}

// 子类继承后链式调用仍然返回正确的子类类型
class AdvancedQueryBuilder extends QueryBuilder {
  private orderByField: string = "created_at";

  orderBy(field: string): this {
    this.orderByField = field;
    return this;
  }
}

const query = new AdvancedQueryBuilder()
  .where("score > 0.8")
  .where("language = 'zh'")
  .orderBy("relevance")
  .limit(20)
  .offset(40)
  .build();

console.log(query);
// SELECT * FROM documents WHERE score > 0.8 AND language = 'zh' LIMIT 20 OFFSET 40
```

### 3.5.4 泛型函数

泛型让函数在保持类型安全的前提下处理多种类型：

```typescript
// 泛型函数基础
function identity<T>(value: T): T {
  return value;
}

const str = identity("hello");   // T 推断为 string
const num = identity(42);        // T 推断为 number

// 多个泛型参数
function zip<T, U>(arr1: T[], arr2: U[]): [T, U][] {
  const len = Math.min(arr1.length, arr2.length);
  return Array.from({ length: len }, (_, i) => [arr1[i], arr2[i]]);
}

const pairs = zip([1, 2, 3], ["a", "b", "c"]);
// [[1, "a"], [2, "b"], [3, "c"]]

// 泛型约束
interface HasLength {
  length: number;
}

function longest<T extends HasLength>(a: T, b: T): T {
  return a.length >= b.length ? a : b;
}

longest("hello", "hi");         // "hello"
longest([1, 2, 3], [1, 2]);    // [1, 2, 3]
```

---

## 本章小结

| 概念 | 关键语法 | 核心要点 |
|------|---------|---------|
| if/else | `if (cond) {} else {}` | 支持类型收窄，配合 `typeof`/`instanceof` 使用 |
| switch | `switch (val) { case x: }` | 适合离散值，可用 `assertNever` 做穷尽检查 |
| 三元运算符 | `cond ? a : b` | 简洁赋值，避免超过两层嵌套 |
| for | `for (let i=0; i<n; i++)` | 需精确控制索引时使用 |
| for...of | `for (const x of arr)` | 遍历数组/可迭代对象的首选方式 |
| for...in | `for (const k in obj)` | 遍历对象属性键，注意原型链 |
| while | `while (cond) {}` | 循环次数不确定时使用 |
| 函数声明 | `function f() {}` | 会提升，适合模块级工具函数 |
| 箭头函数 | `(x) => expr` | 词法 `this`，适合回调和方法简写 |
| 可选参数 | `f(a?: T)` | 类型变为 `T \| undefined`，放在必选参数之后 |
| 默认参数 | `f(a = val)` | 传 `undefined` 时也会使用默认值 |
| 剩余参数 | `f(...args: T[])` | 收集多余参数为数组，必须是最后一个 |
| 函数重载 | 多个签名 + 一个实现 | 实现签名必须兼容所有重载签名 |
| this 类型 | `f(this: T, ...)` | 仅用于类型检查，不影响运行时 |

---

## AI应用实战：重试逻辑与错误处理

在调用外部 AI API 时，网络抖动、频率限制（Rate Limit）和临时服务错误是家常便饭。一个健壮的**指数退避重试（Exponential Backoff Retry）**函数是每个 AI 应用的必备工具。

本节将综合运用本章所学的所有知识点，实现一个完整的、类型安全的重试工具。

```typescript
// ============================================================
// 类型定义
// ============================================================

/** 重试配置选项 */
interface RetryOptions {
  /** 最大重试次数（不含首次调用），默认 3 */
  maxRetries?: number;
  /** 初始等待时间（毫秒），默认 1000 */
  initialDelayMs?: number;
  /** 退避倍数，默认 2（每次翻倍） */
  backoffMultiplier?: number;
  /** 最大等待时间上限（毫秒），默认 30000 */
  maxDelayMs?: number;
  /** 是否在等待时间上添加随机抖动，默认 true */
  jitter?: boolean;
  /** 判断是否应该重试的函数，默认对所有错误重试 */
  shouldRetry?: (error: Error, attemptNumber: number) => boolean;
  /** 每次重试前的回调（用于日志记录） */
  onRetry?: (error: Error, attemptNumber: number, delayMs: number) => void;
}

/** 重试执行结果 */
type RetryResult<T> =
  | { success: true; value: T; attempts: number }
  | { success: false; error: Error; attempts: number };

/** 可重试的异步函数类型 */
type AsyncFn<T> = () => Promise<T>;

// ============================================================
// 工具函数
// ============================================================

/**
 * 等待指定毫秒数
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * 计算第 n 次重试的等待时间（带抖动的指数退避）
 */
function calculateDelay(
  attempt: number,
  initialDelayMs: number,
  backoffMultiplier: number,
  maxDelayMs: number,
  jitter: boolean
): number {
  // 指数退避：delay = initialDelay * multiplier^(attempt-1)
  const exponentialDelay = initialDelayMs * Math.pow(backoffMultiplier, attempt - 1);
  const cappedDelay = Math.min(exponentialDelay, maxDelayMs);

  if (!jitter) {
    return cappedDelay;
  }

  // 添加 ±25% 的随机抖动，避免多个请求同时重试（惊群效应）
  const jitterRange = cappedDelay * 0.25;
  const jitterOffset = (Math.random() * 2 - 1) * jitterRange;
  return Math.round(cappedDelay + jitterOffset);
}

/**
 * 判断 HTTP 错误是否值得重试
 * 5xx 服务端错误通常值得重试，4xx 客户端错误通常不应重试
 */
function isRetryableHttpError(error: Error): boolean {
  const retryableCodes = new Set([408, 429, 500, 502, 503, 504]);
  const match = error.message.match(/HTTP (\d{3})/);
  if (match) {
    return retryableCodes.has(parseInt(match[1], 10));
  }
  // 网络错误（无状态码）也值得重试
  return error.name === "TypeError" || error.message.includes("network");
}

// ============================================================
// 核心重试函数（函数重载）
// ============================================================

/**
 * 带指数退避的异步重试（简化版：使用默认选项）
 */
async function withRetry<T>(fn: AsyncFn<T>): Promise<T>;

/**
 * 带指数退避的异步重试（完整版：自定义选项）
 */
async function withRetry<T>(fn: AsyncFn<T>, options: RetryOptions): Promise<T>;

/**
 * 带指数退避的异步重试（结果版：返回包含元数据的结果对象，不抛出异常）
 */
async function withRetry<T>(
  fn: AsyncFn<T>,
  options: RetryOptions,
  returnResult: true
): Promise<RetryResult<T>>;

// 实现签名
async function withRetry<T>(
  fn: AsyncFn<T>,
  options: RetryOptions = {},
  returnResult?: boolean
): Promise<T | RetryResult<T>> {
  const {
    maxRetries = 3,
    initialDelayMs = 1000,
    backoffMultiplier = 2,
    maxDelayMs = 30_000,
    jitter = true,
    shouldRetry = isRetryableHttpError,
    onRetry,
  } = options;

  let lastError: Error = new Error("未知错误");

  for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
    try {
      const value = await fn();

      if (returnResult) {
        return { success: true, value, attempts: attempt };
      }
      return value;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));

      const isLastAttempt = attempt === maxRetries + 1;

      // 最后一次尝试失败，或者判断不应重试
      if (isLastAttempt || !shouldRetry(lastError, attempt)) {
        break;
      }

      const delayMs = calculateDelay(
        attempt,
        initialDelayMs,
        backoffMultiplier,
        maxDelayMs,
        jitter
      );

      // 触发重试回调（通常用于日志）
      onRetry?.(lastError, attempt, delayMs);

      await sleep(delayMs);
    }
  }

  if (returnResult) {
    return { success: false, error: lastError, attempts: maxRetries + 1 };
  }
  throw lastError;
}

// ============================================================
// 实际使用示例
// ============================================================

// 模拟一个不稳定的 API 调用
let callCount = 0;
async function unstableApiCall(): Promise<{ data: string }> {
  callCount++;
  if (callCount < 3) {
    throw new Error("HTTP 503: Service Temporarily Unavailable");
  }
  return { data: "AI模型响应内容" };
}

// 示例1：使用默认选项重试
async function example1(): Promise<void> {
  try {
    const result = await withRetry(unstableApiCall);
    console.log("成功:", result.data);
  } catch (error) {
    console.error("所有重试均失败:", error);
  }
}

// 示例2：自定义重试配置，附带日志
async function example2(): Promise<void> {
  const result = await withRetry(
    unstableApiCall,
    {
      maxRetries: 5,
      initialDelayMs: 500,
      backoffMultiplier: 1.5,
      maxDelayMs: 10_000,
      jitter: true,
      shouldRetry: (error, attempt) => {
        // 自定义：超过3次不再重试 429 错误
        if (error.message.includes("429") && attempt > 3) return false;
        return isRetryableHttpError(error);
      },
      onRetry: (error, attempt, delayMs) => {
        console.warn(
          `[重试] 第 ${attempt} 次失败: ${error.message}，` +
          `等待 ${(delayMs / 1000).toFixed(1)}s 后重试...`
        );
      },
    },
    true  // 返回结果对象而非抛出异常
  );

  if (result.success) {
    console.log(`成功（共尝试 ${result.attempts} 次）:`, result.value.data);
  } else {
    console.error(`最终失败（共尝试 ${result.attempts} 次）:`, result.error.message);
  }
}

// 示例3：包装 OpenAI API 调用
interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
}

interface ChatCompletionResponse {
  id: string;
  choices: Array<{
    message: ChatMessage;
    finish_reason: string;
  }>;
}

// 用 withRetry 包装 API 客户端方法
async function chat(
  messages: ChatMessage[],
  model = "gpt-4o"
): Promise<string> {
  const response = await withRetry(
    async () => {
      // 实际项目中替换为真实的 fetch 调用
      const res = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
        },
        body: JSON.stringify({
          model,
          messages,
          temperature: 0.7,
        } satisfies ChatCompletionRequest),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      return res.json() as Promise<ChatCompletionResponse>;
    },
    {
      maxRetries: 3,
      initialDelayMs: 2000,
      backoffMultiplier: 2,
      onRetry: (err, attempt, delay) =>
        console.warn(`API 调用失败，第 ${attempt} 次重试（${delay}ms 后）: ${err.message}`),
    }
  );

  return response.choices[0].message.content;
}
```

**关键设计决策解析**：

| 设计点 | 实现方式 | 理由 |
|--------|---------|------|
| 函数重载 | 3个重载签名 | 根据是否传 `returnResult` 返回不同类型 |
| 指数退避 | `delay = init * multiplier^(n-1)` | 避免在服务过载时雪上加霜 |
| 随机抖动 | ±25% 随机偏移 | 防止多个客户端同时重试（惊群效应） |
| shouldRetry 回调 | 默认只重试可恢复错误 | 4xx 错误重试无意义，浪费配额 |
| RetryResult 联合类型 | `success: true/false` 区分 | 调用方可以无异常地处理失败情况 |

---

## 练习题

### 基础题

**题目 1**：编写一个函数 `classifyNumber`，接受一个数字参数，返回以下字符串之一：
- `"negative"`（负数）
- `"zero"`（零）
- `"small positive"`（1-100）
- `"large positive"`（大于100）

要求：使用 `if/else` 实现，为参数和返回值添加正确的类型注解。

---

**题目 2**：实现一个 `countWords` 函数，接受一段文本（`string`），返回每个单词出现次数的 `Map<string, number>`。要求：
- 忽略大小写（统一转为小写）
- 忽略标点符号（使用 `.replace(/[^\w\s]/g, "")` 去除）
- 使用 `for...of` 遍历单词列表

---

### 进阶题

**题目 3**：实现一个泛型函数 `pipe`，接受一个初始值和一组转换函数，将初始值依次传入每个函数，返回最终结果。

```typescript
// 期望用法示例
const result = pipe(
  "  Hello, World!  ",
  (s: string) => s.trim(),
  (s: string) => s.toLowerCase(),
  (s: string) => s.replace(/,/g, ""),
);
// result 应为 "hello world!"

// 所有函数的输入输出类型必须一致
```

要求：使用剩余参数和泛型，确保类型安全。

---

**题目 4**：为以下 `formatOutput` 函数编写函数重载签名：

```typescript
// 该函数有以下三种调用方式：
// 1. formatOutput(42)           -> 返回字符串 "Number: 42"
// 2. formatOutput(true)         -> 返回字符串 "Boolean: true"
// 3. formatOutput(42, true)     -> 返回对象 { value: 42, formatted: "Number: 42" }
```

要求：实现函数重载使调用时类型推断正确。

---

### 挑战题

**题目 5**：基于本章实战部分的 `withRetry` 函数，实现一个**并发限制重试器**`withConcurrentRetry`：

```typescript
// 接受一个任务列表（异步函数数组）和并发限制数
// 同时最多运行 concurrency 个任务
// 每个任务独立重试（最多 maxRetries 次）
// 所有任务完成后，返回结果数组（成功的返回值，失败的返回 Error 对象）

async function withConcurrentRetry<T>(
  tasks: Array<() => Promise<T>>,
  concurrency: number,
  maxRetries?: number
): Promise<Array<T | Error>>
```

要求：
1. 使用 `while` 循环和计数器控制并发
2. 失败的任务不应影响其他任务的执行
3. 保持结果顺序与输入任务顺序一致

---

## 练习答案

### 题目 1 答案

```typescript
function classifyNumber(n: number): "negative" | "zero" | "small positive" | "large positive" {
  if (n < 0) {
    return "negative";
  } else if (n === 0) {
    return "zero";
  } else if (n <= 100) {
    return "small positive";
  } else {
    return "large positive";
  }
}

console.log(classifyNumber(-5));   // "negative"
console.log(classifyNumber(0));    // "zero"
console.log(classifyNumber(42));   // "small positive"
console.log(classifyNumber(200));  // "large positive"
```

### 题目 2 答案

```typescript
function countWords(text: string): Map<string, number> {
  const cleaned = text.replace(/[^\w\s]/g, "").toLowerCase();
  const words = cleaned.split(/\s+/).filter((w) => w.length > 0);
  const counts = new Map<string, number>();

  for (const word of words) {
    const current = counts.get(word) ?? 0;
    counts.set(word, current + 1);
  }

  return counts;
}

const result = countWords("To be, or not to be, that is the question.");
console.log(result.get("be"));   // 2
console.log(result.get("to"));   // 2
console.log(result.get("that")); // 1
```

### 题目 3 答案

```typescript
// 使用剩余参数和泛型约束，确保函数链类型一致
function pipe<T>(value: T, ...fns: Array<(input: T) => T>): T {
  let result = value;
  for (const fn of fns) {
    result = fn(result);
  }
  return result;
}

const result = pipe(
  "  Hello, World!  ",
  (s: string) => s.trim(),
  (s: string) => s.toLowerCase(),
  (s: string) => s.replace(/,/g, ""),
);

console.log(result); // "hello world!"

// 数字管道
const numResult = pipe(
  1,
  (n: number) => n + 1,   // 2
  (n: number) => n * 3,   // 6
  (n: number) => n - 1    // 5
);
console.log(numResult); // 5
```

### 题目 4 答案

```typescript
// 重载签名
function formatOutput(value: number): string;
function formatOutput(value: boolean): string;
function formatOutput(value: number, withMeta: true): { value: number; formatted: string };
// 实现签名
function formatOutput(
  value: number | boolean,
  withMeta?: true
): string | { value: number; formatted: string } {
  if (typeof value === "boolean") {
    return `Boolean: ${value}`;
  }
  const formatted = `Number: ${value}`;
  if (withMeta) {
    return { value, formatted };
  }
  return formatted;
}

const s1 = formatOutput(42);       // 类型: string → "Number: 42"
const s2 = formatOutput(true);     // 类型: string → "Boolean: true"
const s3 = formatOutput(42, true); // 类型: { value: number; formatted: string }

console.log(s1); // "Number: 42"
console.log(s2); // "Boolean: true"
console.log(s3); // { value: 42, formatted: "Number: 42" }
```

### 题目 5 答案

```typescript
async function withConcurrentRetry<T>(
  tasks: Array<() => Promise<T>>,
  concurrency: number,
  maxRetries = 3
): Promise<Array<T | Error>> {
  const results: Array<T | Error> = new Array(tasks.length);
  let nextIndex = 0;
  let activeCount = 0;

  return new Promise((resolve) => {
    // 检查是否所有任务都已完成
    function checkDone(): void {
      if (nextIndex >= tasks.length && activeCount === 0) {
        resolve(results);
      }
    }

    // 启动一个任务槽
    function startNext(): void {
      if (nextIndex >= tasks.length) {
        checkDone();
        return;
      }

      const taskIndex = nextIndex++;
      const task = tasks[taskIndex];
      activeCount++;

      // 为每个任务独立重试
      withRetry(task, { maxRetries })
        .then((value) => {
          results[taskIndex] = value;
        })
        .catch((error: unknown) => {
          results[taskIndex] = error instanceof Error ? error : new Error(String(error));
        })
        .finally(() => {
          activeCount--;
          startNext(); // 任务完成后启动下一个
        });
    }

    // 初始启动 min(concurrency, tasks.length) 个任务
    const initialBatch = Math.min(concurrency, tasks.length);
    for (let i = 0; i < initialBatch; i++) {
      startNext();
    }

    // 处理空任务列表
    if (tasks.length === 0) {
      resolve(results);
    }
  });
}

// 使用示例
async function testConcurrentRetry(): Promise<void> {
  const tasks = Array.from({ length: 8 }, (_, i) => async () => {
    // 模拟部分任务失败
    if (i % 3 === 0 && Math.random() > 0.3) {
      throw new Error(`HTTP 503: Task ${i} failed`);
    }
    await sleep(Math.random() * 500);
    return `Task ${i} result`;
  });

  console.log("开始执行8个任务，并发限制3个...");
  const results = await withConcurrentRetry(tasks, 3, 2);

  results.forEach((result, i) => {
    if (result instanceof Error) {
      console.log(`任务 ${i}: 失败 - ${result.message}`);
    } else {
      console.log(`任务 ${i}: 成功 - ${result}`);
    }
  });
}

testConcurrentRetry();
```

---

> **下一章预告**：第4章将深入 TypeScript 的类型系统核心——**接口与类型别名**，学习如何设计 AI 应用中复杂数据结构的类型契约，以及联合类型、交叉类型、映射类型的高级用法。
