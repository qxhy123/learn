# 前言：如何使用本教程

## 教程设计理念

TypeScript已成为现代Web开发和AI应用开发的首选语言。本教程的设计基于以下理念：

1. **类型思维优先**：从一开始就建立正确的类型思维方式
2. **实战驱动学习**：每个概念都配有AI应用场景的代码示例
3. **渐进式深入**：从基础到高级，逐步构建完整的知识体系
4. **现代化工具链**：使用最新的构建工具和开发实践
5. **中文优先**：专为中文读者设计，术语准确，表达清晰

## 为什么选择TypeScript？

### 相比JavaScript的优势

```typescript
// JavaScript: 运行时才发现错误
function greet(name) {
  return "Hello, " + name.toUpperCase();
}
greet(123); // 运行时错误：name.toUpperCase is not a function

// TypeScript: 编译时就能发现错误
function greet(name: string): string {
  return "Hello, " + name.toUpperCase();
}
greet(123); // 编译错误：Argument of type 'number' is not assignable to parameter of type 'string'
```

### 在AI应用开发中的价值

```typescript
// 类型安全的API响应处理
interface ChatCompletionResponse {
  id: string;
  choices: {
    message: {
      role: "assistant";
      content: string;
    };
    finish_reason: "stop" | "length" | "tool_calls";
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
  };
}

// 编辑器自动补全，减少查文档的时间
const response: ChatCompletionResponse = await callLLM(prompt);
console.log(response.choices[0].message.content); // 完整的类型提示
```

## 章节结构

每章采用统一的结构，便于系统学习：

### 1. 学习目标
每章开头列出5个学习目标，帮助读者明确本章重点。

### 2. 正文内容
每章包含5个小节，循序渐进地展开主题：
- 概念引入与语法详解
- 类型系统相关知识
- 常见模式与最佳实践
- 进阶技巧与注意事项
- 综合案例演示

### 3. 本章小结
以表格或列表形式总结核心概念与语法。

### 4. AI应用实战
每章的特色部分，包含：
- 概念在AI应用开发中的应用场景
- 完整的代码示例（可直接运行）
- 实际项目中的使用建议

### 5. 练习题
每章5道练习题，分三个难度级别：
- 基础题（2道）：检验概念理解
- 中级题（2道）：应用与实践
- 提高题（1道）：综合项目或架构设计

### 6. 练习答案
详细的解答过程，帮助自学者检验学习效果。

## 代码约定

本教程使用以下代码约定：

```typescript
// 类型注解使用明确的类型
const apiKey: string = process.env.API_KEY ?? "";
const maxTokens: number = 1000;

// 接口使用大驼峰命名，前缀I可选
interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

// 类型别名用于复杂类型
type MessageRole = "user" | "assistant" | "system";

// 枚举使用大驼峰
enum HttpStatus {
  OK = 200,
  BadRequest = 400,
  Unauthorized = 401,
}

// 常量使用UPPER_SNAKE_CASE
const DEFAULT_MODEL = "gpt-4";
const MAX_RETRIES = 3;
```

## 输出格式说明

代码示例的输出使用以下格式：

```typescript
console.log("Hello, TypeScript!");
// 输出: Hello, TypeScript!

const sum = (a: number, b: number): number => a + b;
console.log(sum(1, 2));
// 输出: 3
```

类型错误示例：

```typescript
const name: string = 123;
// ❌ 错误: Type 'number' is not assignable to type 'string'.

const name: string = "Alice";
// ✅ 正确
```

## 环境准备

### 推荐开发环境

1. **Node.js**：18.0或更高版本（推荐使用LTS版本）
2. **包管理器**：pnpm（推荐）、npm或yarn
3. **编辑器**：VS Code（强烈推荐，TypeScript支持最佳）
4. **VS Code插件**：
   - TypeScript Importer
   - Error Lens
   - Pretty TypeScript Errors

### 快速搭建开发环境

```bash
# 1. 安装Node.js（使用nvm管理版本）
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
nvm use 20

# 2. 安装pnpm
npm install -g pnpm

# 3. 全局安装TypeScript
pnpm add -g typescript ts-node

# 4. 创建项目
mkdir ts-ai-demo && cd ts-ai-demo
pnpm init
pnpm add typescript @types/node -D
npx tsc --init

# 5. 配置tsconfig.json（推荐配置）
cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "declaration": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
EOF

# 6. 创建源码目录
mkdir src
echo 'console.log("Hello, TypeScript!")' > src/index.ts

# 7. 运行
npx ts-node src/index.ts
```

## 学习建议

### 对于JavaScript开发者

1. **拥抱类型**：不要把类型当作负担，而是把它当作文档和安全网
2. **善用推断**：TypeScript的类型推断很强大，不需要处处标注类型
3. **渐进迁移**：可以从`any`开始，逐步收紧类型
4. **阅读错误**：TypeScript的错误信息很详细，学会阅读它们

### 对于想开发AI应用的学习者

1. **重点关注第16-21章**：这些章节直接涉及AI应用开发
2. **理解异步编程**：LLM API都是异步的，必须掌握async/await
3. **学习流式处理**：SSE是现代AI应用的标配
4. **实践Function Calling**：这是构建AI Agent的核心能力

### 对于团队技术负责人

1. **关注工程化章节**：第13-15章帮助建立团队规范
2. **统一类型定义**：第12章教你如何管理项目类型
3. **建立代码规范**：结合ESLint和Prettier使用TypeScript

## 与其他教程的关系

本教程是"从零到高阶"系列的一部分：

```
TypeScript教程 ────→ AI应用开发
       ↑
JavaScript基础（前置知识）
```

TypeScript教程是AI应用开发的基础。如果你对Python版本的深度学习感兴趣，可以参考本系列的Python教程。

## 排版说明

- **粗体**：重要概念、关键词
- *斜体*：术语首次出现、强调
- `代码字体`：代码、类型名、变量名
- > 引用块：重要提示、注意事项

## 常见问题

**Q: 学习本教程需要多长时间？**
A: 因人而异。有JavaScript基础的学习者完整学习约需1-2个月，重点学习AI应用部分约需2-3周。

**Q: 遇到问题怎么办？**
A: 首先仔细阅读错误信息，然后查阅TypeScript官方文档，也可以在项目Issue中提问。

**Q: 需要先学JavaScript吗？**
A: 强烈建议先掌握JavaScript基础。TypeScript是JavaScript的超集，理解JavaScript有助于理解TypeScript。

**Q: 可以直接学习AI应用章节吗？**
A: 如果已有TypeScript基础，可以直接从第16章开始。否则建议先完成基础章节。

---

*准备好了吗？让我们开始TypeScript之旅！*

[下一章：环境搭建与基本语法](./part1-basics/01-environment-syntax.md)
