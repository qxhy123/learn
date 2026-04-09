# 第13章：tsconfig配置详解

> **适用读者**：已完成第12章学习，希望深入掌握 TypeScript 项目配置、构建优化与多环境管理能力的开发者。

---

## 学习目标

完成本章学习后，你将能够：

1. 理解 `tsconfig.json` 的完整结构，区分 `compilerOptions`、`include`、`exclude`、`files` 各字段的作用
2. 掌握 `target`、`module`、`strict` 等核心编译选项的含义，能够根据目标运行环境选择合适的配置
3. 使用 `paths` 与 `baseUrl` 配置路径别名，消除项目中的相对路径地狱
4. 理解项目引用（`references` + `composite`）机制，能够将大型 monorepo 拆分为相互依赖的子项目
5. 掌握开发、测试、生产三套环境的配置模板，并能为 AI 应用项目设计合理的多环境配置体系

---

## 13.1 tsconfig.json 结构概览

### 13.1.1 tsconfig.json 的作用

`tsconfig.json` 是 TypeScript 项目的"总控中心"。当你运行 `tsc`、`ts-node`、`eslint`、Vite 或 Next.js 时，它们都会读取这个文件来决定：

- **编译哪些文件**（`include` / `exclude` / `files`）
- **如何编译**（`compilerOptions`）
- **输出到哪里**（`outDir` / `rootDir`）
- **如何与其他子项目协作**（`references`）

```bash
# 在项目根目录初始化一个 tsconfig.json
npx tsc --init

# 使用特定配置文件编译
tsc --project tsconfig.prod.json

# 仅做类型检查，不输出文件
tsc --noEmit
```

### 13.1.2 顶层字段总览

一个完整的 `tsconfig.json` 包含以下顶层字段：

```json
{
  "compilerOptions": {
    // 编译器行为的核心开关，选项最多，见 13.2 节
  },
  "include": [
    // 明确包含哪些文件（支持 glob）
    "src/**/*",
    "types/**/*"
  ],
  "exclude": [
    // 排除哪些文件（默认已排除 node_modules、outDir）
    "node_modules",
    "dist",
    "**/*.test.ts"
  ],
  "files": [
    // 精确列举要包含的文件（与 include 互补，优先级更高）
    "src/main.ts",
    "src/global.d.ts"
  ],
  "references": [
    // 项目引用：指向其他子项目的 tsconfig，见 13.4 节
    { "path": "./packages/core" }
  ],
  "extends": "../tsconfig.base.json"
  // 继承另一个配置文件，见 13.5 节
}
```

### 13.1.3 include、exclude、files 的优先级

```
files 精确列举  >  include + exclude 模式匹配
```

具体规则如下：

| 字段 | 说明 | 默认行为 |
|------|------|----------|
| `files` | 精确列举文件，不支持 glob | 未指定时忽略 |
| `include` | glob 模式白名单 | 未指定时包含根目录下所有 `.ts`/`.tsx`/`.d.ts` |
| `exclude` | glob 模式黑名单，从 `include` 结果中排除 | 默认排除 `node_modules`、`outDir`、`jspm_packages` |

```json
{
  "include": ["src/**/*", "tests/**/*"],
  "exclude": [
    "tests/fixtures/**",   // 排除测试夹具目录
    "**/*.spec.ts",        // 排除所有 spec 文件
    "src/legacy/**"        // 排除遗留代码目录
  ]
}
```

> **注意**：`exclude` 只能排除被 `include` 纳入的文件。如果某个文件被 `files` 明确列举，`exclude` 无法将其排除。

### 13.1.4 extends 继承机制

`extends` 字段允许一个配置文件继承另一个的选项，是管理多环境配置的核心工具：

```json
// tsconfig.base.json（基础配置）
{
  "compilerOptions": {
    "target": "ES2022",
    "strict": true,
    "esModuleInterop": true
  }
}
```

```json
// tsconfig.json（开发配置，继承基础配置）
{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    "sourceMap": true,     // 覆盖或追加选项
    "noEmit": true
  },
  "include": ["src/**/*"]
}
```

继承时的合并规则：
- `compilerOptions` 中的标量字段（字符串、布尔值）：子配置**覆盖**父配置
- `compilerOptions` 中的数组字段（`lib`、`types`）：子配置**完整替换**父配置（不合并）
- `include`、`exclude`、`files`：子配置**完整替换**父配置

---

## 13.2 compilerOptions 核心选项

### 13.2.1 target：编译目标版本

`target` 控制 TypeScript 将代码编译到哪个 ECMAScript 版本。它同时影响：
- 语法降级（如 `async/await` 是否转换为 Promise 链）
- 内置类型（如 `ES2022` 会包含 `Array.prototype.at()` 的类型定义）

```json
{
  "compilerOptions": {
    "target": "ES2022"
  }
}
```

常见目标的选择指南：

| target | 适用场景 | 说明 |
|--------|----------|------|
| `ES5` | 需要兼容 IE11 的老旧项目 | 所有现代语法全部降级，输出冗长 |
| `ES2017` | Node.js 8+ / 大多数现代浏览器 | 支持原生 async/await |
| `ES2020` | Node.js 12+ / 现代浏览器 | 支持可选链 `?.`、空值合并 `??` |
| `ES2022` | Node.js 16+ / 2022+ 浏览器 | 支持顶层 await、类字段 |
| `ESNext` | 始终追踪最新特性 | 慎用，可能包含实验性特性 |

```typescript
// 源代码（TypeScript）
class AIClient {
  #apiKey: string;          // 私有字段（ES2022+）

  constructor(key: string) {
    this.#apiKey = key;
  }

  async chat(prompt: string) {
    return await fetch("/api/chat", {
      body: JSON.stringify({ prompt }),
    });
  }
}
```

```javascript
// target: "ES2022" 输出 —— 基本保持原样
class AIClient {
  #apiKey;
  constructor(key) { this.#apiKey = key; }
  async chat(prompt) { ... }
}

// target: "ES5" 输出 —— 大量降级代码
var AIClient = /** @class */ (function () {
  function AIClient(key) { this._apiKey = key; }
  AIClient.prototype.chat = function (prompt) {
    return __awaiter(this, void 0, void 0, function () { ... });
  };
  return AIClient;
}());
```

### 13.2.2 module：模块系统

`module` 控制编译后的模块格式，与 `target` 独立配置：

```json
{
  "compilerOptions": {
    "module": "NodeNext"
  }
}
```

| module | 适用场景 |
|--------|----------|
| `CommonJS` | Node.js 传统项目（`require` / `module.exports`） |
| `ESNext` / `ES2022` | 现代前端（Vite / Webpack），保留 `import`/`export` |
| `NodeNext` | Node.js 16+ 原生 ESM（`.mjs` / `package.json "type":"module"`） |
| `None` | 无模块系统（全局脚本） |

`module` 与 `moduleResolution` 的关系：

```json
// Node.js 原生 ESM 项目
{
  "compilerOptions": {
    "module": "NodeNext",
    "moduleResolution": "NodeNext"
  }
}

// 前端打包工具项目（Vite / Webpack）
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "bundler"
  }
}

// 传统 Node.js 项目
{
  "compilerOptions": {
    "module": "CommonJS",
    "moduleResolution": "node"
  }
}
```

### 13.2.3 strict 系列选项

`"strict": true` 是一个快捷开关，同时启用以下所有严格检查：

```json
{
  "compilerOptions": {
    "strict": true
    // 等价于同时开启以下所有选项：
    // "strictNullChecks": true,
    // "strictFunctionTypes": true,
    // "strictBindCallApply": true,
    // "strictPropertyInitialization": true,
    // "noImplicitAny": true,
    // "noImplicitThis": true,
    // "alwaysStrict": true,
    // "useUnknownInCatchVariables": true  （TypeScript 4.4+）
  }
}
```

最常影响代码的两项：

**strictNullChecks**：禁止将 `null`/`undefined` 赋值给非空类型

```typescript
// strictNullChecks: false（宽松模式，不推荐）
let model: string = null;          // 允许（潜在运行时错误）
let tokens: number = undefined;    // 允许

// strictNullChecks: true（严格模式，推荐）
let model: string = null;          // 错误：Type 'null' is not assignable to type 'string'
let model: string | null = null;   // 正确：显式声明可空
let tokens: number | undefined;    // 正确：显式声明可选

// 实际影响：AI API 响应处理
interface ModelResponse {
  content: string | null;   // 必须显式处理 null
  stopReason: string;
}

function processResponse(res: ModelResponse): string {
  // 不处理 null 会报错：
  // return res.content.trim(); // 错误：res.content 可能为 null

  // 必须先检查
  return res.content?.trim() ?? "(空响应)";
}
```

**noImplicitAny**：禁止隐式推断 `any` 类型

```typescript
// noImplicitAny: false
function buildPrompt(messages) {    // messages 隐式为 any，不报错
  return messages.map(m => m.content).join("\n");
}

// noImplicitAny: true
function buildPrompt(messages: Array<{ role: string; content: string }>) {
  // 必须显式标注参数类型
  return messages.map(m => m.content).join("\n");
}
```

### 13.2.4 输入/输出相关选项

```json
{
  "compilerOptions": {
    // 根目录：TypeScript 源文件的根
    "rootDir": "src",

    // 输出目录：编译后 JS 文件的目标位置
    "outDir": "dist",

    // 源映射：将运行时错误栈映射回 TypeScript 源码行号
    "sourceMap": true,

    // 内联源映射：将 source map 嵌入 JS 文件（无需单独 .map 文件）
    "inlineSourceMap": false,

    // 内联源代码：将 TypeScript 源码内嵌到 source map 中（调试最方便）
    "inlineSources": false,

    // 仅类型检查，不输出 JS 文件（配合打包工具使用）
    "noEmit": true,

    // 声明文件：生成 .d.ts 类型声明文件（发布 npm 包时必须）
    "declaration": true,

    // 声明文件目录（默认与 outDir 相同）
    "declarationDir": "types",

    // 为声明文件生成 source map
    "declarationMap": true
  }
}
```

典型场景配置：

```json
// 场景一：应用项目（配合 Vite / Next.js）
{
  "compilerOptions": {
    "noEmit": true,           // 让打包工具处理转译，tsc 只做类型检查
    "sourceMap": false        // 打包工具会自己生成 source map
  }
}

// 场景二：发布 npm 包
{
  "compilerOptions": {
    "declaration": true,      // 生成 .d.ts，让用户享受类型提示
    "declarationMap": true,   // 生成 .d.ts.map，支持"跳转到源码"
    "sourceMap": true,        // 生成 .js.map，方便调试
    "outDir": "dist"
  }
}
```

### 13.2.5 lib：内置类型库

`lib` 控制 TypeScript 包含哪些内置 API 的类型声明（不影响运行时，只影响类型检查）：

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": [
      "ES2022",          // ES2022 语言特性（Array.at、Object.hasOwn 等）
      "DOM",             // 浏览器 DOM API（document、window、fetch 等）
      "DOM.Iterable"     // 可迭代 DOM 集合（NodeList 等）
    ]
  }
}
```

`target` 与 `lib` 的关系：
- 不指定 `lib` 时，`lib` 默认等于 `target` 对应的标准库集合
- 指定 `lib` 后，`target` 的默认 `lib` 被完全替换

```json
// Node.js 后端项目：不需要 DOM 类型
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022"]   // 明确排除 DOM，避免 document 等全局类型干扰
  }
}

// AI 全栈项目（前后端同一 tsconfig）
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"]
  }
}
```

### 13.2.6 其他重要选项速查

```json
{
  "compilerOptions": {
    // 模块互操作性
    "esModuleInterop": true,       // 允许 import React from 'react'（而非 import * as React）
    "allowSyntheticDefaultImports": true, // 允许没有默认导出的模块被 default import

    // 大小写与路径
    "forceConsistentCasingInFileNames": true, // 强制文件名大小写一致（跨平台必备）

    // 跳过类型检查（加速构建）
    "skipLibCheck": true,          // 跳过 .d.ts 文件的类型检查（推荐开启，第三方库类型问题不影响你）

    // JSX 支持
    "jsx": "react-jsx",            // React 17+ 新 JSX 转换
    // "jsx": "preserve",          // 保留 JSX 语法（让打包工具处理，如 Vite）

    // 装饰器（NestJS、TypeORM 等框架需要）
    "experimentalDecorators": true,
    "emitDecoratorMetadata": true,

    // 额外严格检查（比 strict 更严）
    "noUnusedLocals": true,        // 禁止未使用的局部变量
    "noUnusedParameters": true,    // 禁止未使用的函数参数
    "noImplicitReturns": true,     // 函数所有路径必须有返回值
    "noFallthroughCasesInSwitch": true, // 禁止 switch 语句 case 穿透

    // 解析
    "resolveJsonModule": true,     // 允许 import config from './config.json'
    "allowJs": true,               // 允许编译 .js 文件（JS 项目迁移 TS 时有用）
    "checkJs": true                // 对 .js 文件也做类型检查（配合 allowJs）
  }
}
```

---

## 13.3 路径映射（paths、baseUrl）

### 13.3.1 相对路径的痛点

在大型项目中，深层嵌套的文件路径会产生大量 `../../../` 前缀，既难读又难维护：

```typescript
// src/features/chat/components/ChatPanel.tsx
// 要引用 src/utils/api/client.ts

// 没有路径别名时
import { apiClient } from "../../../utils/api/client";
import { formatMessage } from "../../../utils/string/format";
import { AIConfig } from "../../../core/config/ai";
import type { ChatMessage } from "../../../types/chat";
```

文件移动时，所有相对路径都需要手动更新，极易出错。

### 13.3.2 baseUrl 配置

`baseUrl` 设置模块解析的基础目录，使绝对路径导入成为可能：

```json
{
  "compilerOptions": {
    "baseUrl": "."   // 以 tsconfig.json 所在目录为基础
  }
}
```

配置 `baseUrl` 后，可以直接用项目根目录相对路径导入：

```typescript
// 设置 baseUrl 为项目根目录后
import { apiClient } from "src/utils/api/client";
import { formatMessage } from "src/utils/string/format";
```

### 13.3.3 paths 路径别名

`paths` 在 `baseUrl` 基础上定义路径别名（glob 映射规则）：

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@ai/*": ["src/ai/*"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"],
      "@config": ["src/config/index.ts"],
      "~test/*": ["tests/*"]
    }
  }
}
```

```typescript
// 使用路径别名后的清晰导入
import { apiClient } from "@/utils/api/client";
import { formatMessage } from "@utils/string/format";
import { AIConfig } from "@ai/config";
import type { ChatMessage } from "@types/chat";
import { testFixtures } from "~test/fixtures";
```

### 13.3.4 paths 的 glob 语法规则

`paths` 的键支持 `*` 通配符，映射到值数组中的第一个匹配项：

```json
{
  "compilerOptions": {
    "baseUrl": "src",
    "paths": {
      // 单一别名（无通配符）：精确映射一个路径
      "@config": ["config/index.ts"],

      // 带通配符：* 匹配任意子路径
      "@components/*": ["components/*"],

      // 多个备选路径（按顺序尝试，取第一个存在的）
      "@shared/*": [
        "shared/*",
        "../packages/shared/src/*"  // 相对于 baseUrl 的路径
      ]
    }
  }
}
```

### 13.3.5 运行时路径解析的注意事项

`tsconfig.json` 中的 `paths` **只影响 TypeScript 类型检查**，不影响运行时。需要额外配置运行时路径解析：

**Node.js（ts-node）**：

```json
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": { "@/*": ["src/*"] }
  },
  "ts-node": {
    "require": ["tsconfig-paths/register"]
  }
}
```

```bash
npm install --save-dev tsconfig-paths
```

**Vite 前端项目**：

```typescript
// vite.config.ts
import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
      "@ai": path.resolve(__dirname, "src/ai"),
      "@utils": path.resolve(__dirname, "src/utils"),
    },
  },
});
```

**Jest 测试**：

```json
// jest.config.json
{
  "moduleNameMapper": {
    "^@/(.*)$": "<rootDir>/src/$1",
    "^@ai/(.*)$": "<rootDir>/src/ai/$1",
    "^@utils/(.*)$": "<rootDir>/src/utils/$1"
  }
}
```

**Next.js**：Next.js 自动读取 `tsconfig.json` 中的 `paths` 配置，无需额外设置。

### 13.3.6 AI 项目的路径别名实践

针对 AI 应用项目的典型目录结构：

```
src/
├── ai/
│   ├── clients/        # LLM 客户端（OpenAI、Anthropic 等）
│   ├── prompts/        # Prompt 模板
│   ├── chains/         # LLM 调用链
│   └── tools/          # Function calling 工具
├── api/                # API 路由层
├── core/               # 核心业务逻辑
├── config/             # 配置管理
├── utils/              # 通用工具函数
└── types/              # 全局类型定义
```

推荐的 `paths` 配置：

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@ai/*": ["src/ai/*"],
      "@api/*": ["src/api/*"],
      "@core/*": ["src/core/*"],
      "@config": ["src/config/index.ts"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"]
    }
  }
}
```

```typescript
// 使用路径别名的 AI 应用代码
import { AnthropicClient } from "@ai/clients/anthropic";
import { ChatPromptTemplate } from "@ai/prompts/chat";
import { systemPrompt } from "@ai/prompts/system";
import { AppConfig } from "@config";
import { retry } from "@utils/retry";
import type { ChatMessage, ModelResponse } from "@types/ai";

// 清晰、不依赖相对层级的导入
```

---

## 13.4 项目引用（references、composite）

### 13.4.1 为什么需要项目引用

在 monorepo 或大型多包项目中，如果所有代码共用一个 `tsconfig.json`，会遇到以下问题：

1. **构建缓慢**：每次修改都重新编译所有代码，即使未改动的部分也要重新检查
2. **循环依赖**：多个子项目彼此依赖时，单一 `tsconfig.json` 无法表达依赖顺序
3. **缺乏隔离**：核心库的类型错误会直接影响所有依赖它的项目

项目引用（Project References）通过将大项目拆分为多个互相引用的子项目来解决上述问题，实现**增量构建**。

### 13.4.2 composite 选项

子项目的 `tsconfig.json` 必须开启 `"composite": true`，这会强制以下约束：

```json
// packages/core/tsconfig.json
{
  "compilerOptions": {
    "composite": true,       // 启用项目引用支持（必须）
    "declaration": true,     // 必须生成 .d.ts（被引用项目需要类型信息）
    "declarationMap": true,  // 生成 .d.ts.map（支持"跳转到源码"）
    "outDir": "dist",
    "rootDir": "src"
  },
  "include": ["src/**/*"]
}
```

`composite: true` 自动启用的约束：
- `declaration: true`（如未手动设置，自动开启）
- `rootDir` 必须包含所有输入文件（不能引用 rootDir 外的文件）
- 每次构建后生成 `.tsbuildinfo` 文件（增量构建的"快照"）

### 13.4.3 references 配置

父项目通过 `references` 声明对子项目的依赖：

```
ai-platform/
├── tsconfig.json               # 根配置（聚合所有子项目）
├── packages/
│   ├── core/                   # 核心类型与工具
│   │   ├── src/
│   │   └── tsconfig.json
│   ├── ai-client/              # AI 客户端封装（依赖 core）
│   │   ├── src/
│   │   └── tsconfig.json
│   └── api-server/             # API 服务器（依赖 core 和 ai-client）
│       ├── src/
│       └── tsconfig.json
└── apps/
    └── web/                    # 前端应用（依赖 core 和 ai-client）
        ├── src/
        └── tsconfig.json
```

```json
// packages/core/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "declaration": true,
    "outDir": "dist",
    "rootDir": "src"
  },
  "include": ["src/**/*"]
}
```

```json
// packages/ai-client/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "declaration": true,
    "outDir": "dist",
    "rootDir": "src"
  },
  "include": ["src/**/*"],
  "references": [
    { "path": "../core" }   // 声明对 core 的依赖
  ]
}
```

```json
// packages/api-server/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "outDir": "dist",
    "rootDir": "src"
  },
  "include": ["src/**/*"],
  "references": [
    { "path": "../core" },
    { "path": "../ai-client" }  // 同时依赖 core 和 ai-client
  ]
}
```

```json
// tsconfig.json（根配置：聚合所有子项目，自身不包含源文件）
{
  "files": [],               // 根配置不直接编译任何文件
  "references": [
    { "path": "./packages/core" },
    { "path": "./packages/ai-client" },
    { "path": "./packages/api-server" },
    { "path": "./apps/web" }
  ]
}
```

### 13.4.4 增量构建命令

项目引用配合 `--build`（简写 `-b`）标志实现增量构建：

```bash
# 构建所有子项目（按依赖顺序）
tsc --build

# 等价写法
tsc -b

# 构建特定子项目及其依赖
tsc -b packages/api-server

# 强制重新构建（忽略 .tsbuildinfo 缓存）
tsc -b --force

# 清除所有构建产物
tsc -b --clean

# 监听模式：文件变化时自动增量构建
tsc -b --watch
```

增量构建的工作原理：

```
第一次构建：
  core: 完整编译 → 生成 dist/ 和 .tsbuildinfo
  ai-client: 完整编译（依赖 core 的 .d.ts）→ 生成 dist/ 和 .tsbuildinfo
  api-server: 完整编译 → 生成 dist/ 和 .tsbuildinfo

修改 core/src/config.ts 后再次构建：
  core: 检测到变化 → 重新编译 → 更新 .tsbuildinfo
  ai-client: core 的 .d.ts 变化 → 重新编译
  api-server: ai-client 的 .d.ts 变化 → 重新编译

只修改 api-server/src/route.ts 后再次构建：
  core: .tsbuildinfo 未变 → 跳过
  ai-client: .tsbuildinfo 未变 → 跳过
  api-server: 检测到变化 → 仅重新编译 api-server
```

### 13.4.5 prepend 选项（旧版全局脚本）

在非模块（全局脚本）场景下，`prepend: true` 可将被引用项目的输出拼接到当前项目输出的开头。该选项仅用于遗留的全局脚本项目，现代 ESM 项目无需使用。

---

## 13.5 常见配置模板与最佳实践

### 13.5.1 Node.js 后端服务模板

```json
// tsconfig.json（Node.js + TypeScript 后端）
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "CommonJS",
    "moduleResolution": "node",
    "lib": ["ES2022"],

    // 严格模式全开
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,

    // 输入输出
    "rootDir": "src",
    "outDir": "dist",
    "sourceMap": true,
    "declaration": true,
    "declarationMap": true,

    // 其他
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true,
    "resolveJsonModule": true,

    // 路径别名
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts", "**/*.spec.ts"]
}
```

### 13.5.2 前端应用模板（Vite + React）

```json
// tsconfig.json（Vite + React + TypeScript）
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],

    // React JSX
    "jsx": "react-jsx",

    // 严格模式
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,

    // 让 Vite 处理转译，tsc 只做类型检查
    "noEmit": true,

    // 其他
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true,
    "resolveJsonModule": true,
    "allowImportingTsExtensions": true,  // Vite 环境支持 .ts 扩展名导入

    // 路径别名（Vite 侧需同步配置 resolve.alias）
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@hooks/*": ["src/hooks/*"]
    }
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist"]
}
```

### 13.5.3 npm 包发布模板

```json
// tsconfig.json（发布 npm 包）
{
  "compilerOptions": {
    "target": "ES2018",     // 较广的兼容性
    "module": "CommonJS",
    "moduleResolution": "node",
    "lib": ["ES2018"],

    "strict": true,

    // 发布必须生成声明文件
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,

    "rootDir": "src",
    "outDir": "dist",

    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true
  },
  "include": ["src/**/*"],
  "exclude": [
    "node_modules",
    "dist",
    "**/*.test.ts",
    "**/*.spec.ts",
    "tests/**"
  ]
}
```

### 13.5.4 最佳实践总结

**配置组织原则**：

| 原则 | 说明 |
|------|------|
| 抽取 `tsconfig.base.json` | 将通用选项提取到基础配置，各环境/用途配置通过 `extends` 继承 |
| 开启 `strict: true` | 新项目从第一行代码开始就启用严格模式，越晚迁移越痛苦 |
| 开启 `skipLibCheck: true` | 避免第三方库的类型错误阻断你的项目构建 |
| 配置 `paths` 别名 | 超过 3 层目录嵌套就值得配置路径别名 |
| 大型项目用项目引用 | 超过 5 万行代码的项目考虑拆分为 composite 子项目 |
| `noEmit` 搭配打包工具 | 使用 Vite/Webpack 等工具时，让它们负责转译，`tsc` 只做类型检查 |

**常见错误避坑**：

```json
// 错误：target 与 lib 不匹配
{
  "compilerOptions": {
    "target": "ES5",
    "lib": ["ES2022"]   // 包含了 ES5 运行时不支持的 API 类型
                         // TypeScript 会认为 Array.prototype.at() 存在，但运行时会报错
  }
}

// 正确：target 与 lib 保持一致
{
  "compilerOptions": {
    "target": "ES5",
    "lib": ["ES5", "DOM"]
  }
}
```

```json
// 错误：paths 没有配套 baseUrl
{
  "compilerOptions": {
    "paths": { "@/*": ["src/*"] }
    // 缺少 baseUrl，paths 中的相对路径无法解析
  }
}

// 正确：paths 必须与 baseUrl 配合
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": { "@/*": ["src/*"] }
  }
}
```

---

## 本章小结

| 配置项 | 关键点 | 推荐值/场景 |
|--------|--------|------------|
| `target` | 编译输出的 ES 版本，影响语法降级和内置类型 | Node.js 16+：`ES2022`；现代浏览器：`ES2020` |
| `module` | 编译输出的模块格式 | Node.js ESM：`NodeNext`；打包工具：`ESNext` |
| `moduleResolution` | 模块路径解析策略 | 配合 `module` 选择；打包工具项目用 `bundler` |
| `strict` | 一键启用所有严格检查 | 新项目始终为 `true` |
| `noEmit` | 只做类型检查不输出文件 | 使用打包工具时开启 |
| `declaration` | 生成 `.d.ts` 类型声明文件 | 发布 npm 包时必须为 `true` |
| `sourceMap` | 生成调试源映射 | 非生产环境开启；生产环境视需要 |
| `baseUrl` + `paths` | 路径别名，消除相对路径地狱 | 推荐所有中大型项目配置 |
| `composite` | 启用项目引用支持 | monorepo 子项目必须开启 |
| `references` | 声明对其他子项目的依赖 | monorepo 根配置及有依赖关系的子项目 |
| `extends` | 继承另一个配置文件 | 多环境配置的核心机制 |
| `skipLibCheck` | 跳过第三方声明文件类型检查 | 几乎所有项目建议开启 |

---

## AI 应用实战：多环境配置管理

在 AI 应用开发中，开发、测试和生产环境的需求差异显著：

- **开发环境**：需要详细源映射、宽松的类型检查（方便快速迭代）、热重载支持
- **测试环境**：需要类型严格检查、测试文件包含、模拟模块支持
- **生产环境**：需要最小化输出、严格类型检查、不含测试代码

本节展示一套完整的多环境 TypeScript 配置体系。

### 项目结构

```
ai-chat-app/
├── tsconfig.base.json          # 基础配置（所有环境共享）
├── tsconfig.json               # 开发配置（默认，继承 base）
├── tsconfig.test.json          # 测试配置（继承 base）
├── tsconfig.prod.json          # 生产配置（继承 base）
├── src/
│   ├── main.ts
│   ├── ai/
│   │   ├── client.ts
│   │   └── prompts.ts
│   ├── config/
│   │   ├── index.ts
│   │   ├── development.ts
│   │   ├── test.ts
│   │   └── production.ts
│   └── types/
│       └── env.d.ts
├── tests/
│   ├── unit/
│   └── integration/
└── scripts/
    └── build.ts
```

### 基础配置

```json
// tsconfig.base.json
// 所有环境共享的稳定选项
{
  "compilerOptions": {
    // 目标环境
    "target": "ES2022",
    "lib": ["ES2022"],

    // 模块系统（Node.js 服务端）
    "module": "CommonJS",
    "moduleResolution": "node",

    // 严格模式：所有环境统一开启
    "strict": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true,

    // 互操作性
    "esModuleInterop": true,
    "resolveJsonModule": true,
    "skipLibCheck": true,

    // 路径别名（所有环境使用相同别名）
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@ai/*": ["src/ai/*"],
      "@config": ["src/config/index.ts"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"]
    }
  }
}
```

### 开发环境配置

```json
// tsconfig.json（开发环境，也是编辑器使用的默认配置）
{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    // 开发环境：详细调试信息
    "sourceMap": true,
    "inlineSources": true,       // 源码内嵌到 source map，调试更方便

    // 开发环境：仅类型检查，由 ts-node 或 nodemon 处理执行
    "noEmit": true,

    // 开发环境：稍宽松（允许未使用变量，快速原型开发）
    "noUnusedLocals": false,
    "noUnusedParameters": false,

    // 开发时包含类型声明
    "types": ["node"]
  },
  "include": [
    "src/**/*",
    "scripts/**/*"               // 开发脚本也需要类型检查
  ],
  "exclude": [
    "node_modules",
    "dist"
  ],
  // ts-node 配置（通过 tsconfig-paths 支持路径别名）
  "ts-node": {
    "require": ["tsconfig-paths/register"],
    "transpileOnly": true        // 开发时跳过类型检查，加快启动速度
  }
}
```

### 测试环境配置

```json
// tsconfig.test.json
{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    // 测试环境：严格类型检查（提前发现问题）
    "noUnusedLocals": true,
    "noUnusedParameters": true,

    // 测试环境：需要 Node.js 和 Jest 的类型
    "types": ["node", "jest"],

    // 测试环境：生成 source map 用于错误定位
    "sourceMap": true,

    // 测试通过 ts-jest 运行，不需要 tsc 输出文件
    "noEmit": true
  },
  "include": [
    "src/**/*",
    "tests/**/*"                 // 包含测试文件
  ],
  "exclude": [
    "node_modules",
    "dist"
  ]
}
```

### 生产构建配置

```json
// tsconfig.prod.json
{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    // 生产环境：最严格的类型检查
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "exactOptionalPropertyTypes": true,  // 严格可选属性类型检查（TypeScript 4.4+）
    "noPropertyAccessFromIndexSignature": true,

    // 生产环境：输出优化
    "outDir": "dist",
    "rootDir": "src",
    "removeComments": true,      // 删除注释，减小输出体积

    // 生产环境：声明文件（如果是库的话）
    "declaration": true,
    "declarationMap": false,     // 生产不需要 .d.ts.map

    // 生产环境：source map（按需，有些团队不在生产中暴露）
    "sourceMap": false,

    // 生产环境：只需要 node 类型
    "types": ["node"]
  },
  "include": [
    "src/**/*"
    // 不包含 tests/** 和 scripts/**
  ],
  "exclude": [
    "node_modules",
    "dist",
    "tests/**",
    "**/*.test.ts",
    "**/*.spec.ts",
    "scripts/**"
  ]
}
```

### 环境配置代码（TypeScript 实现）

```typescript
// src/types/env.d.ts
// 扩展 Node.js 的 ProcessEnv 类型，为所有环境变量提供类型安全

declare namespace NodeJS {
  interface ProcessEnv {
    NODE_ENV: "development" | "test" | "production";

    // AI 服务配置
    ANTHROPIC_API_KEY: string;
    OPENAI_API_KEY?: string;
    AI_BASE_URL?: string;
    AI_TIMEOUT?: string;

    // 服务器配置
    PORT?: string;
    LOG_LEVEL?: "debug" | "info" | "warn" | "error";

    // 数据库
    DATABASE_URL: string;
  }
}
```

```typescript
// src/config/index.ts
// 根据当前 NODE_ENV 加载并合并对应的配置

export interface AIConfig {
  provider: "anthropic" | "openai";
  apiKey: string;
  baseUrl: string;
  timeout: number;
  maxRetries: number;
  defaultModel: string;
  temperature: number;
}

export interface ServerConfig {
  port: number;
  logLevel: "debug" | "info" | "warn" | "error";
  corsOrigins: string[];
}

export interface DatabaseConfig {
  url: string;
  poolSize: number;
  queryTimeout: number;
}

export interface AppConfig {
  env: "development" | "test" | "production";
  ai: AIConfig;
  server: ServerConfig;
  database: DatabaseConfig;
}

// 基础配置（所有环境共享的默认值）
const baseConfig = {
  ai: {
    provider: "anthropic" as const,
    baseUrl: "https://api.anthropic.com",
    timeout: 30_000,
    maxRetries: 3,
    defaultModel: "claude-sonnet-4-5-20251101",
    temperature: 0.7,
  },
  server: {
    port: 3000,
    logLevel: "info" as const,
    corsOrigins: ["http://localhost:3000"],
  },
  database: {
    poolSize: 5,
    queryTimeout: 10_000,
  },
};

// 单例缓存
let configCache: AppConfig | null = null;

export async function loadConfig(): Promise<AppConfig> {
  if (configCache !== null) return configCache;

  const env = process.env.NODE_ENV ?? "development";

  // 动态加载环境特定配置（生产环境不打包开发/测试配置）
  let envOverride: Partial<AppConfig>;
  switch (env) {
    case "production": {
      const { productionOverride } = await import("./production");
      envOverride = productionOverride;
      break;
    }
    case "test": {
      const { testOverride } = await import("./test");
      envOverride = testOverride;
      break;
    }
    default: {
      const { developmentOverride } = await import("./development");
      envOverride = developmentOverride;
    }
  }

  // 深度合并配置（优先使用环境变量，其次是环境配置，最后是基础配置）
  configCache = {
    env: env as AppConfig["env"],
    ai: {
      ...baseConfig.ai,
      ...(envOverride.ai ?? {}),
      // 环境变量优先级最高
      apiKey: process.env.ANTHROPIC_API_KEY ?? "",
      ...(process.env.AI_BASE_URL ? { baseUrl: process.env.AI_BASE_URL } : {}),
      ...(process.env.AI_TIMEOUT ? { timeout: parseInt(process.env.AI_TIMEOUT, 10) } : {}),
    },
    server: {
      ...baseConfig.server,
      ...(envOverride.server ?? {}),
      ...(process.env.PORT ? { port: parseInt(process.env.PORT, 10) } : {}),
      ...(process.env.LOG_LEVEL ? { logLevel: process.env.LOG_LEVEL } : {}),
    },
    database: {
      ...baseConfig.database,
      ...(envOverride.database ?? {}),
      url: process.env.DATABASE_URL ?? "",
    },
  };

  validateConfig(configCache);
  return configCache;
}

// 同步获取（仅在应用启动完成后使用）
export function getConfig(): AppConfig {
  if (configCache === null) {
    throw new Error("配置未初始化，请先在应用启动时调用 await loadConfig()");
  }
  return configCache;
}

// 配置校验：确保关键配置不为空
function validateConfig(config: AppConfig): void {
  const errors: string[] = [];

  if (!config.ai.apiKey) {
    errors.push("缺少 ANTHROPIC_API_KEY 环境变量");
  }
  if (!config.database.url) {
    errors.push("缺少 DATABASE_URL 环境变量");
  }

  if (errors.length > 0) {
    throw new Error(`配置校验失败：\n${errors.map(e => `  - ${e}`).join("\n")}`);
  }
}

// 重置缓存（仅用于测试）
export function resetConfigCache(): void {
  configCache = null;
}
```

```typescript
// src/config/development.ts
import type { AppConfig } from "./index";

export const developmentOverride: Partial<AppConfig> = {
  ai: {
    provider: "anthropic",
    apiKey: "",            // 从环境变量读取
    baseUrl: "https://api.anthropic.com",
    timeout: 60_000,       // 开发时增大超时，方便调试
    maxRetries: 1,         // 开发时减少重试，快速失败
    defaultModel: "claude-sonnet-4-5-20251101",
    temperature: 0.7,
  },
  server: {
    port: 3000,
    logLevel: "debug",     // 开发时输出详细日志
    corsOrigins: [
      "http://localhost:3000",
      "http://localhost:5173", // Vite 开发服务器
    ],
  },
  database: {
    url: "",               // 从环境变量读取
    poolSize: 2,           // 开发时小连接池
    queryTimeout: 30_000,
  },
};
```

```typescript
// src/config/test.ts
import type { AppConfig } from "./index";

export const testOverride: Partial<AppConfig> = {
  ai: {
    provider: "anthropic",
    apiKey: "test-api-key-mock",  // 测试使用 mock key（不发真实请求）
    baseUrl: "http://localhost:4000",   // 指向本地 mock 服务器
    timeout: 5_000,               // 测试使用短超时，快速失败
    maxRetries: 0,                // 测试不重试（确定性）
    defaultModel: "claude-sonnet-4-5-20251101",
    temperature: 0,               // 测试使用 temperature=0（确定性输出）
  },
  server: {
    port: 4001,
    logLevel: "warn",             // 测试时只输出警告以上日志，减少噪音
    corsOrigins: ["*"],
  },
  database: {
    url: "postgresql://localhost:5432/test_db",
    poolSize: 1,
    queryTimeout: 5_000,
  },
};
```

```typescript
// src/config/production.ts
import type { AppConfig } from "./index";

export const productionOverride: Partial<AppConfig> = {
  ai: {
    provider: "anthropic",
    apiKey: "",            // 严格从环境变量读取，绝不硬编码
    baseUrl: "https://api.anthropic.com",
    timeout: 30_000,
    maxRetries: 3,         // 生产时允许重试
    defaultModel: "claude-opus-4-5-20251101",  // 生产使用更强大的模型
    temperature: 0.7,
  },
  server: {
    port: 8080,            // 生产端口
    logLevel: "warn",      // 生产只记录警告和错误
    corsOrigins: [
      "https://app.example.com",
      "https://api.example.com",
    ],
  },
  database: {
    url: "",               // 从环境变量读取
    poolSize: 20,          // 生产使用更大连接池
    queryTimeout: 10_000,
  },
};
```

### 应用启动入口

```typescript
// src/main.ts
import { loadConfig, getConfig } from "@config";

async function bootstrap(): Promise<void> {
  // 应用启动时加载配置（确保环境变量已就绪）
  const config = await loadConfig();

  console.log(`启动 AI 聊天服务`);
  console.log(`环境：${config.env}`);
  console.log(`AI 模型：${config.ai.defaultModel}`);
  console.log(`监听端口：${config.server.port}`);

  // 后续所有模块通过同步 getConfig() 获取配置
  const { AIClient } = await import("@ai/client");
  const client = new AIClient(getConfig().ai);

  // 启动 HTTP 服务器...
  const { startServer } = await import("./server");
  await startServer(config.server);
}

bootstrap().catch((error) => {
  console.error("启动失败：", error.message);
  process.exit(1);
});
```

### package.json 脚本配置

```json
{
  "scripts": {
    "dev": "nodemon --exec ts-node src/main.ts",
    "dev:check": "tsc --noEmit --watch",
    "test": "jest --config jest.config.json",
    "typecheck": "tsc -p tsconfig.test.json --noEmit",
    "build": "tsc -p tsconfig.prod.json",
    "build:check": "tsc -p tsconfig.prod.json --noEmit"
  }
}
```

### 设计要点总结

| 环境 | tsconfig | 关键特点 | 目的 |
|------|----------|----------|------|
| 开发 | `tsconfig.json` | `noEmit: true`，宽松检查，详细 source map | 快速迭代，方便调试 |
| 测试 | `tsconfig.test.json` | 包含测试文件，严格检查，Jest 类型 | 确保测试代码类型正确 |
| 生产 | `tsconfig.prod.json` | 严格检查，输出优化，排除测试文件 | 最小化、高质量输出 |
| 基础 | `tsconfig.base.json` | 通用选项，路径别名 | 避免配置重复，统一约束 |

---

## 练习题

### 基础题

**练习 13-1**：给定以下项目结构，完成 `tsconfig.json` 的缺失配置，使其满足要求：

```
my-ai-app/
├── src/
│   ├── index.ts
│   ├── ai/
│   │   └── client.ts
│   └── utils/
│       └── helpers.ts
├── tests/
│   └── client.test.ts
└── tsconfig.json
```

要求：
1. 目标为 Node.js 18，使用 CommonJS 模块
2. 开启所有严格检查
3. 输出目录为 `dist/`，保留源映射
4. 不包含测试文件
5. 配置路径别名 `@ai/*` 指向 `src/ai/*`

**练习 13-2**：下面的 `tsconfig.json` 有 3 处错误或不良配置，找出并说明原因及修正方式：

```json
{
  "compilerOptions": {
    "target": "ES5",
    "lib": ["ES2022", "DOM"],
    "module": "ESNext",
    "moduleResolution": "Classic",
    "paths": {
      "@/*": ["src/*"]
    },
    "strict": false,
    "skipLibCheck": false
  },
  "include": ["src/**/*", "node_modules/@types/**/*"]
}
```

### 进阶题

**练习 13-3**：为一个有以下依赖关系的 monorepo 设计项目引用配置：

```
packages/
├── shared-types/    # 共享类型定义（无依赖）
├── ai-core/         # AI 核心逻辑（依赖 shared-types）
├── api-gateway/     # API 网关（依赖 ai-core 和 shared-types）
└── web-client/      # 前端客户端（依赖 shared-types）
```

写出 `packages/ai-core/tsconfig.json` 和根目录 `tsconfig.json` 的完整内容。

**练习 13-4**：实现一个 TypeScript 函数 `loadTypedConfig<T>()` 的配置加载器，利用泛型确保：
- 从 JSON 文件加载配置时，返回值类型与传入的 schema 类型完全匹配
- 缺少必填字段时抛出带有详细信息的类型错误（运行时验证）
- 支持通过环境变量覆盖任意字段

函数签名：
```typescript
async function loadTypedConfig<T extends Record<string, unknown>>(
  filePath: string,
  schema: { [K in keyof T]: { required: boolean; type: string; envKey?: string } }
): Promise<T>
```

### 挑战题

**练习 13-5**：设计并实现一个 **TypeScript 配置验证工具**，在构建时自动检测 `tsconfig.json` 中的常见问题：

1. 检测 `target` 与 `lib` 不匹配（如 `target: ES5` 但 `lib` 包含 `ES2022`）
2. 检测 `paths` 有别名但缺少 `baseUrl`
3. 检测开发配置（`noEmit: true`）被误用于生产构建脚本
4. 检测 `composite: true` 但未开启 `declaration: true`

实现一个 `validateTsConfig(configPath: string): ValidationResult[]` 函数，返回所有检测到的问题列表，每个问题包含：
- `severity`：`"error"` | `"warning"` | `"suggestion"`
- `rule`：规则名称
- `message`：问题描述
- `fix`：建议的修复方式

---

## 练习答案

### 练习 13-1 答案

```json
{
  "compilerOptions": {
    // 1. 目标 Node.js 18 使用 ES2022，CommonJS 模块
    "target": "ES2022",
    "module": "CommonJS",
    "moduleResolution": "node",
    "lib": ["ES2022"],

    // 2. 所有严格检查
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,

    // 3. 输出到 dist/，保留 source map
    "rootDir": "src",
    "outDir": "dist",
    "sourceMap": true,

    // 其他推荐配置
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true,
    "resolveJsonModule": true,

    // 5. 路径别名
    "baseUrl": ".",
    "paths": {
      "@ai/*": ["src/ai/*"]
    }
  },
  // 4. 不包含测试文件
  "include": ["src/**/*"],
  "exclude": [
    "node_modules",
    "dist",
    "tests/**"
  ]
}
```

### 练习 13-2 答案

**问题一：`target` 与 `lib` 不匹配（错误）**

```json
// 错误
"target": "ES5",
"lib": ["ES2022", "DOM"]

// 问题：target 为 ES5，但 lib 包含 ES2022 的 API 类型（如 Array.at()）。
// TypeScript 认为这些 API 存在，但运行时（ES5 环境）实际没有，会导致运行时错误。

// 修正：lib 应与 target 保持一致
"target": "ES5",
"lib": ["ES5", "DOM"]
// 或者升级 target
"target": "ES2022",
"lib": ["ES2022", "DOM"]
```

**问题二：`paths` 缺少 `baseUrl`（错误）**

```json
// 错误
"paths": {
  "@/*": ["src/*"]
}
// 没有 baseUrl，TypeScript 无法解析 paths 中的相对路径

// 修正：添加 baseUrl
"baseUrl": ".",
"paths": {
  "@/*": ["src/*"]
}
```

**问题三：`moduleResolution: "Classic"` 已过时（不良配置）**

```json
// 错误
"moduleResolution": "Classic"
// Classic 策略不查找 node_modules，与 module: "ESNext" 也不匹配

// 修正：使用现代解析策略
"module": "ESNext",
"moduleResolution": "bundler"  // 对应打包工具项目
// 或
"module": "CommonJS",
"moduleResolution": "node"
```

**附加问题：`include` 包含 `node_modules`（严重错误）**

```json
// 错误
"include": ["src/**/*", "node_modules/@types/**/*"]
// node_modules 永远不应该出现在 include 中
// @types 包会被 TypeScript 自动发现，无需手动包含

// 修正
"include": ["src/**/*"]
// 如需指定 @types，使用 compilerOptions.types
"types": ["node", "jest"]
```

### 练习 13-3 答案

```json
// packages/ai-core/tsconfig.json
{
  "compilerOptions": {
    "composite": true,           // 必须开启项目引用支持
    "declaration": true,         // 必须生成 .d.ts 供其他项目引用
    "declarationMap": true,
    "sourceMap": true,

    "target": "ES2022",
    "module": "CommonJS",
    "moduleResolution": "node",
    "lib": ["ES2022"],

    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,

    "rootDir": "src",
    "outDir": "dist",

    "esModuleInterop": true,
    "skipLibCheck": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"],
  "references": [
    { "path": "../shared-types" }  // 声明对 shared-types 的依赖
  ]
}
```

```json
// tsconfig.json（根目录，聚合所有子项目）
{
  "files": [],          // 根配置本身不编译任何文件
  "references": [
    { "path": "./packages/shared-types" },   // 无依赖，最先构建
    { "path": "./packages/ai-core" },        // 依赖 shared-types
    { "path": "./packages/api-gateway" },    // 依赖 ai-core 和 shared-types
    { "path": "./packages/web-client" }      // 依赖 shared-types
  ]
}
```

构建命令：

```bash
# 按依赖顺序构建所有子项目
tsc -b

# 只修改了 ai-core 后，只有 ai-core 和 api-gateway 会重新编译
tsc -b packages/api-gateway
```

### 练习 13-4 答案

```typescript
import fs from "fs/promises";
import path from "path";

// 字段 schema 类型定义
interface FieldSchema {
  required: boolean;
  type: "string" | "number" | "boolean" | "object";
  envKey?: string;         // 对应的环境变量名
  default?: unknown;       // 默认值
}

// 验证错误
class ConfigValidationError extends Error {
  constructor(
    public readonly field: string,
    public readonly expected: string,
    public readonly received: unknown
  ) {
    super(`配置字段 "${field}" 验证失败：期望 ${expected}，实际为 ${JSON.stringify(received)}`);
    this.name = "ConfigValidationError";
  }
}

async function loadTypedConfig<T extends Record<string, unknown>>(
  filePath: string,
  schema: { [K in keyof T]: FieldSchema }
): Promise<T> {
  // 1. 读取 JSON 配置文件
  const absolutePath = path.resolve(filePath);
  let rawConfig: Record<string, unknown> = {};

  try {
    const content = await fs.readFile(absolutePath, "utf-8");
    rawConfig = JSON.parse(content) as Record<string, unknown>;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      console.warn(`配置文件 ${absolutePath} 不存在，使用默认配置`);
    } else {
      throw new Error(`读取配置文件失败：${(error as Error).message}`);
    }
  }

  // 2. 合并环境变量（优先级高于文件）
  const merged: Record<string, unknown> = { ...rawConfig };

  for (const [key, fieldSchema] of Object.entries(schema) as [keyof T, FieldSchema][]) {
    const { envKey } = fieldSchema;
    if (envKey && process.env[envKey] !== undefined) {
      const envValue = process.env[envKey]!;

      // 按照 schema 中声明的类型转换环境变量（环境变量始终是字符串）
      switch (fieldSchema.type) {
        case "number":
          merged[key as string] = parseFloat(envValue);
          break;
        case "boolean":
          merged[key as string] = envValue === "true" || envValue === "1";
          break;
        case "object":
          try {
            merged[key as string] = JSON.parse(envValue);
          } catch {
            throw new ConfigValidationError(
              key as string,
              "valid JSON (from env)",
              envValue
            );
          }
          break;
        default:
          merged[key as string] = envValue;
      }
    }
  }

  // 3. 运行时验证：检查必填字段和类型
  const result: Record<string, unknown> = {};

  for (const [key, fieldSchema] of Object.entries(schema) as [keyof T, FieldSchema][]) {
    const value = merged[key as string] ?? fieldSchema.default;

    // 检查必填字段
    if (fieldSchema.required && (value === undefined || value === null || value === "")) {
      throw new ConfigValidationError(
        key as string,
        `non-empty ${fieldSchema.type}`,
        value
      );
    }

    // 跳过可选的未定义字段
    if (value === undefined) {
      continue;
    }

    // 类型检查
    const actualType = Array.isArray(value) ? "array" : typeof value;
    if (actualType !== fieldSchema.type) {
      throw new ConfigValidationError(
        key as string,
        fieldSchema.type,
        value
      );
    }

    result[key as string] = value;
  }

  return result as T;
}

// 使用示例
interface AIAppConfig {
  apiKey: string;
  model: string;
  temperature: number;
  debug: boolean;
  retries: number;
}

// 调用时 schema 的键必须与 T 的键完全匹配（类型安全）
async function main() {
  const config = await loadTypedConfig<AIAppConfig>("./config.json", {
    apiKey: {
      required: true,
      type: "string",
      envKey: "ANTHROPIC_API_KEY",
    },
    model: {
      required: false,
      type: "string",
      envKey: "AI_MODEL",
      default: "claude-sonnet-4-5-20251101",
    },
    temperature: {
      required: false,
      type: "number",
      envKey: "AI_TEMPERATURE",
      default: 0.7,
    },
    debug: {
      required: false,
      type: "boolean",
      envKey: "DEBUG",
      default: false,
    },
    retries: {
      required: false,
      type: "number",
      default: 3,
    },
  });

  // config 类型为 AIAppConfig，完全类型安全
  console.log(`API Key: ${config.apiKey.slice(0, 8)}...`);
  console.log(`模型: ${config.model}`);
  console.log(`Temperature: ${config.temperature}`);
}
```

### 练习 13-5 答案

```typescript
import fs from "fs";
import path from "path";

// 验证结果类型
interface ValidationResult {
  severity: "error" | "warning" | "suggestion";
  rule: string;
  message: string;
  fix: string;
}

// tsconfig.json 的简化类型定义（实际更复杂）
interface TsConfigCompilerOptions {
  target?: string;
  lib?: string[];
  module?: string;
  moduleResolution?: string;
  paths?: Record<string, string[]>;
  baseUrl?: string;
  strict?: boolean;
  noEmit?: boolean;
  composite?: boolean;
  declaration?: boolean;
  skipLibCheck?: boolean;
  [key: string]: unknown;
}

interface TsConfig {
  compilerOptions?: TsConfigCompilerOptions;
  include?: string[];
  exclude?: string[];
  files?: string[];
}

// ES 版本的最低要求映射（lib 中的 ES 版本不应高于 target）
const ES_VERSION_ORDER: Record<string, number> = {
  ES3: 3, ES5: 5, ES2015: 2015, ES2016: 2016, ES2017: 2017,
  ES2018: 2018, ES2019: 2019, ES2020: 2020, ES2021: 2021,
  ES2022: 2022, ES2023: 2023, ESNEXT: 9999,
};

function parseEsVersion(versionStr: string): number {
  const normalized = versionStr.toUpperCase().replace("ES6", "ES2015").replace("ES7", "ES2016");
  return ES_VERSION_ORDER[normalized] ?? 0;
}

// 规则检查函数

function checkTargetLibMismatch(options: TsConfigCompilerOptions): ValidationResult | null {
  const { target, lib } = options;
  if (!target || !lib) return null;

  const targetVersion = parseEsVersion(target);
  if (targetVersion === 0) return null;

  const highestLib = lib
    .map(l => parseEsVersion(l))
    .filter(v => v > 0)
    .reduce((max, v) => Math.max(max, v), 0);

  if (highestLib > targetVersion && highestLib !== 9999) {
    return {
      severity: "error",
      rule: "target-lib-mismatch",
      message: `target "${target}" 低于 lib 中包含的 ES 版本 (ES${highestLib})。` +
                `运行时可能不存在 lib 中声明的 API，导致运行时错误。`,
      fix: `将 target 升级为 "ES${highestLib}" 或更高，或将 lib 中的 ES 版本降至 "${target}"。`,
    };
  }
  return null;
}

function checkPathsWithoutBaseUrl(options: TsConfigCompilerOptions): ValidationResult | null {
  const { paths, baseUrl } = options;
  if (paths && !baseUrl) {
    return {
      severity: "error",
      rule: "paths-requires-baseUrl",
      message: `配置了 "paths" 路径别名，但缺少 "baseUrl"。paths 依赖 baseUrl 来解析路径，缺少时别名无法正常工作。`,
      fix: `在 compilerOptions 中添加 "baseUrl": "."（或其他合适的基础目录）。`,
    };
  }
  return null;
}

function checkNoEmitInProdConfig(
  options: TsConfigCompilerOptions,
  configPath: string
): ValidationResult | null {
  const filename = path.basename(configPath).toLowerCase();
  const isProdConfig =
    filename.includes("prod") ||
    filename.includes("production") ||
    filename.includes("build");

  if (isProdConfig && options.noEmit === true) {
    return {
      severity: "warning",
      rule: "noEmit-in-prod-config",
      message: `生产构建配置文件 "${path.basename(configPath)}" 中 noEmit 为 true，` +
                `这意味着构建不会输出任何 JS 文件，通常是开发/类型检查配置的设置。`,
      fix: `确认这是有意为之（如使用外部打包工具），否则将 "noEmit" 改为 false 或删除该选项。`,
    };
  }
  return null;
}

function checkCompositeWithoutDeclaration(options: TsConfigCompilerOptions): ValidationResult | null {
  if (options.composite === true && options.declaration === false) {
    return {
      severity: "error",
      rule: "composite-requires-declaration",
      message: `启用了 "composite: true"（项目引用），但显式设置了 "declaration: false"。` +
                `项目引用要求生成 .d.ts 声明文件，其他子项目需要它来获取类型信息。`,
      fix: `删除 "declaration: false" 或将其改为 "declaration: true"。composite 模式会自动启用 declaration。`,
    };
  }
  return null;
}

function checkClassicModuleResolution(options: TsConfigCompilerOptions): ValidationResult | null {
  if (options.moduleResolution?.toLowerCase() === "classic") {
    return {
      severity: "warning",
      rule: "avoid-classic-resolution",
      message: `"moduleResolution: Classic" 是已过时的解析策略，不查找 node_modules，` +
                `不适合现代项目。`,
      fix: `改用 "node"（CommonJS 项目）、"bundler"（Vite/Webpack 项目）或 "NodeNext"（Node.js ESM 项目）。`,
    };
  }
  return null;
}

function checkNodeModulesInInclude(config: TsConfig): ValidationResult | null {
  const hasNodeModulesInInclude = config.include?.some(
    pattern => pattern.includes("node_modules")
  );
  if (hasNodeModulesInInclude) {
    return {
      severity: "error",
      rule: "no-node-modules-in-include",
      message: `"include" 中包含 "node_modules" 路径。这会让 TypeScript 尝试编译所有第三方包，` +
                `极大降低编译速度，并可能引入大量类型错误。`,
      fix: `从 "include" 中删除所有包含 "node_modules" 的模式。@types 会被自动发现。`,
    };
  }
  return null;
}

// 主验证函数
function validateTsConfig(configPath: string): ValidationResult[] {
  const absolutePath = path.resolve(configPath);

  if (!fs.existsSync(absolutePath)) {
    return [{
      severity: "error",
      rule: "file-not-found",
      message: `找不到配置文件：${absolutePath}`,
      fix: `确认文件路径正确，或运行 tsc --init 创建默认配置。`,
    }];
  }

  let config: TsConfig;
  try {
    const content = fs.readFileSync(absolutePath, "utf-8");
    // 移除 JSON 注释（tsconfig 支持注释，但标准 JSON.parse 不支持）
    const stripped = content.replace(/\/\/[^\n]*/g, "").replace(/\/\*[\s\S]*?\*\//g, "");
    config = JSON.parse(stripped) as TsConfig;
  } catch (error) {
    return [{
      severity: "error",
      rule: "invalid-json",
      message: `配置文件 JSON 解析失败：${(error as Error).message}`,
      fix: `检查 JSON 语法错误，注意尾逗号（trailing comma）等常见问题。`,
    }];
  }

  const options = config.compilerOptions ?? {};
  const results: ValidationResult[] = [];

  // 运行所有检查规则
  const checks = [
    checkTargetLibMismatch(options),
    checkPathsWithoutBaseUrl(options),
    checkNoEmitInProdConfig(options, configPath),
    checkCompositeWithoutDeclaration(options),
    checkClassicModuleResolution(options),
    checkNodeModulesInInclude(config),
  ];

  for (const result of checks) {
    if (result !== null) {
      results.push(result);
    }
  }

  return results;
}

// 使用示例
function main() {
  const configPath = process.argv[2] ?? "tsconfig.json";
  const results = validateTsConfig(configPath);

  if (results.length === 0) {
    console.log(`✓ ${configPath} 通过所有检查`);
    return;
  }

  console.log(`\n${configPath} 发现 ${results.length} 个问题：\n`);

  for (const result of results) {
    const icon = result.severity === "error" ? "X" :
                 result.severity === "warning" ? "!" : "i";
    console.log(`[${result.severity.toUpperCase()}] ${icon} ${result.rule}`);
    console.log(`  问题：${result.message}`);
    console.log(`  修复：${result.fix}`);
    console.log();
  }

  // 有错误时退出码为 1（可用于 CI/CD 门禁）
  const hasErrors = results.some(r => r.severity === "error");
  if (hasErrors) process.exit(1);
}

main();
```

---

*下一章：[第14章：代码质量工具（ESLint + Prettier）](./14-linting-formatting.md)*
