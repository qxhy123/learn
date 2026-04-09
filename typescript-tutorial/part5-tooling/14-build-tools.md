# 第14章：构建工具集成

> **系列定位**：本章深入讲解 TypeScript 生态中主流构建工具的配置与使用，包括 tsc、ts-node、esbuild 和 Vite，并通过一个完整的 AI 应用构建流程展示从开发到生产的最佳实践。

---

## 学习目标

完成本章学习后，你将能够：

1. 深入理解 `tsc` 编译器的核心配置项，能够根据项目需求定制 `tsconfig.json`，掌握项目引用（Project References）等高级特性
2. 使用 `ts-node` 与 `tsx` 直接运行 TypeScript 文件，配置合理的开发调试环境
3. 使用 `esbuild` 对 TypeScript 项目进行高性能打包，理解其与 `tsc` 在类型检查方面的分工
4. 在 Vite 项目中集成 TypeScript，利用 Vite 的热更新与按需编译提升 AI 前端应用的开发体验
5. 掌握 tree-shaking、代码分割、懒加载等打包优化技术，将其应用于 AI SDK 的生产部署

---

## 14.1 tsc编译器深入

### 14.1.1 tsconfig.json 核心结构

`tsconfig.json` 是 TypeScript 项目的核心配置文件。一个完整的配置文件由以下几个顶层字段组成：

```json
{
  "compilerOptions": { ... },   // 编译器行为
  "include": [ ... ],           // 纳入编译的文件/目录
  "exclude": [ ... ],           // 排除的文件/目录
  "files": [ ... ],             // 显式指定文件列表（优先级最高）
  "extends": "...",             // 继承另一个配置文件
  "references": [ ... ]         // 项目引用（多包项目）
}
```

对于 AI 应用后端服务，一个典型的起点配置如下：

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

### 14.1.2 关键编译选项详解

**`target`：编译目标**

`target` 决定输出 JavaScript 的语法版本，同时影响哪些语法会被降级（downleveled）：

```json
// Node.js 18+ 后端服务：充分利用现代特性
"target": "ES2022"

// 浏览器兼容性优先：覆盖旧版浏览器
"target": "ES2017"

// 最新特性（仅限支持最新 V8 的环境）
"target": "ESNext"
```

**`module` 与 `moduleResolution`：模块系统**

```json
// Node.js ESM 项目（package.json 中 "type": "module"）
"module": "Node16",
"moduleResolution": "Node16"

// Node.js CommonJS 项目
"module": "CommonJS",
"moduleResolution": "Node10"

// 浏览器 / Vite / esbuild 项目
"module": "ESNext",
"moduleResolution": "Bundler"
```

> **注意**：`"moduleResolution": "Bundler"` 是 TypeScript 5.0 引入的新选项，专为打包工具设计，允许省略扩展名、支持 `exports` 字段，但不强制要求 `.js` 后缀。

**`strict` 系列选项**

`"strict": true` 是一个快捷开关，等价于同时启用以下所有选项：

```json
{
  "strictNullChecks": true,          // null/undefined 安全
  "strictFunctionTypes": true,       // 函数参数逆变检查
  "strictBindCallApply": true,       // bind/call/apply 类型检查
  "strictPropertyInitialization": true, // 类属性初始化检查
  "noImplicitAny": true,             // 禁止隐式 any
  "noImplicitThis": true,            // 禁止隐式 this
  "alwaysStrict": true               // 输出文件中始终加 "use strict"
}
```

也可以在启用 `strict` 的基础上单独关闭某项：

```json
{
  "strict": true,
  "strictPropertyInitialization": false  // 单独关闭某项
}
```

**`paths`：路径别名**

在 AI 应用中，路径别名极为常用：

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@ai/*": ["src/ai/*"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"]
    }
  }
}
```

对应的使用方式：

```typescript
// 使用路径别名
import { ChatClient } from "@ai/client";
import { formatPrompt } from "@utils/prompt";
import type { ModelConfig } from "@types/ai";
```

> **注意**：`tsconfig.json` 中的 `paths` 仅影响 TypeScript 类型检查，不影响实际运行时模块解析。运行时需要对应打包工具（esbuild、Vite）或 `tsconfig-paths` 包配合处理。

### 14.1.3 项目引用（Project References）

当 AI 应用拆分为多个子包（如 `shared`、`backend`、`frontend`）时，项目引用能实现**增量编译**和**类型隔离**：

```
ai-monorepo/
├── packages/
│   ├── shared/          # 共享类型与工具
│   │   ├── src/
│   │   └── tsconfig.json
│   ├── backend/         # Node.js API 服务
│   │   ├── src/
│   │   └── tsconfig.json
│   └── frontend/        # React 前端
│       ├── src/
│       └── tsconfig.json
└── tsconfig.json        # 根配置
```

`packages/shared/tsconfig.json`：

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "composite": true,       // 必须：启用项目引用支持
    "declaration": true,     // 必须：生成 .d.ts 文件
    "outDir": "./dist"
  },
  "include": ["src/**/*"]
}
```

`packages/backend/tsconfig.json`：

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "composite": true,
    "outDir": "./dist"
  },
  "include": ["src/**/*"],
  "references": [
    { "path": "../shared" }    // 声明对 shared 包的依赖
  ]
}
```

根目录 `tsconfig.json`（汇总配置）：

```json
{
  "files": [],             // 根配置不直接编译任何文件
  "references": [
    { "path": "packages/shared" },
    { "path": "packages/backend" },
    { "path": "packages/frontend" }
  ]
}
```

使用 `--build` 模式编译整个 monorepo：

```bash
# 增量构建（只重新编译有变更的包）
tsc --build

# 强制全量重建
tsc --build --force

# 查看构建详情
tsc --build --verbose

# 清理构建产物
tsc --build --clean
```

### 14.1.4 常用 tsc 命令行选项

```bash
# 类型检查（不输出文件）——CI 管道中最常用
tsc --noEmit

# 监听模式（开发时自动重新编译）
tsc --watch

# 指定配置文件
tsc --project tsconfig.prod.json

# 覆盖单个配置项（临时调试用）
tsc --noEmit --strict false

# 查看编译器版本
tsc --version
```

---

## 14.2 ts-node与tsx（直接运行TypeScript）

### 14.2.1 ts-node 基础

`ts-node` 允许在 Node.js 中直接运行 TypeScript 文件，无需预先编译：

```bash
# 安装
npm install -D ts-node typescript @types/node

# 直接运行 TypeScript 文件
npx ts-node src/index.ts

# 启动交互式 REPL
npx ts-node
```

`ts-node` 的基本配置通过 `tsconfig.json` 中的 `ts-node` 字段设置：

```json
{
  "compilerOptions": {
    "module": "CommonJS",
    "target": "ES2022",
    "strict": true
  },
  "ts-node": {
    "transpileOnly": true,    // 跳过类型检查，只做语法转译（速度更快）
    "files": true,            // 加载 tsconfig 中 files/include 指定的文件
    "experimentalSpecifierResolution": "node"
  }
}
```

用于 AI 脚本的典型场景：

```typescript
// scripts/test-model.ts
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

async function testModel(): Promise<void> {
  const message = await client.messages.create({
    model: "claude-opus-4-5-20251101",
    max_tokens: 256,
    messages: [{ role: "user", content: "用一句话介绍 TypeScript 构建工具。" }],
  });

  const content = message.content[0];
  if (content.type === "text") {
    console.log("模型响应：", content.text);
  }
}

testModel().catch(console.error);
```

```bash
# 直接运行，无需编译步骤
npx ts-node scripts/test-model.ts
```

### 14.2.2 ESM 模式下的 ts-node

当项目使用 ESM（`"type": "module"`）时，需要使用 `ts-node` 的 ESM 加载器：

```json
// package.json
{
  "type": "module",
  "scripts": {
    "dev": "node --loader ts-node/esm src/index.ts",
    "script": "node --loader ts-node/esm"
  }
}
```

对应的 `tsconfig.json`：

```json
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "Node16",
    "target": "ES2022"
  },
  "ts-node": {
    "esm": true,
    "experimentalSpecifierResolution": "node"
  }
}
```

### 14.2.3 tsx：更快的替代方案

`tsx`（基于 esbuild）是 `ts-node` 的现代替代品，启动速度明显更快，且原生支持 ESM/CJS 双模式，无需额外配置：

```bash
# 安装
npm install -D tsx

# 直接运行
npx tsx src/index.ts

# 监听模式（文件变化时自动重新运行）
npx tsx watch src/server.ts
```

`tsx` 与 `ts-node` 的对比：

| 特性 | ts-node | tsx |
|------|---------|-----|
| 底层转译器 | TypeScript 编译器 | esbuild |
| 启动速度 | 较慢（~500ms） | 极快（~50ms） |
| 类型检查 | 支持（可选关闭） | 不支持（纯转译） |
| ESM 支持 | 需要额外配置 | 开箱即用 |
| tsconfig 支持 | 完整 | 部分 |
| 适用场景 | 需要类型检查的脚本 | 开发服务、快速脚本 |

推荐的 `package.json` 脚本配置：

```json
{
  "scripts": {
    "dev": "tsx watch src/server.ts",
    "script": "tsx",
    "typecheck": "tsc --noEmit",
    "build": "tsc --build"
  }
}
```

### 14.2.4 调试配置

在 VS Code 中调试 TypeScript 文件，`.vscode/launch.json`：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "launch",
      "name": "调试当前文件 (tsx)",
      "runtimeExecutable": "npx",
      "runtimeArgs": ["tsx"],
      "args": ["${file}"],
      "console": "integratedTerminal",
      "sourceMaps": true
    },
    {
      "type": "node",
      "request": "launch",
      "name": "调试服务器",
      "runtimeExecutable": "npx",
      "runtimeArgs": ["tsx", "watch"],
      "args": ["src/server.ts"],
      "console": "integratedTerminal",
      "restart": true
    }
  ]
}
```

---

## 14.3 esbuild与TypeScript

### 14.3.1 esbuild 的定位与核心优势

esbuild 是一款用 Go 编写的极高性能 JavaScript/TypeScript 打包工具。它对 TypeScript 的处理方式与 `tsc` 有本质区别：

- **esbuild**：纯语法转译，将 TypeScript 语法剥除转换为 JavaScript，**不做类型检查**
- **tsc**：完整编译器，既做类型检查，也做代码转换

这种分工使得 esbuild 在构建速度上比 `tsc` 快 10~100 倍。正确的工作流是：

```
开发时：tsx（快速运行）
CI 检查：tsc --noEmit（类型检查）
生产构建：esbuild（快速打包）
```

### 14.3.2 命令行使用

```bash
# 安装
npm install -D esbuild

# 打包单个文件（CommonJS 输出）
npx esbuild src/index.ts --bundle --platform=node --outfile=dist/index.js

# 打包为 ESM 格式
npx esbuild src/index.ts --bundle --format=esm --outfile=dist/index.mjs

# 生产模式（压缩 + tree-shaking）
npx esbuild src/index.ts \
  --bundle \
  --minify \
  --platform=node \
  --target=node18 \
  --outfile=dist/index.js

# 监听模式（开发用）
npx esbuild src/index.ts --bundle --watch --outfile=dist/index.js
```

### 14.3.3 编程式 API

对于复杂项目，推荐使用编程式 API 编写构建脚本：

```typescript
// scripts/build.ts
import * as esbuild from "esbuild";
import { nodeExternalsPlugin } from "esbuild-node-externals";

async function build(): Promise<void> {
  const result = await esbuild.build({
    // 入口文件
    entryPoints: {
      index: "src/index.ts",
      "workers/embedding": "src/workers/embedding.ts",
    },

    // 输出配置
    outdir: "dist",
    format: "esm",
    platform: "node",
    target: "node18",

    // 打包与优化
    bundle: true,
    minify: process.env.NODE_ENV === "production",
    sourcemap: true,
    treeShaking: true,

    // 将 node_modules 标记为外部依赖（不打入 bundle）
    plugins: [nodeExternalsPlugin()],

    // 路径别名（对应 tsconfig paths）
    alias: {
      "@ai": "./src/ai",
      "@utils": "./src/utils",
      "@types": "./src/types",
    },

    // 元数据输出（用于分析 bundle 体积）
    metafile: true,
  });

  if (result.metafile) {
    const text = await esbuild.analyzeMetafile(result.metafile);
    console.log(text);
  }

  console.log("构建完成");
}

build().catch((err) => {
  console.error("构建失败：", err);
  process.exit(1);
});
```

### 14.3.4 处理 TypeScript 路径别名

esbuild 本身不读取 `tsconfig.json` 中的 `paths` 配置，需要使用插件或 `alias` 选项：

```typescript
// scripts/build.ts
import * as esbuild from "esbuild";
import { readFileSync } from "fs";

// 读取 tsconfig.json 中的 paths，自动转换为 esbuild alias
function tsconfigPathsToAlias(tsconfigPath: string): Record<string, string> {
  const tsconfig = JSON.parse(readFileSync(tsconfigPath, "utf-8"));
  const paths: Record<string, string[]> = tsconfig.compilerOptions?.paths ?? {};
  const alias: Record<string, string> = {};

  for (const [key, values] of Object.entries(paths)) {
    // "@ai/*" -> "@ai"，"./src/ai/*" -> "./src/ai"
    const aliasKey = key.replace("/*", "");
    const aliasValue = (values[0] ?? "").replace("/*", "");
    alias[aliasKey] = aliasValue;
  }

  return alias;
}

await esbuild.build({
  entryPoints: ["src/index.ts"],
  bundle: true,
  outfile: "dist/index.js",
  alias: tsconfigPathsToAlias("tsconfig.json"),
});
```

### 14.3.5 与 tsc 的分工实践

最佳实践是将类型检查与构建完全解耦：

```json
// package.json
{
  "scripts": {
    "typecheck": "tsc --noEmit",
    "build": "tsx scripts/build.ts",
    "build:check": "npm run typecheck && npm run build",
    "ci": "npm run typecheck && npm run build && npm test"
  }
}
```

---

## 14.4 Vite与TypeScript

### 14.4.1 Vite 对 TypeScript 的支持

Vite 对 TypeScript 提供开箱即用的支持，底层使用 esbuild 进行转译（同样不做类型检查），速度极快。典型的 Vite + TypeScript AI 前端项目结构：

```
ai-chat-app/
├── src/
│   ├── main.ts            # 应用入口
│   ├── App.tsx            # 根组件（使用 React + TSX）
│   ├── ai/
│   │   ├── client.ts      # AI API 客户端封装
│   │   └── stream.ts      # SSE 流式响应处理
│   └── components/
│       └── ChatWindow.tsx
├── index.html
├── vite.config.ts         # Vite 配置
├── tsconfig.json          # 基础 TypeScript 配置
└── tsconfig.node.json     # Vite 配置文件专用 TypeScript 配置
```

### 14.4.2 vite.config.ts 配置详解

```typescript
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig({
  plugins: [react()],

  // 路径别名（需与 tsconfig.json 的 paths 保持一致）
  resolve: {
    alias: {
      "@": resolve(__dirname, "./src"),
      "@ai": resolve(__dirname, "./src/ai"),
      "@components": resolve(__dirname, "./src/components"),
    },
  },

  // 开发服务器配置
  server: {
    port: 5173,
    // 代理 AI API 请求，避免 CORS 问题
    proxy: {
      "/api": {
        target: "http://localhost:3000",
        changeOrigin: true,
      },
    },
  },

  // 构建配置
  build: {
    outDir: "dist",
    sourcemap: true,
    // 代码分割
    rollupOptions: {
      output: {
        manualChunks: {
          // 将大型 AI SDK 单独打包
          "ai-sdk": ["@anthropic-ai/sdk"],
          vendor: ["react", "react-dom"],
        },
      },
    },
  },

  // 环境变量类型提示（配合 src/env.d.ts）
  envPrefix: "VITE_",
});
```

### 14.4.3 双 tsconfig 结构

Vite 项目通常维护两个 `tsconfig.json` 文件，职责明确：

`tsconfig.json`（应用源码）：

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx",
    "strict": true,
    "noEmit": true,
    "isolatedModules": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@ai/*": ["src/ai/*"]
    }
  },
  "include": ["src/**/*", "src/env.d.ts"]
}
```

`tsconfig.node.json`（Vite 配置文件本身）：

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "lib": ["ES2022"],
    "strict": true,
    "noEmit": true
  },
  "include": ["vite.config.ts"]
}
```

### 14.4.4 环境变量类型安全

Vite 使用 `import.meta.env` 访问环境变量，通过声明文件实现类型安全：

```typescript
// src/env.d.ts
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_AI_API_URL: string;
  readonly VITE_AI_MODEL: string;
  readonly VITE_MAX_TOKENS: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

使用时即可获得完整的类型提示：

```typescript
// src/ai/client.ts
const AI_API_URL = import.meta.env.VITE_AI_API_URL;  // 类型：string
const AI_MODEL = import.meta.env.VITE_AI_MODEL;       // 类型：string
```

### 14.4.5 热更新与 AI 流式响应

Vite 的 HMR（热模块替换）与 TypeScript 深度集成，修改类型定义后无需刷新页面即可看到效果：

```typescript
// src/ai/stream.ts
export interface StreamChunk {
  type: "text" | "done" | "error";
  content?: string;
  error?: string;
}

export async function* streamChat(
  prompt: string,
  signal?: AbortSignal
): AsyncGenerator<StreamChunk> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
    signal,
  });

  if (!response.ok) {
    yield { type: "error", error: `HTTP ${response.status}` };
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value, { stream: true });
      yield { type: "text", content: text };
    }
    yield { type: "done" };
  } finally {
    reader.releaseLock();
  }
}
```

---

## 14.5 打包优化与tree-shaking

### 14.5.1 什么是 tree-shaking

tree-shaking 是打包工具在构建时**静态分析模块的导入导出关系**，移除未被使用的代码（dead code）的过程。它要求代码使用 ES 模块（`import`/`export`），因为 CommonJS 的 `require` 是动态的，无法静态分析。

```typescript
// utils/ai-helpers.ts（导出多个函数）
export function formatPrompt(text: string): string {
  return text.trim();
}

export function parseResponse(raw: string): object {
  return JSON.parse(raw);
}

export function calculateTokenCost(tokens: number): number {
  return tokens * 0.00002;
}
```

```typescript
// main.ts（只用了其中一个）
import { formatPrompt } from "./utils/ai-helpers";

console.log(formatPrompt("  hello  "));
// parseResponse 和 calculateTokenCost 不会出现在最终 bundle 中
```

### 14.5.2 确保代码 tree-shaking 友好

TypeScript 代码要享受 tree-shaking，需要遵守以下规则：

**1. 使用命名导出，避免无差别的命名空间导入**

```typescript
// 不推荐：导入整个命名空间
import * as AIUtils from "./utils/ai-helpers";
AIUtils.formatPrompt("hello");  // 打包工具无法判断哪些函数用到了

// 推荐：按需命名导入
import { formatPrompt } from "./utils/ai-helpers";
formatPrompt("hello");
```

**2. 避免副作用导入（除非必要）**

```typescript
// 有副作用的导入（打包工具会保留整个模块）
import "./polyfills";         // 副作用：修改全局对象

// 无副作用的导入（可以安全 tree-shake）
import { base64Encode } from "./utils/encoding";
```

在 `package.json` 中声明无副作用（让打包工具更激进地优化）：

```json
{
  "name": "my-ai-sdk",
  "sideEffects": false
}
```

或指定有副作用的文件：

```json
{
  "sideEffects": [
    "src/polyfills.ts",
    "**/*.css"
  ]
}
```

**3. 避免使用 `export default` 导出对象**

```typescript
// 不推荐：导出一个大对象，打包工具无法拆分内部成员
export default {
  formatPrompt,
  parseResponse,
  calculateTokenCost,
};

// 推荐：单独命名导出
export { formatPrompt, parseResponse, calculateTokenCost };
```

### 14.5.3 代码分割（Code Splitting）

代码分割将 bundle 拆分为多个小文件，实现按需加载，减少首屏加载时间：

**Vite / Rollup 的动态导入**

```typescript
// 静态导入（始终打进主 bundle）
import { HeavyChart } from "./components/HeavyChart";

// 动态导入（独立 chunk，按需加载）
const { HeavyChart } = await import("./components/HeavyChart");
```

在 React AI 应用中，对重型 AI 组件使用懒加载：

```typescript
// src/App.tsx
import React, { Suspense, lazy } from "react";

// 代码分割：AI 聊天界面单独打包
const ChatWindow = lazy(() => import("./components/ChatWindow"));
const EmbeddingVisualizer = lazy(() => import("./components/EmbeddingVisualizer"));

export function App(): React.ReactElement {
  return (
    <Suspense fallback={<div>加载中...</div>}>
      <ChatWindow />
    </Suspense>
  );
}
```

**手动 chunk 划分（vite.config.ts）**

```typescript
// vite.config.ts
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          // AI SDK 单独打包（变化频率低，利于缓存）
          if (id.includes("@anthropic-ai") || id.includes("openai")) {
            return "ai-vendors";
          }
          // React 生态单独打包
          if (id.includes("node_modules/react")) {
            return "react-vendor";
          }
          // 其余 node_modules 打进通用 vendor chunk
          if (id.includes("node_modules")) {
            return "vendor";
          }
        },
      },
    },
  },
});
```

### 14.5.4 分析 bundle 体积

**使用 esbuild 元数据分析**

```typescript
// scripts/analyze.ts
import * as esbuild from "esbuild";
import { writeFileSync } from "fs";

const result = await esbuild.build({
  entryPoints: ["src/index.ts"],
  bundle: true,
  outfile: "dist/index.js",
  metafile: true,
});

// 输出可读的分析报告
const analysis = await esbuild.analyzeMetafile(result.metafile!, {
  verbose: true,
});
console.log(analysis);

// 保存元数据供 esbuild bundle visualizer 使用
writeFileSync("dist/meta.json", JSON.stringify(result.metafile));
```

**使用 rollup-plugin-visualizer（Vite 项目）**

```typescript
// vite.config.ts
import { visualizer } from "rollup-plugin-visualizer";

export default defineConfig({
  plugins: [
    react(),
    visualizer({
      filename: "dist/stats.html",
      open: true,       // 构建后自动打开报告
      gzipSize: true,   // 显示 gzip 压缩后体积
    }),
  ],
});
```

### 14.5.5 优化 AI SDK 的引入体积

AI SDK 往往体积较大，以下策略可以显著减小最终 bundle：

```typescript
// 不推荐：导入整个 SDK
import Anthropic from "@anthropic-ai/sdk";

// 推荐：仅导入需要的子模块（如 SDK 支持）
import { Anthropic } from "@anthropic-ai/sdk/core";
import { Messages } from "@anthropic-ai/sdk/resources";

// 推荐：在 Node.js 后端将 SDK 标记为外部依赖（不打入 bundle）
// esbuild.config.ts
await esbuild.build({
  entryPoints: ["src/server.ts"],
  bundle: true,
  platform: "node",
  external: [
    "@anthropic-ai/sdk",
    "openai",
    // 其他不需要打包的 npm 包
  ],
});
```

---

## 本章小结

| 工具 | 类型检查 | 构建速度 | 适用阶段 | 典型命令 |
|------|----------|----------|----------|---------|
| `tsc` | 完整 | 较慢 | CI 检查 / 生产构建 | `tsc --noEmit` / `tsc --build` |
| `ts-node` | 可选 | 中等 | 开发脚本 / 调试 | `ts-node src/script.ts` |
| `tsx` | 无 | 极快 | 开发服务器 / 快速脚本 | `tsx watch src/server.ts` |
| `esbuild` | 无 | 极快 | 生产打包 | `esbuild src/index.ts --bundle` |
| `Vite` | 无 | 极快 | 前端开发 + 生产构建 | `vite dev` / `vite build` |

**关键原则**：

- 类型检查与代码转译**分开执行**：`tsc --noEmit` 负责类型安全，esbuild/Vite 负责速度
- 开发环境优先速度：使用 `tsx` 或 `Vite dev server`，即时反馈
- 生产环境优先体积：启用 tree-shaking、代码分割、minify
- AI SDK 等大型依赖：后端不打包（`external`），前端单独 chunk（利于缓存）
- `tsconfig.json` 的 `paths` 需要在打包工具侧同步配置 `alias`

---

## AI应用实战：AI应用构建流程

本节展示一个完整的 AI 聊天应用从开发到生产的构建流程配置，综合运用本章所有工具。

### 项目结构

```
ai-chat-service/
├── src/
│   ├── server.ts             # Express API 服务入口
│   ├── ai/
│   │   ├── client.ts         # Anthropic SDK 封装
│   │   ├── stream.ts         # 流式响应处理
│   │   └── types.ts          # AI 相关类型定义
│   ├── routes/
│   │   └── chat.ts           # 聊天路由
│   └── utils/
│       └── logger.ts         # 日志工具
├── scripts/
│   └── build.ts              # esbuild 构建脚本
├── tsconfig.json
├── tsconfig.build.json       # 生产构建专用配置
└── package.json
```

### tsconfig.json（开发配置）

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "sourceMap": true,
    "declaration": true,
    "baseUrl": ".",
    "paths": {
      "@ai/*": ["src/ai/*"],
      "@utils/*": ["src/utils/*"]
    }
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### 核心业务代码

```typescript
// src/ai/types.ts
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  model?: string;
  maxTokens?: number;
  stream?: boolean;
}

export interface ChatResponse {
  id: string;
  content: string;
  usage: {
    inputTokens: number;
    outputTokens: number;
  };
}
```

```typescript
// src/ai/client.ts
import Anthropic from "@anthropic-ai/sdk";
import type { ChatRequest, ChatResponse } from "./types";

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const DEFAULT_MODEL = "claude-opus-4-5-20251101";

export async function chat(request: ChatRequest): Promise<ChatResponse> {
  const response = await client.messages.create({
    model: request.model ?? DEFAULT_MODEL,
    max_tokens: request.maxTokens ?? 1024,
    messages: request.messages,
  });

  const textContent = response.content.find((c) => c.type === "text");

  return {
    id: response.id,
    content: textContent?.type === "text" ? textContent.text : "",
    usage: {
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
    },
  };
}

export async function* chatStream(
  request: ChatRequest
): AsyncGenerator<string> {
  const stream = await client.messages.create({
    model: request.model ?? DEFAULT_MODEL,
    max_tokens: request.maxTokens ?? 1024,
    messages: request.messages,
    stream: true,
  });

  for await (const event of stream) {
    if (
      event.type === "content_block_delta" &&
      event.delta.type === "text_delta"
    ) {
      yield event.delta.text;
    }
  }
}
```

```typescript
// src/routes/chat.ts
import { Router, type Request, type Response } from "express";
import { chat, chatStream } from "@ai/client";
import type { ChatRequest } from "@ai/types";

export const chatRouter = Router();

// 普通聊天端点
chatRouter.post("/", async (req: Request, res: Response): Promise<void> => {
  try {
    const chatReq = req.body as ChatRequest;
    const result = await chat(chatReq);
    res.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "未知错误";
    res.status(500).json({ error: message });
  }
});

// 流式聊天端点（Server-Sent Events）
chatRouter.post("/stream", async (req: Request, res: Response): Promise<void> => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  try {
    const chatReq = req.body as ChatRequest;

    for await (const chunk of chatStream(chatReq)) {
      res.write(`data: ${JSON.stringify({ text: chunk })}\n\n`);
    }

    res.write("data: [DONE]\n\n");
    res.end();
  } catch (err) {
    const message = err instanceof Error ? err.message : "未知错误";
    res.write(`data: ${JSON.stringify({ error: message })}\n\n`);
    res.end();
  }
});
```

### esbuild 构建脚本

```typescript
// scripts/build.ts
import * as esbuild from "esbuild";
import { nodeExternalsPlugin } from "esbuild-node-externals";
import { execSync } from "child_process";

const isProduction = process.env.NODE_ENV === "production";

async function typecheck(): Promise<void> {
  console.log("正在执行类型检查...");
  execSync("tsc --noEmit", { stdio: "inherit" });
  console.log("类型检查通过");
}

async function bundle(): Promise<void> {
  console.log("正在打包...");

  const result = await esbuild.build({
    entryPoints: { server: "src/server.ts" },
    outdir: "dist",
    format: "esm",
    platform: "node",
    target: "node18",
    bundle: true,
    minify: isProduction,
    sourcemap: !isProduction,
    treeShaking: true,
    plugins: [nodeExternalsPlugin()],
    alias: {
      "@ai": "./src/ai",
      "@utils": "./src/utils",
    },
    metafile: isProduction,
  });

  if (result.metafile) {
    const analysis = await esbuild.analyzeMetafile(result.metafile);
    console.log("\n=== Bundle 分析 ===");
    console.log(analysis);
  }

  console.log("打包完成");
}

async function main(): Promise<void> {
  try {
    if (isProduction) {
      await typecheck();  // 生产构建时先做类型检查
    }
    await bundle();
    console.log("构建成功");
  } catch (err) {
    console.error("构建失败：", err);
    process.exit(1);
  }
}

main();
```

### package.json 完整脚本配置

```json
{
  "name": "ai-chat-service",
  "version": "1.0.0",
  "type": "module",
  "engines": {
    "node": ">=18.0.0"
  },
  "scripts": {
    "dev": "tsx watch src/server.ts",
    "typecheck": "tsc --noEmit",
    "build": "tsx scripts/build.ts",
    "build:prod": "NODE_ENV=production tsx scripts/build.ts",
    "start": "node dist/server.js",
    "test": "vitest run",
    "test:watch": "vitest",
    "ci": "npm run typecheck && npm run build:prod && npm test",
    "clean": "rm -rf dist"
  },
  "dependencies": {
    "@anthropic-ai/sdk": "^0.35.0",
    "express": "^4.18.0"
  },
  "devDependencies": {
    "@types/express": "^4.17.0",
    "@types/node": "^20.0.0",
    "esbuild": "^0.24.0",
    "esbuild-node-externals": "^1.14.0",
    "tsx": "^4.0.0",
    "typescript": "^5.7.0",
    "vitest": "^2.0.0"
  }
}
```

### 开发与构建工作流总结

```
开发阶段
┌─────────────────────────────────────────┐
│  tsx watch src/server.ts                │
│  → esbuild 极速转译                     │
│  → 文件变化时自动重启                   │
│  → VS Code 调试器可附加                 │
└─────────────────────────────────────────┘

代码提交前
┌─────────────────────────────────────────┐
│  tsc --noEmit                           │
│  → 完整类型检查                         │
│  → 确保零类型错误                       │
└─────────────────────────────────────────┘

CI/CD 生产构建
┌─────────────────────────────────────────┐
│  1. tsc --noEmit    （类型检查）        │
│  2. vitest run      （单元测试）        │
│  3. esbuild bundle  （极速打包）        │
│     - minify                            │
│     - tree-shaking                      │
│     - sourcemap                         │
│  4. node dist/server.js （验证启动）    │
└─────────────────────────────────────────┘
```

---

## 练习题

### 练习 14-1（基础）

**题目**：创建一个 `tsconfig.json`，满足以下要求：
1. 目标为 Node.js 18+（`target: "ES2022"`，`module: "Node16"`）
2. 开启所有 `strict` 模式
3. 将 `src/` 目录下的所有文件纳入编译，排除 `dist/` 和 `node_modules/`
4. 配置路径别名：`@utils/*` 映射到 `src/utils/*`，`@ai/*` 映射到 `src/ai/*`
5. 输出目录为 `dist/`，并生成 source map 和声明文件

---

### 练习 14-2（基础）

**题目**：解释以下三种运行 TypeScript 文件方式的区别，分别适用于哪些场景：

```bash
# 方式 A
npx tsc && node dist/server.js

# 方式 B
npx ts-node src/server.ts

# 方式 C
npx tsx watch src/server.ts
```

---

### 练习 14-3（中级）

**题目**：编写一个 esbuild 构建脚本 `scripts/build.ts`，要求：
1. 入口文件为 `src/index.ts`，输出到 `dist/index.js`
2. 格式为 ESM，平台为 Node.js，目标为 Node 18
3. 启用 bundle、minify（仅生产环境）、sourcemap、treeShaking
4. 将所有 `node_modules` 标记为外部依赖（不打入 bundle）
5. 配置路径别名：`@utils` 指向 `./src/utils`

---

### 练习 14-4（中级）

**题目**：在 Vite + React 项目中，你引入了一个体积较大的 AI 图表组件 `AIChartDashboard`。请：
1. 使用 `React.lazy` 和动态 `import()` 对该组件实现代码分割
2. 用 `Suspense` 提供加载态
3. 在 `vite.config.ts` 中，将所有 `@anthropic-ai/sdk` 相关代码单独打入名为 `ai-sdk` 的 chunk

---

### 练习 14-5（高级）

**题目**：构建一个完整的 monorepo 构建配置，包含两个包：
- `packages/shared`：共享类型定义（`ModelConfig`、`ChatMessage`）
- `packages/backend`：依赖 `shared` 包，提供 AI API 服务

要求：
1. `shared` 包使用 `composite: true` 支持项目引用
2. `backend` 包通过 `references` 引用 `shared` 包
3. 根目录提供汇总的 `tsconfig.json`
4. 编写 `package.json` 脚本，支持 `build:all`（增量构建全部包）和 `typecheck:all`（检查全部包类型）

---

## 练习答案

### 练习 14-1 答案

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "declaration": true,
    "sourceMap": true,
    "baseUrl": ".",
    "paths": {
      "@utils/*": ["src/utils/*"],
      "@ai/*": ["src/ai/*"]
    }
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

**关键点说明**：
- `module: "Node16"` 配合 `moduleResolution: "Node16"` 是 Node.js ESM 的正确组合
- `baseUrl: "."` 是 `paths` 生效的前提，基准目录为 `tsconfig.json` 所在目录
- `declaration: true` 生成 `.d.ts` 文件，供其他包或工具消费
- `sourceMap: true` 使调试器可以定位到 TypeScript 源码行

---

### 练习 14-2 答案

| | 方式 A（tsc + node） | 方式 B（ts-node） | 方式 C（tsx watch） |
|--|--|--|--|
| 步骤 | 先编译再运行 | 边转译边运行 | 边转译边运行 |
| 类型检查 | 有（编译阶段） | 可选 | 无 |
| 速度 | 慢（全量编译） | 中 | 极快 |
| 热重载 | 无（需手动重编译） | 无 | 有（watch 模式） |
| 适用场景 | 生产部署前验证 | 一次性脚本 | 日常开发服务器 |

**补充说明**：
- 方式 A 适合 **CI 构建验证**，能确保类型零错误才运行
- 方式 B 适合 **运行一次性数据处理脚本**（如数据库迁移、API 测试）
- 方式 C 适合 **日常开发**，文件保存后服务立即重启，反馈最快

---

### 练习 14-3 答案

```typescript
// scripts/build.ts
import * as esbuild from "esbuild";
import { nodeExternalsPlugin } from "esbuild-node-externals";

const isProduction = process.env.NODE_ENV === "production";

async function build(): Promise<void> {
  const result = await esbuild.build({
    // 入口与输出
    entryPoints: { index: "src/index.ts" },
    outdir: "dist",
    outExtension: { ".js": ".js" },

    // 格式与平台
    format: "esm",
    platform: "node",
    target: "node18",

    // 打包策略
    bundle: true,
    minify: isProduction,
    sourcemap: true,
    treeShaking: true,

    // 将 node_modules 标记为外部依赖
    plugins: [nodeExternalsPlugin()],

    // 路径别名（对应 tsconfig.json paths）
    alias: {
      "@utils": "./src/utils",
    },
  });

  console.log("构建完成", isProduction ? "（生产模式）" : "（开发模式）");
}

build().catch((err) => {
  console.error(err);
  process.exit(1);
});
```

---

### 练习 14-4 答案

```typescript
// src/App.tsx
import React, { Suspense, lazy } from "react";

// 代码分割：AIChartDashboard 单独打包为独立 chunk
const AIChartDashboard = lazy(
  () => import("./components/AIChartDashboard")
);

function LoadingFallback(): React.ReactElement {
  return (
    <div style={{ padding: "2rem", textAlign: "center" }}>
      AI 图表加载中...
    </div>
  );
}

export function App(): React.ReactElement {
  return (
    <div>
      <h1>AI 数据分析平台</h1>
      <Suspense fallback={<LoadingFallback />}>
        <AIChartDashboard />
      </Suspense>
    </div>
  );
}
```

```typescript
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          // AI SDK 单独打包
          if (id.includes("@anthropic-ai/sdk")) {
            return "ai-sdk";
          }
          // React 生态单独打包
          if (id.includes("node_modules/react")) {
            return "react-vendor";
          }
        },
      },
    },
  },
});
```

---

### 练习 14-5 答案

**`packages/shared/tsconfig.json`**：

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "composite": true,
    "declaration": true,
    "declarationMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true
  },
  "include": ["src/**/*"]
}
```

**`packages/shared/src/types.ts`**：

```typescript
export interface ModelConfig {
  model: string;
  maxTokens: number;
  temperature: number;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}
```

**`packages/backend/tsconfig.json`**：

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "composite": true,
    "declaration": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true
  },
  "include": ["src/**/*"],
  "references": [
    { "path": "../shared" }
  ]
}
```

**`packages/backend/src/service.ts`**：

```typescript
// 直接引用 shared 包的类型（项目引用确保类型安全）
import type { ModelConfig, ChatMessage } from "../../shared/src/types";

export async function runChat(
  config: ModelConfig,
  messages: ChatMessage[]
): Promise<string> {
  // 实现省略
  return `使用模型 ${config.model} 处理了 ${messages.length} 条消息`;
}
```

**根目录 `tsconfig.json`**：

```json
{
  "files": [],
  "references": [
    { "path": "packages/shared" },
    { "path": "packages/backend" }
  ]
}
```

**根目录 `package.json`**：

```json
{
  "name": "ai-monorepo",
  "private": true,
  "scripts": {
    "build:all": "tsc --build",
    "build:all:force": "tsc --build --force",
    "typecheck:all": "tsc --build --noEmit",
    "clean:all": "tsc --build --clean",
    "dev:backend": "tsx watch packages/backend/src/server.ts"
  },
  "devDependencies": {
    "typescript": "^5.7.0",
    "tsx": "^4.0.0"
  }
}
```

**关键点说明**：
- `composite: true` 必须与 `declaration: true` 同时设置，否则项目引用无法工作
- `tsc --build` 会自动按依赖顺序编译（先 `shared` 再 `backend`），且支持增量构建
- `--noEmit` 与 `--build` 组合使用时，仅做类型检查，不生成输出文件，适合 CI 快速校验
- 如需在 `backend` 中使用路径别名引用 `shared`，还需配合打包工具的 `alias` 配置

---
