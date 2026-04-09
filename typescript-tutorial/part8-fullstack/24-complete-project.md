# 第24章：完整项目实战

## 学习目标

完成本章学习后，你将能够：

1. 设计并实现一个基于 Monorepo 的全栈 AI 聊天应用，掌握前后端代码共享的架构模式
2. 使用 pnpm workspace 管理多包项目，配置共享类型、API 契约与构建流水线
3. 定义严格的 API 契约与共享类型，确保前后端类型一致性与接口稳定性
4. 完成生产级部署配置，包括环境变量管理、Docker 容器化与 CI/CD 流水线
5. 集成结构化日志与监控体系，实现可观测性与线上问题的快速定位

---

## 24.1 项目架构设计

### 全栈 AI 应用的架构全貌

完整的 AI 聊天应用由多个相互协作的层次组成。在本章，我们将构建一个名为 **TypeChat** 的 AI 聊天平台，它具备以下能力：

- 多会话管理（创建、切换、删除对话）
- 流式消息响应（打字机效果）
- 用户认证（JWT）
- 多模型支持（OpenAI、Anthropic）
- 完整的日志与监控

**整体架构图：**

```
┌──────────────────────────────────────────────────────────────┐
│                       TypeChat Monorepo                      │
│                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────┐   │
│  │  packages/  │   │    apps/    │   │   packages/      │   │
│  │   shared    │   │   backend   │   │     ui           │   │
│  │  (types,    │◄──│  (Fastify + │   │  (React +        │   │
│  │   utils,    │   │   OpenAI)   │   │   Tailwind)      │   │
│  │   schemas)  │   │             │   │                  │   │
│  └─────────────┘   └──────┬──────┘   └────────┬─────────┘   │
│                           │                   │             │
│                    ┌──────▼──────────────────▼──────┐       │
│                    │         packages/config         │       │
│                    │   (tsconfig, eslint, tailwind)  │       │
│                    └─────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘

外部依赖：
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ OpenAI   │  │Anthropic │  │PostgreSQL│  │  Redis   │
  │   API    │  │   API    │  │ (会话存储)│  │ (速率限制)│
  └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

### 技术选型决策

| 层次 | 技术选型 | 选型理由 |
|------|---------|---------|
| 包管理 | pnpm workspace | 磁盘高效、依赖提升可控、速度快 |
| 后端框架 | Fastify | 高性能、原生 TypeScript 支持、Schema 验证内置 |
| 前端框架 | React 18 + Vite | 流式渲染支持、快速 HMR、生态成熟 |
| 类型共享 | packages/shared | 单一来源，前后端类型零漂移 |
| 数据库 | PostgreSQL + Drizzle ORM | 强类型 ORM、迁移工具完善 |
| 认证 | JWT + Refresh Token | 无状态、易水平扩展 |
| 部署 | Docker + GitHub Actions | 可重现构建、自动化 CI/CD |
| 监控 | Pino + OpenTelemetry | 结构化日志、分布式追踪 |

### 目录结构规划

```
typechat/
├── package.json                 # 根 package.json（workspace 配置）
├── pnpm-workspace.yaml          # pnpm workspace 声明
├── turbo.json                   # Turborepo 构建缓存配置
├── docker-compose.yml           # 本地开发环境
├── .env.example                 # 环境变量示例
│
├── packages/
│   ├── shared/                  # 共享类型、工具函数、Schema
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   └── src/
│   │       ├── types/           # 核心类型定义
│   │       │   ├── api.ts       # API 请求/响应类型
│   │       │   ├── chat.ts      # 聊天相关类型
│   │       │   └── user.ts      # 用户类型
│   │       ├── schemas/         # Zod 验证 Schema
│   │       │   ├── chat.ts
│   │       │   └── auth.ts
│   │       └── utils/           # 前后端通用工具
│   │           ├── errors.ts
│   │           └── constants.ts
│   │
│   └── config/                  # 共享配置
│       ├── tsconfig/
│       │   ├── base.json
│       │   ├── react.json
│       │   └── node.json
│       └── eslint/
│           └── index.js
│
├── apps/
│   ├── backend/                 # Fastify 后端服务
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   ├── Dockerfile
│   │   └── src/
│   │       ├── main.ts          # 入口
│   │       ├── app.ts           # Fastify 实例配置
│   │       ├── routes/          # 路由层
│   │       ├── services/        # 业务逻辑层
│   │       ├── repositories/    # 数据访问层
│   │       ├── middleware/      # 认证、日志中间件
│   │       └── config/          # 环境配置
│   │
│   └── frontend/                # React 前端应用
│       ├── package.json
│       ├── tsconfig.json
│       ├── vite.config.ts
│       └── src/
│           ├── main.tsx
│           ├── App.tsx
│           ├── components/      # UI 组件
│           ├── hooks/           # 自定义 Hooks
│           ├── stores/          # 状态管理（Zustand）
│           ├── api/             # API 客户端层
│           └── types/           # 前端专属类型
│
└── scripts/
    ├── setup.sh                 # 初始化脚本
    └── deploy.sh                # 部署脚本
```

---

## 24.2 Monorepo 配置（pnpm workspace）

### 根级配置文件

**`pnpm-workspace.yaml`** — 声明工作空间范围：

```yaml
packages:
  - 'apps/*'
  - 'packages/*'
```

**根 `package.json`** — 工作空间脚本与工具依赖：

```json
{
  "name": "typechat",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "turbo run dev --parallel",
    "build": "turbo run build",
    "test": "turbo run test",
    "lint": "turbo run lint",
    "type-check": "turbo run type-check",
    "clean": "turbo run clean && rm -rf node_modules",
    "db:migrate": "pnpm --filter backend db:migrate",
    "db:seed": "pnpm --filter backend db:seed"
  },
  "devDependencies": {
    "turbo": "^2.0.0",
    "typescript": "^5.4.0",
    "@types/node": "^20.0.0",
    "eslint": "^8.57.0",
    "prettier": "^3.2.0"
  },
  "engines": {
    "node": ">=18.0.0",
    "pnpm": ">=8.0.0"
  }
}
```

**`turbo.json`** — 构建流水线配置（利用增量缓存）：

```json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": [".env"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "test": {
      "dependsOn": ["^build"],
      "outputs": ["coverage/**"]
    },
    "type-check": {
      "dependsOn": ["^build"]
    },
    "lint": {
      "outputs": []
    },
    "clean": {
      "cache": false
    }
  }
}
```

### 共享 tsconfig 配置

**`packages/config/tsconfig/base.json`** — 所有包的基础 TypeScript 配置：

```json
{
  "$schema": "https://json.schemastore.org/tsconfig",
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["ES2022"],
    "strict": true,
    "exactOptionalPropertyTypes": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  }
}
```

**`packages/config/tsconfig/node.json`** — Node.js 服务专用：

```json
{
  "$schema": "https://json.schemastore.org/tsconfig",
  "extends": "./base.json",
  "compilerOptions": {
    "outDir": "dist",
    "rootDir": "src"
  }
}
```

**`packages/config/tsconfig/react.json`** — React 应用专用：

```json
{
  "$schema": "https://json.schemastore.org/tsconfig",
  "extends": "./base.json",
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx",
    "allowImportingTsExtensions": true,
    "noEmit": true
  }
}
```

### shared 包配置

**`packages/shared/package.json`**：

```json
{
  "name": "@typechat/shared",
  "version": "1.0.0",
  "private": true,
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "types": "./dist/index.d.ts"
    },
    "./types": {
      "import": "./dist/types/index.js",
      "types": "./dist/types/index.d.ts"
    },
    "./schemas": {
      "import": "./dist/schemas/index.js",
      "types": "./dist/schemas/index.d.ts"
    }
  },
  "scripts": {
    "build": "tsc -p tsconfig.json",
    "dev": "tsc -p tsconfig.json --watch",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "zod": "^3.22.0"
  },
  "devDependencies": {
    "@typechat/config": "workspace:*",
    "typescript": "^5.4.0"
  }
}
```

---

## 24.3 共享类型与 API 契约

### 核心类型定义

**`packages/shared/src/types/chat.ts`** — 聊天领域核心类型：

```typescript
// 消息角色
export type MessageRole = 'user' | 'assistant' | 'system';

// 单条消息
export interface Message {
  id: string;
  conversationId: string;
  role: MessageRole;
  content: string;
  createdAt: string;          // ISO 8601
  tokenCount?: number;
  modelId?: string;
}

// 对话会话
export interface Conversation {
  id: string;
  userId: string;
  title: string;
  modelId: string;
  systemPrompt?: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
  messageCount: number;
  totalTokens: number;
}

// 对话列表项（不含消息，用于侧边栏）
export type ConversationSummary = Omit<Conversation, 'messages'>;

// 支持的模型
export type ModelId =
  | 'gpt-4o'
  | 'gpt-4o-mini'
  | 'gpt-4-turbo'
  | 'claude-3-5-sonnet-20241022'
  | 'claude-3-haiku-20240307';

export interface ModelInfo {
  id: ModelId;
  name: string;
  provider: 'openai' | 'anthropic';
  contextWindow: number;
  inputCostPer1K: number;   // USD
  outputCostPer1K: number;  // USD
  supportsVision: boolean;
}

// 流式响应事件（SSE 数据结构）
export type StreamEvent =
  | { type: 'start'; messageId: string }
  | { type: 'delta'; content: string }
  | { type: 'done'; tokenCount: number; cost: number }
  | { type: 'error'; code: string; message: string };
```

**`packages/shared/src/types/api.ts`** — API 请求/响应类型契约：

```typescript
import type { Conversation, ConversationSummary, Message, ModelId } from './chat.js';
import type { User } from './user.js';

// 通用 API 响应包装
export interface ApiResponse<T> {
  success: true;
  data: T;
  meta?: {
    page?: number;
    total?: number;
    hasMore?: boolean;
  };
}

export interface ApiError {
  success: false;
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
}

export type ApiResult<T> = ApiResponse<T> | ApiError;

// ── 认证 API ──────────────────────────────────────────────
export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  user: User;
  accessToken: string;
  refreshToken: string;
  expiresAt: number;    // Unix timestamp
}

export interface RegisterRequest {
  email: string;
  password: string;
  displayName: string;
}

// ── 对话 API ──────────────────────────────────────────────
export interface CreateConversationRequest {
  title?: string;
  modelId: ModelId;
  systemPrompt?: string;
}

export interface SendMessageRequest {
  conversationId: string;
  content: string;
  stream?: boolean;
}

export interface ListConversationsRequest {
  page?: number;
  limit?: number;
  search?: string;
}

export interface ListConversationsResponse {
  conversations: ConversationSummary[];
  total: number;
  hasMore: boolean;
}

// ── API 路径常量（前后端共用）────────────────────────────
export const API_ROUTES = {
  AUTH: {
    LOGIN: '/api/auth/login',
    REGISTER: '/api/auth/register',
    LOGOUT: '/api/auth/logout',
    REFRESH: '/api/auth/refresh',
    ME: '/api/auth/me',
  },
  CONVERSATIONS: {
    LIST: '/api/conversations',
    CREATE: '/api/conversations',
    GET: (id: string) => `/api/conversations/${id}` as const,
    DELETE: (id: string) => `/api/conversations/${id}` as const,
    MESSAGES: (id: string) => `/api/conversations/${id}/messages` as const,
    STREAM: (id: string) => `/api/conversations/${id}/stream` as const,
  },
  MODELS: {
    LIST: '/api/models',
  },
} as const;
```

### Zod Schema 验证

在 shared 包中定义 Zod Schema，后端用于验证请求体，前端用于表单校验：

**`packages/shared/src/schemas/chat.ts`**：

```typescript
import { z } from 'zod';

export const ModelIdSchema = z.enum([
  'gpt-4o',
  'gpt-4o-mini',
  'gpt-4-turbo',
  'claude-3-5-sonnet-20241022',
  'claude-3-haiku-20240307',
]);

export const CreateConversationSchema = z.object({
  title: z.string().min(1).max(200).optional(),
  modelId: ModelIdSchema,
  systemPrompt: z.string().max(4000).optional(),
});

export const SendMessageSchema = z.object({
  content: z.string().min(1).max(32000),
  stream: z.boolean().default(true),
});

export const ListConversationsSchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
  search: z.string().max(200).optional(),
});

// 从 Zod Schema 推导 TypeScript 类型（单一来源原则）
export type CreateConversationInput = z.infer<typeof CreateConversationSchema>;
export type SendMessageInput = z.infer<typeof SendMessageSchema>;
export type ListConversationsQuery = z.infer<typeof ListConversationsSchema>;
```

**`packages/shared/src/schemas/auth.ts`**：

```typescript
import { z } from 'zod';

export const LoginSchema = z.object({
  email: z.string().email('邮箱格式不正确'),
  password: z.string().min(8, '密码至少8位'),
});

export const RegisterSchema = z.object({
  email: z.string().email('邮箱格式不正确'),
  password: z
    .string()
    .min(8, '密码至少8位')
    .regex(/[A-Z]/, '密码须包含大写字母')
    .regex(/[0-9]/, '密码须包含数字'),
  displayName: z.string().min(2).max(50),
});

export type LoginInput = z.infer<typeof LoginSchema>;
export type RegisterInput = z.infer<typeof RegisterSchema>;
```

### 错误码规范

**`packages/shared/src/utils/errors.ts`**：

```typescript
// 业务错误码枚举（字符串常量，便于日志搜索）
export const ErrorCodes = {
  // 认证错误 (AUTH_xxx)
  AUTH_INVALID_CREDENTIALS: 'AUTH_INVALID_CREDENTIALS',
  AUTH_TOKEN_EXPIRED: 'AUTH_TOKEN_EXPIRED',
  AUTH_TOKEN_INVALID: 'AUTH_TOKEN_INVALID',
  AUTH_INSUFFICIENT_PERMISSIONS: 'AUTH_INSUFFICIENT_PERMISSIONS',

  // 资源错误 (RESOURCE_xxx)
  RESOURCE_NOT_FOUND: 'RESOURCE_NOT_FOUND',
  RESOURCE_ALREADY_EXISTS: 'RESOURCE_ALREADY_EXISTS',
  RESOURCE_FORBIDDEN: 'RESOURCE_FORBIDDEN',

  // AI 错误 (AI_xxx)
  AI_PROVIDER_UNAVAILABLE: 'AI_PROVIDER_UNAVAILABLE',
  AI_RATE_LIMIT_EXCEEDED: 'AI_RATE_LIMIT_EXCEEDED',
  AI_CONTEXT_LENGTH_EXCEEDED: 'AI_CONTEXT_LENGTH_EXCEEDED',
  AI_CONTENT_FILTERED: 'AI_CONTENT_FILTERED',

  // 验证错误 (VALIDATION_xxx)
  VALIDATION_FAILED: 'VALIDATION_FAILED',

  // 系统错误 (SYSTEM_xxx)
  SYSTEM_INTERNAL_ERROR: 'SYSTEM_INTERNAL_ERROR',
} as const;

export type ErrorCode = (typeof ErrorCodes)[keyof typeof ErrorCodes];

// 业务异常基类（后端抛出，经中间件转换为 ApiError 响应）
export class AppError extends Error {
  constructor(
    public readonly code: ErrorCode,
    message: string,
    public readonly statusCode: number = 400,
    public readonly details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = 'AppError';
  }

  static notFound(resource: string): AppError {
    return new AppError(ErrorCodes.RESOURCE_NOT_FOUND, `${resource} 不存在`, 404);
  }

  static forbidden(message = '无权访问此资源'): AppError {
    return new AppError(ErrorCodes.RESOURCE_FORBIDDEN, message, 403);
  }

  static aiRateLimit(): AppError {
    return new AppError(ErrorCodes.AI_RATE_LIMIT_EXCEEDED, 'AI 请求频率超限，请稍后重试', 429);
  }
}
```

---

## 24.4 部署与环境配置

### 环境变量管理

**`.env.example`** — 完整的环境变量说明：

```bash
# ─── 应用配置 ───────────────────────────────────────
NODE_ENV=development          # development | production | test
APP_PORT=3000
APP_HOST=0.0.0.0
APP_URL=http://localhost:3000
FRONTEND_URL=http://localhost:5173

# ─── 数据库 ─────────────────────────────────────────
DATABASE_URL=postgresql://user:password@localhost:5432/typechat
DATABASE_POOL_MIN=2
DATABASE_POOL_MAX=10

# ─── Redis ──────────────────────────────────────────
REDIS_URL=redis://localhost:6379

# ─── JWT ────────────────────────────────────────────
JWT_ACCESS_SECRET=your-super-secret-access-key-min-32-chars
JWT_REFRESH_SECRET=your-super-secret-refresh-key-min-32-chars
JWT_ACCESS_EXPIRES_IN=15m
JWT_REFRESH_EXPIRES_IN=30d

# ─── AI 提供商 ───────────────────────────────────────
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=org-...          # 可选
ANTHROPIC_API_KEY=sk-ant-...

# ─── 速率限制 ────────────────────────────────────────
RATE_LIMIT_WINDOW_MS=60000    # 1分钟窗口
RATE_LIMIT_MAX_REQUESTS=60    # 每窗口最大请求数

# ─── 日志 ────────────────────────────────────────────
LOG_LEVEL=info                # trace | debug | info | warn | error
LOG_PRETTY=true               # 开发时启用美化输出
```

**`apps/backend/src/config/env.ts`** — 类型安全的环境变量解析：

```typescript
import { z } from 'zod';

const EnvSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  APP_PORT: z.coerce.number().default(3000),
  APP_HOST: z.string().default('0.0.0.0'),
  APP_URL: z.string().url(),
  FRONTEND_URL: z.string().url(),

  DATABASE_URL: z.string().url(),
  DATABASE_POOL_MIN: z.coerce.number().default(2),
  DATABASE_POOL_MAX: z.coerce.number().default(10),

  REDIS_URL: z.string().url().optional(),

  JWT_ACCESS_SECRET: z.string().min(32),
  JWT_REFRESH_SECRET: z.string().min(32),
  JWT_ACCESS_EXPIRES_IN: z.string().default('15m'),
  JWT_REFRESH_EXPIRES_IN: z.string().default('30d'),

  OPENAI_API_KEY: z.string().startsWith('sk-').optional(),
  ANTHROPIC_API_KEY: z.string().startsWith('sk-ant-').optional(),

  RATE_LIMIT_WINDOW_MS: z.coerce.number().default(60_000),
  RATE_LIMIT_MAX_REQUESTS: z.coerce.number().default(60),

  LOG_LEVEL: z.enum(['trace', 'debug', 'info', 'warn', 'error']).default('info'),
  LOG_PRETTY: z.coerce.boolean().default(false),
});

// 启动时验证，配置错误立即崩溃（fail fast）
const parsed = EnvSchema.safeParse(process.env);
if (!parsed.success) {
  console.error('环境变量配置错误：');
  console.error(parsed.error.flatten().fieldErrors);
  process.exit(1);
}

export const env = parsed.data;
export type Env = typeof env;
```

### Docker 配置

**`apps/backend/Dockerfile`** — 多阶段构建：

```dockerfile
# ─── 构建阶段 ───────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app

# 安装 pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

# 复制 workspace 配置
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY packages/config/package.json ./packages/config/
COPY packages/shared/package.json ./packages/shared/
COPY apps/backend/package.json ./apps/backend/

# 安装依赖（仅生产+构建所需）
RUN pnpm install --frozen-lockfile

# 复制源码并构建
COPY packages/ ./packages/
COPY apps/backend/ ./apps/backend/

RUN pnpm --filter @typechat/shared build
RUN pnpm --filter backend build

# 裁剪 node_modules（仅保留生产依赖）
RUN pnpm deploy --filter backend --prod /prod/backend

# ─── 运行阶段 ───────────────────────────────────────
FROM node:20-alpine AS runner
WORKDIR /app

# 创建非 root 用户
RUN addgroup --system --gid 1001 nodejs \
  && adduser --system --uid 1001 fastify

# 复制构建产物与精简依赖
COPY --from=builder --chown=fastify:nodejs /prod/backend ./
COPY --from=builder --chown=fastify:nodejs /app/apps/backend/dist ./dist

USER fastify
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD wget -qO- http://localhost:3000/health || exit 1

CMD ["node", "dist/main.js"]
```

**`docker-compose.yml`** — 本地开发环境：

```yaml
version: '3.9'

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: typechat
      POSTGRES_PASSWORD: typechat_dev
      POSTGRES_DB: typechat
    ports:
      - '5432:5432'
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ['CMD-SHELL', 'pg_isready -U typechat']
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - '6379:6379'
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  backend:
    build:
      context: .
      dockerfile: apps/backend/Dockerfile
    ports:
      - '3000:3000'
    env_file: .env
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - ./apps/backend/src:/app/src:ro  # 开发时挂载源码

volumes:
  postgres_data:
  redis_data:
```

### GitHub Actions CI/CD

**`.github/workflows/deploy.yml`**：

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/backend

jobs:
  quality:
    name: Type Check & Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v3
        with:
          version: 8
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'pnpm'
      - run: pnpm install --frozen-lockfile
      - run: pnpm type-check
      - run: pnpm lint

  test:
    name: Test
    runs-on: ubuntu-latest
    needs: quality
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: typechat_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v3
        with:
          version: 8
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'pnpm'
      - run: pnpm install --frozen-lockfile
      - run: pnpm test
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/typechat_test

  build-and-push:
    name: Build & Push Image
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: apps/backend/Dockerfile
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## 24.5 监控与日志

### 结构化日志体系

使用 Pino 实现高性能结构化日志，每条日志都是 JSON，便于 ELK/Loki 等日志系统解析：

**`apps/backend/src/config/logger.ts`**：

```typescript
import pino, { type Logger } from 'pino';
import { env } from './env.js';

// 日志上下文接口（所有请求日志必须包含的字段）
export interface LogContext {
  requestId: string;
  userId?: string;
  conversationId?: string;
  modelId?: string;
  durationMs?: number;
}

// 创建根 Logger
export const logger: Logger = pino({
  level: env.LOG_LEVEL,
  // 生产环境输出 JSON，开发环境输出美化文本
  transport: env.LOG_PRETTY
    ? {
        target: 'pino-pretty',
        options: {
          colorize: true,
          translateTime: 'SYS:standard',
          ignore: 'pid,hostname',
        },
      }
    : undefined,
  // 基础字段，每条日志都会携带
  base: {
    service: 'typechat-backend',
    version: process.env['npm_package_version'] ?? 'unknown',
    env: env.NODE_ENV,
  },
  // 字段重命名（符合 ECS 规范）
  formatters: {
    level(label) {
      return { level: label };
    },
  },
  // 时间戳格式
  timestamp: pino.stdTimeFunctions.isoTime,
  // 序列化 Error 对象
  serializers: {
    err: pino.stdSerializers.err,
  },
});

// 创建带上下文的子 Logger（用于单次请求生命周期）
export function createRequestLogger(context: LogContext): Logger {
  return logger.child(context);
}
```

**`apps/backend/src/middleware/request-logger.ts`** — Fastify 请求日志中间件：

```typescript
import type { FastifyInstance } from 'fastify';
import { randomUUID } from 'node:crypto';
import { createRequestLogger } from '../config/logger.js';

export async function requestLoggerMiddleware(app: FastifyInstance): Promise<void> {
  // 为每个请求注入唯一 ID 和专属 Logger
  app.addHook('onRequest', async (request, _reply) => {
    const requestId = (request.headers['x-request-id'] as string) ?? randomUUID();
    request.id = requestId;
    request.log = createRequestLogger({
      requestId,
      userId: undefined, // 认证后填充
    });

    request.log.info({
      msg: 'request_start',
      method: request.method,
      url: request.url,
      ip: request.ip,
    });
  });

  // 记录响应结果和耗时
  app.addHook('onResponse', async (request, reply) => {
    request.log.info({
      msg: 'request_end',
      method: request.method,
      url: request.url,
      statusCode: reply.statusCode,
      durationMs: Math.round(reply.elapsedTime),
    });
  });

  // 记录未捕获的错误
  app.addHook('onError', async (request, _reply, error) => {
    request.log.error({
      msg: 'request_error',
      err: error,
      method: request.method,
      url: request.url,
    });
  });
}
```

### 关键指标监控

**`apps/backend/src/services/metrics.ts`** — 自定义业务指标收集：

```typescript
// 简单的内存指标收集器（生产环境可替换为 Prometheus）
interface Counter {
  increment(labels?: Record<string, string>): void;
  value(labels?: Record<string, string>): number;
}

interface Histogram {
  observe(value: number, labels?: Record<string, string>): void;
  percentile(p: number, labels?: Record<string, string>): number;
}

class MetricsRegistry {
  private counters = new Map<string, Map<string, number>>();
  private histograms = new Map<string, number[]>();

  counter(name: string): Counter {
    if (!this.counters.has(name)) {
      this.counters.set(name, new Map());
    }
    const store = this.counters.get(name)!;
    const labelKey = (labels?: Record<string, string>) =>
      labels ? JSON.stringify(labels) : '_default';

    return {
      increment: (labels) => {
        const key = labelKey(labels);
        store.set(key, (store.get(key) ?? 0) + 1);
      },
      value: (labels) => store.get(labelKey(labels)) ?? 0,
    };
  }

  histogram(name: string): Histogram {
    if (!this.histograms.has(name)) {
      this.histograms.set(name, []);
    }
    const values = this.histograms.get(name)!;

    return {
      observe: (value) => {
        values.push(value);
        // 只保留最近 10000 个样本
        if (values.length > 10_000) values.shift();
      },
      percentile: (p) => {
        if (values.length === 0) return 0;
        const sorted = [...values].sort((a, b) => a - b);
        const index = Math.ceil((p / 100) * sorted.length) - 1;
        return sorted[Math.max(0, index)] ?? 0;
      },
    };
  }
}

export const metrics = new MetricsRegistry();

// 预定义业务指标
export const chatMetrics = {
  requestsTotal: metrics.counter('chat_requests_total'),
  streamDurationMs: metrics.histogram('chat_stream_duration_ms'),
  tokensGenerated: metrics.counter('chat_tokens_generated_total'),
  errorsTotal: metrics.counter('chat_errors_total'),
};
```

**`apps/backend/src/routes/health.ts`** — 健康检查端点：

```typescript
import type { FastifyInstance } from 'fastify';
import { chatMetrics } from '../services/metrics.js';

export async function healthRoutes(app: FastifyInstance): Promise<void> {
  // 简单存活检查
  app.get('/health', async () => {
    return { status: 'ok', timestamp: new Date().toISOString() };
  });

  // 详细就绪检查（负载均衡器用于流量切换判断）
  app.get('/ready', async (request, reply) => {
    const checks: Record<string, 'ok' | 'fail'> = {};

    // TODO: 检查数据库、Redis 连接
    checks['database'] = 'ok';
    checks['redis'] = 'ok';

    const allOk = Object.values(checks).every((v) => v === 'ok');
    reply.status(allOk ? 200 : 503);

    return {
      status: allOk ? 'ready' : 'not_ready',
      checks,
      timestamp: new Date().toISOString(),
    };
  });

  // 指标端点（供 Prometheus/Grafana 抓取）
  app.get('/metrics', async () => {
    return {
      chat: {
        requests_total: chatMetrics.requestsTotal.value(),
        tokens_generated_total: chatMetrics.tokensGenerated.value(),
        errors_total: chatMetrics.errorsTotal.value(),
        stream_p50_ms: chatMetrics.streamDurationMs.percentile(50),
        stream_p95_ms: chatMetrics.streamDurationMs.percentile(95),
        stream_p99_ms: chatMetrics.streamDurationMs.percentile(99),
      },
    };
  });
}
```

---

## 本章小结

本章是全教程的收官之作，将前 23 章的所有知识融合为一个完整的生产级项目：

| 主题 | 核心要点 | 对应知识点 |
|------|---------|-----------|
| 项目架构 | Monorepo 分层设计，前后端代码共享 | 第9章模块系统 |
| Monorepo 配置 | pnpm workspace + Turborepo 增量构建 | 第13-14章工程化 |
| 共享类型 | @typechat/shared 包，Zod Schema 单一来源 | 第6章泛型、第10-11章高级类型 |
| API 契约 | 类型安全的路由常量、请求/响应接口 | 第4-5章类型系统 |
| 部署配置 | 多阶段 Docker 构建、环境变量验证 | 第13章 tsconfig |
| CI/CD | GitHub Actions 类型检查+测试+构建流水线 | 第15章测试 |
| 结构化日志 | Pino + 请求追踪 ID，便于生产排查 | 第8章装饰器 |
| 指标监控 | 业务指标收集、健康检查端点 | 第3章函数 |

---

## AI 应用实战：AI 聊天应用全流程

本节展示 TypeChat 的核心实现代码，将全部知识融汇贯通。

### 后端：流式聊天服务

**`apps/backend/src/services/chat.service.ts`**：

```typescript
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import type { Message, ModelId, StreamEvent } from '@typechat/shared/types';
import { ErrorCodes, AppError } from '@typechat/shared/utils/errors';
import { env } from '../config/env.js';
import { logger } from '../config/logger.js';
import { chatMetrics } from './metrics.js';

// 模型路由：根据 modelId 选择底层 Provider
type StreamOptions = {
  messages: Pick<Message, 'role' | 'content'>[];
  modelId: ModelId;
  systemPrompt?: string;
  signal?: AbortSignal;
  onEvent: (event: StreamEvent) => void | Promise<void>;
};

export class ChatService {
  private openai: OpenAI;
  private anthropic: Anthropic;

  constructor() {
    this.openai = new OpenAI({ apiKey: env.OPENAI_API_KEY });
    this.anthropic = new Anthropic({ apiKey: env.ANTHROPIC_API_KEY });
  }

  async streamChat(options: StreamOptions): Promise<void> {
    const { messages, modelId, systemPrompt, signal, onEvent } = options;
    const startTime = Date.now();

    try {
      if (modelId.startsWith('gpt-')) {
        await this.streamOpenAI({ messages, modelId, systemPrompt, signal, onEvent });
      } else if (modelId.startsWith('claude-')) {
        await this.streamAnthropic({ messages, modelId, systemPrompt, signal, onEvent });
      } else {
        throw new AppError(ErrorCodes.AI_PROVIDER_UNAVAILABLE, `不支持的模型：${modelId}`);
      }

      chatMetrics.streamDurationMs.observe(Date.now() - startTime, { modelId });
    } catch (error) {
      chatMetrics.errorsTotal.increment({ modelId });
      if (error instanceof AppError) throw error;

      // 处理 AI Provider 特有错误
      if (error instanceof OpenAI.APIError) {
        logger.warn({ msg: 'openai_api_error', status: error.status, code: error.code });
        if (error.status === 429) throw AppError.aiRateLimit();
        throw new AppError(ErrorCodes.AI_PROVIDER_UNAVAILABLE, 'OpenAI 服务暂时不可用', 503);
      }

      throw error;
    }
  }

  private async streamOpenAI(options: StreamOptions): Promise<void> {
    const { messages, modelId, systemPrompt, signal, onEvent } = options;
    let totalTokens = 0;

    const systemMessages: OpenAI.ChatCompletionMessageParam[] = systemPrompt
      ? [{ role: 'system', content: systemPrompt }]
      : [];

    const stream = await this.openai.chat.completions.create(
      {
        model: modelId,
        stream: true,
        messages: [
          ...systemMessages,
          ...messages.map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content })),
        ],
      },
      { signal },
    );

    const messageId = crypto.randomUUID();
    await onEvent({ type: 'start', messageId });

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content;
      if (delta) {
        await onEvent({ type: 'delta', content: delta });
        chatMetrics.tokensGenerated.increment({ modelId });
      }

      // 获取最终 token 用量
      if (chunk.usage) {
        totalTokens = chunk.usage.completion_tokens ?? 0;
      }
    }

    // 估算成本（以 gpt-4o-mini 为例：$0.15/1M input, $0.60/1M output）
    const cost = (totalTokens / 1_000_000) * 0.6;
    await onEvent({ type: 'done', tokenCount: totalTokens, cost });
  }

  private async streamAnthropic(options: StreamOptions): Promise<void> {
    const { messages, modelId, systemPrompt, signal, onEvent } = options;
    let totalTokens = 0;

    const stream = this.anthropic.messages.stream(
      {
        model: modelId,
        max_tokens: 4096,
        system: systemPrompt,
        messages: messages.map((m) => ({
          role: m.role as 'user' | 'assistant',
          content: m.content,
        })),
      },
      { signal },
    );

    const messageId = crypto.randomUUID();
    await onEvent({ type: 'start', messageId });

    stream.on('text', async (text: string) => {
      await onEvent({ type: 'delta', content: text });
    });

    const finalMessage = await stream.finalMessage();
    totalTokens = finalMessage.usage.output_tokens;

    const cost = (totalTokens / 1_000_000) * 15; // claude-3-5-sonnet 约 $15/1M output
    await onEvent({ type: 'done', tokenCount: totalTokens, cost });
  }
}

export const chatService = new ChatService();
```

### 后端：SSE 流式路由

**`apps/backend/src/routes/chat.ts`**：

```typescript
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import { SendMessageSchema } from '@typechat/shared/schemas';
import type { StreamEvent } from '@typechat/shared/types';
import { chatService } from '../services/chat.service.js';

export async function chatRoutes(app: FastifyInstance): Promise<void> {
  // POST /api/conversations/:id/stream — 流式消息
  app.post<{ Params: { id: string }; Body: z.infer<typeof SendMessageSchema> }>(
    '/api/conversations/:id/stream',
    {
      schema: {
        params: { type: 'object', properties: { id: { type: 'string' } }, required: ['id'] },
        body: SendMessageSchema,
      },
      preHandler: [app.authenticate],  // JWT 认证 hook
    },
    async (request, reply) => {
      const { id: conversationId } = request.params;
      const { content } = request.body;
      const userId = request.user.id;

      // 获取对话历史（省略具体实现）
      const conversation = await app.conversationRepo.findById(conversationId, userId);
      if (!conversation) {
        return reply.status(404).send({ success: false, error: { code: 'RESOURCE_NOT_FOUND', message: '对话不存在' } });
      }

      // 设置 SSE 响应头
      reply.raw.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',    // 禁用 Nginx 缓冲
      });

      const abortController = new AbortController();
      request.raw.on('close', () => abortController.abort());

      try {
        await chatService.streamChat({
          messages: [
            ...conversation.messages.map((m) => ({ role: m.role, content: m.content })),
            { role: 'user' as const, content },
          ],
          modelId: conversation.modelId,
          systemPrompt: conversation.systemPrompt,
          signal: abortController.signal,
          onEvent: (event: StreamEvent) => {
            // SSE 格式：data: <JSON>\n\n
            reply.raw.write(`data: ${JSON.stringify(event)}\n\n`);
          },
        });
      } finally {
        reply.raw.write('data: [DONE]\n\n');
        reply.raw.end();
      }
    },
  );
}
```

### 前端：流式聊天 Hook

**`apps/frontend/src/hooks/useStreamChat.ts`**：

```typescript
import { useState, useCallback, useRef } from 'react';
import type { StreamEvent } from '@typechat/shared/types';
import { API_ROUTES } from '@typechat/shared/types/api';

type ChatState = 'idle' | 'streaming' | 'done' | 'error';

interface UseStreamChatOptions {
  conversationId: string;
  onMessageComplete?: (content: string, tokenCount: number) => void;
}

export function useStreamChat({ conversationId, onMessageComplete }: UseStreamChatOptions) {
  const [state, setState] = useState<ChatState>('idle');
  const [streamingContent, setStreamingContent] = useState('');
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (content: string) => {
      setState('streaming');
      setStreamingContent('');
      setError(null);

      abortRef.current = new AbortController();

      try {
        const response = await fetch(API_ROUTES.CONVERSATIONS.STREAM(conversationId), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${localStorage.getItem('accessToken')}`,
          },
          body: JSON.stringify({ content }),
          signal: abortRef.current.signal,
        });

        if (!response.ok) {
          const err = await response.json() as { error: { message: string } };
          throw new Error(err.error.message);
        }

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let accumulated = '';
        let finalTokenCount = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6).trim();
            if (data === '[DONE]') break;

            const event: StreamEvent = JSON.parse(data);

            if (event.type === 'delta') {
              accumulated += event.content;
              setStreamingContent(accumulated);
            } else if (event.type === 'done') {
              finalTokenCount = event.tokenCount;
            } else if (event.type === 'error') {
              throw new Error(event.message);
            }
          }
        }

        setState('done');
        onMessageComplete?.(accumulated, finalTokenCount);
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          setState('idle');
          return;
        }
        setState('error');
        setError(err instanceof Error ? err.message : '未知错误');
      }
    },
    [conversationId, onMessageComplete],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setState('idle');
  }, []);

  return { state, streamingContent, error, sendMessage, stop };
}
```

### 前端：聊天界面组件

**`apps/frontend/src/components/ChatWindow.tsx`**：

```typescript
import React, { useState, useEffect, useRef } from 'react';
import type { Message, Conversation } from '@typechat/shared/types';
import { useStreamChat } from '../hooks/useStreamChat.js';

interface ChatWindowProps {
  conversation: Conversation;
  onNewMessage: (message: Message) => void;
}

export function ChatWindow({ conversation, onNewMessage }: ChatWindowProps) {
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  const { state, streamingContent, error, sendMessage, stop } = useStreamChat({
    conversationId: conversation.id,
    onMessageComplete: (content, tokenCount) => {
      onNewMessage({
        id: crypto.randomUUID(),
        conversationId: conversation.id,
        role: 'assistant',
        content,
        createdAt: new Date().toISOString(),
        tokenCount,
      });
    },
  });

  // 自动滚动到底部
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation.messages, streamingContent]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || state === 'streaming') return;

    onNewMessage({
      id: crypto.randomUUID(),
      conversationId: conversation.id,
      role: 'user',
      content: input,
      createdAt: new Date().toISOString(),
    });

    sendMessage(input);
    setInput('');
  };

  return (
    <div className="flex flex-col h-full">
      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {conversation.messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {/* 流式输出中的消息气泡 */}
        {state === 'streaming' && streamingContent && (
          <MessageBubble
            message={{
              id: 'streaming',
              conversationId: conversation.id,
              role: 'assistant',
              content: streamingContent,
              createdAt: new Date().toISOString(),
            }}
            isStreaming
          />
        )}

        {error && (
          <div className="text-red-500 text-sm px-4 py-2 bg-red-50 rounded-lg">
            错误：{error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* 输入框 */}
      <form onSubmit={handleSubmit} className="border-t p-4 flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
          placeholder="输入消息，Enter 发送，Shift+Enter 换行..."
          disabled={state === 'streaming'}
          rows={1}
          className="flex-1 resize-none rounded-lg border px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        {state === 'streaming' ? (
          <button
            type="button"
            onClick={stop}
            className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
          >
            停止
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
          >
            发送
          </button>
        )}
      </form>
    </div>
  );
}

// 消息气泡子组件
interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
}

function MessageBubble({ message, isStreaming = false }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[70%] rounded-2xl px-4 py-2 ${
          isUser
            ? 'bg-blue-500 text-white'
            : 'bg-gray-100 text-gray-900'
        }`}
      >
        <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
        {isStreaming && (
          <span className="inline-block w-1 h-4 bg-current ml-0.5 animate-pulse" />
        )}
        {message.tokenCount !== undefined && !isStreaming && (
          <p className="text-xs opacity-60 mt-1">{message.tokenCount} tokens</p>
        )}
      </div>
    </div>
  );
}
```

### 完整启动流程

```bash
# 1. 克隆项目并安装依赖
git clone https://github.com/example/typechat
cd typechat
pnpm install

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填写 OPENAI_API_KEY、数据库连接等

# 3. 启动本地基础设施（PostgreSQL、Redis）
docker-compose up -d postgres redis

# 4. 执行数据库迁移
pnpm db:migrate

# 5. 启动全部服务（并行，支持热更新）
pnpm dev

# 后端运行在 http://localhost:3000
# 前端运行在 http://localhost:5173
```

---

## 练习题

### 基础题

**1. 完善 shared 包的用户类型**

在 `packages/shared/src/types/user.ts` 中，定义 `User` 类型和 `UserProfile` 类型，要求：
- `User` 包含 `id`、`email`、`displayName`、`createdAt`、`role`（`'user' | 'admin'`）字段
- `UserProfile` 是 `User` 的公开展示版本，不含敏感信息（不含 `role`，增加 `avatarUrl` 可选字段）
- 使用 `Omit` 和扩展来避免重复定义

**2. 为 Zod Schema 添加登录次数限制**

扩展 `packages/shared/src/schemas/auth.ts` 中的 `LoginSchema`，添加一个 `rememberMe` 可选布尔字段（默认 `false`），并在 `apps/backend/src/routes/auth.ts` 中根据该字段决定 `refreshToken` 的有效期：
- `rememberMe: true` → 有效期 30 天
- `rememberMe: false` → 有效期 1 天

### 中级题

**3. 实现对话标题自动生成**

在 `ChatService` 中添加 `generateTitle` 方法，当对话的第一条用户消息发送后，自动调用 LLM 根据消息内容生成一个简短标题（不超过 20 字），并更新数据库中的 `title` 字段。要求：
- 使用非流式调用（因为标题生成无需打字机效果）
- 超时时间设为 5 秒，超时则使用消息前 20 个字符作为标题
- 在后台异步执行，不阻塞用户的流式消息响应

**4. 添加前端 Token 用量统计**

在 `useStreamChat` Hook 中，累计记录当前会话的总 token 用量和估算费用（从 `done` 事件的 `cost` 字段获取）。在 `ChatWindow` 组件底部显示会话统计信息，格式为：
```
本次会话：共 1,234 tokens | 估算费用：$0.0012
```
要求使用 `useReducer` 管理累计状态，而非多个独立 `useState`。

### 提高题

**5. 实现多模型并行对比功能**

扩展项目，实现一个"模型对比"模式：用户输入同一条消息，同时向两个不同模型发送请求，并排展示两个模型的流式响应。要求：

- 在 shared 包中添加 `CompareRequest` 类型，包含 `content`、`modelA: ModelId`、`modelB: ModelId` 字段
- 后端新增 `POST /api/compare` 端点，返回一个 SSE 流，事件中包含 `modelId` 字段区分来源：
  ```typescript
  type CompareStreamEvent =
    | { model: ModelId; type: 'delta'; content: string }
    | { model: ModelId; type: 'done'; tokenCount: number }
    | { type: 'all_done' };
  ```
- 使用 `Promise.allSettled` 并行发起两个模型的请求，任一完成均推送事件
- 前端使用两列布局，每列实时展示对应模型的流式输出，并在完成后显示各自的 token 用量与耗时

---

## 练习答案

### 答案1：完善用户类型

```typescript
// packages/shared/src/types/user.ts

export type UserRole = 'user' | 'admin';

export interface User {
  id: string;
  email: string;
  displayName: string;
  role: UserRole;
  createdAt: string;    // ISO 8601
  updatedAt: string;
}

// UserProfile 去掉 role 字段，增加 avatarUrl
export type UserProfile = Omit<User, 'role'> & {
  avatarUrl?: string;
};

// 辅助函数：将 User 转换为公开 Profile（去掉 role）
export function toUserProfile(user: User, avatarUrl?: string): UserProfile {
  const { role: _role, ...profile } = user;
  return avatarUrl ? { ...profile, avatarUrl } : profile;
}
```

### 答案2：添加 rememberMe 字段

```typescript
// packages/shared/src/schemas/auth.ts（扩展部分）
export const LoginSchema = z.object({
  email: z.string().email('邮箱格式不正确'),
  password: z.string().min(8, '密码至少8位'),
  rememberMe: z.boolean().default(false),
});

// apps/backend/src/routes/auth.ts（关键部分）
const { email, password, rememberMe } = request.body;
// ... 验证用户凭据 ...
const refreshTokenExpiresIn = rememberMe ? '30d' : '1d';
const refreshToken = jwt.sign({ userId: user.id }, env.JWT_REFRESH_SECRET, {
  expiresIn: refreshTokenExpiresIn,
});
```

### 答案3：对话标题自动生成

```typescript
// apps/backend/src/services/chat.service.ts（新增方法）

async generateTitle(firstUserMessage: string, modelId: ModelId): Promise<string> {
  const TIMEOUT_MS = 5_000;
  const fallback = firstUserMessage.slice(0, 20) + (firstUserMessage.length > 20 ? '...' : '');

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    const completion = await this.openai.chat.completions.create(
      {
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'system',
            content: '根据用户消息生成一个简短的对话标题，不超过20个字，只输出标题本身，不加引号。',
          },
          { role: 'user', content: firstUserMessage },
        ],
        max_tokens: 30,
      },
      { signal: controller.signal },
    );

    clearTimeout(timer);
    return completion.choices[0]?.message?.content?.trim() ?? fallback;
  } catch {
    return fallback;
  }
}

// 在路由中异步触发（不 await，不阻塞流式响应）
void chatService.generateTitle(content, conversation.modelId).then((title) => {
  return conversationRepo.updateTitle(conversationId, title);
});
```

### 答案4：Token 用量统计

```typescript
// apps/frontend/src/hooks/useStreamChat.ts（扩展部分）

type StatsState = {
  totalTokens: number;
  totalCostUsd: number;
  messageCount: number;
};

type StatsAction =
  | { type: 'ADD_USAGE'; tokenCount: number; cost: number }
  | { type: 'RESET' };

function statsReducer(state: StatsState, action: StatsAction): StatsState {
  switch (action.type) {
    case 'ADD_USAGE':
      return {
        ...state,
        totalTokens: state.totalTokens + action.tokenCount,
        totalCostUsd: state.totalCostUsd + action.cost,
        messageCount: state.messageCount + 1,
      };
    case 'RESET':
      return { totalTokens: 0, totalCostUsd: 0, messageCount: 0 };
  }
}

// 在 useStreamChat Hook 内：
const [stats, dispatchStats] = useReducer(statsReducer, {
  totalTokens: 0,
  totalCostUsd: 0,
  messageCount: 0,
});

// 在 done 事件处理中：
if (event.type === 'done') {
  dispatchStats({ type: 'ADD_USAGE', tokenCount: event.tokenCount, cost: event.cost });
}

// 在 ChatWindow 底部展示：
<div className="text-xs text-gray-400 px-4 py-1 border-t">
  本次会话：共 {stats.totalTokens.toLocaleString()} tokens ｜
  估算费用：${stats.totalCostUsd.toFixed(4)}
</div>
```

### 答案5：多模型并行对比

```typescript
// packages/shared/src/types/api.ts（新增类型）
export interface CompareRequest {
  content: string;
  modelA: ModelId;
  modelB: ModelId;
  systemPrompt?: string;
}

export type CompareStreamEvent =
  | { model: ModelId; type: 'start' }
  | { model: ModelId; type: 'delta'; content: string }
  | { model: ModelId; type: 'done'; tokenCount: number; durationMs: number }
  | { type: 'all_done' };

// apps/backend/src/routes/compare.ts（核心逻辑）
app.post('/api/compare', { preHandler: [app.authenticate] }, async (request, reply) => {
  const { content, modelA, modelB, systemPrompt } = request.body as CompareRequest;

  reply.raw.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });

  const send = (event: CompareStreamEvent) => {
    reply.raw.write(`data: ${JSON.stringify(event)}\n\n`);
  };

  const runModel = async (modelId: ModelId): Promise<void> => {
    const start = Date.now();
    let tokenCount = 0;
    send({ model: modelId, type: 'start' });

    await chatService.streamChat({
      messages: [{ role: 'user', content }],
      modelId,
      systemPrompt,
      onEvent: (event) => {
        if (event.type === 'delta') {
          send({ model: modelId, type: 'delta', content: event.content });
        } else if (event.type === 'done') {
          tokenCount = event.tokenCount;
        }
      },
    });

    send({ model: modelId, type: 'done', tokenCount, durationMs: Date.now() - start });
  };

  // 并行运行，任一完成均推送事件（不等另一个）
  await Promise.allSettled([runModel(modelA), runModel(modelB)]);
  send({ type: 'all_done' });
  reply.raw.end();
});
```

---

*本章是《从零到高阶的 TypeScript 教程（AI 应用实战版）》的最终章。感谢你完成了全部 24 章的学习旅程。从第 1 章的环境搭建，到如今完整的全栈 AI 聊天应用，你已经掌握了 TypeScript 在现代 AI 应用开发中所需的全套技能。祝你在未来的开发旅途中，写出既优雅又健壮的代码。*
