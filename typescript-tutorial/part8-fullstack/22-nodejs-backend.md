# 第22章：Node.js后端开发

> **适用读者**：已掌握 TypeScript 基础与异步编程，希望使用 TypeScript 构建类型安全的 Node.js 后端服务，并了解如何将 LLM 能力集成到后端 API 中的开发者。

---

## 学习目标

完成本章学习后，你将能够：

1. 使用 Express 或 Fastify 搭建 TypeScript 类型安全的 HTTP 服务，理解两者在类型集成上的差异
2. 为路由处理器和中间件编写精确的类型标注，消除 `any` 类型的滥用
3. 使用 Zod 对请求体、查询参数和路径参数进行运行时验证，并自动推导 TypeScript 类型
4. 设计统一的 API 响应结构与错误处理体系，实现对外一致、对内可扩展的错误管理
5. 实现 SSE（Server-Sent Events）端点，将 LLM 流式输出代理给前端客户端

---

## 22.1 Express/Fastify 与 TypeScript

### 22.1.1 项目初始化

在开始之前，先搭建一个标准的 TypeScript + Node.js 后端项目结构：

```bash
mkdir ts-backend && cd ts-backend
npm init -y

# 安装 Express 及其类型声明
npm install express
npm install -D @types/express typescript ts-node-dev

# 初始化 TypeScript 配置
npx tsc --init
```

推荐的 `tsconfig.json` 配置（针对 Node.js 后端）：

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "commonjs",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "resolveJsonModule": true,
    "declaration": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

`package.json` 中添加开发脚本：

```json
{
  "scripts": {
    "dev": "ts-node-dev --respawn --transpile-only src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js"
  }
}
```

### 22.1.2 Express 与 TypeScript

Express 的类型声明由 `@types/express` 提供，核心类型包括 `Request`、`Response`、`NextFunction` 和 `Application`。

最基础的 Express 服务：

```typescript
// src/index.ts
import express, { Application, Request, Response } from "express";

const app: Application = express();

// 解析 JSON 请求体
app.use(express.json());

app.get("/health", (req: Request, res: Response) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

const PORT = process.env.PORT ?? 3000;

app.listen(PORT, () => {
  console.log(`服务已启动，监听端口 ${PORT}`);
});
```

Express 的泛型参数允许对路由进行更精确的类型约束。`Request` 接受四个泛型参数：

```typescript
// Request<Params, ResBody, ReqBody, Query>
import { Request, Response } from "express";

interface UserParams {
  id: string;
}

interface UserQuery {
  include?: "profile" | "posts";
}

interface UpdateUserBody {
  name?: string;
  email?: string;
}

// 完全类型化的路由处理器
async function getUserHandler(
  req: Request<UserParams, unknown, unknown, UserQuery>,
  res: Response
): Promise<void> {
  const userId = req.params.id;         // string
  const include = req.query.include;    // "profile" | "posts" | undefined

  // ... 业务逻辑
  res.json({ id: userId, include });
}
```

### 22.1.3 Fastify 与 TypeScript

Fastify 是一个更现代的 Node.js 框架，其泛型路由设计天然贴合 TypeScript，类型推断能力更强：

```bash
npm install fastify
```

Fastify 使用 Schema 对象描述路由的输入输出，并通过泛型直接推导类型：

```typescript
// src/fastify-app.ts
import Fastify, { FastifyRequest, FastifyReply } from "fastify";

const fastify = Fastify({ logger: true });

// 定义路由 Schema（用于运行时验证 + 类型推导）
interface GetUserParams {
  id: string;
}

interface GetUserQuery {
  include?: string;
}

fastify.get<{
  Params: GetUserParams;
  Querystring: GetUserQuery;
}>(
  "/users/:id",
  {
    schema: {
      params: {
        type: "object",
        properties: { id: { type: "string" } },
        required: ["id"],
      },
      querystring: {
        type: "object",
        properties: { include: { type: "string" } },
      },
    },
  },
  async (request, reply) => {
    const { id } = request.params;        // string，完全类型安全
    const { include } = request.query;   // string | undefined
    return { id, include };
  }
);

fastify.listen({ port: 3000 }, (err) => {
  if (err) {
    fastify.log.error(err);
    process.exit(1);
  }
});
```

### 22.1.4 Express vs Fastify 选型参考

| 维度 | Express | Fastify |
|------|---------|---------|
| TypeScript 集成 | 需要手动标注泛型 | 泛型路由内置，推断更强 |
| 生态成熟度 | 极其丰富，中间件众多 | 插件生态快速增长 |
| 性能 | 良好 | 优秀（比 Express 快约 2-3 倍） |
| 学习曲线 | 平缓 | 中等 |
| 适用场景 | 快速原型、传统项目迁移 | 新项目、高并发 API |

本章后续示例以 **Express** 为主（生态更广泛），但所有概念均适用于 Fastify。

---

## 22.2 路由与中间件类型

### 22.2.1 路由器模块化

将路由按业务域拆分是大型应用的标准做法：

```typescript
// src/routes/users.ts
import { Router, Request, Response, NextFunction } from "express";

export const usersRouter = Router();

// GET /users
usersRouter.get("/", async (req: Request, res: Response) => {
  // 返回用户列表
  res.json({ users: [] });
});

// GET /users/:id
usersRouter.get("/:id", async (req: Request, res: Response) => {
  const { id } = req.params;
  res.json({ id });
});

// POST /users
usersRouter.post("/", async (req: Request, res: Response) => {
  const body = req.body;
  res.status(201).json({ created: body });
});
```

```typescript
// src/index.ts
import express from "express";
import { usersRouter } from "./routes/users";

const app = express();
app.use(express.json());

app.use("/users", usersRouter);

app.listen(3000);
```

### 22.2.2 中间件类型标注

Express 中间件的类型签名为 `(req, res, next) => void`，使用 `RequestHandler` 类型可以让标注更简洁：

```typescript
import { RequestHandler, Request, Response, NextFunction } from "express";

// 方式1：使用 RequestHandler 类型别名
const loggerMiddleware: RequestHandler = (req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
};

// 方式2：显式标注参数类型
function authMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const token = req.headers.authorization?.replace("Bearer ", "");

  if (!token) {
    res.status(401).json({ error: "缺少认证令牌" });
    return; // 注意：必须 return，不能调用 next()
  }

  // 将解析后的用户信息挂载到 req 上（需要扩展类型）
  next();
}
```

### 22.2.3 扩展 Request 类型

实际项目中常需要在 `req` 上挂载自定义属性（如认证用户信息）。Express 提供了声明合并机制：

```typescript
// src/types/express.d.ts
import { User } from "../models/user";

declare global {
  namespace Express {
    interface Request {
      user?: User;           // 认证中间件解析后挂载
      requestId?: string;    // 请求追踪 ID
    }
  }
}
```

```typescript
// src/middleware/auth.ts
import { RequestHandler } from "express";
import jwt from "jsonwebtoken";

interface JwtPayload {
  userId: string;
  email: string;
  role: "admin" | "user";
}

export const authMiddleware: RequestHandler = (req, res, next) => {
  const token = req.headers.authorization?.replace("Bearer ", "");

  if (!token) {
    res.status(401).json({ error: "缺少认证令牌" });
    return;
  }

  try {
    const payload = jwt.verify(
      token,
      process.env.JWT_SECRET ?? "secret"
    ) as JwtPayload;

    // 类型安全地挂载到 req（利用声明合并）
    req.user = {
      id: payload.userId,
      email: payload.email,
      role: payload.role,
    };

    next();
  } catch {
    res.status(401).json({ error: "令牌无效或已过期" });
  }
};
```

### 22.2.4 错误处理中间件

Express 的错误处理中间件有特殊的四参数签名：

```typescript
import { ErrorRequestHandler } from "express";

// 错误处理中间件必须有四个参数，否则 Express 不会识别为错误处理器
export const globalErrorHandler: ErrorRequestHandler = (
  err,
  req,
  res,
  next
) => {
  console.error("未捕获的错误：", err);

  if (res.headersSent) {
    return next(err);
  }

  res.status(500).json({
    success: false,
    error: {
      code: "INTERNAL_SERVER_ERROR",
      message: "服务器内部错误",
    },
  });
};
```

```typescript
// src/index.ts — 错误处理中间件必须最后注册
app.use("/users", usersRouter);
app.use(globalErrorHandler); // 放在所有路由之后
```

---

## 22.3 请求体验证（Zod）

### 22.3.1 为什么需要运行时验证

TypeScript 的类型检查发生在编译时，运行时接收到的 HTTP 请求体是完全未经验证的 `unknown` 数据。Zod 提供了运行时 Schema 验证，并能自动推导对应的 TypeScript 类型，实现"一个定义，两种保证"。

```bash
npm install zod
```

### 22.3.2 基础 Zod Schema

```typescript
import { z } from "zod";

// 定义用户创建的请求体 Schema
const CreateUserSchema = z.object({
  name: z.string().min(2, "姓名至少2个字符").max(50, "姓名最多50个字符"),
  email: z.string().email("邮箱格式不正确"),
  age: z.number().int().min(0).max(150).optional(),
  role: z.enum(["admin", "user", "guest"]).default("user"),
  tags: z.array(z.string()).max(10, "标签最多10个").default([]),
});

// 自动推导 TypeScript 类型
type CreateUserInput = z.infer<typeof CreateUserSchema>;
// 等价于：
// type CreateUserInput = {
//   name: string;
//   email: string;
//   age?: number | undefined;
//   role: "admin" | "user" | "guest";
//   tags: string[];
// }
```

### 22.3.3 Zod 验证中间件工厂

将 Zod 验证逻辑封装为可复用的中间件工厂函数：

```typescript
// src/middleware/validate.ts
import { Request, Response, NextFunction, RequestHandler } from "express";
import { ZodSchema, ZodError } from "zod";

type ValidateTarget = "body" | "query" | "params";

export function validate<T>(
  schema: ZodSchema<T>,
  target: ValidateTarget = "body"
): RequestHandler {
  return (req: Request, res: Response, next: NextFunction): void => {
    const result = schema.safeParse(req[target]);

    if (!result.success) {
      const errors = result.error.errors.map((e) => ({
        field: e.path.join("."),
        message: e.message,
      }));

      res.status(400).json({
        success: false,
        error: {
          code: "VALIDATION_ERROR",
          message: "请求参数验证失败",
          details: errors,
        },
      });
      return;
    }

    // 将验证并转换后的数据写回 req
    (req as Request & Record<string, unknown>)[target] = result.data;
    next();
  };
}
```

### 22.3.4 在路由中使用验证中间件

```typescript
// src/routes/users.ts
import { Router } from "express";
import { z } from "zod";
import { validate } from "../middleware/validate";

export const usersRouter = Router();

// 定义 Schema
const CreateUserSchema = z.object({
  name: z.string().min(2).max(50),
  email: z.string().email(),
  role: z.enum(["admin", "user"]).default("user"),
});

const UserParamsSchema = z.object({
  id: z.string().uuid("用户ID必须是有效的UUID"),
});

const ListUsersQuerySchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  pageSize: z.coerce.number().int().min(1).max(100).default(20),
  role: z.enum(["admin", "user"]).optional(),
});

type CreateUserInput = z.infer<typeof CreateUserSchema>;
type ListUsersQuery = z.infer<typeof ListUsersQuerySchema>;

// GET /users?page=1&pageSize=20
usersRouter.get(
  "/",
  validate(ListUsersQuerySchema, "query"),
  async (req, res) => {
    const query = req.query as unknown as ListUsersQuery;
    // query.page 已经是 number 类型（coerce 完成了字符串到数字的转换）
    res.json({
      success: true,
      data: { users: [], page: query.page, pageSize: query.pageSize },
    });
  }
);

// POST /users
usersRouter.post(
  "/",
  validate(CreateUserSchema, "body"),
  async (req, res) => {
    const body = req.body as CreateUserInput;
    // body 已通过验证，类型安全
    res.status(201).json({ success: true, data: { user: body } });
  }
);

// GET /users/:id
usersRouter.get(
  "/:id",
  validate(UserParamsSchema, "params"),
  async (req, res) => {
    const { id } = req.params; // 已验证为有效 UUID
    res.json({ success: true, data: { id } });
  }
);
```

### 22.3.5 常用 Zod Schema 模式

```typescript
import { z } from "zod";

// 嵌套对象
const AddressSchema = z.object({
  street: z.string(),
  city: z.string(),
  country: z.string().length(2, "国家代码必须是2位"),
});

const UserWithAddressSchema = z.object({
  name: z.string(),
  address: AddressSchema,          // 嵌套
  addresses: z.array(AddressSchema).optional(), // 嵌套数组
});

// 联合类型
const EventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("click"), x: z.number(), y: z.number() }),
  z.object({ type: z.literal("keydown"), key: z.string() }),
]);

// 自定义精炼（refine）
const PasswordSchema = z
  .object({
    password: z.string().min(8),
    confirmPassword: z.string(),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "两次密码不一致",
    path: ["confirmPassword"],
  });

// 转换（transform）
const DateStringSchema = z
  .string()
  .datetime()
  .transform((s) => new Date(s)); // 字符串 → Date 对象

type DateInput = z.infer<typeof DateStringSchema>; // Date
```

---

## 22.4 API设计与错误处理

### 22.4.1 统一响应结构

良好的 API 应该有一致的响应格式，便于前端统一处理：

```typescript
// src/types/api.ts

// 成功响应
interface ApiSuccess<T> {
  success: true;
  data: T;
  meta?: {
    page?: number;
    pageSize?: number;
    total?: number;
  };
}

// 错误响应
interface ApiError {
  success: false;
  error: {
    code: string;
    message: string;
    details?: Array<{ field: string; message: string }>;
  };
}

type ApiResponse<T> = ApiSuccess<T> | ApiError;

// 响应工厂函数
export function successResponse<T>(data: T, meta?: ApiSuccess<T>["meta"]): ApiSuccess<T> {
  return { success: true, data, ...(meta ? { meta } : {}) };
}

export function errorResponse(
  code: string,
  message: string,
  details?: ApiError["error"]["details"]
): ApiError {
  return {
    success: false,
    error: { code, message, ...(details ? { details } : {}) },
  };
}
```

### 22.4.2 自定义错误类体系

```typescript
// src/errors/AppError.ts

export class AppError extends Error {
  constructor(
    public readonly code: string,
    public readonly message: string,
    public readonly statusCode: number,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = "AppError";
    // 修复 TypeScript 中继承 Error 的 prototype 链问题
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class NotFoundError extends AppError {
  constructor(resource: string, id?: string) {
    super(
      "NOT_FOUND",
      id ? `${resource} (id: ${id}) 不存在` : `${resource} 不存在`,
      404
    );
  }
}

export class ValidationError extends AppError {
  constructor(
    message: string,
    public readonly fields?: Array<{ field: string; message: string }>
  ) {
    super("VALIDATION_ERROR", message, 400, fields);
  }
}

export class UnauthorizedError extends AppError {
  constructor(message = "未经授权的访问") {
    super("UNAUTHORIZED", message, 401);
  }
}

export class ForbiddenError extends AppError {
  constructor(message = "权限不足") {
    super("FORBIDDEN", message, 403);
  }
}

export class ConflictError extends AppError {
  constructor(resource: string) {
    super("CONFLICT", `${resource} 已存在`, 409);
  }
}
```

### 22.4.3 全局错误处理中间件

```typescript
// src/middleware/errorHandler.ts
import { ErrorRequestHandler } from "express";
import { ZodError } from "zod";
import { AppError } from "../errors/AppError";
import { errorResponse } from "../types/api";

export const globalErrorHandler: ErrorRequestHandler = (
  err,
  req,
  res,
  _next
) => {
  // 记录错误日志
  console.error(`[ERROR] ${req.method} ${req.path}:`, err);

  // 已经发送过响应头，交给默认处理
  if (res.headersSent) return;

  // 处理 Zod 验证错误
  if (err instanceof ZodError) {
    res.status(400).json(
      errorResponse(
        "VALIDATION_ERROR",
        "请求参数验证失败",
        err.errors.map((e) => ({
          field: e.path.join("."),
          message: e.message,
        }))
      )
    );
    return;
  }

  // 处理自定义应用错误
  if (err instanceof AppError) {
    res.status(err.statusCode).json(
      errorResponse(err.code, err.message)
    );
    return;
  }

  // 处理其他未知错误（生产环境不暴露细节）
  const isDev = process.env.NODE_ENV === "development";
  res.status(500).json(
    errorResponse(
      "INTERNAL_SERVER_ERROR",
      isDev ? (err as Error).message : "服务器内部错误"
    )
  );
};
```

### 22.4.4 异步路由错误捕获

Express 默认不捕获 async 路由中抛出的错误，需要包装处理：

```typescript
// src/utils/asyncHandler.ts
import { Request, Response, NextFunction, RequestHandler } from "express";

type AsyncRequestHandler = (
  req: Request,
  res: Response,
  next: NextFunction
) => Promise<void>;

// 将 async 函数包装为 Express 兼容的处理器
export function asyncHandler(fn: AsyncRequestHandler): RequestHandler {
  return (req, res, next) => {
    fn(req, res, next).catch(next); // 捕获 Promise rejection 并传递给错误处理器
  };
}
```

```typescript
// src/routes/users.ts — 使用 asyncHandler
import { asyncHandler } from "../utils/asyncHandler";
import { NotFoundError } from "../errors/AppError";

usersRouter.get(
  "/:id",
  asyncHandler(async (req, res) => {
    const { id } = req.params;
    const user = await userService.findById(id);

    if (!user) {
      throw new NotFoundError("用户", id); // 错误自动传递给 globalErrorHandler
    }

    res.json(successResponse(user));
  })
);
```

---

## 22.5 SSE 端点实现

### 22.5.1 SSE 响应头设置

在 Express 中实现 SSE 端点，核心是设置正确的响应头并保持连接：

```typescript
// src/utils/sse.ts

import { Response } from "express";

// SSE 辅助工具
export class SSEWriter {
  private isClosed = false;

  constructor(private readonly res: Response) {
    // 设置 SSE 必要的响应头
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    // 允许跨域（如果需要）
    res.setHeader("Access-Control-Allow-Origin", "*");
    // 禁用 Nginx 等代理的缓冲
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders(); // 立即发送响应头
  }

  // 发送普通数据事件
  send(data: unknown, eventId?: string): void {
    if (this.isClosed) return;

    if (eventId) {
      this.res.write(`id: ${eventId}\n`);
    }
    const payload = typeof data === "string" ? data : JSON.stringify(data);
    this.res.write(`data: ${payload}\n\n`);
  }

  // 发送命名事件
  sendEvent(event: string, data: unknown, eventId?: string): void {
    if (this.isClosed) return;

    if (eventId) {
      this.res.write(`id: ${eventId}\n`);
    }
    this.res.write(`event: ${event}\n`);
    const payload = typeof data === "string" ? data : JSON.stringify(data);
    this.res.write(`data: ${payload}\n\n`);
  }

  // 发送心跳注释（防止代理超时断开）
  ping(): void {
    if (this.isClosed) return;
    this.res.write(": ping\n\n");
  }

  // 关闭连接
  close(): void {
    if (this.isClosed) return;
    this.isClosed = true;
    this.res.end();
  }

  get closed(): boolean {
    return this.isClosed;
  }
}
```

### 22.5.2 基础 SSE 端点

```typescript
// src/routes/events.ts
import { Router, Request, Response } from "express";
import { SSEWriter } from "../utils/sse";

export const eventsRouter = Router();

// GET /events/counter — 每秒推送计数器
eventsRouter.get("/counter", (req: Request, res: Response) => {
  const sse = new SSEWriter(res);
  let count = 0;

  const interval = setInterval(() => {
    count++;
    sse.send({ count, time: new Date().toISOString() });

    if (count >= 10) {
      sse.sendEvent("done", { message: "计数完成" });
      sse.close();
      clearInterval(interval);
    }
  }, 1000);

  // 客户端断开连接时清理资源
  req.on("close", () => {
    clearInterval(interval);
    sse.close();
  });
});
```

### 22.5.3 代理 LLM 流式输出的 SSE 端点

```typescript
// src/routes/chat.ts
import { Router } from "express";
import { z } from "zod";
import OpenAI from "openai";
import { SSEWriter } from "../utils/sse";
import { validate } from "../middleware/validate";
import { asyncHandler } from "../utils/asyncHandler";

export const chatRouter = Router();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const ChatRequestSchema = z.object({
  messages: z
    .array(
      z.object({
        role: z.enum(["user", "assistant", "system"]),
        content: z.string().min(1),
      })
    )
    .min(1, "消息列表不能为空"),
  model: z.string().default("gpt-4o-mini"),
  temperature: z.number().min(0).max(2).default(0.7),
});

type ChatRequest = z.infer<typeof ChatRequestSchema>;

// POST /chat/stream
chatRouter.post(
  "/stream",
  validate(ChatRequestSchema, "body"),
  asyncHandler(async (req, res) => {
    const body = req.body as ChatRequest;
    const sse = new SSEWriter(res);

    // 通知客户端开始
    sse.sendEvent("start", { model: body.model });

    try {
      // 创建流式请求
      const stream = await openai.chat.completions.create({
        model: body.model,
        messages: body.messages,
        temperature: body.temperature,
        stream: true,
      });

      let totalTokens = 0;

      // 遍历流式响应
      for await (const chunk of stream) {
        // 检查客户端是否已断开
        if (sse.closed) break;

        const delta = chunk.choices[0]?.delta;
        const finishReason = chunk.choices[0]?.finish_reason;

        if (delta?.content) {
          // 推送文本增量
          sse.sendEvent("delta", { content: delta.content });
        }

        if (finishReason === "stop") {
          totalTokens = chunk.usage?.total_tokens ?? 0;
        }
      }

      // 推送完成事件
      sse.sendEvent("done", {
        totalTokens,
        finishReason: "stop",
      });
    } catch (err) {
      // 将错误通知客户端
      sse.sendEvent("error", {
        code: "LLM_ERROR",
        message: err instanceof Error ? err.message : "LLM 调用失败",
      });
    } finally {
      sse.close();
    }
  })
);
```

### 22.5.4 心跳保活机制

对于长时间运行的 SSE 连接，需要定期发送心跳防止代理超时：

```typescript
// src/utils/sse.ts — 扩展 SSEWriter，支持自动心跳

export class SSEWriter {
  private isClosed = false;
  private heartbeatTimer?: NodeJS.Timeout;

  constructor(
    private readonly res: Response,
    options: { heartbeatInterval?: number } = {}
  ) {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders();

    // 启动心跳
    if (options.heartbeatInterval) {
      this.heartbeatTimer = setInterval(() => {
        this.ping();
      }, options.heartbeatInterval);
    }
  }

  // ... 其余方法不变 ...

  close(): void {
    if (this.isClosed) return;
    this.isClosed = true;

    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }

    this.res.end();
  }
}

// 使用时启用 30 秒心跳
const sse = new SSEWriter(res, { heartbeatInterval: 30_000 });
```

---

## 本章小结

| 知识点 | 关键 API / 工具 | 核心作用 |
|--------|----------------|---------|
| Express 类型系统 | `Request<P,R,B,Q>`、`Response`、`RequestHandler` | 为路由参数提供编译时类型保障 |
| Fastify 泛型路由 | `fastify.get<{Params,Querystring,Body}>` | 通过 Schema 同时完成验证和类型推导 |
| 请求扩展 | `declare global { namespace Express { interface Request } }` | 在请求对象上安全挂载自定义属性 |
| Zod 验证 | `z.object()`、`z.infer<>`、`safeParse()` | 运行时验证 + TypeScript 类型双重保障 |
| 验证中间件 | `validate(schema, target)` 工厂函数 | 复用验证逻辑，统一错误格式 |
| 自定义错误类 | `AppError`、`NotFoundError` 等 | 语义清晰、携带状态码和错误码 |
| 全局错误处理 | 四参数 `ErrorRequestHandler` | 统一捕获所有错误，避免信息泄露 |
| 异步路由包装 | `asyncHandler(fn)` | 捕获 async 路由中的 Promise 错误 |
| SSE 端点 | `text/event-stream`、`SSEWriter` 工具类 | 服务器到客户端的单向实时推流 |
| LLM 流式代理 | `openai.stream` + SSE | 将 LLM Token 流实时转发给前端 |

---

## AI 应用实战：AI 网关服务

本节将构建一个完整的 **AI 网关服务**——一个代理和管理 LLM API 请求的后端服务，支持多模型路由、请求限流、使用量统计和流式输出代理。

### 项目结构

```
src/
├── index.ts               # 应用入口
├── config.ts              # 配置管理
├── types/
│   ├── api.ts             # 通用 API 类型
│   └── gateway.ts         # 网关专用类型
├── errors/
│   └── AppError.ts        # 错误类体系
├── middleware/
│   ├── validate.ts        # Zod 验证中间件
│   ├── auth.ts            # API Key 认证
│   ├── rateLimit.ts       # 请求限流
│   └── errorHandler.ts    # 全局错误处理
├── services/
│   ├── openai.service.ts  # OpenAI 封装
│   └── usage.service.ts   # 使用量统计
├── utils/
│   ├── asyncHandler.ts    # 异步路由包装
│   └── sse.ts             # SSE 工具
└── routes/
    ├── chat.ts            # 聊天路由
    └── usage.ts           # 使用量查询
```

### 类型定义

```typescript
// src/types/gateway.ts
import { z } from "zod";

// 支持的模型列表
export const SUPPORTED_MODELS = [
  "gpt-4o",
  "gpt-4o-mini",
  "gpt-3.5-turbo",
] as const;

export type SupportedModel = (typeof SUPPORTED_MODELS)[number];

// 聊天消息 Schema
export const MessageSchema = z.object({
  role: z.enum(["system", "user", "assistant"]),
  content: z.string().min(1, "消息内容不能为空"),
});

// 聊天请求 Schema
export const ChatCompletionRequestSchema = z.object({
  model: z.enum(SUPPORTED_MODELS).default("gpt-4o-mini"),
  messages: z.array(MessageSchema).min(1),
  temperature: z.number().min(0).max(2).default(0.7),
  maxTokens: z.number().int().min(1).max(4096).default(1024),
  stream: z.boolean().default(false),
});

export type ChatCompletionRequest = z.infer<typeof ChatCompletionRequestSchema>;

// 使用量记录
export interface UsageRecord {
  apiKeyId: string;
  model: SupportedModel;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  latencyMs: number;
  timestamp: Date;
}

// SSE 事件类型
export type GatewaySSEEvent =
  | { type: "start"; model: string }
  | { type: "delta"; content: string }
  | { type: "done"; usage: { promptTokens: number; completionTokens: number; totalTokens: number } }
  | { type: "error"; code: string; message: string };
```

### API Key 认证中间件

```typescript
// src/middleware/auth.ts
import { RequestHandler } from "express";
import crypto from "crypto";

// 模拟的 API Key 存储（生产环境应使用数据库）
const API_KEYS = new Map<string, { id: string; name: string; tier: "free" | "pro" }>([
  ["gw-test-key-001", { id: "key_001", name: "测试应用", tier: "free" }],
  ["gw-prod-key-002", { id: "key_002", name: "生产应用", tier: "pro" }],
]);

// 扩展 Request 类型（在 src/types/express.d.ts 中声明）
declare global {
  namespace Express {
    interface Request {
      apiKey?: { id: string; name: string; tier: "free" | "pro" };
    }
  }
}

export const apiKeyAuth: RequestHandler = (req, res, next) => {
  const authHeader = req.headers.authorization;
  const key = authHeader?.startsWith("Bearer ")
    ? authHeader.slice(7)
    : req.headers["x-api-key"] as string | undefined;

  if (!key) {
    res.status(401).json({
      success: false,
      error: { code: "MISSING_API_KEY", message: "请提供 API Key" },
    });
    return;
  }

  const keyInfo = API_KEYS.get(key);

  if (!keyInfo) {
    res.status(401).json({
      success: false,
      error: { code: "INVALID_API_KEY", message: "API Key 无效" },
    });
    return;
  }

  req.apiKey = keyInfo;
  next();
};
```

### 限流中间件

```typescript
// src/middleware/rateLimit.ts
import { RequestHandler } from "express";

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

// 简单的内存限流（生产环境应使用 Redis）
const rateLimitStore = new Map<string, RateLimitEntry>();

const TIER_LIMITS = {
  free: { requestsPerMinute: 10 },
  pro: { requestsPerMinute: 100 },
};

export const rateLimiter: RequestHandler = (req, res, next) => {
  const apiKey = req.apiKey;
  if (!apiKey) return next();

  const limit = TIER_LIMITS[apiKey.tier].requestsPerMinute;
  const now = Date.now();
  const windowMs = 60_000; // 1分钟

  const entry = rateLimitStore.get(apiKey.id);

  if (!entry || now > entry.resetAt) {
    // 新窗口
    rateLimitStore.set(apiKey.id, { count: 1, resetAt: now + windowMs });
    res.setHeader("X-RateLimit-Limit", limit);
    res.setHeader("X-RateLimit-Remaining", limit - 1);
    return next();
  }

  if (entry.count >= limit) {
    res.setHeader("X-RateLimit-Limit", limit);
    res.setHeader("X-RateLimit-Remaining", 0);
    res.setHeader("Retry-After", Math.ceil((entry.resetAt - now) / 1000));
    res.status(429).json({
      success: false,
      error: {
        code: "RATE_LIMIT_EXCEEDED",
        message: `请求频率超限，每分钟最多 ${limit} 次`,
      },
    });
    return;
  }

  entry.count++;
  res.setHeader("X-RateLimit-Limit", limit);
  res.setHeader("X-RateLimit-Remaining", limit - entry.count);
  next();
};
```

### 使用量统计服务

```typescript
// src/services/usage.service.ts
import { UsageRecord } from "../types/gateway";

class UsageService {
  private records: UsageRecord[] = [];

  record(entry: UsageRecord): void {
    this.records.push(entry);
    // 生产环境：写入数据库或消息队列
  }

  getStats(apiKeyId: string): {
    totalRequests: number;
    totalTokens: number;
    avgLatencyMs: number;
    byModel: Record<string, { requests: number; tokens: number }>;
  } {
    const keyRecords = this.records.filter((r) => r.apiKeyId === apiKeyId);

    const byModel: Record<string, { requests: number; tokens: number }> = {};

    for (const record of keyRecords) {
      if (!byModel[record.model]) {
        byModel[record.model] = { requests: 0, tokens: 0 };
      }
      byModel[record.model].requests++;
      byModel[record.model].tokens += record.totalTokens;
    }

    const totalTokens = keyRecords.reduce((sum, r) => sum + r.totalTokens, 0);
    const avgLatencyMs =
      keyRecords.length > 0
        ? keyRecords.reduce((sum, r) => sum + r.latencyMs, 0) / keyRecords.length
        : 0;

    return {
      totalRequests: keyRecords.length,
      totalTokens,
      avgLatencyMs: Math.round(avgLatencyMs),
      byModel,
    };
  }
}

export const usageService = new UsageService();
```

### 核心聊天路由

```typescript
// src/routes/chat.ts
import { Router } from "express";
import OpenAI from "openai";
import { ChatCompletionRequestSchema, GatewaySSEEvent } from "../types/gateway";
import { validate } from "../middleware/validate";
import { asyncHandler } from "../utils/asyncHandler";
import { SSEWriter } from "../utils/sse";
import { usageService } from "../services/usage.service";

export const chatRouter = Router();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// POST /chat/completions — 统一入口，自动识别是否流式
chatRouter.post(
  "/completions",
  validate(ChatCompletionRequestSchema, "body"),
  asyncHandler(async (req, res) => {
    const body = req.body as import("../types/gateway").ChatCompletionRequest;
    const apiKeyId = req.apiKey!.id;
    const startTime = Date.now();

    if (body.stream) {
      // ---- 流式响应 ----
      const sse = new SSEWriter(res, { heartbeatInterval: 20_000 });

      const startEvent: GatewaySSEEvent = { type: "start", model: body.model };
      sse.sendEvent("message", startEvent);

      let promptTokens = 0;
      let completionTokens = 0;

      try {
        const stream = await openai.chat.completions.create({
          model: body.model,
          messages: body.messages,
          temperature: body.temperature,
          max_tokens: body.maxTokens,
          stream: true,
          stream_options: { include_usage: true },
        });

        for await (const chunk of stream) {
          if (sse.closed) break;

          const delta = chunk.choices[0]?.delta?.content;

          if (delta) {
            const deltaEvent: GatewaySSEEvent = { type: "delta", content: delta };
            sse.sendEvent("message", deltaEvent);
          }

          if (chunk.usage) {
            promptTokens = chunk.usage.prompt_tokens;
            completionTokens = chunk.usage.completion_tokens;
          }
        }

        const doneEvent: GatewaySSEEvent = {
          type: "done",
          usage: {
            promptTokens,
            completionTokens,
            totalTokens: promptTokens + completionTokens,
          },
        };
        sse.sendEvent("message", doneEvent);

        // 记录使用量
        usageService.record({
          apiKeyId,
          model: body.model,
          promptTokens,
          completionTokens,
          totalTokens: promptTokens + completionTokens,
          latencyMs: Date.now() - startTime,
          timestamp: new Date(),
        });
      } catch (err) {
        const errorEvent: GatewaySSEEvent = {
          type: "error",
          code: "LLM_ERROR",
          message: err instanceof Error ? err.message : "未知错误",
        };
        sse.sendEvent("message", errorEvent);
      } finally {
        sse.close();
      }
    } else {
      // ---- 非流式响应 ----
      const completion = await openai.chat.completions.create({
        model: body.model,
        messages: body.messages,
        temperature: body.temperature,
        max_tokens: body.maxTokens,
        stream: false,
      });

      const usage = completion.usage;

      usageService.record({
        apiKeyId,
        model: body.model,
        promptTokens: usage?.prompt_tokens ?? 0,
        completionTokens: usage?.completion_tokens ?? 0,
        totalTokens: usage?.total_tokens ?? 0,
        latencyMs: Date.now() - startTime,
        timestamp: new Date(),
      });

      res.json({
        success: true,
        data: {
          content: completion.choices[0].message.content,
          model: body.model,
          usage: {
            promptTokens: usage?.prompt_tokens,
            completionTokens: usage?.completion_tokens,
            totalTokens: usage?.total_tokens,
          },
        },
      });
    }
  })
);
```

### 使用量查询路由

```typescript
// src/routes/usage.ts
import { Router } from "express";
import { usageService } from "../services/usage.service";

export const usageRouter = Router();

// GET /usage/stats
usageRouter.get("/stats", (req, res) => {
  const apiKeyId = req.apiKey!.id;
  const stats = usageService.getStats(apiKeyId);

  res.json({
    success: true,
    data: stats,
  });
});
```

### 应用入口

```typescript
// src/index.ts
import "dotenv/config";
import express from "express";
import { chatRouter } from "./routes/chat";
import { usageRouter } from "./routes/usage";
import { apiKeyAuth } from "./middleware/auth";
import { rateLimiter } from "./middleware/rateLimit";
import { globalErrorHandler } from "./middleware/errorHandler";

const app = express();

// 全局中间件
app.use(express.json({ limit: "1mb" }));
app.use(express.urlencoded({ extended: true }));

// 健康检查（无需认证）
app.get("/health", (_req, res) => {
  res.json({ status: "ok", version: "1.0.0", timestamp: new Date().toISOString() });
});

// API 路由（需要认证 + 限流）
app.use("/v1", apiKeyAuth, rateLimiter, chatRouter);
app.use("/v1", apiKeyAuth, usageRouter);

// 404 处理
app.use((_req, res) => {
  res.status(404).json({
    success: false,
    error: { code: "NOT_FOUND", message: "接口不存在" },
  });
});

// 全局错误处理（必须最后注册）
app.use(globalErrorHandler);

const PORT = process.env.PORT ?? 3000;
app.listen(PORT, () => {
  console.log(`AI 网关服务已启动：http://localhost:${PORT}`);
  console.log(`健康检查：http://localhost:${PORT}/health`);
});
```

### 测试网关服务

```bash
# 非流式请求
curl -X POST http://localhost:3000/v1/completions \
  -H "Authorization: Bearer gw-test-key-001" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "用一句话介绍TypeScript"}],
    "stream": false
  }'

# 流式请求
curl -X POST http://localhost:3000/v1/completions \
  -H "Authorization: Bearer gw-prod-key-002" \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "写一首关于TypeScript的短诗"}],
    "stream": true
  }'

# 查询使用量
curl http://localhost:3000/v1/usage/stats \
  -H "Authorization: Bearer gw-prod-key-002"
```

---

## 练习题

### 基础题

**练习 22-1**：为以下路由添加完整的 TypeScript 类型标注，消除所有 `any` 类型，并使用 Zod 验证请求体：

```typescript
// 待完善：商品搜索接口
app.get("/products", async (req, res) => {
  const { keyword, minPrice, maxPrice, category, page } = req.query;
  // 要求：keyword 为字符串，minPrice/maxPrice 为数字，
  // category 为 "electronics"|"clothing"|"food" 之一，
  // page 默认为 1
  res.json({ products: [], query: req.query });
});
```

**练习 22-2**：实现一个 `retry` 工具函数，在异步操作失败时自动重试，并为其编写完整的 TypeScript 类型：

```typescript
// 要求：
// - 接受异步函数、最大重试次数、延迟毫秒数
// - 使用指数退避（每次重试延迟翻倍）
// - 超过最大重试次数后抛出最后一次的错误
// - 完全类型安全，返回类型与传入函数的返回类型一致

async function retry<T>(/* 补全参数 */): Promise<T> {
  // 实现
}
```

### 进阶题

**练习 22-3**：扩展本章的 SSE 网关服务，添加请求日志中间件，记录每次请求的以下信息，并将日志按 JSON 格式写入文件：

- 请求时间、方法、路径、查询参数
- API Key ID（不记录原始 Key）
- 响应状态码、响应时间（毫秒）
- 对于流式请求，记录最终的 Token 使用量

**练习 22-4**：基于 Zod 实现一个更强大的验证中间件，支持以下功能：

- 同时验证 `body`、`query`、`params` 三个来源
- 验证失败时合并所有错误，而不是遇到第一个就停止
- 支持自定义错误消息的国际化（接受一个 `locale` 参数，提供中文和英文两套错误消息）

### 挑战题

**练习 22-5**：为 AI 网关服务设计并实现一个**多模型负载均衡路由器**，满足以下需求：

1. 支持同时配置多个 LLM 提供商（OpenAI、Anthropic），每个提供商可以配置多个 API Key
2. 实现以下路由策略（通过请求头 `X-Routing-Strategy` 指定）：
   - `round-robin`：轮询分发
   - `lowest-latency`：选择当前平均延迟最低的 Key
   - `random`：随机选择
3. 当某个 Key 连续失败 3 次后，自动将其标记为"熔断"状态，30 秒后自动恢复
4. 所有状态使用 TypeScript 严格类型建模，不使用任何 `any`

---

## 练习答案

### 练习 22-1 答案

```typescript
import { Router } from "express";
import { z } from "zod";
import { validate } from "./middleware/validate";

const router = Router();

const ProductSearchQuerySchema = z.object({
  keyword: z.string().min(1).optional(),
  minPrice: z.coerce.number().min(0).optional(),
  maxPrice: z.coerce.number().min(0).optional(),
  category: z.enum(["electronics", "clothing", "food"]).optional(),
  page: z.coerce.number().int().min(1).default(1),
}).refine(
  (data) =>
    data.minPrice === undefined ||
    data.maxPrice === undefined ||
    data.minPrice <= data.maxPrice,
  {
    message: "minPrice 不能大于 maxPrice",
    path: ["minPrice"],
  }
);

type ProductSearchQuery = z.infer<typeof ProductSearchQuerySchema>;

router.get(
  "/products",
  validate(ProductSearchQuerySchema, "query"),
  async (req, res) => {
    const query = req.query as unknown as ProductSearchQuery;
    // 所有字段已验证并转换类型
    res.json({
      products: [],
      query: {
        keyword: query.keyword,
        minPrice: query.minPrice,
        maxPrice: query.maxPrice,
        category: query.category,
        page: query.page, // number 类型
      },
    });
  }
);
```

### 练习 22-2 答案

```typescript
interface RetryOptions {
  maxAttempts: number;
  initialDelayMs: number;
  onRetry?: (attempt: number, error: unknown) => void;
}

async function retry<T>(
  fn: () => Promise<T>,
  options: RetryOptions
): Promise<T> {
  const { maxAttempts, initialDelayMs, onRetry } = options;
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;

      if (attempt === maxAttempts) break;

      // 指数退避：1倍、2倍、4倍...
      const delayMs = initialDelayMs * Math.pow(2, attempt - 1);
      onRetry?.(attempt, err);

      await new Promise<void>((resolve) => setTimeout(resolve, delayMs));
    }
  }

  throw lastError;
}

// 使用示例
const result = await retry(
  () => fetch("https://api.example.com/data").then((r) => r.json()),
  {
    maxAttempts: 3,
    initialDelayMs: 500,
    onRetry: (attempt, err) =>
      console.warn(`第 ${attempt} 次重试，错误：`, err),
  }
);
```

### 练习 22-3 答案

```typescript
// src/middleware/requestLogger.ts
import { RequestHandler } from "express";
import fs from "fs";
import path from "path";

interface LogEntry {
  timestamp: string;
  method: string;
  path: string;
  query: Record<string, unknown>;
  apiKeyId: string | null;
  statusCode: number;
  latencyMs: number;
  streamTokens?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

// 在 res.locals 中传递流式 Token 信息
declare module "express" {
  interface Locals {
    streamTokens?: LogEntry["streamTokens"];
  }
}

const logStream = fs.createWriteStream(
  path.join(process.cwd(), "logs", "access.jsonl"),
  { flags: "a" }
);

export const requestLogger: RequestHandler = (req, res, next) => {
  const startTime = Date.now();

  res.on("finish", () => {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      method: req.method,
      path: req.path,
      query: req.query as Record<string, unknown>,
      apiKeyId: req.apiKey?.id ?? null,
      statusCode: res.statusCode,
      latencyMs: Date.now() - startTime,
      ...(res.locals.streamTokens
        ? { streamTokens: res.locals.streamTokens }
        : {}),
    };

    logStream.write(JSON.stringify(entry) + "\n");
  });

  next();
};
```

### 练习 22-4 答案

```typescript
// src/middleware/validateMulti.ts
import { Request, Response, NextFunction, RequestHandler } from "express";
import { ZodSchema } from "zod";

type Locale = "zh" | "en";

const ERROR_MESSAGES: Record<Locale, Record<string, string>> = {
  zh: {
    VALIDATION_FAILED: "请求参数验证失败",
    BODY: "请求体",
    QUERY: "查询参数",
    PARAMS: "路径参数",
  },
  en: {
    VALIDATION_FAILED: "Request validation failed",
    BODY: "body",
    QUERY: "query",
    PARAMS: "params",
  },
};

interface MultiValidateSchemas {
  body?: ZodSchema;
  query?: ZodSchema;
  params?: ZodSchema;
}

export function validateMulti(
  schemas: MultiValidateSchemas,
  locale: Locale = "zh"
): RequestHandler {
  return (req: Request, res: Response, next: NextFunction): void => {
    const allErrors: Array<{ source: string; field: string; message: string }> = [];
    const msgs = ERROR_MESSAGES[locale];

    for (const [target, schema] of Object.entries(schemas) as Array<
      ["body" | "query" | "params", ZodSchema]
    >) {
      const result = schema.safeParse(req[target]);
      if (!result.success) {
        for (const e of result.error.errors) {
          allErrors.push({
            source: msgs[target.toUpperCase()],
            field: e.path.join("."),
            message: e.message,
          });
        }
      } else {
        (req as Request & Record<string, unknown>)[target] = result.data;
      }
    }

    if (allErrors.length > 0) {
      res.status(400).json({
        success: false,
        error: {
          code: "VALIDATION_ERROR",
          message: msgs.VALIDATION_FAILED,
          details: allErrors,
        },
      });
      return;
    }

    next();
  };
}
```

### 练习 22-5 答案（核心实现）

```typescript
// src/services/router.service.ts
type Provider = "openai" | "anthropic";
type RoutingStrategy = "round-robin" | "lowest-latency" | "random";

interface ApiKeyEntry {
  id: string;
  provider: Provider;
  key: string;
  consecutiveFailures: number;
  circuitOpenUntil: number | null; // 时间戳，null 表示正常
  avgLatencyMs: number;
  requestCount: number;
}

class LLMRouter {
  private keys: ApiKeyEntry[] = [];
  private roundRobinIndex = 0;

  addKey(provider: Provider, key: string): void {
    this.keys.push({
      id: `${provider}_${this.keys.length}`,
      provider,
      key,
      consecutiveFailures: 0,
      circuitOpenUntil: null,
      avgLatencyMs: 0,
      requestCount: 0,
    });
  }

  private getAvailableKeys(): ApiKeyEntry[] {
    const now = Date.now();
    return this.keys.filter(
      (k) => k.circuitOpenUntil === null || now > k.circuitOpenUntil
    );
  }

  select(strategy: RoutingStrategy): ApiKeyEntry | null {
    const available = this.getAvailableKeys();
    if (available.length === 0) return null;

    switch (strategy) {
      case "round-robin": {
        const key = available[this.roundRobinIndex % available.length];
        this.roundRobinIndex++;
        return key;
      }
      case "lowest-latency": {
        return available.reduce((best, k) =>
          k.avgLatencyMs < best.avgLatencyMs ? k : best
        );
      }
      case "random": {
        return available[Math.floor(Math.random() * available.length)];
      }
    }
  }

  recordSuccess(keyId: string, latencyMs: number): void {
    const key = this.keys.find((k) => k.id === keyId);
    if (!key) return;

    key.consecutiveFailures = 0;
    key.circuitOpenUntil = null;
    // 滑动平均延迟
    key.avgLatencyMs =
      key.requestCount === 0
        ? latencyMs
        : (key.avgLatencyMs * key.requestCount + latencyMs) /
          (key.requestCount + 1);
    key.requestCount++;
  }

  recordFailure(keyId: string): void {
    const key = this.keys.find((k) => k.id === keyId);
    if (!key) return;

    key.consecutiveFailures++;

    // 连续失败 3 次，触发熔断（30 秒后恢复）
    if (key.consecutiveFailures >= 3) {
      key.circuitOpenUntil = Date.now() + 30_000;
      console.warn(
        `Key ${keyId} 已熔断，将在 30 秒后自动恢复`
      );
    }
  }

  getStatus(): Array<Omit<ApiKeyEntry, "key">> {
    return this.keys.map(({ key: _key, ...rest }) => rest);
  }
}

export const llmRouter = new LLMRouter();

// 初始化（从环境变量读取）
if (process.env.OPENAI_API_KEY) {
  llmRouter.addKey("openai", process.env.OPENAI_API_KEY);
}
```
