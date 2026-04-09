# 第23章：React前端集成

## 学习目标

完成本章学习后，你将能够：

1. 使用 Vite 或 Create React App 搭建 React+TypeScript 项目，正确配置 `tsconfig.json` 与 JSX 支持
2. 为 React 组件的 Props 定义精确类型，区分 `interface` 与 `type` 的使用场景，并处理可选属性与默认值
3. 掌握常用 Hooks 的 TypeScript 类型标注，包括 `useState`、`useReducer`、`useRef` 的泛型用法
4. 使用 Context API 与 Zustand 实现带完整类型约束的全局状态管理
5. 封装类型安全的 API 调用层，处理响应数据类型、加载状态与错误状态

---

## 23.1 React+TypeScript 项目配置

### 使用 Vite 创建项目

Vite 是目前最主流的 React+TypeScript 项目脚手架，比 Create React App 启动速度快得多。

```bash
# 创建新项目
npm create vite@latest my-ai-app -- --template react-ts

# 进入目录并安装依赖
cd my-ai-app
npm install
npm run dev
```

生成的项目结构如下：

```
my-ai-app/
├── public/
├── src/
│   ├── assets/
│   ├── App.tsx          # 根组件
│   ├── App.css
│   ├── main.tsx         # 入口文件
│   └── vite-env.d.ts    # Vite 环境类型声明
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts
```

### tsconfig.json 关键配置

Vite 模板生成的 `tsconfig.json` 已包含合理的默认值，但在 AI 应用开发中需要注意以下配置：

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,

    /* 模块解析 */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,

    /* JSX */
    "jsx": "react-jsx",

    /* 严格模式（强烈推荐开启） */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,

    /* 路径别名（可选） */
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### 安装 AI 应用常用依赖

```bash
# 类型定义（Vite 模板已包含 @types/react 和 @types/react-dom）
npm install @types/react @types/react-dom

# 状态管理
npm install zustand

# API 请求
npm install axios
npm install @types/axios  # 通常 axios 自带类型

# UI 组件库（按需选择）
npm install @headlessui/react
npm install lucide-react        # 图标库（自带 TS 类型）

# AI SDK
npm install openai
```

### vite.config.ts 配置路径别名

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    // 开发时代理 API 请求，避免 CORS 问题
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
      },
    },
  },
});
```

### 环境变量类型声明

在 `src/vite-env.d.ts` 中扩展环境变量类型，获得自动补全：

```typescript
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_OPENAI_API_KEY: string;
  readonly VITE_APP_TITLE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

使用时：

```typescript
// TypeScript 现在知道这些变量的类型
const apiUrl = import.meta.env.VITE_API_BASE_URL; // string
const title = import.meta.env.VITE_APP_TITLE;     // string
```

---

## 23.2 组件 Props 类型定义

### 基础 Props 类型

React 组件的 Props 可以用 `interface` 或 `type` 来定义。推荐使用 `interface`，因为它支持声明合并，更符合 React 生态惯例。

```typescript
// 基础按钮组件
interface ButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;          // 可选属性
  variant?: 'primary' | 'secondary' | 'danger';  // 字面量联合类型
  size?: 'sm' | 'md' | 'lg';
}

const Button: React.FC<ButtonProps> = ({
  label,
  onClick,
  disabled = false,
  variant = 'primary',
  size = 'md',
}) => {
  const baseClass = 'rounded font-medium transition-colors';
  const sizeClass = { sm: 'px-2 py-1 text-sm', md: 'px-4 py-2', lg: 'px-6 py-3 text-lg' }[size];
  const variantClass = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    secondary: 'bg-gray-200 text-gray-800 hover:bg-gray-300',
    danger: 'bg-red-600 text-white hover:bg-red-700',
  }[variant];

  return (
    <button
      className={`${baseClass} ${sizeClass} ${variantClass}`}
      onClick={onClick}
      disabled={disabled}
    >
      {label}
    </button>
  );
};
```

### 扩展原生 HTML 元素属性

使用 `React.ComponentPropsWithoutRef` 继承原生属性，避免重复声明 `className`、`id` 等通用属性：

```typescript
// 扩展原生 input 元素
interface TextInputProps extends React.ComponentPropsWithoutRef<'input'> {
  label: string;
  error?: string;
  helperText?: string;
}

const TextInput: React.FC<TextInputProps> = ({
  label,
  error,
  helperText,
  className,
  ...restProps  // 透传所有原生 input 属性
}) => {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-sm font-medium text-gray-700">{label}</label>
      <input
        className={`border rounded px-3 py-2 ${error ? 'border-red-500' : 'border-gray-300'} ${className ?? ''}`}
        {...restProps}
      />
      {error && <span className="text-sm text-red-500">{error}</span>}
      {helperText && !error && <span className="text-sm text-gray-500">{helperText}</span>}
    </div>
  );
};
```

### children 的类型处理

```typescript
// 方式一：使用 React.PropsWithChildren 工具类型
interface CardProps extends React.PropsWithChildren {
  title: string;
  footer?: React.ReactNode;
}

// 方式二：手动声明 children
interface PanelProps {
  children: React.ReactNode;      // 最通用，接受任何可渲染内容
  header: React.ReactElement;     // 必须是 React 元素
  actions?: React.ReactNode;
}

// 方式三：render prop 模式
interface ListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  keyExtractor: (item: T) => string;
  emptyText?: string;
}

function List<T>({ items, renderItem, keyExtractor, emptyText = '暂无数据' }: ListProps<T>) {
  if (items.length === 0) {
    return <div className="text-gray-400 text-center py-8">{emptyText}</div>;
  }
  return (
    <ul>
      {items.map((item, index) => (
        <li key={keyExtractor(item)}>{renderItem(item, index)}</li>
      ))}
    </ul>
  );
}
```

### 事件处理器类型

```typescript
interface FormProps {
  onSubmit: (data: FormData) => void;
  onChange?: (field: string, value: string) => void;
}

// 常用 React 事件类型速查
type EventHandlers = {
  onClick: React.MouseEventHandler<HTMLButtonElement>;
  onChange: React.ChangeEventHandler<HTMLInputElement>;
  onSubmit: React.FormEventHandler<HTMLFormElement>;
  onKeyDown: React.KeyboardEventHandler<HTMLInputElement>;
  onFocus: React.FocusEventHandler<HTMLInputElement>;
  onBlur: React.FocusEventHandler<HTMLInputElement>;
};

// 实际使用示例
const SearchBar: React.FC = () => {
  const handleChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const value = e.target.value; // TypeScript 知道这是 string
    console.log(value);
  };

  const handleKeyDown: React.KeyboardEventHandler<HTMLInputElement> = (e) => {
    if (e.key === 'Enter') {
      // 处理回车搜索
    }
  };

  return <input onChange={handleChange} onKeyDown={handleKeyDown} />;
};
```

---

## 23.3 Hooks 类型（useState、useReducer、useRef）

### useState 类型标注

大多数情况下 TypeScript 能从初始值推断类型，但在处理复杂对象或 `null` 初始值时需要显式标注：

```typescript
// 简单类型：自动推断
const [count, setCount] = useState(0);           // number
const [name, setName] = useState('');            // string
const [visible, setVisible] = useState(false);  // boolean

// 复杂对象：显式泛型
interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'user';
}

// 初始值为 null 时必须显式标注
const [user, setUser] = useState<User | null>(null);

// 数组类型
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

const [messages, setMessages] = useState<Message[]>([]);

// 使用时 TypeScript 知道 user 可能为 null
function UserProfile() {
  const [user, setUser] = useState<User | null>(null);

  // 必须做 null 检查才能访问属性
  return (
    <div>
      {user ? (
        <span>{user.name} ({user.role})</span>
      ) : (
        <span>未登录</span>
      )}
    </div>
  );
}
```

### useReducer 类型标注

`useReducer` 适合管理具有多种操作的复杂状态，TypeScript 的可辨识联合类型让 action 类型推断非常精确：

```typescript
// 定义状态类型
interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  inputText: string;
}

// 定义 Action 可辨识联合类型
type ChatAction =
  | { type: 'ADD_MESSAGE'; payload: Message }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_INPUT'; payload: string }
  | { type: 'CLEAR_MESSAGES' };

// 初始状态
const initialChatState: ChatState = {
  messages: [],
  isLoading: false,
  error: null,
  inputText: '',
};

// Reducer 函数（每个 case 中 action 类型自动收窄）
function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'ADD_MESSAGE':
      // action.payload 类型为 Message
      return { ...state, messages: [...state.messages, action.payload] };

    case 'SET_LOADING':
      // action.payload 类型为 boolean
      return { ...state, isLoading: action.payload };

    case 'SET_ERROR':
      // action.payload 类型为 string | null
      return { ...state, error: action.payload, isLoading: false };

    case 'SET_INPUT':
      return { ...state, inputText: action.payload };

    case 'CLEAR_MESSAGES':
      // 这个 action 没有 payload
      return { ...state, messages: [] };

    default:
      return state;
  }
}

// 在组件中使用
function ChatApp() {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);

  const sendMessage = () => {
    const newMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: state.inputText,
      timestamp: Date.now(),
    };
    dispatch({ type: 'ADD_MESSAGE', payload: newMessage });
    dispatch({ type: 'SET_INPUT', payload: '' });
    dispatch({ type: 'SET_LOADING', payload: true });
  };

  return (
    <div>
      <p>消息数：{state.messages.length}</p>
      {state.isLoading && <p>加载中...</p>}
    </div>
  );
}
```

### useRef 类型标注

`useRef` 有两种用途，类型标注方式不同：

```typescript
// 用途一：引用 DOM 元素（初始值为 null，类型为 HTMLElement）
function AutoFocusInput() {
  // 必须传 null 作为初始值，类型参数指定 DOM 元素类型
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // inputRef.current 可能为 null，需要检查
    inputRef.current?.focus();
  }, []);

  const scrollToInput = () => {
    inputRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  return <input ref={inputRef} placeholder="自动聚焦" />;
}

// 用途二：存储可变值（不触发重渲染）
function StreamingText() {
  const [displayText, setDisplayText] = useState('');
  // 存储累积的完整文本，不需要触发渲染
  const fullTextRef = useRef<string>('');
  // 存储定时器 ID
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // 存储 AbortController
  const abortControllerRef = useRef<AbortController | null>(null);

  const startStream = async () => {
    abortControllerRef.current = new AbortController();
    fullTextRef.current = '';

    // 模拟流式输出
    const words = ['Hello', ', ', 'world', '! ', 'TypeScript', ' is', ' awesome.'];
    for (const word of words) {
      if (abortControllerRef.current.signal.aborted) break;
      fullTextRef.current += word;
      setDisplayText(fullTextRef.current);
      await new Promise(resolve => setTimeout(resolve, 150));
    }
  };

  const stopStream = () => {
    abortControllerRef.current?.abort();
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
    }
  };

  return (
    <div>
      <p>{displayText}</p>
      <button onClick={startStream}>开始</button>
      <button onClick={stopStream}>停止</button>
    </div>
  );
}
```

### 自定义 Hook 类型

```typescript
// 封装 useLocalStorage
function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T) => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? (JSON.parse(item) as T) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setValue = (value: T): void => {
    try {
      setStoredValue(value);
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error('Failed to save to localStorage:', error);
    }
  };

  return [storedValue, setValue];
}

// 使用时完整的类型推断
function Settings() {
  const [theme, setTheme] = useLocalStorage<'light' | 'dark'>('theme', 'light');
  const [fontSize, setFontSize] = useLocalStorage<number>('fontSize', 14);

  return (
    <div>
      <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
        切换主题: {theme}
      </button>
      <input
        type="number"
        value={fontSize}
        onChange={e => setFontSize(Number(e.target.value))}
      />
    </div>
  );
}
```

---

## 23.4 状态管理类型（Context、Zustand）

### Context API 类型安全模式

原生 Context API 在 TypeScript 中需要处理 `undefined` 默认值问题。推荐使用自定义 Hook 封装，强制在 Provider 内部使用：

```typescript
// 定义 Context 数据类型
interface ThemeContextType {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  primaryColor: string;
}

// 创建 Context（默认值设为 undefined，通过 Hook 强制检查）
const ThemeContext = React.createContext<ThemeContextType | undefined>(undefined);

// Provider 组件
interface ThemeProviderProps {
  children: React.ReactNode;
  defaultTheme?: 'light' | 'dark';
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({
  children,
  defaultTheme = 'light',
}) => {
  const [theme, setTheme] = useState<'light' | 'dark'>(defaultTheme);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'light' ? 'dark' : 'light'));
  };

  const value: ThemeContextType = {
    theme,
    toggleTheme,
    primaryColor: theme === 'light' ? '#1d4ed8' : '#60a5fa',
  };

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
};

// 自定义 Hook，在 Provider 外部使用时抛出明确错误
export function useTheme(): ThemeContextType {
  const context = React.useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme 必须在 ThemeProvider 内部使用');
  }
  return context;
}

// 使用
function Header() {
  const { theme, toggleTheme } = useTheme();
  return (
    <header className={theme === 'dark' ? 'bg-gray-900 text-white' : 'bg-white text-gray-900'}>
      <button onClick={toggleTheme}>切换主题</button>
    </header>
  );
}
```

### Zustand 状态管理

Zustand 是轻量级状态管理库，比 Redux 更简洁，天然支持 TypeScript：

```typescript
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// 定义消息类型
interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

// 定义 Store 类型（包含状态和操作）
interface ChatStore {
  // 状态
  messages: Message[];
  isStreaming: boolean;
  error: string | null;
  model: string;

  // 操作
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void;
  updateLastMessage: (content: string) => void;
  clearMessages: () => void;
  setStreaming: (isStreaming: boolean) => void;
  setError: (error: string | null) => void;
  setModel: (model: string) => void;
}

// 创建 Store
export const useChatStore = create<ChatStore>()(
  devtools(
    persist(
      (set, get) => ({
        // 初始状态
        messages: [],
        isStreaming: false,
        error: null,
        model: 'gpt-4o-mini',

        // 操作实现
        addMessage: (messageData) => {
          const message: Message = {
            ...messageData,
            id: crypto.randomUUID(),
            timestamp: Date.now(),
          };
          set((state) => ({ messages: [...state.messages, message] }));
        },

        updateLastMessage: (content) => {
          set((state) => {
            const messages = [...state.messages];
            const lastIndex = messages.length - 1;
            if (lastIndex >= 0 && messages[lastIndex].role === 'assistant') {
              messages[lastIndex] = { ...messages[lastIndex], content };
            }
            return { messages };
          });
        },

        clearMessages: () => set({ messages: [] }),

        setStreaming: (isStreaming) => set({ isStreaming }),

        setError: (error) => set({ error, isStreaming: false }),

        setModel: (model) => set({ model }),
      }),
      {
        name: 'chat-storage',                          // localStorage key
        partialize: (state) => ({ messages: state.messages, model: state.model }), // 只持久化部分状态
      }
    )
  )
);

// 在组件中使用（支持选择器，避免不必要的重渲染）
function MessageList() {
  // 只订阅 messages，其他状态变化不触发重渲染
  const messages = useChatStore((state) => state.messages);

  return (
    <ul>
      {messages.map((msg) => (
        <li key={msg.id} className={msg.role === 'user' ? 'text-right' : 'text-left'}>
          {msg.content}
        </li>
      ))}
    </ul>
  );
}

function ChatInput() {
  const { addMessage, setStreaming, isStreaming } = useChatStore();
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (!input.trim() || isStreaming) return;
    addMessage({ role: 'user', content: input });
    setInput('');
    setStreaming(true);
  };

  return (
    <div className="flex gap-2">
      <input value={input} onChange={e => setInput(e.target.value)} />
      <button onClick={handleSend} disabled={isStreaming}>发送</button>
    </div>
  );
}
```

---

## 23.5 API 调用与数据类型

### 定义 API 响应类型

建立统一的 API 类型定义文件，是大型项目的最佳实践：

```typescript
// src/types/api.ts

// 通用 API 响应包装类型
interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
}

interface ApiError {
  code: string;
  message: string;
  details?: Record<string, string[]>;
}

// 分页响应
interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

// 聊天相关类型
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: string;   // ISO 8601 字符串
}

interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  model: string;
  createdAt: string;
  updatedAt: string;
}

interface CreateChatRequest {
  message: string;
  model?: string;
  sessionId?: string;
  systemPrompt?: string;
}

interface CreateChatResponse {
  sessionId: string;
  message: ChatMessage;
}
```

### 封装类型安全的 API 客户端

```typescript
// src/lib/apiClient.ts
import axios, { AxiosInstance, AxiosError } from 'axios';

class ApiClient {
  private readonly client: AxiosInstance;

  constructor(baseURL: string) {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: { 'Content-Type': 'application/json' },
    });

    // 请求拦截器：自动添加认证 token
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // 响应拦截器：统一错误处理
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError<ApiError>) => {
        const message = error.response?.data?.message ?? '请求失败，请重试';
        return Promise.reject(new Error(message));
      }
    );
  }

  async get<T>(url: string, params?: Record<string, unknown>): Promise<T> {
    const response = await this.client.get<ApiResponse<T>>(url, { params });
    return response.data.data;
  }

  async post<TRequest, TResponse>(url: string, data: TRequest): Promise<TResponse> {
    const response = await this.client.post<ApiResponse<TResponse>>(url, data);
    return response.data.data;
  }

  async delete<T>(url: string): Promise<T> {
    const response = await this.client.delete<ApiResponse<T>>(url);
    return response.data.data;
  }
}

export const apiClient = new ApiClient(import.meta.env.VITE_API_BASE_URL);
```

### 封装 API 服务层

```typescript
// src/services/chatService.ts
import { apiClient } from '@/lib/apiClient';

export const chatService = {
  async getSessions(): Promise<PaginatedResponse<ChatSession>> {
    return apiClient.get<PaginatedResponse<ChatSession>>('/chat/sessions');
  },

  async getSession(sessionId: string): Promise<ChatSession> {
    return apiClient.get<ChatSession>(`/chat/sessions/${sessionId}`);
  },

  async sendMessage(request: CreateChatRequest): Promise<CreateChatResponse> {
    return apiClient.post<CreateChatRequest, CreateChatResponse>('/chat/messages', request);
  },

  async deleteSession(sessionId: string): Promise<void> {
    return apiClient.delete<void>(`/chat/sessions/${sessionId}`);
  },
};
```

### 使用自定义 Hook 管理请求状态

```typescript
// src/hooks/useAsync.ts

interface AsyncState<T> {
  data: T | null;
  isLoading: boolean;
  error: Error | null;
}

type AsyncAction<T> =
  | { type: 'LOADING' }
  | { type: 'SUCCESS'; payload: T }
  | { type: 'ERROR'; payload: Error };

function asyncReducer<T>(state: AsyncState<T>, action: AsyncAction<T>): AsyncState<T> {
  switch (action.type) {
    case 'LOADING':
      return { data: null, isLoading: true, error: null };
    case 'SUCCESS':
      return { data: action.payload, isLoading: false, error: null };
    case 'ERROR':
      return { data: null, isLoading: false, error: action.payload };
  }
}

function useAsync<T>(asyncFn: () => Promise<T>, deps: React.DependencyList = []) {
  const [state, dispatch] = useReducer(asyncReducer<T>, {
    data: null,
    isLoading: false,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    dispatch({ type: 'LOADING' });

    asyncFn()
      .then((data) => {
        if (!cancelled) dispatch({ type: 'SUCCESS', payload: data });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          dispatch({
            type: 'ERROR',
            payload: error instanceof Error ? error : new Error(String(error)),
          });
        }
      });

    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}

// 使用示例
function SessionList() {
  const { data, isLoading, error } = useAsync(
    () => chatService.getSessions(),
    []  // 只在组件挂载时执行
  );

  if (isLoading) return <div>加载会话列表...</div>;
  if (error) return <div>错误：{error.message}</div>;
  if (!data) return null;

  return (
    <ul>
      {data.items.map((session) => (
        <li key={session.id}>{session.title}</li>
      ))}
    </ul>
  );
}
```

---

## 本章小结

| 知识点 | 核心要点 | 推荐实践 |
|--------|----------|----------|
| 项目配置 | Vite + `react-ts` 模板，开启 `strict: true` | 配置路径别名 `@/`，声明环境变量类型 |
| Props 类型 | 用 `interface` 定义，扩展原生 HTML 属性 | 使用 `React.ComponentPropsWithoutRef` 继承原生属性 |
| useState | 复杂类型和 `null` 初始值需显式泛型标注 | `useState<User \| null>(null)` |
| useReducer | 可辨识联合类型让 action 精确收窄 | 每个 action 类型独立声明 payload |
| useRef | DOM 引用用 `null` 初始值，可变值用具体初始值 | 区分两种用途，类型参数不同 |
| Context | 用 `undefined` 默认值 + 自定义 Hook 封装 | Hook 内部抛出明确错误，强制使用 Provider |
| Zustand | `create<StoreType>()` 定义完整 Store 类型 | 用选择器订阅，`persist` 中间件持久化 |
| API 调用 | 统一 `ApiResponse<T>` 包装类型 | 封装 API 客户端和服务层，Hook 管理请求状态 |

---

## AI 应用实战：聊天界面组件

下面实现一个完整的 AI 聊天界面，整合本章所有知识点：Props 类型、Hooks、Zustand 状态管理和 API 调用。

```typescript
// src/components/ChatInterface.tsx
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { create } from 'zustand';

// ====== 类型定义 ======

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  isStreaming?: boolean;
}

interface ChatStore {
  messages: Message[];
  isStreaming: boolean;
  error: string | null;
  model: string;
  addMessage: (msg: Omit<Message, 'id' | 'timestamp'>) => string;
  updateMessage: (id: string, content: string, isStreaming?: boolean) => void;
  setStreaming: (v: boolean) => void;
  setError: (e: string | null) => void;
  clearMessages: () => void;
}

// ====== Zustand Store ======

const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  isStreaming: false,
  error: null,
  model: 'gpt-4o-mini',

  addMessage: (msgData) => {
    const id = crypto.randomUUID();
    const message: Message = { ...msgData, id, timestamp: Date.now() };
    set((state) => ({ messages: [...state.messages, message] }));
    return id;
  },

  updateMessage: (id, content, isStreaming = false) => {
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === id ? { ...m, content, isStreaming } : m
      ),
    }));
  },

  setStreaming: (isStreaming) => set({ isStreaming }),
  setError: (error) => set({ error, isStreaming: false }),
  clearMessages: () => set({ messages: [] }),
}));

// ====== 子组件：单条消息 ======

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs font-bold mr-2 flex-shrink-0">
          AI
        </div>
      )}
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? 'bg-blue-600 text-white rounded-br-sm'
            : 'bg-gray-100 text-gray-800 rounded-bl-sm'
        }`}
      >
        <p className="whitespace-pre-wrap break-words">{message.content}</p>
        {message.isStreaming && (
          <span className="inline-block w-2 h-4 bg-gray-500 animate-pulse ml-1 align-text-bottom" />
        )}
        <p className={`text-xs mt-1 ${isUser ? 'text-blue-200' : 'text-gray-400'}`}>
          {new Date(message.timestamp).toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </p>
      </div>
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-gray-400 flex items-center justify-center text-white text-xs font-bold ml-2 flex-shrink-0">
          我
        </div>
      )}
    </div>
  );
};

// ====== 子组件：输入区域 ======

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled: boolean;
  onStop: () => void;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSend, disabled, onStop }) => {
  const [inputText, setInputText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // 自动调整高度
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  }, [inputText]);

  const handleSend = useCallback(() => {
    const text = inputText.trim();
    if (!text || disabled) return;
    onSend(text);
    setInputText('');
  }, [inputText, disabled, onSend]);

  const handleKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t bg-white px-4 py-3">
      <div className="flex items-end gap-2 max-w-3xl mx-auto">
        <textarea
          ref={textareaRef}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入消息，Enter 发送，Shift+Enter 换行..."
          rows={1}
          disabled={disabled}
          className="flex-1 resize-none rounded-xl border border-gray-300 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-400 max-h-[120px] overflow-y-auto"
        />
        {disabled ? (
          <button
            onClick={onStop}
            className="flex-shrink-0 w-10 h-10 rounded-full bg-red-500 text-white flex items-center justify-center hover:bg-red-600 transition-colors"
            title="停止生成"
          >
            ■
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!inputText.trim()}
            className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-600 text-white flex items-center justify-center hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            title="发送"
          >
            ▶
          </button>
        )}
      </div>
      <p className="text-xs text-gray-400 text-center mt-1">
        AI 回复仅供参考，请自行判断信息准确性
      </p>
    </div>
  );
};

// ====== 子组件：空状态 ======

const EmptyState: React.FC = () => (
  <div className="flex flex-col items-center justify-center h-full text-center px-8">
    <div className="w-16 h-16 rounded-full bg-blue-100 flex items-center justify-center mb-4">
      <span className="text-2xl">💬</span>
    </div>
    <h2 className="text-xl font-semibold text-gray-700 mb-2">开始 AI 对话</h2>
    <p className="text-gray-500 text-sm max-w-sm">
      向 AI 提问任何问题。支持代码分析、文字创作、数据解读等多种任务。
    </p>
    <div className="mt-6 grid grid-cols-2 gap-2 w-full max-w-sm">
      {['解释 TypeScript 泛型', '用 React 写一个 Todo 应用', '分析这段代码的性能问题', '帮我写一封商务邮件'].map(
        (suggestion) => (
          <button
            key={suggestion}
            className="text-xs text-left px-3 py-2 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-colors text-gray-600"
          >
            {suggestion}
          </button>
        )
      )}
    </div>
  </div>
);

// ====== 主组件：聊天界面 ======

interface ChatInterfaceProps {
  systemPrompt?: string;
  initialMessages?: Message[];
  className?: string;
  onMessageSent?: (message: Message) => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  systemPrompt = '你是一个专业的 TypeScript 和 AI 应用开发助手。',
  initialMessages = [],
  className = '',
  onMessageSent,
}) => {
  const { messages, isStreaming, error, addMessage, updateMessage, setStreaming, setError, clearMessages } =
    useChatStore();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // 初始化消息
  useEffect(() => {
    if (initialMessages.length > 0) {
      initialMessages.forEach((msg) => addMessage(msg));
    }
    return () => {
      abortControllerRef.current?.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // 发送消息并调用 AI API
  const handleSend = useCallback(
    async (text: string) => {
      // 添加用户消息
      const userMessage: Omit<Message, 'id' | 'timestamp'> = {
        role: 'user',
        content: text,
      };
      addMessage(userMessage);
      onMessageSent?.({ ...userMessage, id: '', timestamp: Date.now() });

      // 添加占位助手消息
      const assistantMsgId = addMessage({ role: 'assistant', content: '', isStreaming: true });
      setStreaming(true);
      setError(null);

      abortControllerRef.current = new AbortController();

      try {
        // 调用流式 API（假设后端代理，避免暴露 API Key）
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: [
              { role: 'system', content: systemPrompt },
              ...messages.map((m) => ({ role: m.role, content: m.content })),
              { role: 'user', content: text },
            ],
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error('无法读取响应流');

        const decoder = new TextDecoder();
        let accumulatedContent = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          // 解析 SSE 格式
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6).trim();
              if (data === '[DONE]') continue;
              try {
                const parsed = JSON.parse(data) as {
                  choices: Array<{ delta: { content?: string } }>;
                };
                const delta = parsed.choices[0]?.delta?.content ?? '';
                accumulatedContent += delta;
                updateMessage(assistantMsgId, accumulatedContent, true);
              } catch {
                // 忽略无法解析的行
              }
            }
          }
        }

        // 流式完成，更新最终内容
        updateMessage(assistantMsgId, accumulatedContent, false);
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          // 用户主动停止
          updateMessage(assistantMsgId, messages.find((m) => m.id === assistantMsgId)?.content ?? '', false);
        } else {
          const errorMsg = err instanceof Error ? err.message : '请求失败';
          setError(errorMsg);
          updateMessage(assistantMsgId, `[错误] ${errorMsg}`, false);
        }
      } finally {
        setStreaming(false);
      }
    },
    [messages, systemPrompt, addMessage, updateMessage, setStreaming, setError, onMessageSent]
  );

  const handleStop = () => {
    abortControllerRef.current?.abort();
  };

  return (
    <div className={`flex flex-col h-full bg-white rounded-xl shadow-lg overflow-hidden ${className}`}>
      {/* 顶部工具栏 */}
      <div className="flex items-center justify-between px-4 py-3 border-b bg-gray-50">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500" />
          <span className="text-sm font-medium text-gray-700">AI 助手</span>
          <span className="text-xs text-gray-400 bg-gray-200 px-2 py-0.5 rounded-full">
            gpt-4o-mini
          </span>
        </div>
        <button
          onClick={clearMessages}
          disabled={messages.length === 0 || isStreaming}
          className="text-xs text-gray-500 hover:text-red-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          清空对话
        </button>
      </div>

      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <>
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            {error && (
              <div className="text-center text-sm text-red-500 bg-red-50 rounded-lg px-4 py-2 mb-4">
                {error}
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* 输入区域 */}
      <ChatInput onSend={handleSend} disabled={isStreaming} onStop={handleStop} />
    </div>
  );
};

// ====== 使用示例 ======

export default function App() {
  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl h-[80vh]">
        <ChatInterface
          systemPrompt="你是一个专业的 TypeScript 开发助手，擅长解答编程问题并给出最佳实践建议。"
          onMessageSent={(msg) => console.log('新消息:', msg)}
        />
      </div>
    </div>
  );
}
```

上述代码整合了本章全部核心知识点：

- `ChatStore` 使用 Zustand 管理全局状态，完整定义了 `interface ChatStore`
- `MessageBubble` 和 `ChatInput` 通过 `interface Props` 严格定义 Props 类型
- `useRef<HTMLTextAreaElement>` 操作 DOM 元素，`useRef<AbortController>` 存储可变值
- `useReducer` 风格的 `asyncReducer` 处理加载状态
- `fetch` 调用流式 API，解析 SSE 数据并实时更新 UI

---

## 练习题

### 基础题

**练习 1**：定义一个 `AvatarProps` 接口，要求：
- `src`：图片 URL，必填
- `alt`：替代文本，必填
- `size`：枚举 `'sm' | 'md' | 'lg' | 'xl'`，可选，默认 `'md'`
- `shape`：`'circle' | 'square'`，可选，默认 `'circle'`
- 继承 `<img>` 元素的所有原生属性（排除 `src` 和 `alt`，因为已单独定义）

编写对应的 `Avatar` 组件实现。

---

**练习 2**：使用 `useReducer` 实现一个表单状态管理器，支持以下操作：
- `SET_FIELD`：更新指定字段值
- `SET_ERROR`：设置指定字段的错误信息
- `CLEAR_ERRORS`：清除所有错误
- `RESET`：重置到初始值

表单包含字段：`username: string`、`email: string`、`password: string`。

---

### 进阶题

**练习 3**：基于本章的 `useAsync` Hook，扩展实现一个 `usePagedData<T>` Hook，支持：
- 分页加载（`page`、`pageSize` 参数）
- `loadMore()` 方法加载下一页
- `refresh()` 方法重新从第一页加载
- 返回 `{ items, isLoading, hasMore, error, loadMore, refresh }` 结构
- 接受一个异步函数 `fetcher: (page: number, pageSize: number) => Promise<PaginatedResponse<T>>`

---

**练习 4**：为 `ChatInterface` 组件新增**历史会话**功能，使用 Zustand 实现：
- `SessionStore`：管理多个会话列表，包含 `sessions: ChatSession[]`、`currentSessionId: string | null`
- 操作：`createSession()`、`switchSession(id: string)`、`deleteSession(id: string)`
- `currentSession` 计算属性，返回当前会话对象
- 使用 `persist` 中间件持久化到 `localStorage`

---

### 挑战题

**练习 5**：实现一个类型安全的**插件系统**，允许为 `ChatInterface` 注册插件：

```typescript
interface ChatPlugin {
  name: string;
  // 消息发送前的钩子，可以修改消息或阻止发送
  beforeSend?: (message: string) => string | false;
  // 消息接收后的钩子，可以处理响应内容
  afterReceive?: (content: string) => string;
  // 渲染额外的 UI 元素（工具栏按钮等）
  renderToolbarItem?: () => React.ReactNode;
}
```

要求：
1. 创建 `PluginManager` 类，方法 `register(plugin: ChatPlugin)`、`unregister(name: string)`、`runBeforeSend(msg: string): string | false`、`runAfterReceive(content: string): string`
2. 将 `PluginManager` 通过 Context 提供给子组件
3. 创建两个示例插件：`WordCountPlugin`（统计字数并显示在工具栏）和 `CensorPlugin`（过滤敏感词）
4. 修改 `ChatInterface` 的 `handleSend` 逻辑，在发送前调用 `runBeforeSend`，在接收后调用 `runAfterReceive`

---

## 练习答案

### 练习 1 答案

```typescript
// 使用 Omit 排除已单独定义的 src 和 alt 属性
interface AvatarProps extends Omit<React.ComponentPropsWithoutRef<'img'>, 'src' | 'alt'> {
  src: string;
  alt: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  shape?: 'circle' | 'square';
}

const sizeMap: Record<NonNullable<AvatarProps['size']>, string> = {
  sm: 'w-6 h-6',
  md: 'w-10 h-10',
  lg: 'w-16 h-16',
  xl: 'w-24 h-24',
};

const Avatar: React.FC<AvatarProps> = ({
  src,
  alt,
  size = 'md',
  shape = 'circle',
  className,
  ...restProps
}) => {
  return (
    <img
      src={src}
      alt={alt}
      className={`object-cover ${sizeMap[size]} ${shape === 'circle' ? 'rounded-full' : 'rounded-md'} ${className ?? ''}`}
      {...restProps}
    />
  );
};
```

### 练习 2 答案

```typescript
interface FormState {
  values: {
    username: string;
    email: string;
    password: string;
  };
  errors: Partial<Record<'username' | 'email' | 'password', string>>;
}

type FormField = keyof FormState['values'];

type FormAction =
  | { type: 'SET_FIELD'; field: FormField; value: string }
  | { type: 'SET_ERROR'; field: FormField; error: string }
  | { type: 'CLEAR_ERRORS' }
  | { type: 'RESET' };

const initialFormState: FormState = {
  values: { username: '', email: '', password: '' },
  errors: {},
};

function formReducer(state: FormState, action: FormAction): FormState {
  switch (action.type) {
    case 'SET_FIELD':
      return {
        ...state,
        values: { ...state.values, [action.field]: action.value },
      };
    case 'SET_ERROR':
      return {
        ...state,
        errors: { ...state.errors, [action.field]: action.error },
      };
    case 'CLEAR_ERRORS':
      return { ...state, errors: {} };
    case 'RESET':
      return initialFormState;
  }
}

function RegistrationForm() {
  const [state, dispatch] = useReducer(formReducer, initialFormState);

  const handleSubmit: React.FormEventHandler<HTMLFormElement> = (e) => {
    e.preventDefault();
    dispatch({ type: 'CLEAR_ERRORS' });

    if (!state.values.username) {
      dispatch({ type: 'SET_ERROR', field: 'username', error: '用户名不能为空' });
      return;
    }
    if (!state.values.email.includes('@')) {
      dispatch({ type: 'SET_ERROR', field: 'email', error: '邮箱格式不正确' });
      return;
    }
    if (state.values.password.length < 8) {
      dispatch({ type: 'SET_ERROR', field: 'password', error: '密码至少8位' });
      return;
    }
    console.log('提交：', state.values);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        value={state.values.username}
        onChange={(e) => dispatch({ type: 'SET_FIELD', field: 'username', value: e.target.value })}
        placeholder="用户名"
      />
      {state.errors.username && <span>{state.errors.username}</span>}

      <input
        type="email"
        value={state.values.email}
        onChange={(e) => dispatch({ type: 'SET_FIELD', field: 'email', value: e.target.value })}
        placeholder="邮箱"
      />
      {state.errors.email && <span>{state.errors.email}</span>}

      <input
        type="password"
        value={state.values.password}
        onChange={(e) => dispatch({ type: 'SET_FIELD', field: 'password', value: e.target.value })}
        placeholder="密码"
      />
      {state.errors.password && <span>{state.errors.password}</span>}

      <button type="submit">注册</button>
      <button type="button" onClick={() => dispatch({ type: 'RESET' })}>重置</button>
    </form>
  );
}
```

### 练习 3 答案

```typescript
import { useState, useCallback, useRef } from 'react';

interface PagedState<T> {
  items: T[];
  isLoading: boolean;
  error: Error | null;
  hasMore: boolean;
  currentPage: number;
}

function usePagedData<T>(
  fetcher: (page: number, pageSize: number) => Promise<PaginatedResponse<T>>,
  pageSize = 20
) {
  const [state, setState] = useState<PagedState<T>>({
    items: [],
    isLoading: false,
    error: null,
    hasMore: true,
    currentPage: 0,
  });

  // 使用 ref 跟踪当前加载状态，避免竞态条件
  const loadingRef = useRef(false);

  const loadPage = useCallback(
    async (page: number, append: boolean) => {
      if (loadingRef.current) return;
      loadingRef.current = true;
      setState((prev) => ({ ...prev, isLoading: true, error: null }));

      try {
        const result = await fetcher(page, pageSize);
        setState((prev) => ({
          items: append ? [...prev.items, ...result.items] : result.items,
          isLoading: false,
          error: null,
          hasMore: result.hasMore,
          currentPage: page,
        }));
      } catch (err) {
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: err instanceof Error ? err : new Error(String(err)),
        }));
      } finally {
        loadingRef.current = false;
      }
    },
    [fetcher, pageSize]
  );

  const loadMore = useCallback(() => {
    if (!state.hasMore || state.isLoading) return;
    loadPage(state.currentPage + 1, true);
  }, [state.hasMore, state.isLoading, state.currentPage, loadPage]);

  const refresh = useCallback(() => {
    loadPage(1, false);
  }, [loadPage]);

  // 初始加载
  useEffect(() => {
    loadPage(1, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    items: state.items,
    isLoading: state.isLoading,
    hasMore: state.hasMore,
    error: state.error,
    loadMore,
    refresh,
  };
}
```

### 练习 4 答案

```typescript
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

interface SessionStore {
  sessions: ChatSession[];
  currentSessionId: string | null;
  currentSession: ChatSession | null;
  createSession: (title?: string) => string;
  switchSession: (id: string) => void;
  deleteSession: (id: string) => void;
  addMessageToSession: (sessionId: string, message: Message) => void;
}

const useSessionStore = create<SessionStore>()(
  persist(
    (set, get) => ({
      sessions: [],
      currentSessionId: null,

      get currentSession() {
        const { sessions, currentSessionId } = get();
        return sessions.find((s) => s.id === currentSessionId) ?? null;
      },

      createSession: (title = '新对话') => {
        const id = crypto.randomUUID();
        const session: ChatSession = {
          id,
          title,
          messages: [],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        set((state) => ({
          sessions: [session, ...state.sessions],
          currentSessionId: id,
        }));
        return id;
      },

      switchSession: (id) => {
        set({ currentSessionId: id });
      },

      deleteSession: (id) => {
        set((state) => {
          const filtered = state.sessions.filter((s) => s.id !== id);
          const newCurrentId =
            state.currentSessionId === id
              ? (filtered[0]?.id ?? null)
              : state.currentSessionId;
          return { sessions: filtered, currentSessionId: newCurrentId };
        });
      },

      addMessageToSession: (sessionId, message) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId
              ? { ...s, messages: [...s.messages, message], updatedAt: Date.now() }
              : s
          ),
        }));
      },
    }),
    {
      name: 'chat-sessions',
      partialize: (state) => ({
        sessions: state.sessions,
        currentSessionId: state.currentSessionId,
      }),
    }
  )
);
```

### 练习 5 答案

```typescript
// ====== 类型定义 ======
interface ChatPlugin {
  name: string;
  beforeSend?: (message: string) => string | false;
  afterReceive?: (content: string) => string;
  renderToolbarItem?: () => React.ReactNode;
}

// ====== PluginManager 类 ======
class PluginManager {
  private plugins = new Map<string, ChatPlugin>();

  register(plugin: ChatPlugin): void {
    if (this.plugins.has(plugin.name)) {
      console.warn(`插件 "${plugin.name}" 已注册，将覆盖旧版本`);
    }
    this.plugins.set(plugin.name, plugin);
  }

  unregister(name: string): void {
    this.plugins.delete(name);
  }

  runBeforeSend(message: string): string | false {
    let current: string | false = message;
    for (const plugin of this.plugins.values()) {
      if (plugin.beforeSend) {
        const result = plugin.beforeSend(current as string);
        if (result === false) return false;  // 阻止发送
        current = result;
      }
    }
    return current;
  }

  runAfterReceive(content: string): string {
    let current = content;
    for (const plugin of this.plugins.values()) {
      if (plugin.afterReceive) {
        current = plugin.afterReceive(current);
      }
    }
    return current;
  }

  getToolbarItems(): React.ReactNode[] {
    return Array.from(this.plugins.values())
      .filter((p) => p.renderToolbarItem)
      .map((p) => p.renderToolbarItem!());
  }
}

// ====== Context ======
const PluginContext = React.createContext<PluginManager | undefined>(undefined);

interface PluginProviderProps {
  children: React.ReactNode;
  plugins?: ChatPlugin[];
}

export const PluginProvider: React.FC<PluginProviderProps> = ({ children, plugins = [] }) => {
  const manager = useRef(new PluginManager()).current;

  useEffect(() => {
    plugins.forEach((p) => manager.register(p));
    return () => plugins.forEach((p) => manager.unregister(p.name));
  }, []); // 仅初始化一次

  return <PluginContext.Provider value={manager}>{children}</PluginContext.Provider>;
};

export function usePluginManager(): PluginManager {
  const ctx = React.useContext(PluginContext);
  if (!ctx) throw new Error('usePluginManager 必须在 PluginProvider 内使用');
  return ctx;
}

// ====== 示例插件 ======

// 字数统计插件
const wordCountPlugin: ChatPlugin = {
  name: 'word-count',
  beforeSend: (message) => {
    console.log(`发送字数：${message.length}`);
    return message;  // 不修改，直接透传
  },
  renderToolbarItem: () => (
    <span key="word-count" className="text-xs text-gray-400">字数统计已启用</span>
  ),
};

// 敏感词过滤插件
const censorPlugin: ChatPlugin = {
  name: 'censor',
  beforeSend: (message) => {
    const sensitiveWords = ['违禁词1', '违禁词2'];
    const hasSensitive = sensitiveWords.some((w) => message.includes(w));
    if (hasSensitive) {
      alert('消息包含不当内容，已阻止发送');
      return false;  // 阻止发送
    }
    return message;
  },
  afterReceive: (content) => {
    // 对 AI 响应也进行过滤
    return content.replace(/违禁词1|违禁词2/g, '***');
  },
};

// ====== 使用示例 ======
export default function AppWithPlugins() {
  return (
    <PluginProvider plugins={[wordCountPlugin, censorPlugin]}>
      <ChatInterface systemPrompt="你是一个助手" />
    </PluginProvider>
  );
}
```
