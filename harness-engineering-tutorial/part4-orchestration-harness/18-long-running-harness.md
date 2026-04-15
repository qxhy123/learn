# 第18章：长时运行 Agent Harness

> 让 AI 写一个函数只需 20 秒，让 AI 构建一个完整系统需要 6 小时。从分钟级到小时级，agent 面临的不再是能力问题，而是记忆问题。长时运行 Harness 的核心使命是：**让 agent 在遗忘中依然能做对事情**。

---

## 学习目标

学完本章，你将能够：

1. 理解长时运行 agent 的上下文窗口瓶颈
2. 区分 context reset 与 context compaction 的适用场景
3. 设计结构化 artifact 实现跨会话状态传递
4. 实现 Initializer + Coding Agent 双 harness 模式
5. 评估长时运行 agent 的成本效益（单次 $9 vs 完整 harness $200）

---

## 18.1 上下文窗口问题：离散会话与无记忆

### 根本矛盾

LLM 的上下文窗口是一个"短期记忆"——它的大小是固定的，但任务的复杂度不是：

```
任务持续时间        上下文需求        模型上下文窗口

5 分钟任务    ═══                ╔════════════════╗
              ▏                  ║                ║
30 分钟任务   ═══════════        ║    200K tokens  ║
              ▏                  ║    (Claude)     ║
2 小时任务    ═══════════════    ║                ║
              ▏ ← 溢出 →        ╚════════════════╝
6 小时任务    ═════════════════════════════╸
```

### 会话断裂的后果

```python
# 模拟上下文断裂的影响
class ContextLossSimulator:
    """模拟 agent 在不同阶段丢失上下文的后果"""

    LOSS_PATTERNS = {
        "architecture_decision": {
            "lost": "为什么选 PostgreSQL 而不是 MongoDB",
            "consequence": "后续 agent 可能重新选型或写出不兼容的代码",
            "severity": "CRITICAL",
        },
        "naming_convention": {
            "lost": "项目使用 snake_case 而非 camelCase",
            "consequence": "生成的新代码风格不一致",
            "severity": "MEDIUM",
        },
        "partial_implementation": {
            "lost": "已实现了 3/5 个模块",
            "consequence": "重复实现已完成的模块或遗漏未完成的",
            "severity": "HIGH",
        },
        "error_history": {
            "lost": "之前尝试过方案A但因X原因失败",
            "consequence": "重复犯相同的错误",
            "severity": "HIGH",
        },
    }
```

### 三种上下文策略对比

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| Context Stuffing | 把所有信息塞进 prompt | 简单直接 | 窗口不够大 |
| Context Compaction | 压缩/摘要旧上下文 | 保留关键信息 | 压缩损失 |
| Context Reset | 清空重来，靠 artifact 交接 | 干净利落 | 设计成本高 |

---

## 18.2 Context Reset vs Context Compaction

### Context Compaction（上下文压缩）

```
原始上下文（150K tokens）：
┌────────────────────────────────────────────────────┐
│ system prompt │ 历史对话 │ 代码 │ 错误日志 │ 最新任务 │
│   5K          │  80K     │ 40K  │  15K     │  10K    │
└────────────────────────────────────────────────────┘
                        │
                    压缩 ▼
                        │
压缩后上下文（60K tokens）：
┌────────────────────────────────────────────────────┐
│ system prompt │ 历史摘要 │ 关键代码 │ 最新错误 │ 最新任务 │
│   5K          │  15K     │ 20K     │  5K     │  10K    │
└────────────────────────────────────────────────────┘
```

```python
class ContextCompactor:
    """上下文压缩器"""

    def __init__(self, llm_client, target_ratio: float = 0.4):
        self.llm = llm_client
        self.target_ratio = target_ratio  # 压缩到原来的 40%

    def compact(self, context: list[dict]) -> list[dict]:
        """压缩上下文，保留关键信息"""
        # 分类上下文
        system = [m for m in context if m["role"] == "system"]
        history = [m for m in context if m["role"] in ("user", "assistant")]

        # 保留最近 N 轮完整对话
        recent = history[-6:]  # 最近 3 轮
        older = history[:-6]

        if not older:
            return context  # 不需要压缩

        # 用 LLM 摘要旧对话
        summary = self._summarize(older)

        return system + [{
            "role": "system",
            "content": f"[之前对话摘要]\n{summary}",
        }] + recent

    def _summarize(self, messages: list[dict]) -> str:
        """摘要旧对话"""
        content = "\n".join(
            f"{m['role']}: {m['content'][:200]}" for m in messages
        )
        response = self.llm.generate(
            f"请摘要以下对话的关键信息（决策、代码变更、错误记录）：\n{content}"
        )
        return response
```

### Context Reset（上下文重置）

Context Reset 是更激进但更可靠的策略——**完全清空上下文，依靠结构化 artifact 传递状态**：

```
Session 1                 Artifact               Session 2
┌───────────────┐                                ┌───────────────┐
│ Agent 工作了   │    ┌─────────────────┐         │ 新 Agent 读取  │
│ 30 分钟       │───→│ progress.yaml   │───→     │ artifact 后   │
│               │    │ git history     │         │ 继续工作       │
│ 产出代码和    │    │ decisions.json  │         │               │
│ 决策记录      │    │ todo.md         │         │ 不需要历史    │
└───────────────┘    └─────────────────┘         │ 对话上下文     │
                                                  └───────────────┘
```

### 两种策略的选择矩阵

```
                    任务可分解性
                 低 ←─────────→ 高
           ┌──────────┬──────────┐
       低  │  短任务   │ Context  │
上下文     │  不需要   │  Reset   │
需求       │  策略     │          │
       高  │ Context  │ Context  │
           │Compaction│  Reset   │
           └──────────┴──────────┘
```

---

## 18.3 结构化 Artifact 做跨会话交接

### Artifact 设计原则

好的跨会话 artifact 必须满足三个条件：

1. **自包含**：不依赖上下文即可理解
2. **结构化**：机器可解析
3. **版本化**：记录变更历史

### 核心 Artifact 类型

```yaml
# progress.yaml — 进度跟踪 Artifact
project: user-auth-service
session_id: "session-003"
timestamp: "2026-04-15T10:30:00Z"
overall_progress: 65%

completed_tasks:
  - id: T001
    description: "设计数据库 schema"
    files_changed: ["schema.sql", "models.py"]
    decisions:
      - "选择 PostgreSQL，因为需要 JSON 字段支持"
      - "用户表分离 credentials 和 profile"

  - id: T002
    description: "实现注册 API"
    files_changed: ["routes/auth.py", "services/user.py"]
    decisions:
      - "密码使用 bcrypt 哈希"
      - "邮箱验证采用异步发送"

pending_tasks:
  - id: T003
    description: "实现登录 API 和 JWT 签发"
    priority: HIGH
    dependencies: [T001, T002]
    notes: "JWT secret 从环境变量读取"

  - id: T004
    description: "实现密码重置流程"
    priority: MEDIUM
    dependencies: [T002]

known_issues:
  - "邮箱验证的模板还没有设计"
  - "需要确认 JWT 过期时间的产品需求"

architecture_decisions:
  - decision: "REST API 而非 GraphQL"
    reason: "团队更熟悉 REST，项目规模不需要 GraphQL"
    date: "2026-04-15"
  - decision: "不使用 ORM 的自动迁移"
    reason: "生产环境需要手动审核迁移脚本"
    date: "2026-04-15"
```

### Artifact 管理器

```python
import yaml
import json
from pathlib import Path
from datetime import datetime

class ArtifactManager:
    """跨会话 Artifact 管理器"""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.artifact_dir = self.project_dir / ".harness" / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def save_progress(self, progress: dict) -> Path:
        """保存进度 artifact"""
        progress["timestamp"] = datetime.now().isoformat()
        path = self.artifact_dir / "progress.yaml"
        with open(path, "w") as f:
            yaml.dump(progress, f, allow_unicode=True, default_flow_style=False)
        return path

    def load_progress(self) -> dict | None:
        """加载最新进度"""
        path = self.artifact_dir / "progress.yaml"
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return None

    def save_decision_log(self, decision: dict) -> None:
        """追加决策记录"""
        log_path = self.artifact_dir / "decisions.jsonl"
        decision["timestamp"] = datetime.now().isoformat()
        with open(log_path, "a") as f:
            f.write(json.dumps(decision, ensure_ascii=False) + "\n")

    def build_handoff_prompt(self) -> str:
        """构建交接 prompt：新会话读取此 prompt 即可继续工作"""
        progress = self.load_progress()
        if not progress:
            return "这是一个新项目，没有历史进度。"

        prompt = f"""# 项目交接摘要

## 项目：{progress.get('project', 'unknown')}
## 总进度：{progress.get('overall_progress', 0)}%
## 上次会话：{progress.get('session_id', 'unknown')}

## 已完成任务
"""
        for task in progress.get("completed_tasks", []):
            prompt += f"- [{task['id']}] {task['description']}\n"
            for d in task.get("decisions", []):
                prompt += f"  决策：{d}\n"

        prompt += "\n## 待完成任务\n"
        for task in progress.get("pending_tasks", []):
            prompt += f"- [{task['id']}] {task['description']} (优先级: {task['priority']})\n"

        prompt += "\n## 已知问题\n"
        for issue in progress.get("known_issues", []):
            prompt += f"- {issue}\n"

        return prompt
```

### Git 作为 Artifact 的补充

```
Artifact 层                          Git 层
┌──────────────────┐                 ┌──────────────────┐
│ progress.yaml    │  高层抽象        │ git log          │  代码级细节
│ "完成了注册API"  │                 │ "feat: add       │
│                  │                 │  registration    │
│ decisions.json   │  决策记录        │  endpoint"       │
│ "选 PostgreSQL"  │                 │                  │
│                  │                 │ git diff         │  精确变更
│ todo.md          │  待办清单        │ 每一行改动       │
└──────────────────┘                 └──────────────────┘
         │                                    │
         └──── 新 Agent 读取两者来恢复上下文 ────┘
```

---

## 18.4 Initializer Agent + Coding Agent 双 Harness 模式

### 为什么需要两个 Agent

这是长时运行系统的关键设计模式：

```
问题：Coding Agent 擅长写代码，但不擅长理解"此刻应该做什么"
解决：让 Initializer Agent 专门负责"设置上下文"

┌─────────────────┐        ┌──────────────────┐
│  Initializer    │        │   Coding Agent   │
│     Agent       │        │                  │
│                 │ prompt  │                  │
│ 读 artifact    │───────→│  写代码          │
│ 读 git history │        │  跑测试          │
│ 分析当前状态   │        │  修 bug          │
│ 生成任务 brief │        │                  │
│                 │        │  不需要理解      │
│ "你现在应该    │        │  项目全局状态    │
│  做 X 和 Y"    │        │                  │
└─────────────────┘        └──────────────────┘
```

### 双 Harness 实现

```python
class InitializerAgent:
    """初始化 Agent：为 Coding Agent 准备上下文"""

    def __init__(self, llm_client, artifact_manager: ArtifactManager):
        self.llm = llm_client
        self.artifacts = artifact_manager

    def prepare_session(self) -> dict:
        """准备新会话的上下文"""
        # 1. 读取 artifact
        handoff = self.artifacts.build_handoff_prompt()

        # 2. 分析 git 状态
        git_status = self._get_git_status()

        # 3. 生成任务 brief
        brief = self.llm.generate(f"""
基于以下项目状态，确定下一步应该做什么：

{handoff}

Git 状态：
{git_status}

请输出：
1. 当前最高优先级任务
2. 该任务的具体步骤
3. 需要注意的约束和依赖
4. 验收标准
""")
        return {
            "handoff_context": handoff,
            "git_status": git_status,
            "task_brief": brief,
        }

    def _get_git_status(self) -> str:
        import subprocess
        result = subprocess.run(
            ["git", "log", "--oneline", "-10"],
            capture_output=True, text=True,
            cwd=self.artifacts.project_dir,
        )
        return result.stdout


class CodingAgentHarness:
    """Coding Agent Harness：负责执行具体编码任务"""

    def __init__(self, llm_client, artifact_manager: ArtifactManager):
        self.llm = llm_client
        self.artifacts = artifact_manager

    def execute_session(self, session_context: dict) -> dict:
        """执行编码会话"""
        system_prompt = f"""你是一个专业的编码 Agent。

项目背景：
{session_context['handoff_context']}

当前任务：
{session_context['task_brief']}

规则：
1. 严格按照任务 brief 执行
2. 每完成一个子任务，更新 progress artifact
3. 遇到阻塞问题，记录到 known_issues
4. 不要偏离任务范围
"""
        # 执行编码循环
        results = self._coding_loop(system_prompt)

        # 更新 artifact
        self.artifacts.save_progress(results["progress"])
        return results

    def _coding_loop(self, system_prompt: str) -> dict:
        """编码主循环"""
        # 简化：实际实现会包含工具调用循环
        return {
            "progress": {
                "overall_progress": 75,
                "completed_tasks": [],
                "pending_tasks": [],
            },
            "files_changed": [],
        }


class LongRunningOrchestrator:
    """长时运行编排器：协调 Initializer 和 Coding Agent"""

    def __init__(self, project_dir: str, llm_client):
        self.artifacts = ArtifactManager(project_dir)
        self.initializer = InitializerAgent(llm_client, self.artifacts)
        self.coding_agent = CodingAgentHarness(llm_client, self.artifacts)

    def run_session(self) -> dict:
        """运行一个会话周期"""
        # Phase 1: Initializer 准备上下文
        context = self.initializer.prepare_session()

        # Phase 2: Coding Agent 执行
        result = self.coding_agent.execute_session(context)

        return result

    def run_multi_session(self, max_sessions: int = 10) -> dict:
        """运行多个连续会话直到任务完成"""
        all_results = []
        for i in range(max_sessions):
            print(f"=== 会话 {i+1}/{max_sessions} ===")
            result = self.run_session()
            all_results.append(result)

            progress = self.artifacts.load_progress()
            if progress and progress.get("overall_progress", 0) >= 100:
                print("任务完成！")
                break

        return {"sessions": len(all_results), "results": all_results}
```

---

## 18.5 Anthropic 数据：单 Agent $9 vs 完整 Harness $200

### 真实世界的成本-质量权衡

Anthropic 的内部数据揭示了一个残酷的现实：

```
┌─────────────────────────────────────────────────────────────┐
│              单 Agent vs 完整 Harness 对比                    │
├─────────────┬──────────────────┬────────────────────────────┤
│   维度       │   单 Agent       │   完整 Harness              │
├─────────────┼──────────────────┼────────────────────────────┤
│ 时间         │ ~20 分钟         │ ~6 小时                     │
│ 成本         │ ~$9              │ ~$200                      │
│ 成本倍率     │ 1x               │ ~22x                       │
│ 代码质量     │ "能跑"           │ "生产就绪"                  │
│ 测试覆盖     │ 很少或没有        │ 完整的单元/集成测试          │
│ 错误处理     │ 基本              │ 全面的边界和异常处理         │
│ 文档         │ 内联注释          │ 完整的 API 文档             │
│ 可维护性     │ 低                │ 高                         │
└─────────────┴──────────────────┴────────────────────────────┘
```

### 成本是否值得？

```python
def calculate_harness_roi(
    single_agent_cost: float = 9.0,
    harness_cost: float = 200.0,
    manual_fix_hours: float = 8.0,
    developer_hourly_rate: float = 75.0,
    bug_probability_single: float = 0.6,
    bug_probability_harness: float = 0.1,
) -> dict:
    """计算 Harness 的投资回报"""

    # 单 Agent 的真实成本 = Agent 成本 + 修 bug 的人工成本
    single_total = (
        single_agent_cost
        + bug_probability_single * manual_fix_hours * developer_hourly_rate
    )

    # Harness 的真实成本 = Harness 成本 + 修 bug 的人工成本
    harness_total = (
        harness_cost
        + bug_probability_harness * manual_fix_hours * developer_hourly_rate
    )

    return {
        "single_agent_total": single_total,     # $9 + 0.6*8*$75 = $369
        "harness_total": harness_total,          # $200 + 0.1*8*$75 = $260
        "savings": single_total - harness_total, # $109 节省
        "roi": (single_total - harness_total) / harness_cost,  # 54.5% ROI
    }

# 结论：考虑人工修复成本后，Harness 反而更便宜
result = calculate_harness_roi()
# single_agent_total = $369, harness_total = $260, savings = $109
```

### 何时用哪种策略

```
                    错误修复成本
                 低 ←─────────→ 高
           ┌──────────┬──────────┐
       低  │ 单 Agent │ 单 Agent │
任务       │ 足够     │ + 人工审查│
复杂度     │          │          │
       高  │ 轻量     │ 完整     │
           │ Harness  │ Harness  │
           └──────────┴──────────┘
```

---

## 18.6 Managed Agents：Harness 即托管服务

### 从自建到托管

```
发展阶段：

Stage 1: 裸 Agent          → 直接调用 API
Stage 2: 自建 Harness      → 自己实现循环、评估、artifact
Stage 3: Managed Harness   → 平台提供会话管理、状态持久化、编排
Stage 4: Harness as a Service → 声明式配置，平台全托管
```

### Managed Agent 架构

```
┌─────────────────────────────────────────────────────┐
│                  Managed Agent Platform               │
│                                                       │
│  ┌─────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ Session  │  │ Artifact │  │ Orchestration    │    │
│  │ Manager  │  │  Store   │  │    Engine        │    │
│  │          │  │          │  │                  │    │
│  │ 会话生命 │  │ 跨会话   │  │ 自动编排         │    │
│  │ 周期管理 │  │ 状态持久 │  │ Initializer →   │    │
│  │          │  │ 化       │  │ Coding Agent     │    │
│  └─────────┘  └──────────┘  └──────────────────┘    │
│                                                       │
│  ┌──────────────┐  ┌──────────────────────────┐      │
│  │ Cost Control  │  │ Observability Dashboard  │      │
│  │              │  │                          │      │
│  │ Token 预算   │  │ 会话可视化               │      │
│  │ 自动停止     │  │ 进度追踪                 │      │
│  └──────────────┘  └──────────────────────────┘      │
└─────────────────────────────────────────────────────┘
```

### 声明式 Harness 配置

```yaml
# managed-harness.yaml
apiVersion: harness/v1
kind: LongRunningAgent
metadata:
  name: feature-builder
  project: user-auth-service

spec:
  # 会话配置
  session:
    max_duration: 30m        # 单会话最长时间
    max_sessions: 20         # 最多会话数
    cooldown_between: 10s    # 会话间冷却

  # Artifact 配置
  artifacts:
    store: s3://my-bucket/harness-artifacts/
    types:
      - progress.yaml
      - decisions.jsonl
    retention: 30d

  # Agent 配置
  agents:
    initializer:
      model: claude-sonnet-4-20250514
      max_tokens: 2000
    coder:
      model: claude-sonnet-4-20250514
      max_tokens: 8000
      tools: [file_editor, terminal, browser]

  # 成本控制
  budget:
    max_total_cost: $300
    alert_threshold: $150
    auto_stop: true

  # 完成条件
  completion:
    criteria:
      - "所有测试通过"
      - "覆盖率 > 80%"
      - "无 linting 错误"
```

---

## 本章小结

| 概念 | 核心要点 |
|------|----------|
| 上下文瓶颈 | 窗口固定 + 任务增长 = 信息丢失 |
| Context Compaction | 压缩旧上下文，适合不可分解任务 |
| Context Reset | 清空重来 + artifact 交接，适合可分解任务 |
| 结构化 Artifact | progress.yaml + decisions.jsonl + git history |
| 双 Harness 模式 | Initializer 设上下文 + Coding Agent 执行 |
| 成本真相 | 含人工修复后 Harness ($260) < 单 Agent ($369) |
| Managed Harness | 声明式配置，平台托管会话和状态 |

---

## 动手实验

### 实验 1：跨会话状态传递框架

**目标**：实现一个 ArtifactManager，能在会话结束时保存 progress.yaml，在新会话开始时恢复上下文。

```python
# 实验步骤：
# 1. 创建 ArtifactManager 实例
# 2. 模拟第一个会话：完成 2 个任务，保存 progress
# 3. 模拟"会话中断"（清空所有变量）
# 4. 创建新的 ArtifactManager 实例
# 5. 加载 progress，验证信息完整恢复
# 6. 生成 handoff_prompt，确认新 agent 能理解当前状态

manager = ArtifactManager("/tmp/test-project")
# ... 实现并验证
```

**验收标准**：
- progress.yaml 包含已完成和待完成任务
- handoff_prompt 包含所有关键决策
- 新会话能正确识别下一步任务

### 实验 2：双 Harness 模拟

**目标**：实现 Initializer → Coding Agent 的完整流程（可使用 mock LLM）。

**步骤**：
1. Initializer 读取 artifact 生成 task brief
2. Coding Agent 根据 brief 执行（mock 执行）
3. 执行完成后更新 artifact
4. 重复 3 个会话，验证进度递增

### 实验 3：成本效益分析器

**目标**：实现 `calculate_harness_roi` 函数，用不同参数运行，找到 harness 比单 agent 更经济的临界点。

```python
# 在什么任务复杂度下 harness 开始值得？
for complexity in range(1, 11):
    roi = calculate_harness_roi(
        manual_fix_hours=complexity * 2,
        bug_probability_single=min(0.3 + complexity * 0.07, 0.95),
    )
    print(f"复杂度 {complexity}: 节省 ${roi['savings']:.0f}")
```

---

## 练习题

### 基础题

1. **概念题**：解释为什么 context compaction 会导致信息损失？举一个具体的损失场景。

2. **设计题**：一个 progress.yaml 应该包含哪些字段？为什么 `architecture_decisions` 字段是必须的？

3. **计算题**：如果 Claude 上下文窗口 200K tokens，每轮对话消耗约 5K tokens，不做任何上下文管理的情况下，最多能进行多少轮对话？

### 实践题

4. **Artifact 设计**：为一个"构建电商后端"的项目设计完整的 artifact 集合（至少 3 种 artifact），说明每种的用途和更新时机。

5. **双 Harness 配置**：为一个需要 8 小时完成的前端重构任务设计 Initializer + Coding Agent 配置，包括会话时长、artifact 类型、完成条件。

### 思考题

6. **Artifact 过时问题**：如果 Coding Agent 在会话中做了一个与 artifact 记录的架构决策矛盾的变更，artifact 要怎么处理？设计一个一致性保障机制。

7. **无限会话悖论**：如果通过足够好的 artifact 设计，agent 可以无限期运行——那为什么不让 agent 一直跑下去直到完成？讨论"计算成本递增"和"错误累积"两个限制因素。
