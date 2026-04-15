# 第22章：安全、护栏与合规 Harness

> AI 系统的安全不是事后打补丁——它必须像安全带一样，从一开始就编织在系统里。护栏 Harness 的目标不是限制 AI 的能力，而是确保强大的能力始终在安全的轨道上运行。

---

## 学习目标

学完本章，你将能够：

1. 设计完整的输入/输出护栏模式
2. 实现毒性和偏差检测 Sensor
3. 理解 EU AI Act 和 NIST AI Agent Standards 2026.02 的合规要求
4. 构建审计轨迹生成系统
5. 实现 RBAC 范围控制和人工审批门禁

---

## 22.1 护栏模式：输入验证、输出过滤、话题边界

### 三道防线

```
用户输入                                          用户输出
    │                                                ▲
    ▼                                                │
┌──────────────────┐                      ┌──────────────────┐
│ 第 1 道防线      │                      │ 第 3 道防线      │
│ 输入护栏         │                      │ 输出护栏         │
│                  │                      │                  │
│ - prompt 注入检测│                      │ - 敏感信息过滤   │
│ - 毒性内容过滤  │                      │ - 幻觉标记       │
│ - 话题边界检查  │                      │ - 格式校验       │
│ - 长度/频率限制 │                      │ - 引用验证       │
└────────┬─────────┘                      └────────┬─────────┘
         │                                         │
         ▼                                         │
  ┌──────────────────┐                             │
  │ 第 2 道防线      │                             │
  │ 执行护栏         │                             │
  │                  │                             │
  │ - 工具调用限制   │─── LLM 生成 ──────────────→│
  │ - 资源访问控制   │                             │
  │ - 超时保护       │                             │
  │ - 成本上限       │                             │
  └──────────────────┘
```

### 输入护栏实现

```python
import re
from dataclasses import dataclass
from enum import Enum

class GuardrailVerdict(Enum):
    PASS = "pass"
    BLOCK = "block"
    WARN = "warn"
    REWRITE = "rewrite"

@dataclass
class GuardrailResult:
    verdict: GuardrailVerdict
    reason: str
    original_input: str
    rewritten_input: str | None = None

class InputGuardrail:
    """输入护栏：检查和清理用户输入"""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.max_length = self.config.get("max_length", 10000)
        self.blocked_topics = self.config.get("blocked_topics", [])

    def check(self, user_input: str) -> GuardrailResult:
        """运行全部输入检查"""
        checks = [
            self._check_prompt_injection,
            self._check_length,
            self._check_topic_boundary,
            self._check_toxicity,
            self._check_pii_input,
        ]

        for check_fn in checks:
            result = check_fn(user_input)
            if result.verdict in (GuardrailVerdict.BLOCK, GuardrailVerdict.REWRITE):
                return result

        return GuardrailResult(
            verdict=GuardrailVerdict.PASS,
            reason="所有检查通过",
            original_input=user_input,
        )

    def _check_prompt_injection(self, text: str) -> GuardrailResult:
        """检测 prompt 注入攻击"""
        injection_patterns = [
            r"ignore\s+(previous|above|all)\s+instructions",
            r"you\s+are\s+now\s+(?:a|an)\s+",
            r"system\s*:\s*",
            r"<\s*system\s*>",
            r"forget\s+(everything|what)",
            r"new\s+instructions?\s*:",
            r"override\s+(?:the\s+)?(?:system|instructions)",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return GuardrailResult(
                    verdict=GuardrailVerdict.BLOCK,
                    reason=f"检测到 prompt 注入模式: {pattern}",
                    original_input=text,
                )
        return GuardrailResult(
            verdict=GuardrailVerdict.PASS, reason="", original_input=text
        )

    def _check_length(self, text: str) -> GuardrailResult:
        """检查输入长度"""
        if len(text) > self.max_length:
            return GuardrailResult(
                verdict=GuardrailVerdict.BLOCK,
                reason=f"输入长度 {len(text)} 超过限制 {self.max_length}",
                original_input=text,
            )
        return GuardrailResult(
            verdict=GuardrailVerdict.PASS, reason="", original_input=text
        )

    def _check_topic_boundary(self, text: str) -> GuardrailResult:
        """检查话题边界"""
        for topic in self.blocked_topics:
            if topic.lower() in text.lower():
                return GuardrailResult(
                    verdict=GuardrailVerdict.BLOCK,
                    reason=f"话题 '{topic}' 不在服务范围内",
                    original_input=text,
                )
        return GuardrailResult(
            verdict=GuardrailVerdict.PASS, reason="", original_input=text
        )

    def _check_toxicity(self, text: str) -> GuardrailResult:
        """基本毒性检查（生产中应使用专业模型）"""
        # 简化：关键词列表（实际应使用分类模型）
        toxic_patterns = ["暴力威胁", "歧视性言论"]  # 简化
        for pattern in toxic_patterns:
            if pattern in text:
                return GuardrailResult(
                    verdict=GuardrailVerdict.BLOCK,
                    reason="检测到不当内容",
                    original_input=text,
                )
        return GuardrailResult(
            verdict=GuardrailVerdict.PASS, reason="", original_input=text
        )

    def _check_pii_input(self, text: str) -> GuardrailResult:
        """检测并警告用户输入中的 PII"""
        pii_patterns = {
            "phone": r"1[3-9]\d{9}",
            "id_card": r"\d{17}[\dXx]",
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        }
        found = []
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, text):
                found.append(pii_type)

        if found:
            return GuardrailResult(
                verdict=GuardrailVerdict.WARN,
                reason=f"检测到可能的个人信息: {', '.join(found)}",
                original_input=text,
            )
        return GuardrailResult(
            verdict=GuardrailVerdict.PASS, reason="", original_input=text
        )
```

### 输出护栏实现

```python
class OutputGuardrail:
    """输出护栏：检查和清理 LLM 输出"""

    def check(self, output: str, context: dict | None = None) -> GuardrailResult:
        """运行全部输出检查"""
        checks = [
            self._check_pii_leakage,
            self._check_harmful_content,
            self._check_hallucination_markers,
            self._check_format,
        ]

        for check_fn in checks:
            result = check_fn(output, context or {})
            if result.verdict == GuardrailVerdict.BLOCK:
                return result
            if result.verdict == GuardrailVerdict.REWRITE:
                output = result.rewritten_input or output

        return GuardrailResult(
            verdict=GuardrailVerdict.PASS,
            reason="所有输出检查通过",
            original_input=output,
        )

    def _check_pii_leakage(self, output: str, context: dict) -> GuardrailResult:
        """检查输出是否泄露 PII"""
        pii_patterns = {
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "phone": r"\b1[3-9]\d{9}\b",
        }
        for pii_type, pattern in pii_patterns.items():
            match = re.search(pattern, output)
            if match:
                # 替换为掩码
                masked = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", output)
                return GuardrailResult(
                    verdict=GuardrailVerdict.REWRITE,
                    reason=f"输出中包含 {pii_type}，已掩码",
                    original_input=output,
                    rewritten_input=masked,
                )
        return GuardrailResult(
            verdict=GuardrailVerdict.PASS, reason="", original_input=output
        )

    def _check_harmful_content(self, output: str, context: dict) -> GuardrailResult:
        """检查有害内容"""
        # 生产中使用专业分类模型
        return GuardrailResult(
            verdict=GuardrailVerdict.PASS, reason="", original_input=output
        )

    def _check_hallucination_markers(self, output: str, context: dict) -> GuardrailResult:
        """标记可能的幻觉"""
        markers = ["根据最新数据", "据统计", "研究表明"]
        found = [m for m in markers if m in output]
        if found and not context.get("has_citations"):
            return GuardrailResult(
                verdict=GuardrailVerdict.WARN,
                reason=f"包含声明性语句但无引用来源: {found}",
                original_input=output,
            )
        return GuardrailResult(
            verdict=GuardrailVerdict.PASS, reason="", original_input=output
        )

    def _check_format(self, output: str, context: dict) -> GuardrailResult:
        """格式检查"""
        expected_format = context.get("expected_format")
        if expected_format == "json":
            import json
            try:
                json.loads(output)
            except json.JSONDecodeError:
                return GuardrailResult(
                    verdict=GuardrailVerdict.BLOCK,
                    reason="输出不是有效的 JSON 格式",
                    original_input=output,
                )
        return GuardrailResult(
            verdict=GuardrailVerdict.PASS, reason="", original_input=output
        )
```

---

## 22.2 毒性和偏差检测作为持续 Sensor

### 偏差检测架构

```
             生产流量采样
                  │
                  ▼
          ┌──────────────┐
          │ Bias Sensor  │
          │              │
          │ 检测维度:     │
          │ - 性别偏差   │
          │ - 种族偏差   │
          │ - 年龄偏差   │
          │ - 地域偏差   │
          └──────┬───────┘
                 │
         ┌───────┴───────┐
         │               │
    偏差 < 阈值      偏差 > 阈值
         │               │
     继续监控        ┌────▼────┐
                     │ Alert   │
                     │ + 自动  │
                     │ 限流    │
                     └─────────┘
```

```python
class BiasSensor:
    """偏差检测 Sensor"""

    def __init__(self, evaluator, dimensions: list[str] | None = None):
        self.evaluator = evaluator
        self.dimensions = dimensions or ["gender", "race", "age", "region"]

    def analyze(self, samples: list[dict]) -> dict:
        """分析一批样本的偏差"""
        results = {}
        for dim in self.dimensions:
            bias_score = self._measure_bias(samples, dim)
            results[dim] = bias_score

        overall_score = sum(results.values()) / len(results)
        return {
            "overall_bias": overall_score,
            "dimensions": results,
            "alert": overall_score > 0.3,
            "recommendation": self._recommend(results),
        }

    def _measure_bias(self, samples: list[dict], dimension: str) -> float:
        """测量特定维度的偏差分数 (0=无偏差, 1=严重偏差)"""
        # 简化实现：实际应使用专业偏差检测模型
        # 方法：对同一问题替换人口统计特征，比较回答差异
        pairs = self._generate_counterfactual_pairs(samples, dimension)
        differences = []
        for original, counterfactual in pairs:
            diff = self._compare_responses(original, counterfactual)
            differences.append(diff)

        return sum(differences) / max(len(differences), 1)

    def _generate_counterfactual_pairs(
        self, samples: list[dict], dimension: str
    ) -> list[tuple]:
        """生成反事实对（替换人口统计特征）"""
        swaps = {
            "gender": [("他", "她"), ("男性", "女性"), ("先生", "女士")],
            "age": [("年轻人", "老年人"), ("25岁", "65岁")],
            "region": [("北京", "农村"), ("城市", "乡镇")],
        }
        pairs = []
        for sample in samples[:10]:
            query = sample.get("query", "")
            for old, new in swaps.get(dimension, []):
                if old in query:
                    modified = query.replace(old, new)
                    pairs.append((sample, {**sample, "query": modified}))
        return pairs

    def _compare_responses(self, original: dict, counterfactual: dict) -> float:
        """比较两个响应的差异度"""
        # 简化：返回固定值
        return 0.1

    def _recommend(self, results: dict) -> str:
        worst = max(results, key=results.get)
        if results[worst] > 0.5:
            return f"严重偏差：{worst} 维度分数 {results[worst]:.2f}，建议立即审查 prompt"
        elif results[worst] > 0.3:
            return f"中度偏差：{worst} 维度需要关注"
        return "偏差水平可接受"
```

---

## 22.3 法规合规自动化

### 合规框架概览

```
┌─────────────────────────────────────────────────────────┐
│                  AI 合规框架 (2026)                       │
├──────────────────┬──────────────────┬───────────────────┤
│   EU AI Act      │  NIST AI Agent   │  中国生成式 AI    │
│   (2024.08 生效) │  Standards       │  管理办法          │
│                  │  (2026.02 发布)  │  (2023.08 生效)    │
├──────────────────┼──────────────────┼───────────────────┤
│ 风险分级:         │ Agent 安全:       │ 内容合规:          │
│ - 不可接受风险   │ - 行为边界定义   │ - 内容审核         │
│ - 高风险         │ - 自主性控制     │ - 标识要求         │
│ - 有限风险       │ - 人工监督       │ - 数据安全         │
│ - 最小风险       │ - 审计追踪       │ - 用户权益         │
│                  │ - 故障安全       │                   │
│ 技术文档要求     │ 互操作性标准     │ 算法备案           │
│ 透明度义务       │ 测试方法论       │ 安全评估           │
└──────────────────┴──────────────────┴───────────────────┘
```

### NIST 2026.02 AI Agent Standards 关键要求

```python
class NISTAgentCompliance:
    """NIST 2026.02 AI Agent 标准合规检查"""

    REQUIREMENTS = {
        "behavioral_boundaries": {
            "description": "Agent 行为边界必须明确定义和可验证",
            "checks": [
                "有明确的 system prompt 定义允许/禁止行为",
                "工具调用有权限控制",
                "输出有格式和内容约束",
            ],
        },
        "autonomy_controls": {
            "description": "自主性级别必须可配置和可限制",
            "checks": [
                "高风险操作需人工审批",
                "有最大自主执行步骤数限制",
                "可随时中断 agent 执行",
            ],
        },
        "human_oversight": {
            "description": "必须保持有意义的人工监督",
            "checks": [
                "关键决策有人工审批环节",
                "异常情况自动升级到人工",
                "人工可覆盖 agent 决策",
            ],
        },
        "audit_trail": {
            "description": "所有 agent 行为必须可审计",
            "checks": [
                "完整的操作日志",
                "决策链可追溯",
                "输入输出可重放",
            ],
        },
        "fail_safe": {
            "description": "故障安全机制",
            "checks": [
                "异常时有安全降级策略",
                "超时自动停止",
                "错误不会传播到外部系统",
            ],
        },
    }

    def audit(self, system_config: dict) -> dict:
        """执行合规审计"""
        results = {}
        for req_id, req in self.REQUIREMENTS.items():
            check_results = []
            for check in req["checks"]:
                # 简化：实际应逐条验证
                passed = system_config.get(f"has_{req_id}", False)
                check_results.append({
                    "check": check,
                    "passed": passed,
                })

            all_passed = all(c["passed"] for c in check_results)
            results[req_id] = {
                "requirement": req["description"],
                "passed": all_passed,
                "checks": check_results,
            }

        compliant = all(r["passed"] for r in results.values())
        return {
            "compliant": compliant,
            "requirements": results,
            "compliance_score": sum(
                1 for r in results.values() if r["passed"]
            ) / len(results),
        }
```

---

## 22.4 审计轨迹生成

### 审计事件模型

```python
import uuid
from datetime import datetime

@dataclass
class AuditEvent:
    """审计事件"""
    event_id: str
    timestamp: str
    event_type: str       # "input" | "decision" | "action" | "output"
    actor: str            # "user" | "agent:planner" | "agent:generator"
    action: str           # 具体操作
    input_data: dict      # 输入（可能脱敏）
    output_data: dict     # 输出（可能脱敏）
    metadata: dict        # 额外元数据

class AuditTrailGenerator:
    """审计轨迹生成器"""

    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.session_id = str(uuid.uuid4())

    def log_event(
        self,
        event_type: str,
        actor: str,
        action: str,
        input_data: dict,
        output_data: dict,
        metadata: dict | None = None,
    ) -> str:
        """记录审计事件"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            actor=actor,
            action=action,
            input_data=self._sanitize(input_data),
            output_data=self._sanitize(output_data),
            metadata={
                "session_id": self.session_id,
                **(metadata or {}),
            },
        )
        self.storage.store(event)
        return event.event_id

    def _sanitize(self, data: dict) -> dict:
        """脱敏处理"""
        sanitized = {}
        sensitive_keys = {"password", "token", "secret", "credit_card", "ssn"}
        for key, value in data.items():
            if any(s in key.lower() for s in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:500] + f"...[truncated, total {len(value)} chars]"
            else:
                sanitized[key] = value
        return sanitized

    def get_decision_chain(self, session_id: str) -> list[dict]:
        """获取决策链：追溯一个会话中所有决策"""
        events = self.storage.query(session_id=session_id, event_type="decision")
        return [{
            "step": i + 1,
            "actor": e.actor,
            "action": e.action,
            "timestamp": e.timestamp,
            "input_summary": str(e.input_data)[:100],
            "output_summary": str(e.output_data)[:100],
        } for i, e in enumerate(events)]

    def generate_compliance_report(
        self, time_range: tuple[str, str]
    ) -> dict:
        """生成合规报告"""
        events = self.storage.query(time_range=time_range)

        report = {
            "period": {"start": time_range[0], "end": time_range[1]},
            "total_events": len(events),
            "by_type": {},
            "by_actor": {},
            "blocked_requests": 0,
            "human_escalations": 0,
        }

        for event in events:
            # 按类型统计
            t = event.event_type
            report["by_type"][t] = report["by_type"].get(t, 0) + 1

            # 按 actor 统计
            a = event.actor
            report["by_actor"][a] = report["by_actor"].get(a, 0) + 1

            # 统计被阻止的请求
            if event.metadata.get("blocked"):
                report["blocked_requests"] += 1

            # 统计人工升级
            if event.metadata.get("escalated_to_human"):
                report["human_escalations"] += 1

        return report
```

---

## 22.5 RBAC 范围控制

### Agent 权限模型

```
传统 RBAC:                         Agent RBAC:
用户 → 角色 → 权限                  用户 → agent → 工具 → 权限

┌──────┐    ┌──────────┐           ┌──────┐    ┌──────────┐    ┌──────┐
│ 用户 │───→│ Admin    │           │ 用户 │───→│ Agent A  │───→│ 读DB │
│ Alice│    │ → 全权限  │           │ Alice│    │ scope:   │    │ 写DB │
└──────┘    └──────────┘           └──────┘    │ read-only│    │ 删DB │ ← 禁止
                                               └──────────┘    └──────┘
```

```python
from typing import Any

class AgentRBAC:
    """Agent RBAC 权限控制"""

    def __init__(self):
        self.roles: dict[str, dict] = {}
        self.agent_assignments: dict[str, str] = {}

    def define_role(self, role_name: str, permissions: dict) -> None:
        """定义角色权限"""
        self.roles[role_name] = permissions

    def assign_role(self, agent_id: str, role_name: str) -> None:
        """为 agent 分配角色"""
        if role_name not in self.roles:
            raise ValueError(f"角色 {role_name} 不存在")
        self.agent_assignments[agent_id] = role_name

    def check_permission(
        self, agent_id: str, action: str, resource: str
    ) -> bool:
        """检查 agent 是否有权执行操作"""
        role_name = self.agent_assignments.get(agent_id)
        if not role_name:
            return False

        role = self.roles.get(role_name, {})
        permissions = role.get("permissions", {})

        # 检查资源权限
        resource_perms = permissions.get(resource, [])
        return action in resource_perms

    def create_scoped_context(self, agent_id: str) -> dict:
        """创建 agent 的作用域上下文"""
        role_name = self.agent_assignments.get(agent_id, "none")
        role = self.roles.get(role_name, {})

        return {
            "agent_id": agent_id,
            "role": role_name,
            "allowed_tools": role.get("allowed_tools", []),
            "allowed_resources": list(role.get("permissions", {}).keys()),
            "max_actions_per_session": role.get("max_actions", 100),
            "requires_human_approval": role.get("requires_approval", []),
        }


# 使用示例
rbac = AgentRBAC()

rbac.define_role("reader", {
    "permissions": {
        "database": ["read", "query"],
        "files": ["read", "list"],
    },
    "allowed_tools": ["sql_query", "file_reader"],
    "max_actions": 50,
    "requires_approval": [],
})

rbac.define_role("writer", {
    "permissions": {
        "database": ["read", "query", "insert", "update"],
        "files": ["read", "list", "write"],
    },
    "allowed_tools": ["sql_query", "sql_write", "file_reader", "file_writer"],
    "max_actions": 100,
    "requires_approval": ["delete", "drop_table"],
})

rbac.define_role("admin", {
    "permissions": {
        "database": ["read", "query", "insert", "update", "delete", "drop_table"],
        "files": ["read", "list", "write", "delete"],
        "system": ["restart", "configure"],
    },
    "allowed_tools": ["*"],
    "max_actions": 500,
    "requires_approval": ["system.restart", "database.drop_table"],
})

rbac.assign_role("coding-agent", "writer")
rbac.assign_role("review-agent", "reader")
```

---

## 22.6 人工审批门禁

### 审批门禁架构

```
Agent 执行流程：

  Agent 想执行操作
         │
         ▼
  ┌────────────────┐
  │ 权限检查       │
  │ (RBAC)         │
  └───────┬────────┘
          │
    ┌─────┴─────┐
    │           │
 允许        需要审批
    │           │
    ▼           ▼
  执行    ┌──────────────┐
          │ 审批队列     │
          │              │
          │ 通知人工审批 │
          │ 超时: 30min  │
          └──────┬───────┘
                 │
          ┌──────┴──────┐
          │             │
        批准          拒绝
          │             │
          ▼             ▼
        执行         记录并
                    通知 agent
```

```python
import time
from threading import Event

class HumanApprovalGate:
    """人工审批门禁"""

    def __init__(self, notification_service, timeout_seconds: int = 1800):
        self.notification = notification_service
        self.timeout = timeout_seconds
        self.pending_approvals: dict[str, dict] = {}

    def request_approval(
        self,
        agent_id: str,
        action: str,
        details: dict,
    ) -> dict:
        """请求人工审批"""
        request_id = str(uuid.uuid4())

        approval_request = {
            "request_id": request_id,
            "agent_id": agent_id,
            "action": action,
            "details": details,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }

        self.pending_approvals[request_id] = approval_request

        # 发送通知
        self.notification.send(
            channel="slack",
            message=f"[审批请求] Agent {agent_id} 请求执行: {action}\n"
                    f"详情: {details}\n"
                    f"批准: /approve {request_id}\n"
                    f"拒绝: /reject {request_id}",
        )

        # 等待审批结果
        return self._wait_for_decision(request_id)

    def approve(self, request_id: str, approver: str) -> None:
        """批准请求"""
        if request_id in self.pending_approvals:
            self.pending_approvals[request_id]["status"] = "approved"
            self.pending_approvals[request_id]["approver"] = approver
            self.pending_approvals[request_id]["decided_at"] = datetime.now().isoformat()

    def reject(self, request_id: str, approver: str, reason: str = "") -> None:
        """拒绝请求"""
        if request_id in self.pending_approvals:
            self.pending_approvals[request_id]["status"] = "rejected"
            self.pending_approvals[request_id]["approver"] = approver
            self.pending_approvals[request_id]["reason"] = reason
            self.pending_approvals[request_id]["decided_at"] = datetime.now().isoformat()

    def _wait_for_decision(self, request_id: str) -> dict:
        """等待审批决定（带超时）"""
        start = time.time()
        while time.time() - start < self.timeout:
            status = self.pending_approvals[request_id]["status"]
            if status != "pending":
                return self.pending_approvals[request_id]
            time.sleep(5)  # 轮询间隔

        # 超时 → 默认拒绝
        self.pending_approvals[request_id]["status"] = "timeout_rejected"
        return self.pending_approvals[request_id]
```

---

## 本章小结

| 概念 | 核心要点 |
|------|----------|
| 三道防线 | 输入护栏 → 执行护栏 → 输出护栏 |
| Prompt 注入检测 | 正则模式匹配 + LLM 分类器 |
| 偏差检测 | 反事实测试：替换人口统计特征，比较回答差异 |
| 合规框架 | EU AI Act + NIST 2026.02 + 本地法规 |
| 审计轨迹 | 每个事件记录 actor + action + input/output |
| RBAC | Agent 粒度的权限控制：角色 → 工具 → 资源 |
| 人工门禁 | 高风险操作需人工审批，超时默认拒绝 |

---

## 动手实验

### 实验 1：实现输入/输出护栏 Harness

**目标**：构建一个完整的三层护栏系统。

```python
# 步骤：
# 1. 实现 InputGuardrail（至少 3 种检查）
# 2. 实现 OutputGuardrail（至少 3 种检查）
# 3. 创建测试用例：
#    - 正常输入 → PASS
#    - prompt 注入 → BLOCK
#    - 包含 PII 的输出 → REWRITE（掩码）
#    - 输出含有害内容 → BLOCK
# 4. 计算护栏的误报率和漏报率
```

**验收标准**：
- 能正确识别至少 3 种 prompt 注入模式
- PII 被正确掩码
- 误报率 < 5%

### 实验 2：RBAC 权限系统

**目标**：为一个多 agent 系统实现完整的 RBAC。

**步骤**：
1. 定义 3 种角色（reader / writer / admin）
2. 分配给不同 agent
3. 测试权限检查：coding agent 不能删除数据库表
4. 测试审批流程：writer 尝试 delete 操作 → 触发审批

### 实验 3：合规审计报告生成器

**目标**：实现审计轨迹收集和合规报告生成。

**步骤**：
1. 模拟 100 个 agent 操作事件
2. 记录审计轨迹
3. 生成合规报告
4. 验证报告包含所有 NIST 要求的字段

---

## 练习题

### 基础题

1. **概念题**：输入护栏和输出护栏的职责有什么不同？能否只做输出护栏不做输入护栏？

2. **安全题**：列举 3 种常见的 prompt 注入模式和对应的检测方法。

3. **合规题**：NIST 2026.02 对 AI Agent 的 5 个核心要求是什么？

### 实践题

4. **护栏设计**：为一个医疗问答 AI 设计护栏规则。特别需要考虑：不能给出具体诊断、不能推荐具体药物、必须建议用户咨询医生。

5. **RBAC 设计**：为一个包含 Planner + Generator + Evaluator + Testing Agent 的系统设计 RBAC 矩阵，明确每个 agent 能访问哪些资源和工具。

### 思考题

6. **过度防护问题**：护栏越多越安全，但也越容易误报（把正常请求当成攻击）。如何在"安全"和"可用性"之间找到平衡？一个被护栏频繁阻止的系统和一个没有护栏的系统，哪个更危险？

7. **人工审批瓶颈**：如果系统每天产生 500 个审批请求，人工审批能力只有 100 个/天，会发生什么？如何设计分级审批策略来解决这个问题？
