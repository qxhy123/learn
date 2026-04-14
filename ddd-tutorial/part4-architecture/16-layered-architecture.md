# 第16章：分层架构（Layered Architecture）

## 学习目标

- 理解分层架构的各层职责与依赖规则
- 掌握严格分层与宽松分层的区别
- 识别常见的分层错误并知道如何修正
- 理解分层架构如何支撑DDD的领域模型

---

## 16.1 为什么需要分层

软件中的关注点分离（Separation of Concerns）是核心原则之一。不同的关注点应该在不同的"层"中处理：

```
业务关注点（"做什么"）：
  ├── 订单有哪些状态？
  ├── 什么情况下可以取消？
  └── 优惠券怎么计算？

技术关注点（"怎么实现"）：
  ├── 数据怎么存到数据库？
  ├── HTTP请求怎么解析？
  └── 消息怎么发布？

混在一起的后果：
  - 改数据库schema要改业务代码
  - 单元测试需要连接真实数据库
  - 业务逻辑被技术细节污染，越来越难理解
```

---

## 16.2 经典四层架构

DDD推荐的分层架构：

```
┌──────────────────────────────────────────────────┐
│              接口层（Interface / UI）              │
│                                                  │
│  REST API / GraphQL / CLI / 消息监听器            │
│  职责：协议转换、认证/授权、请求路由                │
└──────────────────────┬───────────────────────────┘
                       │ 调用
                       ▼
┌──────────────────────────────────────────────────┐
│             应用层（Application）                  │
│                                                  │
│  应用服务（用例编排）、命令处理器、查询处理器         │
│  职责：编排用例、管理事务、协调领域层和基础设施层    │
└──────────────────────┬───────────────────────────┘
                       │ 调用
                       ▼
┌──────────────────────────────────────────────────┐
│              领域层（Domain）                      │
│                                                  │
│  实体、值对象、聚合、领域服务、仓储接口、领域事件    │
│  职责：表达业务规则和业务逻辑（纯粹的业务代码）     │
└──────────────────────┬───────────────────────────┘
                       │ 接口定义（实现在基础设施层）
                       ▼
┌──────────────────────────────────────────────────┐
│            基础设施层（Infrastructure）            │
│                                                  │
│  数据库、消息队列、外部API、缓存、文件存储           │
│  职责：实现技术细节（仓储实现、消息发布、外部集成）  │
└──────────────────────────────────────────────────┘
```

### 依赖规则

**关键规则**：**上层依赖下层，但领域层不依赖基础设施层**。

```python
# ✅ 合法的依赖方向
# 接口层 → 应用层
# 应用层 → 领域层
# 应用层 → 基础设施层（通过接口）
# 基础设施层 → 领域层（实现领域层定义的接口）

# ❌ 非法的依赖
# 领域层 → 基础设施层（领域层不依赖持久化技术！）
# 领域层 → 应用层

# 验证：领域层的import语句中，是否有任何数据库、框架相关的import？
# 如果有，就违反了分层规则

# 领域层（domain/model/order.py）
# ✅ 只import Python标准库和其他领域对象
from datetime import datetime
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass

# ❌ 不应该出现的import
from sqlalchemy import ...  # 数据库框架！
from django.db import ...   # Web框架！
import redis                # 缓存！
```

---

## 16.3 各层的代码结构

```
src/
├── interface/                    # 接口层
│   ├── api/
│   │   ├── order_api.py         # REST API路由
│   │   └── payment_api.py
│   └── consumers/
│       └── payment_event_consumer.py  # 消息监听
│
├── application/                  # 应用层
│   ├── commands/
│   │   ├── place_order.py       # 命令对象
│   │   └── cancel_order.py
│   ├── queries/
│   │   └── get_order.py         # 查询对象
│   └── services/
│       └── order_application_service.py
│
├── domain/                       # 领域层（最重要）
│   ├── model/
│   │   ├── order.py             # 聚合根
│   │   ├── order_item.py        # 聚合内实体
│   │   └── value_objects.py     # 值对象
│   ├── events/
│   │   └── order_events.py      # 领域事件
│   ├── services/
│   │   └── pricing_service.py   # 领域服务
│   └── repositories/
│       └── order_repository.py  # 仓储接口（抽象）
│
└── infrastructure/               # 基础设施层
    ├── persistence/
    │   ├── models.py             # ORM模型
    │   └── sqlalchemy_order_repository.py  # 仓储实现
    ├── messaging/
    │   └── kafka_event_publisher.py
    └── external/
        └── alipay_payment_gateway.py
```

---

## 16.4 依赖倒置原则（DIP）

领域层不依赖基础设施层，靠的是**依赖倒置**：

```python
# 领域层定义接口（抽象）
# domain/repositories/order_repository.py
from abc import ABC, abstractmethod

class OrderRepository(ABC):
    @abstractmethod
    def get(self, order_id: OrderId) -> Order: ...
    
    @abstractmethod
    def save(self, order: Order) -> None: ...

# 基础设施层实现接口
# infrastructure/persistence/sqlalchemy_order_repository.py
from domain.repositories.order_repository import OrderRepository

class SqlAlchemyOrderRepository(OrderRepository):  # ← 依赖关系是：基础设施→领域
    def get(self, order_id: OrderId) -> Order:
        ...  # SQL实现
    
    def save(self, order: Order) -> None:
        ...  # SQL实现

# 应用层通过接口类型（而非具体实现）使用仓储
# application/services/order_application_service.py
class OrderApplicationService:
    def __init__(self, order_repo: OrderRepository):  # ← 依赖接口，不依赖具体实现
        self._repo = order_repo

# 在启动时（组合根）注入具体实现
# main.py 或 container.py
def create_order_service() -> OrderApplicationService:
    session = create_db_session()
    repo = SqlAlchemyOrderRepository(session)   # ← 在这里选择具体实现
    return OrderApplicationService(repo)
```

**依赖关系图**：
```
接口层  →  应用层  →  领域层  ←  基础设施层
                              (基础设施实现了领域层定义的接口)
```

---

## 16.5 各层职责对照表

```python
# 接口层示例：HTTP协议转换，不含业务逻辑
@router.post("/orders/{order_id}/cancel")
async def cancel_order(
    order_id: str,
    body: CancelOrderRequest,
    token: str = Depends(verify_token)          # 认证
) -> Response:
    try:
        command = CancelOrderCommand(           # 构建命令
            order_id=order_id,
            reason=body.reason
        )
        order_service.cancel_order(command)     # 调用应用服务
        return Response(status_code=204)        # 协议响应
    except OrderNotFoundException:
        return Response(status_code=404)        # 错误转换
    except OrderException as e:
        return Response(status_code=422, content=str(e))


# 应用层示例：用例编排，不含业务规则
class OrderApplicationService:
    def cancel_order(self, command: CancelOrderCommand) -> None:
        order = self._repo.get(OrderId(command.order_id))  # 加载
        order.cancel(command.reason)                        # 领域操作
        self._repo.save(order)                              # 持久化
        self._publish_events(order)                         # 发布事件


# 领域层示例：纯业务规则，不含技术细节
class Order:
    def cancel(self, reason: str) -> None:
        if self._status == OrderStatus.SHIPPED:
            raise OrderException("已发货订单不可取消")
        self._status = OrderStatus.CANCELLED
        self._cancel_reason = reason
        self._record_event(OrderCancelled(self._id, reason))


# 基础设施层示例：技术实现，不含业务规则
class SqlAlchemyOrderRepository:
    def save(self, order: Order) -> None:
        orm = self._to_orm(order)
        self._session.merge(orm)
        self._session.flush()
```

---

## 16.6 常见分层错误

### 错误1：领域层直接依赖数据库

```python
# ❌ 领域层中出现了数据库操作
class Order:
    def cancel(self, reason: str) -> None:
        self._status = OrderStatus.CANCELLED
        # 直接操作数据库！破坏了分层
        db.execute("UPDATE orders SET status='cancelled' WHERE id=?", self._id)
```

### 错误2：接口层包含业务逻辑

```python
# ❌ 接口层有业务规则判断
@router.post("/orders/{order_id}/cancel")
async def cancel_order(order_id: str, body: dict):
    order_data = db.get("orders", order_id)
    
    # 业务逻辑不应该在接口层
    if order_data["status"] == "shipped":
        raise HTTPException(400, "已发货订单不可取消")
    
    db.update("orders", order_id, {"status": "cancelled"})
```

### 错误3：应用服务直接操作数据库

```python
# ❌ 应用服务绕过仓储，直接操作数据库
class OrderApplicationService:
    def cancel_order(self, command: CancelOrderCommand) -> None:
        # 直接写SQL，绕过了领域模型！
        self._db.execute(
            "UPDATE orders SET status='cancelled' WHERE id=?",
            command.order_id
        )
        # 业务规则在哪里？不存在了！
```

---

## 16.7 宽松分层（Relaxed Layering）

有时候，允许某些层跳过中间层直接调用更底层：

```
严格分层：
  接口层 → 应用层 → 领域层 → 基础设施层（只能相邻层调用）

宽松分层（常见于查询场景）：
  接口层 → 查询服务（直接访问数据库） ← 跳过了应用层和领域层
  
理由：
  对于简单的查询操作，经过全部层次只是增加复杂度
  直接查询数据库（或只读模型）更简单高效
  
这通常与CQRS结合使用（第18章详述）
```

---

## 本章小结

| 层次 | 职责 | 依赖方向 |
|------|------|---------|
| 接口层 | 协议转换、认证路由 | → 应用层 |
| 应用层 | 用例编排、事务管理 | → 领域层、基础设施层（接口） |
| 领域层 | 业务规则（纯粹） | 不依赖其他层 |
| 基础设施层 | 技术实现 | → 领域层（实现接口） |

**核心原则**：领域层是最重要的，它不依赖任何技术框架，可以独立测试，持久保持清洁。

---

**上一章：** [第15章：工厂](../part3-tactical-design/15-factories.md)  
**下一章：** [第17章：六边形架构](./17-hexagonal-architecture.md)
