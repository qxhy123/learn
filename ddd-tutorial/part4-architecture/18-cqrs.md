# 第18章：CQRS（命令查询职责分离）

## 学习目标

- 理解CQRS的核心思想及其解决的问题
- 掌握命令模型与查询模型的分离
- 实现简单CQRS和完整CQRS两种模式
- 了解CQRS与DDD、事件溯源的关系

---

## 18.1 为什么需要CQRS

传统的"同一个模型处理读写"方案存在问题：

```python
# 传统方式：同一个Order聚合同时处理读和写
class Order:
    # 写操作：保护不变量，封装业务规则
    def place(self): ...
    def cancel(self): ...
    def mark_paid(self): ...
    
    # 读操作：提供各种角度的视图
    def to_dict(self): ...
    def to_summary(self): ...
    def to_detail_view(self): ...
    def to_admin_view(self): ...
    def to_export_format(self): ...
```

**问题**：
1. **读写需求不同**：写操作需要强一致性和不变量保护；读操作需要灵活的数据结构和高性能
2. **性能矛盾**：聚合为写而优化（一致性）；查询往往需要跨聚合的非规范化数据（JOIN）
3. **扩展矛盾**：读操作通常比写操作多10-100倍，但耦合在一起无法独立扩展

---

## 18.2 CQRS核心概念

**命令查询职责分离（Command Query Responsibility Segregation，CQRS）**：

> **使用不同的模型来处理读操作（查询）和写操作（命令）。**

```
┌───────────────────────────────────────────────────────┐
│                        客户端                          │
└───────────┬──────────────────────────┬─────────────────┘
            │ 命令（写）                │ 查询（读）
            ▼                          ▼
┌───────────────────┐      ┌───────────────────────────┐
│   命令处理器       │      │      查询处理器/服务        │
│  (Command Stack)  │      │      (Query Stack)        │
│                   │      │                           │
│  聚合根           │      │  读模型（ReadModel）       │
│  领域服务         │      │  直接查询数据库/缓存        │
│  仓储（写）        │      │  非规范化视图              │
└─────────┬─────────┘      └─────────┬─────────────────┘
          │ 写入                      │ 读取
          ▼                          ▼
┌───────────────────┐      ┌───────────────────────────┐
│   写存储          │      │        读存储              │
│  (Write Store)    │      │       (Read Store)        │
│  规范化数据库      │      │  非规范化数据库/缓存/ES     │
└───────────────────┘      └───────────────────────────┘
                    ↑ 同步（事件/定时任务）
```

---

## 18.3 简单CQRS：同一数据库，分离模型

最轻量的CQRS形式，读写使用同一个数据库，但代码层次分离：

```python
# ===== 写侧（命令）：完整的聚合模型 =====

class Order:
    """写模型：包含完整的业务规则和不变量保护"""
    
    def place(self) -> None:
        if not self._items:
            raise OrderException("订单为空")
        self._status = OrderStatus.PLACED
        self._record_event(OrderPlaced(...))
    
    # ... 其他命令方法

class OrderRepository:
    """写仓储：操作聚合根"""
    def get(self, order_id: OrderId) -> Order: ...
    def save(self, order: Order) -> None: ...


# ===== 读侧（查询）：轻量的读模型 =====

@dataclass
class OrderListItem:
    """读模型：订单列表视图（扁平化）"""
    order_id: str
    customer_name: str      # 来自Customer表的JOIN
    status: str
    item_count: int
    total: Decimal
    created_at: datetime

@dataclass
class OrderDetail:
    """读模型：订单详情视图"""
    order_id: str
    customer: dict
    items: List[dict]
    payment_info: dict
    shipping_info: dict
    status_history: List[dict]  # 来自状态历史表
    total: Decimal

class OrderQueryService:
    """读服务：直接查询数据库，返回读模型"""
    
    def __init__(self, db_session):
        self._db = db_session
    
    def list_customer_orders(
        self, 
        customer_id: str, 
        page: int = 1, 
        page_size: int = 20
    ) -> List[OrderListItem]:
        """列表查询：直接写SQL，返回扁平的DTO"""
        rows = self._db.execute("""
            SELECT 
                o.id as order_id,
                c.name as customer_name,
                o.status,
                COUNT(oi.id) as item_count,
                o.total_amount as total,
                o.created_at
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            LEFT JOIN order_items oi ON o.id = oi.order_id
            WHERE o.customer_id = :customer_id
            GROUP BY o.id, c.name, o.status, o.total_amount, o.created_at
            ORDER BY o.created_at DESC
            LIMIT :limit OFFSET :offset
        """, {
            "customer_id": customer_id,
            "limit": page_size,
            "offset": (page - 1) * page_size
        })
        
        return [OrderListItem(**row) for row in rows]
    
    def get_order_detail(self, order_id: str) -> Optional[OrderDetail]:
        """详情查询：JOIN多表，返回完整视图"""
        ...
    
    def count_customer_orders(self, customer_id: str) -> int:
        """计数查询"""
        result = self._db.execute(
            "SELECT COUNT(*) FROM orders WHERE customer_id = :id",
            {"id": customer_id}
        )
        return result.scalar()
```

**关键点**：读服务可以直接写SQL，跳过聚合，返回任意形状的数据！

---

## 18.4 完整CQRS：读写分离的数据存储

进阶形式：写和读使用不同的数据存储：

```python
# 架构图：
# 
#  命令 → 写模型(聚合) → 写DB(PostgreSQL，规范化)
#                     → 发布领域事件
#                                    ↓
#                          事件处理器（投影器）
#                                    ↓
#                           更新读模型（投影）
#                                    ↓
#  查询 ←──────────────── 读DB(Elasticsearch/Redis，非规范化)

# 投影器（Projector）：将领域事件投影到读模型
class OrderProjector:
    """将写侧的领域事件投影（同步）到读模型"""
    
    def __init__(self, read_db):
        self._read_db = read_db
    
    def on_order_placed(self, event: OrderPlaced) -> None:
        """订单下单 → 在读DB创建订单记录"""
        self._read_db.upsert("order_views", {
            "order_id": event.order_id,
            "customer_id": event.customer_id,
            "status": "placed",
            "total": float(event.total_amount),
            "item_count": len(event.items),
            "placed_at": event.occurred_at.isoformat(),
        })
    
    def on_order_paid(self, event: OrderPaid) -> None:
        """支付完成 → 更新读DB中的状态"""
        self._read_db.update(
            "order_views",
            filter={"order_id": event.order_id},
            update={"status": "paid", "paid_at": event.occurred_at.isoformat()}
        )
    
    def on_order_shipped(self, event: OrderShipped) -> None:
        """发货 → 更新读DB，加入物流信息"""
        self._read_db.update(
            "order_views",
            filter={"order_id": event.order_id},
            update={
                "status": "shipped",
                "tracking_number": event.tracking_number,
                "shipped_at": event.occurred_at.isoformat()
            }
        )


# 读服务：直接查询Elasticsearch
class OrderSearchService:
    def __init__(self, es_client):
        self._es = es_client
    
    def search(
        self,
        customer_id: str = None,
        status: str = None,
        date_from: date = None,
        date_to: date = None,
        keyword: str = None,
    ) -> List[OrderView]:
        query = {"bool": {"must": []}}
        
        if customer_id:
            query["bool"]["must"].append({"term": {"customer_id": customer_id}})
        if status:
            query["bool"]["must"].append({"term": {"status": status}})
        if keyword:
            query["bool"]["must"].append({"match": {"items.name": keyword}})
        if date_from or date_to:
            date_range = {}
            if date_from: date_range["gte"] = date_from.isoformat()
            if date_to: date_range["lte"] = date_to.isoformat()
            query["bool"]["must"].append({"range": {"placed_at": date_range}})
        
        result = self._es.search(index="orders", body={"query": query})
        return [OrderView(**hit["_source"]) for hit in result["hits"]["hits"]]
```

---

## 18.5 应用服务中的CQRS实现

```python
class OrderApplicationService:
    """CQRS分离：命令和查询走不同的路径"""
    
    def __init__(
        self,
        # 写侧依赖
        order_repo: OrderRepository,
        event_bus: EventBus,
        # 读侧依赖
        order_query_service: OrderQueryService,
    ):
        self._repo = order_repo
        self._event_bus = event_bus
        self._query_service = order_query_service
    
    # ===== 命令（写）=====
    
    def place_order(self, command: PlaceOrderCommand) -> str:
        """命令：走完整的聚合路径，保护业务规则"""
        order = Order(OrderId(), command.customer_id)
        for item in command.items:
            order.add_item(item["snapshot"], item["quantity"])
        order.place()
        self._repo.save(order)
        self._publish(order)
        return str(order.id)
    
    def cancel_order(self, command: CancelOrderCommand) -> None:
        """命令：通过聚合保护取消规则"""
        order = self._repo.get(OrderId(command.order_id))
        order.cancel(command.reason)
        self._repo.save(order)
        self._publish(order)
    
    # ===== 查询（读）=====
    
    def get_order(self, query: GetOrderQuery) -> OrderDetail:
        """查询：直接走读模型，不加载聚合"""
        return self._query_service.get_order_detail(query.order_id)
    
    def list_orders(self, query: ListOrdersQuery) -> PagedResult[OrderListItem]:
        """查询：直接走读模型"""
        return self._query_service.list_customer_orders(
            query.customer_id,
            query.page,
            query.page_size
        )
```

---

## 18.6 最终一致性与读模型的时延

完整CQRS中，读写模型之间有短暂的时延：

```python
# 场景：用户刚下单，立刻查询订单列表

# 写侧：
order_service.place_order(command)
# 订单已写入写DB，事件已发出
# 投影器异步更新读DB（可能需要几毫秒到几秒）

# 读侧（如果立即查询）：
orders = order_service.list_orders(query)
# 可能看不到刚刚创建的订单！（读DB还未更新）

# 解决方案1：接受最终一致性
# 用户刷新页面后就能看到（对大多数场景可接受）

# 解决方案2：命令执行后立即查询写DB
# 仅针对"刚刚创建的那个对象"走写DB

# 解决方案3：乐观UI更新
# 前端在命令成功后立即更新UI，不等读模型
```

---

## 18.7 何时使用CQRS

```
适合CQRS的场景：
  ✅ 读操作远多于写操作（10:1以上）
  ✅ 查询需要复杂的JOIN或聚合，而领域模型不适合直接查询
  ✅ 读和写有不同的扩展需求
  ✅ 需要支持多种读视图（列表视图、详情视图、报表）
  ✅ 与事件溯源结合使用

不适合CQRS的场景：
  ❌ 简单的CRUD应用
  ❌ 团队规模小，增加的复杂度得不偿失
  ❌ 对最终一致性无法接受的场景（金融核心账务）
  ❌ 没有性能问题需要解决
```

---

## 本章小结

| 概念 | 说明 |
|------|------|
| 命令模型 | 写模型，包含聚合和业务规则 |
| 查询模型 | 读模型，针对查询优化的视图 |
| 投影器 | 将领域事件同步到读模型 |
| 最终一致性 | 读写模型之间有短暂时延 |
| 简单CQRS | 同一数据库，分离代码模型 |
| 完整CQRS | 不同数据库，通过事件同步 |

---

**上一章：** [第17章：六边形架构](./17-hexagonal-architecture.md)  
**下一章：** [第19章：事件溯源](./19-event-sourcing.md)
