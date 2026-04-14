# 第20章：DDD与微服务

## 学习目标

- 理解DDD的限界上下文与微服务边界的天然契合
- 掌握微服务划分的原则与陷阱
- 理解服务间通信的两种模式（同步/异步）
- 从单体到微服务的演化路径

---

## 20.1 为什么说DDD是微服务的最佳指南

微服务架构流行后，最难的问题是：**服务应该怎么拆分？**

DDD给出了天然的答案：

```
限界上下文（Bounded Context）= 微服务边界

原因：
  ├── 限界上下文已经定义了语义一致的边界
  ├── 每个上下文内的统一语言保证了内部一致性
  ├── 上下文之间的显式接口变成了服务API
  └── 上下文映射模式直接对应服务集成模式

Conway定律再次体现：
  组织结构 → 限界上下文 → 微服务
  一个独立的团队 → 一个或多个限界上下文 → 独立部署的服务
```

---

## 20.2 微服务的划分原则

### 原则1：按限界上下文划分，而非按技术层次

```
❌ 按技术层次划分（常见错误）：
  database-service    ← 处理所有数据存储？
  api-service         ← 处理所有HTTP？
  business-service    ← 处理所有业务？

✅ 按业务能力/限界上下文划分：
  order-service       ← 订单上下文
  payment-service     ← 支付上下文
  inventory-service   ← 库存上下文
  catalog-service     ← 商品目录上下文
```

### 原则2：服务拥有自己的数据

```
每个微服务：
  ├── 拥有自己的数据库（或数据库Schema）
  ├── 是该数据的权威来源
  └── 其他服务只能通过它的API访问该数据

❌ 共享数据库反模式：
  service-A                service-B
      ↓                        ↓
  ┌─────────────────────────────────┐
  │           共享数据库             │ ← 紧耦合！A的schema变更影响B
  └─────────────────────────────────┘

✅ 数据库独立：
  service-A    service-B    service-C
      ↓             ↓            ↓
  ┌──────┐      ┌──────┐    ┌──────┐
  │ DB-A │      │ DB-B │    │ DB-C │
  └──────┘      └──────┘    └──────┘
```

### 原则3：单一职责 & 高内聚低耦合

```python
# 判断服务边界是否合理的问题：
# 
# 1. 这个服务是否有一个清晰的业务职责？
#    ✅ 订单服务：管理订单的生命周期
#    ❌ 数据服务：处理各种数据（太模糊）
#
# 2. 修改一个业务功能时，需要更改几个服务？
#    ✅ 理想：只改1个服务（高内聚）
#    ❌ 糟糕：需要改3-4个服务（低内聚，分布式单体）
#
# 3. 服务之间的调用是否频繁？
#    ✅ 合理：服务间偶尔通信，每次操作主要在单服务内完成
#    ❌ 危险：一个请求需要调用5个服务（过细粒度）
```

---

## 20.3 服务间通信：同步 vs 异步

### 同步通信（REST/gRPC）

```python
# 订单服务同步调用库存服务检查可用性
import httpx
from typing import Optional

class InventoryServiceClient:
    """库存服务的HTTP客户端（防腐层）"""
    
    def __init__(self, base_url: str):
        self._base_url = base_url
    
    def check_availability(self, product_id: str, quantity: int) -> bool:
        """检查库存是否充足"""
        try:
            response = httpx.get(
                f"{self._base_url}/inventory/{product_id}/availability",
                params={"quantity": quantity},
                timeout=3.0  # 超时设置至关重要
            )
            response.raise_for_status()
            return response.json()["available"]
        except httpx.TimeoutException:
            # 同步调用的问题：超时会阻塞整个流程
            raise ServiceUnavailableError("库存服务暂时不可用")
        except httpx.HTTPStatusError as e:
            raise ExternalServiceError(f"库存服务返回错误: {e.response.status_code}")

# 在应用服务中使用
class OrderApplicationService:
    def __init__(self, inventory_client: InventoryServiceClient, ...):
        self._inventory = inventory_client
    
    def place_order(self, command: PlaceOrderCommand) -> str:
        # 同步检查库存（问题：如果库存服务挂了，订单也下不了）
        for item in command.items:
            if not self._inventory.check_availability(item["product_id"], item["quantity"]):
                raise InsufficientStockError(f"商品 {item['product_id']} 库存不足")
        
        # ... 创建订单
```

**同步通信适合**：需要即时响应的查询操作（如下单前检查库存）

### 异步通信（消息队列/事件）

```python
# 订单服务异步发布事件，库存服务订阅处理
# 不需要直接调用，解耦两个服务

# 订单服务：发布事件（不关心谁消费）
class OrderApplicationService:
    def place_order(self, command: PlaceOrderCommand) -> str:
        order = Order(OrderId(), command.customer_id)
        # ... 创建订单
        order.place()
        self._order_repo.save(order)
        
        # 发布事件到消息队列
        self._event_publisher.publish(
            topic="order.placed",
            message={
                "order_id": str(order.id),
                "items": [{"product_id": i.product_id, "quantity": i.quantity} 
                          for i in order.items]
            }
        )
        return str(order.id)

# 库存服务：消费事件（不知道是谁发的）
class InventoryEventConsumer:
    def handle_order_placed(self, message: dict) -> None:
        """处理订单已下单事件，锁定库存"""
        for item in message["items"]:
            inventory = self._repo.get(item["product_id"])
            inventory.reserve(item["quantity"])
            self._repo.save(inventory)
```

**异步通信适合**：状态变更的传播（如下单后锁库存、支付后发货通知）

---

## 20.4 服务间数据一致性

微服务面临的核心挑战：**没有跨服务的ACID事务**

```
单体中：
  BEGIN TRANSACTION
    UPDATE orders SET status='placed'
    UPDATE inventory SET reserved=reserved+1
  COMMIT  ← 要么全成功，要么全失败

微服务中：
  order_service.place_order()  ← 成功
  inventory_service.reserve()  ← 失败！
  
  现在怎么办？订单已创建，但库存未锁定！
```

### 解决方案：Saga模式（详见第21章）

```python
# 简化预览：Saga协调多个服务的操作
class PlaceOrderSaga:
    """跨服务的订单下单Saga"""
    
    def execute(self, order_id: str):
        try:
            # Step 1: 创建订单
            order_service.create_order(order_id)
            
            # Step 2: 锁定库存
            inventory_service.reserve_stock(order_id)
            
            # Step 3: 预留支付
            payment_service.prepare_payment(order_id)
            
            # 全部成功
            order_service.confirm_order(order_id)
            
        except InventoryException:
            # 补偿：取消订单
            order_service.cancel_order(order_id, reason="库存不足")
        
        except PaymentException:
            # 补偿：取消订单 + 释放库存
            inventory_service.release_stock(order_id)
            order_service.cancel_order(order_id, reason="支付准备失败")
```

---

## 20.5 从单体到微服务的演化

**不要从一开始就上微服务！** 推荐的演化路径：

```
阶段1：模块化单体（推荐起点）
  └── 在单体中按限界上下文做好模块边界
  └── 模块间只通过接口通信，不直接访问对方数据库
  └── 低复杂度，开发快，容易重构

阶段2：提取高负载服务
  └── 识别哪个模块需要独立扩展（如搜索、推荐）
  └── 将这些模块提取为独立服务
  └── 已有的接口设计让提取变得简单

阶段3：按业务边界继续拆分
  └── 随团队增长，按团队边界拆分更多服务
  └── 每次拆分都有清晰的理由（扩展需求/团队自治）

❌ 不推荐：一开始就设计几十个微服务
  └── 边界未经验证，往往拆错
  └── 分布式复杂度太高，开发效率低
  └── "分布式单体"比单体更糟糕
```

---

## 20.6 微服务的常见陷阱

### 陷阱1：分布式单体（Distributed Monolith）

```
症状：
  - 每次发布需要同步部署多个服务
  - 修改一个功能需要改多个服务
  - 服务间有大量同步调用链
  
原因：
  - 服务边界没有按业务能力划分
  - 共享了数据库
  - 应该在单个服务内完成的业务拆成了多个服务
  
解决：
  - 重新审视服务边界，合并过度拆分的服务
  - 消除共享数据库依赖
```

### 陷阱2：数据饥渴（Data Hunger）

```python
# 症状：服务A需要大量来自服务B的数据
class OrderService:
    def get_order_detail(self, order_id: str) -> dict:
        order = self._repo.get(order_id)
        
        # 每次都要调用外部服务获取数据：
        customer = customer_service.get_customer(order.customer_id)  # 外部调用
        products = [catalog_service.get_product(i.product_id) 
                   for i in order.items]                             # N次外部调用！
        payment = payment_service.get_payment(order.payment_id)      # 外部调用
        
        return {...}  # 组合成完整响应

# 解决方案：
# 1. 在本地存储所需的最小数据快照（如ProductSnapshot）
# 2. 使用CQRS的读模型，在本地维护反规范化视图
# 3. 重新审视服务边界（可能边界划错了）
```

---

## 本章小结

| 原则 | 说明 |
|------|------|
| 边界对齐 | 微服务边界 = 限界上下文边界 |
| 数据独立 | 每个服务拥有自己的数据库 |
| 通信方式 | 查询用同步，状态传播用异步 |
| 演化路径 | 模块化单体 → 按需提取微服务 |
| 常见陷阱 | 分布式单体、共享数据库、过度拆分 |

---

**上一章：** [第19章：事件溯源](../part4-architecture/19-event-sourcing.md)  
**下一章：** [第21章：Saga模式](./21-saga-pattern.md)
