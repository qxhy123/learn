# 第12章：仓储（Repository）

## 学习目标

- 理解仓储的"集合语义"和持久化抽象
- 掌握仓储接口的设计原则
- 实现仓储的多种持久化策略
- 区分仓储与DAO的本质区别

---

## 12.1 仓储解决什么问题

没有仓储时，领域层的代码会直接依赖持久化技术：

```python
# ❌ 领域层直接依赖数据库（糟糕的设计）
class OrderService:
    def place_order(self, order_id: str) -> None:
        # 直接写SQL
        row = db.execute("SELECT * FROM orders WHERE id=?", order_id)
        order_data = dict(row)
        
        # 手动重建领域对象
        order = Order(order_data["id"], order_data["customer_id"])
        order._status = OrderStatus(order_data["status"])
        
        # 业务操作
        order.place()
        
        # 手动持久化
        db.execute(
            "UPDATE orders SET status=? WHERE id=?", 
            order.status.value, order_id
        )
```

问题：
- 领域层与数据库强耦合
- 换数据库要改业务代码
- 单元测试必须连接真实数据库
- 重建领域对象的代码散落各处

---

## 12.2 仓储的本质：集合语义

**仓储（Repository）** 为聚合提供一个类似**内存集合**的持久化抽象。

```
比喻：
  没有仓储：每次需要订单，都要去数据库"抓取"，然后"拼装"
  有仓储：  就像有一个装满Order对象的列表，
           list.find(id)、list.add(order)、list.remove(order)

仓储让领域层"感觉不到"持久化的存在
```

```python
from abc import ABC, abstractmethod
from typing import Optional, List

class OrderRepository(ABC):
    """订单仓储接口——集合语义"""
    
    @abstractmethod
    def get(self, order_id: OrderId) -> Order:
        """根据ID获取订单，不存在则抛出异常"""
        ...
    
    @abstractmethod
    def find(self, order_id: OrderId) -> Optional[Order]:
        """根据ID查找订单，不存在返回None"""
        ...
    
    @abstractmethod
    def save(self, order: Order) -> None:
        """保存订单（新增或更新）"""
        ...
    
    @abstractmethod
    def remove(self, order: Order) -> None:
        """删除订单"""
        ...
    
    @abstractmethod
    def find_by_customer(self, customer_id: str) -> List[Order]:
        """按客户查找订单列表"""
        ...
```

---

## 12.3 仓储的设计原则

### 原则1：每个聚合根一个仓储

```python
# ✅ 每个聚合根有自己的仓储
class OrderRepository(ABC): ...
class CustomerRepository(ABC): ...
class ProductRepository(ABC): ...

# ❌ 为聚合内部实体单独建仓储
class OrderItemRepository(ABC): ...  # 错！通过Order访问OrderItem
```

### 原则2：接口在领域层，实现在基础设施层

```
project/
├── domain/
│   ├── model/
│   │   └── order.py          # 领域模型
│   └── repository/
│       └── order_repository.py   # 仓储接口（抽象）
│
└── infrastructure/
    └── persistence/
        ├── sqlalchemy_order_repository.py   # SQL实现
        ├── in_memory_order_repository.py    # 内存实现（用于测试）
        └── redis_order_repository.py        # 缓存实现
```

```python
# 领域层：只定义接口
# domain/repository/order_repository.py
class OrderRepository(ABC):
    @abstractmethod
    def get(self, order_id: OrderId) -> Order: ...

# 基础设施层：提供实现
# infrastructure/persistence/sqlalchemy_order_repository.py
class SqlAlchemyOrderRepository(OrderRepository):
    def __init__(self, session: Session):
        self._session = session
    
    def get(self, order_id: OrderId) -> Order:
        ...
```

### 原则3：仓储只操作聚合根

```python
class OrderRepository(ABC):
    def get(self, order_id: OrderId) -> Order: ...       # ✅ 操作聚合根
    def save(self, order: Order) -> None: ...            # ✅ 操作聚合根
    
    # ❌ 不应该有针对聚合内部对象的方法
    # def get_order_item(self, item_id: str): ...
    # def save_order_item(self, item: OrderItem): ...
```

### 原则4：查询方法有节制

```python
class OrderRepository(ABC):
    # ✅ 合理的查询方法
    def get(self, order_id: OrderId) -> Order: ...
    def find_by_customer(self, customer_id: str) -> List[Order]: ...
    def find_pending_orders(self) -> List[Order]: ...
    
    # ❌ 过多的查询方法会让仓储膨胀
    # def find_by_customer_and_status_and_date_range_and_amount_greater_than(...)
    # → 复杂查询应该用专门的查询服务或读模型
```

---

## 12.4 具体实现：SQLAlchemy版本

```python
from sqlalchemy import Column, String, Numeric, DateTime, create_engine
from sqlalchemy.orm import Session, relationship
from sqlalchemy.ext.declarative import declarative_base
import json

Base = declarative_base()

# ORM模型（数据库映射，与领域模型分离）
class OrderORM(Base):
    __tablename__ = "orders"
    
    id = Column(String(36), primary_key=True)
    customer_id = Column(String(36), nullable=False)
    status = Column(String(20), nullable=False)
    total_amount = Column(Numeric(10, 2), nullable=False)
    total_currency = Column(String(3), nullable=False)
    created_at = Column(DateTime, nullable=False)
    
    items = relationship("OrderItemORM", back_populates="order", cascade="all, delete-orphan")

class OrderItemORM(Base):
    __tablename__ = "order_items"
    
    id = Column(String(50), primary_key=True)
    order_id = Column(String(36), nullable=False)
    product_id = Column(String(36), nullable=False)
    product_name = Column(String(200), nullable=False)
    unit_price_amount = Column(Numeric(10, 2), nullable=False)
    unit_price_currency = Column(String(3), nullable=False)
    quantity = Column(Numeric(10, 0), nullable=False)
    
    order = relationship("OrderORM", back_populates="items")


# 仓储实现：负责ORM ↔ 领域对象的转换
class SqlAlchemyOrderRepository(OrderRepository):
    
    def __init__(self, session: Session):
        self._session = session
    
    def get(self, order_id: OrderId) -> Order:
        order = self.find(order_id)
        if order is None:
            raise OrderNotFoundError(f"订单 {order_id} 不存在")
        return order
    
    def find(self, order_id: OrderId) -> Optional[Order]:
        orm_obj = self._session.get(OrderORM, str(order_id))
        if orm_obj is None:
            return None
        return self._to_domain(orm_obj)
    
    def save(self, order: Order) -> None:
        orm_obj = self._session.get(OrderORM, str(order.id))
        if orm_obj is None:
            orm_obj = OrderORM()
            self._session.add(orm_obj)
        self._update_orm(orm_obj, order)
    
    def find_by_customer(self, customer_id: str) -> List[Order]:
        orm_list = (
            self._session.query(OrderORM)
            .filter(OrderORM.customer_id == customer_id)
            .all()
        )
        return [self._to_domain(o) for o in orm_list]
    
    # ===== 私有转换方法 =====
    
    def _to_domain(self, orm: OrderORM) -> Order:
        """ORM对象 → 领域对象"""
        # 使用工厂方法重建聚合（绕过正常构造，恢复持久化状态）
        order = Order.reconstitute(
            order_id=OrderId(UUID(orm.id)),
            customer_id=orm.customer_id,
            status=OrderStatus(orm.status),
            items=[self._item_to_domain(item_orm) for item_orm in orm.items],
            created_at=orm.created_at,
        )
        return order
    
    def _item_to_domain(self, orm: OrderItemORM) -> OrderItem:
        product_snapshot = ProductSnapshot(
            product_id=orm.product_id,
            name=orm.product_name,
            unit_price=Money(orm.unit_price_amount, orm.unit_price_currency)
        )
        return OrderItem(orm.id, product_snapshot, int(orm.quantity))
    
    def _update_orm(self, orm: OrderORM, order: Order) -> None:
        """领域对象 → 更新ORM对象"""
        snapshot = order.to_snapshot()  # 领域对象提供快照
        orm.id = str(snapshot.id)
        orm.customer_id = snapshot.customer_id
        orm.status = snapshot.status.value
        orm.total_amount = snapshot.total.amount
        orm.total_currency = snapshot.total.currency
        orm.created_at = snapshot.created_at
        # 更新items（简化处理，实际需要diff）
        orm.items.clear()
        for item_snap in snapshot.items:
            orm.items.append(OrderItemORM(
                id=item_snap.id,
                order_id=str(snapshot.id),
                product_id=item_snap.product_id,
                product_name=item_snap.product_name,
                unit_price_amount=item_snap.unit_price.amount,
                unit_price_currency=item_snap.unit_price.currency,
                quantity=item_snap.quantity
            ))
```

---

## 12.5 内存仓储（用于测试）

```python
class InMemoryOrderRepository(OrderRepository):
    """内存仓储——单元测试用，无需数据库"""
    
    def __init__(self):
        self._store: Dict[str, Order] = {}
    
    def get(self, order_id: OrderId) -> Order:
        order = self.find(order_id)
        if order is None:
            raise OrderNotFoundError(str(order_id))
        return order
    
    def find(self, order_id: OrderId) -> Optional[Order]:
        return self._store.get(str(order_id))
    
    def save(self, order: Order) -> None:
        self._store[str(order.id)] = order
    
    def remove(self, order: Order) -> None:
        self._store.pop(str(order.id), None)
    
    def find_by_customer(self, customer_id: str) -> List[Order]:
        return [o for o in self._store.values() if o.customer_id == customer_id]
    
    def all(self) -> List[Order]:
        """测试辅助方法"""
        return list(self._store.values())


# 使用内存仓储的测试
def test_order_can_be_placed():
    repo = InMemoryOrderRepository()
    
    # Arrange
    order = Order(OrderId(), "customer-001")
    order.add_item(ProductSnapshot("p1", "iPhone", Money(Decimal("7999"))), 1)
    repo.save(order)
    
    # Act
    saved_order = repo.get(order.id)
    saved_order.place()
    repo.save(saved_order)
    
    # Assert
    retrieved = repo.get(order.id)
    assert retrieved.status == OrderStatus.PLACED
```

---

## 12.6 仓储 vs DAO

这是一个常见的混淆点：

```
DAO（Data Access Object）：
  ├── 以数据库表为中心
  ├── 方法对应SQL操作（select, insert, update, delete）
  ├── 通常返回数据传输对象（DTO）或行记录
  └── 不理解领域规则

Repository（仓储）：
  ├── 以聚合为中心
  ├── 方法对应集合操作（get, save, find, remove）
  ├── 返回领域对象（实体/聚合根）
  └── 理解领域边界（只暴露聚合根，不暴露内部实体）

class UserDAO:              class UserRepository:
  find_by_id()               get(user_id: UserId) -> User
  insert()                   save(user: User)
  update_email()             find_by_email(email: str) -> Optional[User]
  delete_by_id()             remove(user: User)
  # 操作数据库字段            # 操作领域对象
```

---

## 本章小结

| 原则 | 内容 |
|------|------|
| 集合语义 | 仓储像内存中的集合，隐藏持久化细节 |
| 接口/实现分离 | 接口在领域层，实现在基础设施层 |
| 每聚合一仓储 | 不为聚合内部实体单独建仓储 |
| 可替换实现 | 内存/SQL/缓存等多种实现，测试用内存版 |

---

## 思考练习

1. 找一个你系统中的DAO，尝试将它改造为仓储——有哪些本质的变化？
2. 为什么仓储的接口要放在领域层，而不是基础设施层？
3. 在什么情况下，一个仓储方法的签名需要改变？

---

**上一章：** [第11章：领域事件](./11-domain-events.md)  
**下一章：** [第13章：领域服务](./13-domain-services.md)
