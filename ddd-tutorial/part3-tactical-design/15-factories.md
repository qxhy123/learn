# 第15章：工厂（Factory）

## 学习目标

- 理解工厂在DDD中的角色（封装复杂创建逻辑）
- 掌握三种工厂形式：工厂方法、工厂类、抽象工厂
- 区分工厂与仓储的职责（创建 vs 重建）
- 知道何时需要工厂，何时简单构造即可

---

## 15.1 为什么需要工厂

当对象的创建逻辑变得复杂时，将创建逻辑放在构造函数中会带来问题：

```python
# ❌ 复杂创建逻辑塞在构造函数里
class Order:
    def __init__(self, raw_data: dict):
        # 构造函数里有大量转换和验证逻辑
        self._id = OrderId(UUID(raw_data["id"]))
        self._customer_id = raw_data["customer_id"]
        
        # 需要调用外部服务获取商品信息？构造函数里做？
        self._items = []
        for item_data in raw_data["items"]:
            product = ProductCatalogAPI.get(item_data["product_id"])  # 外部调用？
            self._items.append(OrderItem(product, item_data["quantity"]))
        
        # 验证业务规则
        if len(self._items) == 0:
            raise ValueError("订单必须有至少一个商品")
        
        # 这个构造函数做了太多事情！

# ✅ 工厂封装复杂创建逻辑，构造函数保持简单
class Order:
    def __init__(self, order_id: OrderId, customer_id: str, items: List[OrderItem]):
        # 构造函数只做最基本的赋值
        self._id = order_id
        self._customer_id = customer_id
        self._items = items
        self._status = OrderStatus.DRAFT

class OrderFactory:
    """封装创建Order的复杂逻辑"""
    
    def __init__(self, catalog: ProductCatalog):
        self._catalog = catalog
    
    def create(self, customer_id: str, item_requests: List[dict]) -> Order:
        items = []
        for req in item_requests:
            product = self._catalog.get_snapshot(req["product_id"])
            items.append(OrderItem(product, req["quantity"]))
        
        if not items:
            raise OrderException("订单必须有至少一个商品")
        
        return Order(OrderId(), customer_id, items)
```

---

## 15.2 三种工厂形式

### 形式1：工厂方法（Factory Method）—— 最常用

在聚合根上定义静态工厂方法，语义清晰：

```python
class User:
    """用户聚合"""
    
    def __init__(self, user_id: UserId, email: Email, password_hash: str):
        self._id = user_id
        self._email = email
        self._password_hash = password_hash
        self._status = UserStatus.INACTIVE
        self._events: List[DomainEvent] = []
    
    @classmethod
    def register(cls, email: Email, password: Password) -> "User":
        """工厂方法：注册新用户"""
        user = cls(
            user_id=UserId(),
            email=email,
            password_hash=password.hash()
        )
        user._record_event(UserRegistered(
            user_id=str(user._id),
            email=str(email),
            registered_at=datetime.now()
        ))
        return user
    
    @classmethod
    def create_admin(cls, email: Email, created_by: UserId) -> "User":
        """工厂方法：创建管理员账户（不同的创建逻辑）"""
        user = cls(
            user_id=UserId(),
            email=email,
            password_hash=Password.generate_temporary().hash()
        )
        user._status = UserStatus.ACTIVE  # 管理员直接激活
        user._role = UserRole.ADMIN
        user._record_event(AdminCreated(
            user_id=str(user._id),
            created_by=str(created_by)
        ))
        return user
    
    @classmethod
    def reconstitute(cls, snapshot: UserSnapshot) -> "User":
        """工厂方法：从持久化数据重建（仓储专用）"""
        user = cls.__new__(cls)  # 绕过 __init__，直接分配内存
        user._id = snapshot.id
        user._email = snapshot.email
        user._password_hash = snapshot.password_hash
        user._status = snapshot.status
        user._events = []  # 重建时不重放事件
        return user
```

**命名约定**：
```
create()        通用创建
register()      注册语义
open()          开户、开始
initiate()      发起
apply()         申请
reconstitute()  从持久化重建（特殊用途）
```

### 形式2：工厂类（Factory Class）

当创建逻辑复杂，需要依赖多个外部服务时：

```python
class MortgageLoanFactory:
    """抵押贷款创建工厂——创建逻辑非常复杂"""
    
    def __init__(
        self,
        property_valuator: PropertyValuator,    # 房产估值服务
        credit_bureau: CreditBureau,             # 征信查询
        risk_calculator: RiskCalculator,         # 风险计算
    ):
        self._valuator = property_valuator
        self._credit_bureau = credit_bureau
        self._risk_calculator = risk_calculator
    
    def create_application(
        self,
        applicant: Customer,
        property: Property,
        requested_amount: Money
    ) -> MortgageLoanApplication:
        """创建抵押贷款申请
        
        需要：
        1. 评估房产价值
        2. 查询申请人信用
        3. 计算初始风险评分
        4. 确定可贷比例
        """
        # 房产估值
        property_value = self._valuator.valuate(property)
        
        # 最高可贷比例（LTV: Loan-To-Value）
        max_ltv = Decimal("0.7")  # 最高贷款70%
        max_loan = property_value * max_ltv
        
        if requested_amount > max_loan:
            raise LoanApplicationException(
                f"申请金额 {requested_amount} 超过最高可贷额度 {max_loan}"
            )
        
        # 信用查询
        credit_report = self._credit_bureau.query(applicant.id_number)
        
        # 风险评分
        risk_score = self._risk_calculator.calculate(
            applicant, credit_report, requested_amount, property_value
        )
        
        # 创建申请聚合
        return MortgageLoanApplication(
            application_id=LoanApplicationId(),
            applicant_id=applicant.id,
            property_id=property.id,
            requested_amount=requested_amount,
            property_value=property_value,
            initial_risk_score=risk_score,
            applied_at=datetime.now()
        )
```

### 形式3：抽象工厂（Abstract Factory）

当需要创建一系列相关对象时：

```python
from abc import ABC, abstractmethod

class NotificationFactory(ABC):
    """通知工厂——根据渠道创建不同的通知对象"""
    
    @abstractmethod
    def create_order_confirmed_notification(
        self, order: Order, recipient: str
    ) -> Notification: ...
    
    @abstractmethod
    def create_payment_failed_notification(
        self, order: Order, recipient: str
    ) -> Notification: ...

class EmailNotificationFactory(NotificationFactory):
    def create_order_confirmed_notification(
        self, order: Order, recipient: str
    ) -> Notification:
        return EmailNotification(
            to=recipient,
            subject=f"订单 {order.id} 确认",
            body=f"您的订单已确认，总金额 {order.total}"
        )
    
    def create_payment_failed_notification(
        self, order: Order, recipient: str
    ) -> Notification:
        return EmailNotification(
            to=recipient,
            subject=f"订单 {order.id} 支付失败",
            body="您的支付未能完成，请重新尝试..."
        )

class SmsNotificationFactory(NotificationFactory):
    def create_order_confirmed_notification(
        self, order: Order, recipient: str
    ) -> Notification:
        return SmsNotification(
            phone=recipient,
            message=f"订单已确认，总计{order.total}"  # 短信要简短
        )
    
    def create_payment_failed_notification(
        self, order: Order, recipient: str
    ) -> Notification:
        return SmsNotification(
            phone=recipient,
            message=f"支付失败，请重试"
        )
```

---

## 15.3 工厂 vs 仓储：创建 vs 重建

这是一个重要的区别：

```
工厂：创建新的对象
  ├── 全新的状态
  ├── 生成新的标识（ID）
  ├── 可能触发领域事件（如 UserRegistered）
  └── 输入：业务数据（命令）

仓储：重建已存在的对象
  ├── 从持久化数据恢复状态
  ├── 使用已有的标识（ID）
  ├── 不触发领域事件
  └── 输入：持久化数据（数据库行/文档）
```

```python
# 工厂：创建新对象
class OrderFactory:
    def create(self, customer_id: str, items: list) -> Order:
        order_id = OrderId()   # 新ID
        order = Order(order_id, customer_id)
        for item in items:
            order.add_item(...)
        # 订单处于 DRAFT 状态，没有历史
        return order

# 仓储：重建已有对象
class OrderRepository:
    def get(self, order_id: OrderId) -> Order:
        orm = self._db.find(str(order_id))
        # 使用 reconstitute 而非 create：
        return Order.reconstitute(
            order_id=OrderId(UUID(orm.id)),
            customer_id=orm.customer_id,
            status=OrderStatus(orm.status),
            items=[...],
        )
```

---

## 15.4 何时需要工厂，何时不需要

```
需要工厂的情况：
  ✅ 创建逻辑复杂（超过3-5行初始化代码）
  ✅ 需要调用外部服务来准备创建所需数据
  ✅ 有多种创建方式（注册/管理员创建/第三方导入）
  ✅ 创建过程有业务规则（创建前的检查）
  ✅ 需要明确表达不同的业务创建语义

不需要工厂的情况：
  ❌ 简单的值对象（直接用构造函数）
  ❌ 只有一种创建方式且逻辑简单
  ❌ 测试用对象（直接构造或用 builder）
```

```python
# 不需要工厂：简单值对象，直接构造
address = Address("上海", "浦东新区", "张江路", "200120")
money = Money(Decimal("99.99"), "CNY")

# 需要工厂：复杂聚合，有业务语义
user = User.register(email, password)          # 工厂方法
loan = loan_factory.create_application(...)    # 工厂类
```

---

## 15.5 测试中的工厂：Test Builder 模式

工厂的思想在测试中也很有用：

```python
class OrderBuilder:
    """测试专用：快速构建测试用的Order对象"""
    
    def __init__(self):
        self._order_id = OrderId()
        self._customer_id = "test-customer"
        self._items = []
        self._status = OrderStatus.DRAFT
    
    def with_id(self, order_id: OrderId) -> "OrderBuilder":
        self._order_id = order_id
        return self
    
    def with_customer(self, customer_id: str) -> "OrderBuilder":
        self._customer_id = customer_id
        return self
    
    def with_item(self, product_id: str, price: Decimal, qty: int = 1) -> "OrderBuilder":
        snapshot = ProductSnapshot(product_id, f"Product-{product_id}", Money(price))
        self._items.append(OrderItem(f"item-{len(self._items)}", snapshot, qty))
        return self
    
    def placed(self) -> "OrderBuilder":
        self._status = OrderStatus.PLACED
        return self
    
    def paid(self) -> "OrderBuilder":
        self._status = OrderStatus.PAID
        return self
    
    def build(self) -> Order:
        return Order.reconstitute(
            order_id=self._order_id,
            customer_id=self._customer_id,
            status=self._status,
            items=self._items
        )


# 在测试中使用
def test_paid_order_cannot_be_cancelled():
    order = (
        OrderBuilder()
        .with_customer("cust-001")
        .with_item("iphone", Decimal("7999"))
        .paid()  # 已支付状态
        .build()
    )
    
    with pytest.raises(OrderException, match="不可取消"):
        order.cancel("changed mind")
```

---

## 本章小结

| 工厂形式 | 使用场景 |
|---------|---------|
| 工厂方法（classmethod） | 聚合上有多种创建语义 |
| 工厂类 | 创建需要依赖外部服务 |
| 抽象工厂 | 创建一族相关对象 |

**工厂 vs 仓储**：工厂创建新对象（新ID，新状态），仓储重建已有对象（旧ID，恢复状态）。

---

**上一章：** [第14章：应用服务](./14-application-services.md)  
**下一章：** [第16章：分层架构](../part4-architecture/16-layered-architecture.md)
