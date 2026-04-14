# 第13章：领域服务（Domain Service）

## 学习目标

- 理解领域服务的使用场景
- 区分领域服务与应用服务的职责边界
- 掌握领域服务的设计原则（无状态）
- 避免将领域逻辑错误地放入服务层

---

## 13.1 领域服务的使用场景

有些领域逻辑**不自然属于任何单个实体或值对象**：

```
问题1：两个账户之间的转账
  扣款逻辑属于 SourceAccount？
  入账逻辑属于 TargetAccount？
  还是两者都有，但"转账"这个动作本身不属于任何一个？
  → 领域服务！

问题2：计算订单的最终价格
  涉及：Order（订单项）+ PricingRules（定价规则）+ Coupons（优惠券）
  三个对象相互作用，"定价"不自然属于任何一个
  → 领域服务！

问题3：检查用户名唯一性
  User不应该依赖UserRepository（违反分层）
  唯一性检查需要查询仓储
  → 领域服务！
```

**领域服务的定义**：
> 表达领域概念、协调多个领域对象、**无状态**的操作。

---

## 13.2 领域服务 vs 应用服务

这是最容易混淆的区别，必须清晰：

```
领域服务（Domain Service）：
  ├── 位置：领域层
  ├── 依赖：只依赖领域层的对象（实体、值对象、其他领域服务）
  ├── 职责：表达复杂的领域概念和规则
  ├── 状态：无状态
  └── 测试：纯领域测试，无需Mock框架基础设施

应用服务（Application Service）：
  ├── 位置：应用层（在领域层之上）
  ├── 依赖：可以依赖仓储、消息队列、外部服务等
  ├── 职责：编排用例（协调领域对象和基础设施）
  ├── 状态：无状态（事务由应用服务管理）
  └── 测试：需要Mock基础设施依赖
```

### 一个容易混淆的例子

```python
# 场景：用户注册

# ❌ 把领域逻辑放在应用服务里（常见错误）
class UserApplicationService:
    def register(self, email: str, password: str) -> str:
        # 验证邮箱格式 ← 这是领域规则，不该在这里
        if not re.match(r'^[\w.-]+@[\w.-]+\.\w+$', email):
            raise ValueError("邮箱格式不正确")
        
        # 检查密码强度 ← 这是领域规则，不该在这里
        if len(password) < 8:
            raise ValueError("密码至少8位")
        
        # 创建用户
        user = User(email, password)
        ...

# ✅ 领域逻辑在领域服务/领域对象中
class UserRegistrationDomainService:
    """领域服务：用户注册时的领域规则"""
    
    def __init__(self, user_repo: UserRepository):
        self._repo = user_repo
    
    def validate_registration(self, email: Email, password: Password) -> None:
        """验证注册条件（领域规则）"""
        if self._repo.exists_by_email(email):
            raise UserAlreadyExistsError(f"邮箱 {email} 已被注册")

class UserApplicationService:
    def __init__(
        self, 
        user_repo: UserRepository,
        domain_service: UserRegistrationDomainService,
        event_bus: EventBus
    ):
        self._repo = user_repo
        self._domain_service = domain_service
        self._event_bus = event_bus
    
    def register(self, command: RegisterUserCommand) -> str:
        # 构建值对象（值对象本身包含格式验证）
        email = Email(command.email)
        password = Password(command.password)
        
        # 使用领域服务检查领域规则
        self._domain_service.validate_registration(email, password)
        
        # 创建聚合
        user = User.register(email, password)
        
        # 持久化
        self._repo.save(user)
        
        # 发布事件
        for event in user.collect_events():
            self._event_bus.publish(event)
        
        return str(user.id)
```

---

## 13.3 领域服务的典型场景

### 场景1：账户转账

```python
class TransferDomainService:
    """账户转账领域服务
    
    转账这个动作不属于任何单个账户——它是两个账户之间的交互。
    """
    
    def transfer(
        self, 
        source: BankAccount, 
        target: BankAccount, 
        amount: Money
    ) -> TransferRecord:
        """执行转账"""
        # 领域规则：来源账户余额必须充足
        if source.balance < amount:
            raise InsufficientFundsError(
                f"账户 {source.id} 余额不足，需要 {amount}，实有 {source.balance}"
            )
        
        # 领域规则：不能转给同一账户
        if source.id == target.id:
            raise TransferException("不能向自己转账")
        
        # 领域规则：单笔转账上限
        MAX_TRANSFER = Money(Decimal("100000"), "CNY")
        if amount > MAX_TRANSFER:
            raise TransferException(f"单笔转账不能超过 {MAX_TRANSFER}")
        
        # 执行转账
        source.debit(amount)     # 扣款
        target.credit(amount)    # 入账
        
        return TransferRecord(
            from_account=source.id,
            to_account=target.id,
            amount=amount,
            transferred_at=datetime.now()
        )
```

### 场景2：订单定价

```python
class OrderPricingService:
    """订单定价领域服务
    
    定价涉及：订单 + 定价规则 + 优惠券 + 会员等级
    这些对象的协作不属于任何单一对象
    """
    
    def calculate_final_price(
        self,
        order: Order,
        customer: Customer,
        applicable_coupons: List[Coupon],
        pricing_rules: PricingRules
    ) -> PriceBreakdown:
        """计算订单最终价格"""
        
        # 基础价格
        base_price = order.total
        
        # 应用会员折扣
        member_discount = pricing_rules.get_member_discount(customer.membership_level)
        after_member = base_price * (Decimal("1") - member_discount)
        
        # 选择最优优惠券（不叠加）
        best_coupon_discount = Money(Decimal("0"), base_price.currency)
        applied_coupon = None
        
        for coupon in applicable_coupons:
            discount = coupon.calculate_discount(after_member)
            if discount > best_coupon_discount:
                best_coupon_discount = discount
                applied_coupon = coupon
        
        final_price = after_member - best_coupon_discount
        
        return PriceBreakdown(
            base_price=base_price,
            member_discount=base_price - after_member,
            coupon_discount=best_coupon_discount,
            applied_coupon=applied_coupon,
            final_price=final_price
        )
```

### 场景3：唯一性检查

```python
class UsernameUniquenessChecker:
    """用户名唯一性检查领域服务
    
    User实体不应该依赖UserRepository（违反依赖规则），
    但唯一性检查是明确的领域规则，放在领域服务中。
    """
    
    def __init__(self, user_repo: UserRepository):
        self._repo = user_repo
    
    def is_available(self, username: Username) -> bool:
        """检查用户名是否可用"""
        return not self._repo.exists_by_username(username)
    
    def ensure_available(self, username: Username) -> None:
        """确保用户名可用，否则抛出异常"""
        if not self.is_available(username):
            raise UsernameTakenError(f"用户名 '{username}' 已被使用")
```

### 场景4：协调多个聚合的复杂规则

```python
class LoanApprovalService:
    """贷款审批领域服务
    
    审批需要：客户信用评分 + 还款历史 + 当前贷款情况
    这是复杂的领域规则，协调多个聚合
    """
    
    def evaluate(
        self,
        applicant: Customer,
        credit_history: CreditHistory,
        existing_loans: List[Loan],
        requested_amount: Money
    ) -> LoanDecision:
        """评估贷款申请"""
        
        # 规则1：信用评分低于600分直接拒绝
        if credit_history.score < 600:
            return LoanDecision.rejected("信用评分不足")
        
        # 规则2：已有贷款总额不能超过年收入的5倍
        total_existing = sum(loan.remaining_balance for loan in existing_loans)
        max_loan_capacity = applicant.annual_income * Decimal("5")
        if total_existing + requested_amount > max_loan_capacity:
            return LoanDecision.rejected("贷款总额超过收入限制")
        
        # 规则3：近12个月没有逾期记录
        if credit_history.has_overdue_in_last_months(12):
            return LoanDecision.rejected("近期有逾期记录")
        
        # 通过审批，计算利率
        interest_rate = self._calculate_rate(credit_history.score, requested_amount)
        return LoanDecision.approved(interest_rate)
    
    def _calculate_rate(self, credit_score: int, amount: Money) -> Decimal:
        """根据信用评分和金额计算利率"""
        base_rate = Decimal("0.05")
        if credit_score >= 800:
            return base_rate - Decimal("0.01")
        elif credit_score >= 700:
            return base_rate
        else:
            return base_rate + Decimal("0.01")
```

---

## 13.4 领域服务的命名

领域服务的命名应该反映业务概念，不要叫 `XxxService`（太通用）：

```python
# ❌ 命名不够清晰
class UserService: ...
class OrderService: ...

# ✅ 清晰反映领域角色
class UsernameUniquenessChecker: ...   # 检查器
class OrderPricingCalculator: ...      # 计算器
class LoanApprovalDecisionMaker: ...   # 决策器
class FraudDetector: ...               # 检测器
class AccountTransferCoordinator: ...  # 协调器
```

---

## 13.5 无状态是关键

```python
# ❌ 有状态的领域服务（错误）
class PricingService:
    def __init__(self):
        self._last_calculated_price = None  # 状态！错误！
    
    def calculate(self, order: Order) -> Money:
        price = ...
        self._last_calculated_price = price  # 保存状态！错误！
        return price

# ✅ 无状态的领域服务（正确）
class PricingService:
    def calculate(self, order: Order, rules: PricingRules) -> Money:
        # 只依赖输入参数，不保存任何状态
        # 相同输入 → 相同输出（纯函数语义）
        ...
```

无状态有两个好处：
1. **可以安全共享**：整个应用中可以使用同一个实例（单例）
2. **容易测试**：不需要重置状态，每次调用独立

---

## 本章小结

| 要点 | 内容 |
|------|------|
| 使用场景 | 不自然属于单个实体的领域逻辑 |
| 无状态 | 领域服务不保存状态 |
| 领域层 | 位于领域层，只依赖领域对象 |
| vs 应用服务 | 领域服务含领域规则，应用服务编排用例 |

---

**上一章：** [第12章：仓储](./12-repositories.md)  
**下一章：** [第14章：应用服务](./14-application-services.md)
