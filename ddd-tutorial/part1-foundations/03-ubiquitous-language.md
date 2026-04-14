# 第03章：统一语言（Ubiquitous Language）

## 学习目标

- 理解统一语言的核心意义和价值
- 掌握统一语言的建立与维护方法
- 识别语言不统一带来的实际危害
- 学会将统一语言映射到代码

---

## 3.1 语言的鸿沟

在大多数软件团队中，存在着一道隐形的语言鸿沟：

```
领域专家说：                    开发者理解：
──────────────────────────────────────────────────
"冻结账户"           →        UPDATE user SET status='frozen'
"撤单"              →        DELETE FROM orders WHERE id=?
"核销优惠券"         →        UPDATE coupon SET used=true
"账期"              →        payment_due_date 字段
"清算"              →        某个批处理脚本
```

两边都没有错，但他们在**用不同的语言描述同一件事**。这带来了巨大的隐患：

- 需求传递中产生误解
- 同一个概念在代码中有多个名字
- 修改需求时难以找到对应的代码
- 新人加入需要"重新翻译"业务知识

---

## 3.2 什么是统一语言

**统一语言（Ubiquitous Language）** 是领域专家和开发者共同使用的、用于描述领域模型的语言。

"Ubiquitous"意为"无处不在"——这套语言应该出现在：
- 需求文档和用户故事中
- 会议讨论和白板图中
- 代码的类名、方法名、变量名中
- API接口和消息事件命名中
- 数据库表名和字段名中
- 测试用例的描述中

```
┌─────────────────────────────────────────────────────┐
│                   统一语言（GL）                     │
│                                                     │
│  ┌─────────────┐                ┌─────────────┐    │
│  │  领域专家   │◄──── GL ────►  │   开发者    │    │
│  └─────────────┘                └─────────────┘    │
│         ↑                              ↑            │
│         │           ← GL →             │            │
│         ↓                              ↓            │
│  ┌─────────────┐                ┌─────────────┐    │
│  │  需求文档   │                │    代码     │    │
│  └─────────────┘                └─────────────┘    │
│                                                     │
│  这套语言在所有这些场合中是一致的！                  │
└─────────────────────────────────────────────────────┘
```

### 统一语言的边界

重要：**统一语言是限界上下文内部的语言**。不同的上下文有不同的统一语言。

```
订单上下文的统一语言：
  Order（订单）、OrderItem（订单项）、Shipment（发货单）
  动词：place（下单）、confirm（确认）、ship（发货）、cancel（取消）

支付上下文的统一语言：
  Payment（支付）、Refund（退款）、Transaction（交易流水）
  动词：charge（扣款）、settle（结算）、refund（退款）

同样是"取消"这个动词，在两个上下文中的含义和操作可能完全不同！
```

---

## 3.3 如何建立统一语言

统一语言不是一次性创建的，而是通过持续协作**逐步精炼**的。

### 第一步：召集会议，共同建模

领域专家和开发者坐在一起，用白板探索业务：

```
场景：电商退款流程讨论

领域专家：客户申请退款后，我们需要审核
开发者：审核是什么意思？谁来审？
领域专家：我们的客服人员看一下退款原因，决定是否同意
开发者：那我们叫它"退款申请审核"，对吗？
领域专家：不对，应该叫"退款受理"，因为大部分情况下我们会受理，
         只有少数情况才会拒绝
开发者：好的。那"受理"之后呢？
领域专家：然后触发退款，退款可能到原支付渠道，也可能到余额
开发者：这两种我们叫"原路退款"和"退至余额"，可以吗？
领域专家：可以，但内部我们叫"渠道退款"和"账户退款"
                                    ↑
                            达成了术语共识！
```

### 第二步：建立词汇表

将讨论中确定的术语记录下来：

```markdown
# 退款上下文词汇表

## 名词
- **退款申请（RefundRequest）**：客户发起的要求退款的请求
- **退款受理（RefundAcceptance）**：客服确认接受退款申请的动作
- **渠道退款（ChannelRefund）**：退回到原支付渠道（银行卡/微信等）
- **账户退款（AccountRefund）**：退到平台内部账户余额
- **退款凭证（RefundVoucher）**：退款完成后生成的凭证

## 动词
- **发起退款（initiate refund）**：客户触发退款申请
- **受理（accept）**：客服同意退款申请
- **拒绝（reject）**：客服拒绝退款申请，附带原因
- **执行退款（process refund）**：实际将钱退回的操作
- **确认到账（confirm receipt）**：退款完成的最终确认

## 状态
- PENDING_REVIEW：待受理
- ACCEPTED：已受理
- REJECTED：已拒绝
- PROCESSING：处理中
- COMPLETED：已完成
- FAILED：退款失败
```

### 第三步：将语言映射到代码

这是统一语言最关键的一步：**代码必须直接使用词汇表中的术语**。

```python
# ❌ 糟糕的代码：语言与业务脱节
class RefundService:
    def submit(self, order_id, amount, type):
        record = RefundRecord(order_id, amount, type, "WAIT")
        db.save(record)
        return record.id
    
    def process(self, record_id, approved):
        record = db.get(record_id)
        if approved:
            record.status = "OK"
            self._do_refund(record)
        else:
            record.status = "NO"

# ✅ 好的代码：直接反映业务语言
class RefundRequest:
    """退款申请"""
    
    def __init__(self, request_id: RefundRequestId, order: Order, amount: Money):
        self._id = request_id
        self._order = order
        self._amount = amount
        self._status = RefundStatus.PENDING_REVIEW  # 待受理
    
    def accept(self, handled_by: Staff) -> None:
        """受理退款申请"""
        if self._status != RefundStatus.PENDING_REVIEW:
            raise RefundException("只有待受理状态的退款申请才能被受理")
        self._status = RefundStatus.ACCEPTED
        self._handled_by = handled_by
        self._record_event(RefundAccepted(self._id))
    
    def reject(self, handled_by: Staff, reason: str) -> None:
        """拒绝退款申请"""
        if self._status != RefundStatus.PENDING_REVIEW:
            raise RefundException("只有待受理状态的退款申请才能被拒绝")
        self._status = RefundStatus.REJECTED
        self._rejection_reason = reason
        self._record_event(RefundRejected(self._id, reason))
```

注意对比两段代码：
- `submit/process` vs `accept/reject`
- `"WAIT"/"OK"/"NO"` vs `PENDING_REVIEW/ACCEPTED/REJECTED`
- 第一段：开发者自己发明的词，业务人员看不懂
- 第二段：直接来自词汇表，领域专家可以review代码

---

## 3.4 统一语言的实践技巧

### 技巧1：从动词开始

业务操作往往比名词更能揭示真实的业务规则：

```
不好的问题：有哪些"用户"相关的实体？

好的问题：
- 用户能做什么？（注册、登录、修改信息、注销）
- 系统对用户做什么？（激活账户、冻结账户、推送通知）
- 什么情况会触发什么？（登录失败3次 → 锁定账户）

从动词出发，自然就能发现需要的名词和规则
```

### 技巧2：用领域专家的话做测试

如果你的代码能被领域专家读懂，统一语言就做到位了：

```python
# 测试用例本身就是统一语言的体现
def test_refund_request_can_be_accepted_when_pending():
    """待受理的退款申请可以被受理"""
    # Given 一个待受理的退款申请
    refund_request = RefundRequest.create(
        order=mock_order,
        amount=Money(100, "CNY"),
        reason="商品质量问题"
    )
    
    # When 客服受理该申请
    staff = Staff("客服张三")
    refund_request.accept(handled_by=staff)
    
    # Then 申请状态变为已受理
    assert refund_request.status == RefundStatus.ACCEPTED


def test_rejected_refund_cannot_be_accepted():
    """已拒绝的退款申请不能被重新受理"""
    ...
```

### 技巧3：警惕翻译行为

当你发现自己在脑海中进行"翻译"时，这是语言不统一的信号：

```
危险信号：
- 领域专家说"冻结账户"，你脑子里自动翻译成 update status
- 代码里的 order_status=2 需要你查文档才知道什么意思
- 接口文档里的参数名和业务文档里的术语完全不同
- 测试用例命名和业务场景描述对不上

正确做法：
- 直接用 freeze_account() 方法
- 直接用 OrderStatus.CONFIRMED 枚举
- API 参数名就用业务术语
- 测试名称直接复制验收标准
```

### 技巧4：持续精炼，不怕改名

随着对业务的理解加深，统一语言会进化：

```
第一周讨论：
  "退款" → Refund（但后来发现有多种退款）

第三周精炼：
  "退款申请" → RefundRequest
  "渠道退款" → ChannelRefund  
  "余额退款" → BalanceRefund

第八周再精炼：
  发现"余额退款"在账务上叫"余额增加"，更准确的叫法是
  "账户补偿" → AccountCompensation
  
  这不是"改名"，而是"模型深化"
```

---

## 3.5 统一语言与代码的深度绑定

让我们看一个完整的例子，展示统一语言如何渗透到代码的每个层次：

### 业务场景：会员卡充值

**领域专家的描述**：
> 用户可以给会员卡充值，充值金额会立即到账，同时会员卡的有效期可能会根据充值档位自动延长。某些特殊活动期间，充值会有赠送金额，赠送金额和充值金额记录在不同的余额里。

**从描述中提取的统一语言**：

```markdown
名词：
- 会员卡（MembershipCard）
- 充值（Recharge/TopUp）
- 有效期（ExpirationDate / ValidityPeriod）
- 充值档位（RechargeTier）
- 赠送金额（BonusAmount）
- 充值余额（RechargeBalance）
- 赠送余额（BonusBalance）

动词：
- 充值（recharge / top_up）
- 到账（credit）
- 延长有效期（extend_validity）
- 赠送（grant_bonus）

规则：
- 充值金额立即到账
- 有效期可能根据档位延长（"可能"意味着并非总是延长）
- 赠送金额和充值金额分开记录
```

**对应的代码**：

```python
from dataclasses import dataclass
from decimal import Decimal
from datetime import date, timedelta
from typing import Optional

@dataclass(frozen=True)
class RechargeTier:
    """充值档位"""
    amount: Decimal           # 充值金额
    bonus_amount: Decimal     # 赠送金额
    validity_extension_days: Optional[int]  # 有效期延长天数（None表示不延长）

class MembershipCard:
    """会员卡"""
    
    def __init__(self, card_id: str, owner_id: str, expiration_date: date):
        self._card_id = card_id
        self._owner_id = owner_id
        self._expiration_date = expiration_date
        self._recharge_balance = Decimal("0")   # 充值余额
        self._bonus_balance = Decimal("0")      # 赠送余额
    
    def recharge(self, tier: RechargeTier, recharge_date: date) -> "RechargeRecord":
        """充值
        
        充值金额立即到账，赠送金额分开记录。
        若档位配置了有效期延长，则延长有效期。
        """
        # 充值金额到账
        self._recharge_balance += tier.amount
        
        # 赠送金额到账（分开记录）
        if tier.bonus_amount > 0:
            self._bonus_balance += tier.bonus_amount
        
        # 延长有效期（若档位有此配置）
        if tier.validity_extension_days:
            self._extend_validity(tier.validity_extension_days, recharge_date)
        
        return RechargeRecord(
            card_id=self._card_id,
            tier=tier,
            recharge_date=recharge_date
        )
    
    def _extend_validity(self, days: int, from_date: date) -> None:
        """延长有效期"""
        base_date = max(self._expiration_date, from_date)
        self._expiration_date = base_date + timedelta(days=days)
    
    @property
    def available_balance(self) -> Decimal:
        """可用余额（充值余额 + 赠送余额）"""
        return self._recharge_balance + self._bonus_balance


# 使用示例：直接读起来就像业务描述
double_eleven_tier = RechargeTier(
    amount=Decimal("100"),
    bonus_amount=Decimal("20"),      # 充100送20
    validity_extension_days=30       # 延长30天
)

card.recharge(tier=double_eleven_tier, recharge_date=date.today())
```

这段代码可以直接给领域专家阅读，他们能理解每个方法和属性的含义。

---

## 3.6 统一语言的反模式

### 反模式1：技术术语污染业务语言

```python
# ❌ 技术术语污染：DTO、CRUD渗入业务层
class OrderDTO:
    pass

def create_order_record(dto: OrderDTO):
    # "create_record"是技术术语，"place_order"才是业务术语
    pass

# ✅ 业务语言纯粹
def place_order(customer: Customer, items: list[OrderItem]) -> Order:
    pass
```

### 反模式2：缩写和代码暗语

```python
# ❌ 充斥着只有老员工才懂的暗语
order.st = 2      # st是status的缩写？2是什么意思？
order.pflag = True  # pflag是什么？

# ✅ 直接表达语义
order.status = OrderStatus.CONFIRMED
order.payment_confirmed = True
```

### 反模式3：一个概念多个名字

```python
# ❌ 同一概念，代码里三种叫法
class User: pass       # 在某个模块
class Member: pass     # 在另一个模块（但其实是同一回事）
class Account: pass    # 又一种叫法

# ✅ 如果是同一个限界上下文内的同一概念，统一名字
# 如果是不同上下文的不同概念，明确区分
```

---

## 本章小结

| 要点 | 说明 |
|------|------|
| 统一语言的本质 | 领域专家与开发者共同协商的语言，无处不在 |
| 建立方式 | 共同建模 → 词汇表 → 代码映射 |
| 边界性 | 统一语言有限界上下文边界，跨边界可以不同 |
| 核心标准 | 领域专家能读懂代码 |
| 持续演化 | 随业务理解加深不断精炼，不怕改名 |

---

## 思考练习

1. 列出你当前项目中5个有歧义或不够准确的类名/方法名，尝试用更贴近业务的语言重命名
2. 找一段你的业务核心代码，让一位领域专家尝试阅读，看他能理解多少
3. 尝试为某个业务模块建立一个词汇表，不超过20个术语

---

**上一章：** [第02章：核心概念](./02-core-concepts.md)  
**下一章：** [第04章：战略设计与战术设计的关系](./04-strategic-vs-tactical.md)
