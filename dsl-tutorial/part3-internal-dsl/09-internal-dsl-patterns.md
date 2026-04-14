# 第9章：内部DSL模式总览

## 核心思维模型

> 内部DSL是一种**语言内的语言**：利用宿主语言的语法特性，在不改变语言本身的前提下，让代码"看起来像"一门领域语言。关键在于**最大化信噪比**——让领域逻辑突出，让语言噪声消隐。

---

## 9.1 什么是语法噪声？

```python
# 高噪声比（纯Python，领域逻辑被淹没在语法中）
rule = Rule(
    name="成年验证",
    condition=Condition(
        left=FieldRef(name="age"),
        operator=Operator.GREATER_THAN,
        right=IntLiteral(value=18)
    ),
    action=Action(
        type=ActionType.SET_FLAG,
        target="is_adult",
        value=BoolLiteral(value=True)
    )
)

# 低噪声比（内部DSL，领域逻辑清晰）
rule("成年验证").when(age > 18).then(set_flag("is_adult", True))
```

**语法噪声**：构造函数调用、类名、参数名、括号、逗号——这些是语言机制，不是领域概念。

内部DSL设计的核心任务：**用宿主语言的元编程特性消除噪声**。

---

## 9.2 Python内部DSL的七种武器

Python提供了丰富的元编程特性，可以用来构建内部DSL：

### 武器一：方法链（Method Chaining）

```python
# 每个方法返回self（或新对象），链式调用
result = (
    Query()
    .from_table("users")
    .where(age > 18)
    .select("name", "email")
    .order_by("name")
    .limit(10)
)
```

### 武器二：操作符重载（Operator Overloading）

```python
# 使用 __gt__, __lt__, __eq__ 等重载运算符
class Field:
    def __gt__(self, value): return Comparison(self, ">", value)
    def __lt__(self, value): return Comparison(self, "<", value)
    def __eq__(self, value): return Comparison(self, "=", value)

age = Field("age")
condition = age > 18  # 创建Comparison对象，而非布尔值
```

### 武器三：装饰器（Decorators）

```python
# 用装饰器注册和元数据化函数
@app.route("/users", methods=["GET"])
def get_users():
    pass

@rule_engine.register("premium_discount")
@when(user.subscription == "premium")
def apply_discount(order):
    order.discount = 0.2
```

### 武器四：上下文管理器（Context Managers）

```python
# with语句创建作用域
with transaction():
    create_user("Alice")
    create_order(user_id=1, amount=100)
    # 异常自动回滚

with pipeline.stage("build"):
    run("docker build .")
    run("docker push registry/app:latest")
```

### 武器五：`__getattr__`动态属性

```python
class Schema:
    def __getattr__(self, name):
        return Field(name)

schema = Schema()
condition = schema.age > 18      # schema.age 动态创建 Field("age")
condition2 = schema.email != ""  # schema.email 动态创建 Field("email")
```

### 武器六：元类（Metaclass）

```python
class ModelMeta(type):
    def __new__(mcs, name, bases, namespace):
        fields = {k: v for k, v in namespace.items() if isinstance(v, Field)}
        namespace['_fields'] = fields
        return super().__new__(mcs, name, bases, namespace)

class User(metaclass=ModelMeta):
    name = StringField()
    age = IntField()
    email = StringField()
# User._fields == {'name': StringField(), 'age': IntField(), ...}
```

### 武器七：`__or__` / `__and__` 管道符

```python
# 用 | 运算符构建管道（类似Unix管道）
result = data | filter(age > 18) | select("name") | limit(10)

# 或者组合验证器
validator = not_empty | max_length(100) | email_format
```

---

## 9.3 内部DSL的四个经典模式

### 模式一：流畅接口（Fluent Interface）

**特征**：每个方法返回当前对象或新对象，支持链式调用。

```python
class EmailBuilder:
    def __init__(self):
        self._to = []
        self._subject = ""
        self._body = ""
        self._attachments = []
    
    def to(self, *addresses: str) -> 'EmailBuilder':
        self._to.extend(addresses)
        return self
    
    def subject(self, text: str) -> 'EmailBuilder':
        self._subject = text
        return self
    
    def body(self, text: str) -> 'EmailBuilder':
        self._body = text
        return self
    
    def attach(self, path: str) -> 'EmailBuilder':
        self._attachments.append(path)
        return self
    
    def send(self) -> bool:
        email = Email(self._to, self._subject, self._body, self._attachments)
        return email_service.send(email)

# 使用：像读英文一样
EmailBuilder()\
    .to("alice@example.com", "bob@example.com")\
    .subject("项目进度报告")\
    .body("本周完成了...")\
    .attach("/reports/week42.pdf")\
    .send()
```

### 模式二：构建者（Builder）

**特征**：将复杂对象的构造分步骤进行，最后调用`build()`生成不可变对象。

```python
# 第10章详细讲解，这里先给出骨架
query = (QueryBuilder()
    .from_("users")
    .where(age > 18)
    .build())  # 返回不可变Query对象
```

### 模式三：表达式树（Expression Tree）

**特征**：通过操作符重载，让Python表达式直接构建AST节点。

```python
# 第11章深入讲解，核心技巧：
age = Column("age")
status = Column("status")

# Python表达式构建条件树
condition = (age > 18) & (status == "active")
# 等价于：BinaryOp(AND, Comparison(age, >, 18), Comparison(status, =, active))
```

### 模式四：闭包注册（Closure Registration）

**特征**：用装饰器或函数调用注册DSL元素，形成声明式配置。

```python
# Flask路由就是这种模式
router = Router()

@router.get("/users")
def list_users():
    return users_service.get_all()

@router.post("/users")
def create_user():
    return users_service.create(request.json)
```

---

## 9.4 内部DSL的噪声最小化技巧

### 技巧一：消除显式类型

```python
# 差：需要显式指定类型
age_field = IntegerField("age")
name_field = StringField("name")

# 好：通过赋值推断
class UserSchema(Schema):
    age: int       # 自动推断为IntegerField
    name: str      # 自动推断为StringField
```

### 技巧二：利用`**kwargs`减少括号

```python
# 差
config(host="localhost", port=8080, debug=True)

# 好（用对象字面量风格）
config ** {
    "host": "localhost",
    "port": 8080,
    "debug": True
}

# 更好（如果可以控制语法）：
# 利用Python的解包语法
server_config = Config(host="localhost") | port(8080) | debug
```

### 技巧三：使用`__class_getitem__`支持泛型风格

```python
class TypedField:
    def __class_getitem__(cls, item):
        return cls(item)

# 使用
field = TypedField[str]    # 等价于 TypedField(str)
field = TypedField[int]
```

### 技巧四：字符串作为标识符（谨慎使用）

```python
# SQLAlchemy风格：字符串字段名
User.query.filter_by(status="active").order_by("name")

# vs 列对象风格（更安全，支持IDE补全）
User.query.filter(User.status == "active").order_by(User.name)
```

---

## 9.5 Python vs Ruby vs Kotlin：内部DSL能力对比

```ruby
# Ruby：DSL天堂（几乎所有东西都是方法调用，可省略括号和分号）
describe "User" do
  it "validates age" do
    user = User.new(age: 17)
    expect(user).not_to be_valid
  end
end

# 等价的Python版本（必须有括号、冒号）
describe("User", lambda: [
    it("validates age", lambda: [
        user := User(age=17),
        expect(user).not_to.be_valid()
    ])
])
```

```kotlin
// Kotlin：通过lambda最后参数和receiver函数，DSL效果接近Ruby
buildString {
    append("Hello")
    append(", ")
    append("World")
}

html {
    head { title { +"My Page" } }
    body { p { +"Content" } }
}
```

**结论**：
- Ruby最适合内部DSL（语法最灵活）
- Kotlin通过`receiver lambda`接近Ruby能力
- Python较好（装饰器、操作符重载、上下文管理器）
- Java/Go内部DSL能力较弱，通常选择外部DSL或构建者模式

---

## 9.6 实战：设计一个测试断言DSL

```python
# 目标：让断言读起来像自然语言
# assert user.age == 18 → expect(user.age).to.equal(18)
# assert len(users) > 0 → expect(users).to.have.length_greater_than(0)

class Expectation:
    def __init__(self, actual, negated=False):
        self._actual = actual
        self._negated = negated
        self.to = self  # 链式占位符
    
    @property
    def not_(self) -> 'Expectation':
        return Expectation(self._actual, not self._negated)
    
    def _assert(self, condition: bool, message: str):
        if self._negated:
            condition = not condition
            message = f"not_{message}"
        if not condition:
            raise AssertionError(
                f"Expected {self._actual!r} to {message}"
            )
    
    def equal(self, expected) -> 'Expectation':
        self._assert(self._actual == expected, f"equal {expected!r}")
        return self
    
    def be_greater_than(self, value) -> 'Expectation':
        self._assert(self._actual > value, f"be greater than {value}")
        return self
    
    def contain(self, item) -> 'Expectation':
        self._assert(item in self._actual, f"contain {item!r}")
        return self
    
    def be_empty(self) -> 'Expectation':
        self._assert(len(self._actual) == 0, "be empty")
        return self
    
    def have_length(self, n: int) -> 'Expectation':
        self._assert(len(self._actual) == n, f"have length {n}")
        return self

def expect(actual) -> Expectation:
    return Expectation(actual)

# 使用：
expect(2 + 2).to.equal(4)
expect("hello").to.contain("ell")
expect([]).to.be_empty()
expect([1, 2, 3]).to.have_length(3)
expect(42).not_.equal(0)
```

---

## 9.7 内部DSL的局限性

| 限制 | 原因 | 应对策略 |
|------|------|---------|
| 语法受宿主语言约束 | 不能自定义语法 | 接受噪声，或改用外部DSL |
| 括号/缩进不可省略 | Python语法要求 | 使用运算符重载减少 |
| 字符串类型不安全 | 字段名用字符串易拼错 | 用列对象（Column）替代字符串 |
| 错误消息来自宿主语言 | Python报AttributeError | 自定义`__getattr__`提供友好错误 |
| 调试体验与普通代码相同 | 既是优点也是局限 | — |

---

## 小结

| 模式 | 宿主语言特性 | 典型用例 |
|------|-------------|---------|
| 流畅接口 | 方法链（返回self） | 查询构建器、配置 |
| 构建者 | 方法链 + build() | 不可变对象构造 |
| 表达式树 | 操作符重载 | 过滤条件、规则 |
| 闭包注册 | 装饰器 | 路由、事件处理、测试 |

---

**上一章**：[解析器组合子](../part2-parsing/08-parser-combinators.md)
**下一章**：[构建者模式DSL](./10-builder-pattern.md)
