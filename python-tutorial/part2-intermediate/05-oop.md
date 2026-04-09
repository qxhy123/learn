# 第5章：面向对象编程

> 本章将系统介绍 Python 面向对象编程（OOP）的核心概念，并展示这些概念如何在深度学习框架（PyTorch）中得到应用。

---

## 学习目标

完成本章学习后，你将能够：

1. 理解类与对象的概念，掌握使用 `class` 关键字定义类并创建实例的方法
2. 区分并正确使用实例属性、类属性、实例方法、类方法和静态方法
3. 运用继承和多态机制构建层次化的类结构，理解 `super()` 的作用
4. 使用常见魔术方法（`__init__`、`__str__`、`__len__` 等）自定义类的行为
5. 通过 `@property` 装饰器实现属性封装，并理解 PyTorch `nn.Module` 的 OOP 设计理念

---

## 5.1 类与对象

### 5.1.1 为什么需要面向对象？

在处理复杂问题时，我们希望把**数据**和**操作数据的方法**打包在一起，形成一个独立的单元。这就是面向对象编程的核心思想。

以神经网络为例：一个"全连接层"既有权重矩阵（数据），又有前向传播计算（方法）。把它们封装成一个类，比分散的函数和变量更易于管理。

### 5.1.2 定义类

使用 `class` 关键字定义类，类名按照 Python 惯例使用大驼峰命名（PascalCase）：

```python
class Dog:
    """一个简单的狗类示例。"""

    # 类体：包含属性和方法的定义
    species = "Canis familiaris"  # 类属性，所有实例共享

    def __init__(self, name, age):
        """初始化方法，创建实例时自动调用。"""
        self.name = name    # 实例属性
        self.age = age      # 实例属性

    def bark(self):
        """实例方法。"""
        return f"{self.name} 说：汪汪！"
```

### 5.1.3 实例化

调用类名（像调用函数一样）来创建实例：

```python
# 创建两个 Dog 实例
dog1 = Dog("旺财", 3)
dog2 = Dog("小黑", 5)

print(dog1.name)        # 旺财
print(dog2.age)         # 5
print(dog1.bark())      # 旺财 说：汪汪！

# 访问类属性
print(dog1.species)     # Canis familiaris
print(Dog.species)      # Canis familiaris（通过类名访问）
```

### 5.1.4 self 参数

`self` 代表当前实例本身，是 Python 实例方法的第一个参数（名称约定为 `self`，但可以是任何合法变量名）：

```python
class Point:
    def __init__(self, x, y):
        self.x = x  # self.x 是实例属性
        self.y = y

    def distance_to_origin(self):
        # self.x 和 self.y 引用当前实例的属性
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def translate(self, dx, dy):
        # 方法可以修改实例的属性
        self.x += dx
        self.y += dy
        return self  # 返回 self 支持链式调用

p = Point(3, 4)
print(p.distance_to_origin())   # 5.0

p.translate(1, 1)
print(p.x, p.y)                 # 4 5
```

> **关键点**：调用 `p.distance_to_origin()` 时，Python 自动将 `p` 作为第一个参数传入，即等价于 `Point.distance_to_origin(p)`。

---

## 5.2 属性与方法

### 5.2.1 实例属性 vs 类属性

```python
class Counter:
    # 类属性：所有实例共享同一份数据
    count = 0

    def __init__(self, name):
        # 实例属性：每个实例独立拥有
        self.name = name
        Counter.count += 1  # 修改类属性需要通过类名

    def reset_class_count(self):
        Counter.count = 0

c1 = Counter("第一个")
c2 = Counter("第二个")
c3 = Counter("第三个")

print(Counter.count)    # 3
print(c1.name)          # 第一个
print(c2.name)          # 第二个
```

**注意陷阱**：在实例上"修改"类属性实际上是创建了同名的实例属性：

```python
class Config:
    debug = False

cfg1 = Config()
cfg2 = Config()

cfg1.debug = True       # 这会在 cfg1 上创建实例属性，不影响类属性

print(cfg1.debug)       # True  （实例属性）
print(cfg2.debug)       # False （仍是类属性）
print(Config.debug)     # False （类属性未变）
```

### 5.2.2 实例方法、类方法与静态方法

```python
import math

class Circle:
    pi = math.pi

    def __init__(self, radius):
        self.radius = radius

    # 实例方法：第一个参数是 self（实例本身）
    def area(self):
        return self.pi * self.radius ** 2

    def circumference(self):
        return 2 * self.pi * self.radius

    # 类方法：第一个参数是 cls（类本身），用 @classmethod 装饰
    @classmethod
    def from_diameter(cls, diameter):
        """通过直径创建圆（替代构造函数）。"""
        return cls(diameter / 2)

    # 静态方法：不接收 self 或 cls，用 @staticmethod 装饰
    @staticmethod
    def is_valid_radius(radius):
        """检查半径是否合法。"""
        return radius > 0


# 使用实例方法
c1 = Circle(5)
print(f"面积: {c1.area():.2f}")            # 面积: 78.54
print(f"周长: {c1.circumference():.2f}")   # 周长: 31.42

# 使用类方法（通过类名或实例均可调用）
c2 = Circle.from_diameter(10)
print(f"半径: {c2.radius}")                # 半径: 5.0

# 使用静态方法
print(Circle.is_valid_radius(5))           # True
print(Circle.is_valid_radius(-1))          # False
```

**三种方法的对比：**

| 方法类型 | 装饰器 | 第一个参数 | 访问实例属性 | 访问类属性 | 典型用途 |
|---------|--------|-----------|------------|----------|---------|
| 实例方法 | 无 | `self` | 是 | 是 | 操作实例数据 |
| 类方法 | `@classmethod` | `cls` | 否 | 是 | 替代构造函数、工厂方法 |
| 静态方法 | `@staticmethod` | 无 | 否 | 否 | 与类相关的工具函数 |

### 5.2.3 综合示例：神经网络层的参数统计

```python
class LinearLayer:
    """模拟一个全连接层（仅示意，非真实实现）。"""

    layer_count = 0  # 类属性：记录创建的层数

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        LinearLayer.layer_count += 1

    def param_count(self):
        """实例方法：计算本层参数量。"""
        w_params = self.in_features * self.out_features
        b_params = self.out_features if self.use_bias else 0
        return w_params + b_params

    @classmethod
    def reset_count(cls):
        """类方法：重置层计数器。"""
        cls.layer_count = 0

    @staticmethod
    def flops(in_features, out_features):
        """静态方法：估算浮点运算次数。"""
        return 2 * in_features * out_features  # 乘加运算

layer1 = LinearLayer(784, 256)
layer2 = LinearLayer(256, 128)
layer3 = LinearLayer(128, 10, bias=False)

print(f"层1参数量: {layer1.param_count()}")   # 200960
print(f"层2参数量: {layer2.param_count()}")   # 32896
print(f"层3参数量: {layer3.param_count()}")   # 1280
print(f"共创建层数: {LinearLayer.layer_count}")  # 3
print(f"层1 FLOPs: {LinearLayer.flops(784, 256)}")  # 401408
```

---

## 5.3 继承与多态

### 5.3.1 单继承

继承允许子类复用父类的代码，并在此基础上进行扩展：

```python
class Animal:
    """所有动物的基类。"""

    def __init__(self, name, sound):
        self.name = name
        self.sound = sound

    def speak(self):
        return f"{self.name} 发出声音：{self.sound}"

    def describe(self):
        return f"我是一只动物，名叫 {self.name}"


class Dog(Animal):
    """继承自 Animal 的狗类。"""

    def __init__(self, name, breed):
        # 调用父类的 __init__
        super().__init__(name, "汪汪")
        self.breed = breed  # 子类特有的属性

    def fetch(self):
        """子类特有的方法。"""
        return f"{self.name} 去捡球了！"


class Cat(Animal):
    """继承自 Animal 的猫类。"""

    def __init__(self, name, indoor=True):
        super().__init__(name, "喵喵")
        self.indoor = indoor

    def speak(self):
        """方法重写（Override）：修改父类方法的行为。"""
        prefix = "慵懒地" if self.indoor else "野性地"
        return f"{self.name} {prefix}叫道：{self.sound}"


dog = Dog("旺财", "金毛")
cat = Cat("咪咪")

print(dog.speak())      # 旺财 发出声音：汪汪
print(cat.speak())      # 咪咪 慵懒地叫道：喵喵
print(dog.fetch())      # 旺财 去捡球了！
print(dog.describe())   # 我是一只动物，名叫 旺财（继承自父类）

# isinstance 检查继承关系
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True（dog 也是 Animal 的实例）
print(isinstance(cat, Dog))     # False
```

### 5.3.2 多态

多态（Polymorphism）意味着不同类的对象可以通过相同的接口调用，产生不同的行为：

```python
class Shape:
    """形状基类。"""

    def area(self):
        raise NotImplementedError("子类必须实现 area() 方法")

    def perimeter(self):
        raise NotImplementedError("子类必须实现 perimeter() 方法")

    def describe(self):
        return (f"{self.__class__.__name__}: "
                f"面积={self.area():.2f}, 周长={self.perimeter():.2f}")


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)


class Circle(Shape):
    import math
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        import math
        return math.pi * self.radius ** 2

    def perimeter(self):
        import math
        return 2 * math.pi * self.radius


class Triangle(Shape):
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def area(self):
        # 海伦公式
        s = (self.a + self.b + self.c) / 2
        return (s * (s - self.a) * (s - self.b) * (s - self.c)) ** 0.5

    def perimeter(self):
        return self.a + self.b + self.c


# 多态：用统一的接口处理不同类型的对象
shapes = [
    Rectangle(4, 6),
    Circle(5),
    Triangle(3, 4, 5),
]

for shape in shapes:
    print(shape.describe())

# 输出：
# Rectangle: 面积=24.00, 周长=20.00
# Circle: 面积=78.54, 周长=31.42
# Triangle: 面积=6.00, 周长=12.00
```

### 5.3.3 多继承

Python 支持一个类同时继承多个父类：

```python
class Flyable:
    """可飞行的混入类（Mixin）。"""

    def fly(self):
        return f"{self.name} 正在飞翔！"

    def max_altitude(self):
        return 1000  # 默认最大高度（米）


class Swimmable:
    """可游泳的混入类。"""

    def swim(self):
        return f"{self.name} 正在游泳！"

    def max_depth(self):
        return 10   # 默认最大深度（米）


class Duck(Animal, Flyable, Swimmable):
    """鸭子：继承多个类。"""

    def __init__(self, name):
        super().__init__(name, "嘎嘎")

    def speak(self):
        return f"{self.name} 叫道：{self.sound}"


duck = Duck("唐老鸭")
print(duck.speak())         # 唐老鸭 叫道：嘎嘎
print(duck.fly())           # 唐老鸭 正在飞翔！
print(duck.swim())          # 唐老鸭 正在游泳！
print(duck.max_altitude())  # 1000

# 查看方法解析顺序（MRO）
print(Duck.__mro__)
# (<class 'Duck'>, <class 'Animal'>, <class 'Flyable'>,
#  <class 'Swimmable'>, <class 'object'>)
```

> **方法解析顺序（MRO）**：Python 使用 C3 线性化算法确定多继承中方法的查找顺序，可通过 `类名.__mro__` 查看。

### 5.3.4 super() 的用法

```python
class Base:
    def greet(self):
        return "Base"


class A(Base):
    def greet(self):
        return "A -> " + super().greet()


class B(Base):
    def greet(self):
        return "B -> " + super().greet()


class C(A, B):
    def greet(self):
        return "C -> " + super().greet()


c = C()
print(c.greet())        # C -> A -> B -> Base
print(C.__mro__)
# (<class 'C'>, <class 'A'>, <class 'B'>, <class 'Base'>, <class 'object'>)
```

`super()` 按照 MRO 顺序查找下一个类，而不是简单地指向"父类"，这使得多继承下的协作方法调用能够正确工作。

---

## 5.4 魔术方法

魔术方法（Magic Methods），也叫特殊方法或 Dunder 方法（Double Underscore），是 Python 用于自定义类行为的钩子函数。

### 5.4.1 对象生命周期：__init__ 与 __del__

```python
class Resource:
    def __init__(self, name):
        self.name = name
        print(f"[创建] 资源 '{name}' 已分配")

    def __del__(self):
        print(f"[销毁] 资源 '{self.name}' 已释放")


r = Resource("GPU内存")    # [创建] 资源 'GPU内存' 已分配
del r                       # [销毁] 资源 'GPU内存' 已释放
```

### 5.4.2 字符串表示：__str__ 与 __repr__

```python
class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        """面向开发者的表示，要求准确、可用于重建对象。"""
        return f"Vector({self.x}, {self.y}, {self.z})"

    def __str__(self):
        """面向用户的表示，要求可读性好。"""
        return f"向量({self.x}, {self.y}, {self.z})"


v = Vector(1, 2, 3)
print(v)            # 向量(1, 2, 3)  （调用 __str__）
print(repr(v))      # Vector(1, 2, 3) （调用 __repr__）

# 在列表中展示时使用 __repr__
vectors = [Vector(1, 0, 0), Vector(0, 1, 0)]
print(vectors)
# [Vector(1, 0, 0), Vector(0, 1, 0)]
```

### 5.4.3 运算符重载：算术与比较

```python
class Vector:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vector({self.x}, {self.y}, {self.z})"

    # 算术运算符
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        """向量与标量相乘。"""
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar):
        """支持 scalar * vector 的写法。"""
        return self.__mul__(scalar)

    def __neg__(self):
        """取反运算符 -v。"""
        return Vector(-self.x, -self.y, -self.z)

    # 比较运算符
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __abs__(self):
        """abs(v) 返回向量模长。"""
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5


v1 = Vector(1, 2, 3)
v2 = Vector(4, 5, 6)

print(v1 + v2)      # Vector(5, 7, 9)
print(v2 - v1)      # Vector(3, 3, 3)
print(v1 * 3)       # Vector(3, 6, 9)
print(2 * v1)       # Vector(2, 4, 6)
print(-v1)          # Vector(-1, -2, -3)
print(abs(v1))      # 3.7416573867739413
print(v1 == Vector(1, 2, 3))  # True
```

### 5.4.4 容器行为：__len__、__getitem__、__setitem__、__contains__

```python
class Dataset:
    """模拟 PyTorch Dataset 类。"""

    def __init__(self, data):
        self._data = list(data)

    def __len__(self):
        """支持 len(dataset)。"""
        return len(self._data)

    def __getitem__(self, index):
        """支持 dataset[i] 和 dataset[1:3]。"""
        return self._data[index]

    def __setitem__(self, index, value):
        """支持 dataset[i] = value。"""
        self._data[index] = value

    def __contains__(self, item):
        """支持 item in dataset。"""
        return item in self._data

    def __iter__(self):
        """支持 for item in dataset。"""
        return iter(self._data)

    def __repr__(self):
        return f"Dataset({self._data})"


ds = Dataset([10, 20, 30, 40, 50])

print(len(ds))          # 5
print(ds[0])            # 10
print(ds[1:3])          # [20, 30]
print(30 in ds)         # True
print(99 in ds)         # False

ds[0] = 100
print(ds)               # Dataset([100, 20, 30, 40, 50])

for item in ds:
    print(item, end=" ")  # 100 20 30 40 50
```

### 5.4.5 上下文管理器：__enter__ 与 __exit__

```python
class Timer:
    """计时上下文管理器。"""
    import time

    def __init__(self, name=""):
        self.name = name
        self.elapsed = 0

    def __enter__(self):
        import time
        self._start = time.time()
        return self  # 赋值给 as 后面的变量

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.elapsed = time.time() - self._start
        label = f"[{self.name}] " if self.name else ""
        print(f"{label}耗时: {self.elapsed:.4f} 秒")
        return False  # 不抑制异常


with Timer("矩阵乘法") as t:
    result = sum(i * i for i in range(1_000_000))

print(f"计算完成，用时 {t.elapsed:.4f} 秒")
```

### 5.4.6 可调用对象：__call__

```python
class Multiplier:
    """可调用对象，像函数一样使用。"""

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor


double = Multiplier(2)
triple = Multiplier(3)

print(double(5))     # 10
print(triple(5))     # 15
print(callable(double))  # True

# 在深度学习中，nn.Module 的 forward 通过 __call__ 触发
# model(input) 等价于调用 model.__call__(input)
```

---

## 5.5 封装与属性装饰器

### 5.5.1 私有属性与名称改写

Python 通过命名约定实现访问控制：

- 单下划线 `_name`：约定为内部使用，外部可访问但不鼓励
- 双下划线 `__name`：触发名称改写（Name Mangling），防止子类意外覆盖

```python
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner          # 公有属性
        self._account_id = "ACC001" # 约定内部使用
        self.__balance = balance    # 私有属性（名称改写）

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return True
        return False

    def get_balance(self):
        return self.__balance


acc = BankAccount("张三", 1000)
print(acc.owner)            # 张三
print(acc._account_id)      # ACC001（可访问，但不推荐）

# print(acc.__balance)       # AttributeError！
print(acc._BankAccount__balance)  # 1000（名称改写后的实际名称）
print(acc.get_balance())    # 1000
```

### 5.5.2 @property 装饰器

`@property` 将方法转换为属性访问，同时支持数据验证：

```python
class Temperature:
    """温度类，支持摄氏度与华氏度互转。"""

    def __init__(self, celsius=0):
        self._celsius = celsius  # 存储实际数据

    @property
    def celsius(self):
        """getter：读取摄氏度。"""
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        """setter：设置摄氏度，带验证。"""
        if value < -273.15:
            raise ValueError(f"温度不能低于绝对零度！({value} < -273.15)")
        self._celsius = value

    @celsius.deleter
    def celsius(self):
        """deleter：删除属性时的处理。"""
        print("删除温度属性")
        del self._celsius

    @property
    def fahrenheit(self):
        """计算属性：根据摄氏度动态计算华氏度。"""
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        """通过华氏度设置摄氏度。"""
        self.celsius = (value - 32) * 5/9  # 复用 celsius setter 的验证


t = Temperature(100)
print(t.celsius)        # 100
print(t.fahrenheit)     # 212.0

t.celsius = 0
print(t.fahrenheit)     # 32.0

t.fahrenheit = 98.6
print(f"{t.celsius:.1f}°C")  # 37.0°C

try:
    t.celsius = -300    # ValueError: 温度不能低于绝对零度！
except ValueError as e:
    print(e)
```

### 5.5.3 综合示例：神经网络参数管理

```python
class Parameter:
    """模拟 PyTorch Parameter 类——带梯度追踪的张量。"""

    def __init__(self, data, requires_grad=True):
        self._data = data
        self._requires_grad = requires_grad
        self._grad = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._grad = None  # 修改数据时清空梯度

    @property
    def grad(self):
        return self._grad

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        if not isinstance(value, bool):
            raise TypeError("requires_grad 必须是布尔值")
        self._requires_grad = value

    def zero_grad(self):
        """清空梯度。"""
        self._grad = None

    def __repr__(self):
        return (f"Parameter(data={self._data}, "
                f"requires_grad={self._requires_grad})")


# 模拟参数的使用
import random
w = Parameter([random.gauss(0, 0.1) for _ in range(3)])
print(w)
print(w.requires_grad)   # True

w.requires_grad = False  # 冻结参数（fine-tuning 时常用）
print(w)
```

---

## 本章小结

| 概念 | 关键语法 | 核心要点 |
|------|---------|---------|
| 类定义 | `class Name:` | 封装数据与方法 |
| 实例化 | `obj = Class(args)` | 调用 `__init__` |
| 实例属性 | `self.attr = val` | 每个实例独立 |
| 类属性 | `Class.attr = val` | 所有实例共享 |
| 实例方法 | `def method(self):` | 操作实例数据 |
| 类方法 | `@classmethod` + `cls` | 工厂方法，操作类数据 |
| 静态方法 | `@staticmethod` | 工具函数，无需访问实例或类 |
| 继承 | `class Child(Parent):` | 代码复用与扩展 |
| 多态 | 方法重写 | 统一接口，不同行为 |
| super() | `super().method()` | 调用父类方法，遵循 MRO |
| 魔术方法 | `__init__`、`__str__` 等 | 自定义内置行为 |
| 属性装饰器 | `@property` | 访问控制与数据验证 |
| 私有属性 | `__name` | 名称改写，防止意外覆盖 |

---

## 深度学习应用：nn.Module 的设计理念

PyTorch 的 `nn.Module` 是深度学习中 OOP 的经典案例，其设计充分运用了本章所学的所有概念。

### nn.Module 的 OOP 核心机制

```python
# 以下是 nn.Module 核心设计的简化演示（非 PyTorch 源码）

class Module:
    """
    PyTorch nn.Module 核心设计的极简模拟。
    真实的 nn.Module 包含更多特性（自动微分、GPU支持等）。
    """

    def __init__(self):
        # 使用有序字典存储子模块和参数
        self._parameters = {}   # 参数（权重、偏置）
        self._modules = {}      # 子模块（嵌套的 Module）
        self.training = True    # 训练/推理模式开关

    def __setattr__(self, name, value):
        """重写属性设置，自动注册 Parameter 和 Module。"""
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        """先查参数字典，再查子模块字典。"""
        if name in self.__dict__.get('_parameters', {}):
            return self._parameters[name]
        if name in self.__dict__.get('_modules', {}):
            return self._modules[name]
        raise AttributeError(f"'{type(self).__name__}' 没有属性 '{name}'")

    def __call__(self, *args, **kwargs):
        """使模块可调用，触发 forward 方法。"""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """子类必须实现此方法。"""
        raise NotImplementedError("子类必须实现 forward() 方法")

    def parameters(self):
        """递归收集所有参数（生成器）。"""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def train(self, mode=True):
        """切换到训练模式。"""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        """切换到评估（推理）模式。"""
        return self.train(False)

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for name, module in self._modules.items():
            lines.append(f"  ({name}): {repr(module)}")
        lines.append(")")
        return "\n".join(lines)
```

### 自定义神经网络层

```python
import math
import random

class SimpleParameter:
    """简化版参数类（替代 torch.nn.Parameter）。"""
    def __init__(self, data):
        self.data = data
        self.grad = None

    def __repr__(self):
        return f"Parameter(shape={len(self.data)}x{len(self.data[0])})"


class LinearLayerModule(Module):
    """
    自定义全连接层，模拟 torch.nn.Linear。

    继承 Module，实现了：
    - __init__：参数初始化（Kaiming 均匀初始化）
    - forward：矩阵乘法 + 偏置
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming 均匀初始化
        bound = 1 / math.sqrt(in_features)
        weight_data = [
            [random.uniform(-bound, bound) for _ in range(in_features)]
            for _ in range(out_features)
        ]
        self.weight = SimpleParameter(weight_data)

        if bias:
            bias_data = [[random.uniform(-bound, bound)] for _ in range(out_features)]
            self.bias = SimpleParameter(bias_data)
        else:
            self.bias = None

    def forward(self, x):
        """
        前向传播：output = x @ weight.T + bias

        参数:
            x: 输入，形状 [batch_size, in_features]
        返回:
            output: 形状 [batch_size, out_features]
        """
        batch_size = len(x)
        # 矩阵乘法（仅作示意）
        out = []
        for sample in x:
            row = []
            for j in range(self.out_features):
                val = sum(sample[k] * self.weight.data[j][k]
                          for k in range(self.in_features))
                if self.bias is not None:
                    val += self.bias.data[j][0]
                row.append(val)
            out.append(row)
        return out

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


class ReLUModule(Module):
    """ReLU 激活函数层（无可学习参数）。"""

    def forward(self, x):
        return [[max(0, val) for val in row] for row in x]

    def extra_repr(self):
        return ""


class SimpleMLPModule(Module):
    """
    两层 MLP，展示如何嵌套 Module（子模块自动注册）。

    结构：Linear -> ReLU -> Linear
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 子模块通过 __setattr__ 自动注册到 _modules
        self.fc1 = LinearLayerModule(input_dim, hidden_dim)
        self.relu = ReLUModule()
        self.fc2 = LinearLayerModule(hidden_dim, output_dim)

    def forward(self, x):
        """前向传播：按顺序调用各层。"""
        x = self.fc1(x)     # 调用 fc1.forward(x)
        x = self.relu(x)    # 调用 relu.forward(x)
        x = self.fc2(x)     # 调用 fc2.forward(x)
        return x


# 使用示例
model = SimpleMLPModule(
    input_dim=4,
    hidden_dim=8,
    output_dim=2
)

# 构造一个 batch（3个样本，每个4维）
batch = [[random.gauss(0, 1) for _ in range(4)] for _ in range(3)]
output = model(batch)

print(f"输入形状: {len(batch)} x {len(batch[0])}")
print(f"输出形状: {len(output)} x {len(output[0])}")
print(f"输出值: {[[round(v, 4) for v in row] for row in output]}")
```

### OOP 概念在 nn.Module 中的映射

| OOP 概念 | 在 nn.Module 中的体现 |
|---------|---------------------|
| 类与对象 | 每个网络层是一个 `Module` 子类的实例 |
| 继承 | 自定义层继承 `nn.Module`，复用参数管理、状态保存等功能 |
| 多态 | 不同层实现各自的 `forward()` 方法，通过统一的 `model(x)` 调用 |
| 魔术方法 `__call__` | `model(x)` 触发钩子函数后调用 `forward(x)` |
| 魔术方法 `__setattr__` | 自动将 `Parameter` 和子 `Module` 注册到相应字典 |
| `@property` | `model.parameters()` 递归获取所有可训练参数 |
| 封装 | 参数存储在 `_parameters` 字典中，通过方法访问 |

---

## 练习题

### 基础题

**题目 1**：定义一个 `Rectangle`（矩形）类，要求：
- 接受 `width` 和 `height` 作为初始化参数
- 使用 `@property` 实现 `area`（面积）和 `perimeter`（周长）只读属性
- 使用 `@width.setter` 和 `@height.setter` 对负值进行验证（抛出 `ValueError`）
- 实现 `__repr__` 方法

示例输出：
```python
r = Rectangle(4, 6)
print(r.area)        # 24
print(r.perimeter)   # 20
r.width = -1         # ValueError: 宽度不能为负数
```

---

**题目 2**：定义一个 `Stack`（栈）类，要求：
- 使用列表作为内部存储（私有属性 `__items`）
- 实现 `push(item)`、`pop()`、`peek()` 方法
- 实现 `__len__`、`__bool__`（空栈为 False）、`__repr__` 魔术方法
- `pop()` 和 `peek()` 在栈为空时抛出 `IndexError`

---

### 进阶题

**题目 3**：设计一个动物园管理系统，要求：
- 基类 `Animal`：包含 `name`、`age` 属性，以及抽象方法 `sound()`
- 子类 `Lion`、`Elephant`、`Parrot`：各自实现 `sound()` 和特有方法
- 容器类 `Zoo`：实现 `__len__`、`__iter__`、`__contains__`、`add_animal()`、`remove_animal()` 方法
- 展示多态：遍历动物园，让所有动物"发声"

---

**题目 4**：实现一个 `LRUCache`（最近最少使用缓存）类，要求：
- 构造函数接受 `capacity`（最大容量）
- 实现 `__getitem__`（读取并更新使用顺序）、`__setitem__`（写入，满时淘汰最旧项）、`__len__`、`__repr__`
- 使用 `collections.OrderedDict` 作为内部存储

示例：
```python
cache = LRUCache(3)
cache["a"] = 1
cache["b"] = 2
cache["c"] = 3
_ = cache["a"]   # 访问 a，a 变为最近使用
cache["d"] = 4   # 容量满，淘汰最旧的 b
print(cache)     # LRUCache({'a': 1, 'c': 3, 'd': 4})
```

---

### 挑战题

**题目 5**：实现一个简化的计算图（Computation Graph），要求：
- 类 `Tensor` 包含 `data`（数值）和 `grad`（梯度，初始为 None）
- 支持运算符重载：`+`、`-`、`*`、`**`（幂运算）
- 每次运算创建新 `Tensor`，并记录运算的两个操作数（`_prev`）和运算类型（`_op`）
- 实现 `backward()` 方法：对标量结果执行反向传播，计算链式法则梯度
- 验证：`z = x * y + x`，`z.backward()`，结果应为 `x.grad = y.data + 1`，`y.grad = x.data`

---

## 练习答案

### 答案 1：Rectangle 类

```python
class Rectangle:
    def __init__(self, width, height):
        # 使用 setter 进行验证
        self.width = width
        self.height = height

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value < 0:
            raise ValueError(f"宽度不能为负数，得到 {value}")
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        if value < 0:
            raise ValueError(f"高度不能为负数，得到 {value}")
        self._height = value

    @property
    def area(self):
        return self._width * self._height

    @property
    def perimeter(self):
        return 2 * (self._width + self._height)

    def __repr__(self):
        return f"Rectangle(width={self._width}, height={self._height})"


# 测试
r = Rectangle(4, 6)
print(r)            # Rectangle(width=4, height=6)
print(r.area)       # 24
print(r.perimeter)  # 20

r.width = 10
print(r.area)       # 60

try:
    r.width = -1
except ValueError as e:
    print(e)        # 宽度不能为负数，得到 -1
```

### 答案 2：Stack 类

```python
class Stack:
    def __init__(self):
        self.__items = []

    def push(self, item):
        self.__items.append(item)

    def pop(self):
        if not self.__items:
            raise IndexError("pop from empty stack")
        return self.__items.pop()

    def peek(self):
        if not self.__items:
            raise IndexError("peek from empty stack")
        return self.__items[-1]

    def __len__(self):
        return len(self.__items)

    def __bool__(self):
        return len(self.__items) > 0

    def __repr__(self):
        return f"Stack({self.__items})"


# 测试
s = Stack()
print(bool(s))      # False（空栈）

s.push(1)
s.push(2)
s.push(3)
print(s)            # Stack([1, 2, 3])
print(len(s))       # 3
print(s.peek())     # 3
print(s.pop())      # 3
print(s)            # Stack([1, 2])
print(bool(s))      # True

try:
    empty = Stack()
    empty.pop()
except IndexError as e:
    print(e)        # pop from empty stack
```

### 答案 3：动物园管理系统

```python
class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def sound(self):
        raise NotImplementedError(f"{self.__class__.__name__} 必须实现 sound()")

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', age={self.age})"


class Lion(Animal):
    def sound(self):
        return f"{self.name} 吼道：ROAAR！"

    def hunt(self):
        return f"{self.name} 正在狩猎"


class Elephant(Animal):
    def sound(self):
        return f"{self.name} 嚎叫：TRUMPET！"

    def spray_water(self):
        return f"{self.name} 用鼻子喷水"


class Parrot(Animal):
    def __init__(self, name, age, phrase="你好"):
        super().__init__(name, age)
        self.phrase = phrase

    def sound(self):
        return f"{self.name} 学舌：{self.phrase}"


class Zoo:
    def __init__(self, name):
        self.name = name
        self._animals = []

    def add_animal(self, animal):
        if not isinstance(animal, Animal):
            raise TypeError("只能添加 Animal 类型")
        self._animals.append(animal)
        print(f"{animal.name} 已加入 {self.name}")

    def remove_animal(self, name):
        for i, a in enumerate(self._animals):
            if a.name == name:
                removed = self._animals.pop(i)
                print(f"{name} 已离开 {self.name}")
                return removed
        raise ValueError(f"未找到名为 '{name}' 的动物")

    def __len__(self):
        return len(self._animals)

    def __iter__(self):
        return iter(self._animals)

    def __contains__(self, name):
        return any(a.name == name for a in self._animals)

    def __repr__(self):
        return f"Zoo('{self.name}', {len(self)} 种动物)"


# 测试
zoo = Zoo("北京动物园")
zoo.add_animal(Lion("辛巴", 5))
zoo.add_animal(Elephant("大象宝宝", 10))
zoo.add_animal(Parrot("鹦鹉波利", 3, "吃了吗"))

print(f"\n{zoo}")
print(f"'辛巴' 在园内: {'辛巴' in zoo}")
print(f"'老虎' 在园内: {'老虎' in zoo}")
print()

# 多态展示
print("=== 动物园演出 ===")
for animal in zoo:
    print(animal.sound())
```

### 答案 4：LRUCache 类

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("容量必须为正整数")
        self.capacity = capacity
        self._cache = OrderedDict()

    def __getitem__(self, key):
        if key not in self._cache:
            raise KeyError(key)
        # 移到末尾（最近使用）
        self._cache.move_to_end(key)
        return self._cache[key]

    def __setitem__(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.capacity:
                # 淘汰最旧的（头部）
                evicted_key, _ = self._cache.popitem(last=False)
                # print(f"淘汰: {evicted_key}")
        self._cache[key] = value

    def __len__(self):
        return len(self._cache)

    def __contains__(self, key):
        return key in self._cache

    def __repr__(self):
        return f"LRUCache({dict(self._cache)})"

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# 测试
cache = LRUCache(3)
cache["a"] = 1
cache["b"] = 2
cache["c"] = 3
print(cache)     # LRUCache({'a': 1, 'b': 2, 'c': 3})

_ = cache["a"]   # 访问 a
cache["d"] = 4   # 淘汰最旧的 b
print(cache)     # LRUCache({'c': 3, 'a': 1, 'd': 4})
print(len(cache))   # 3
print("b" in cache) # False
print("a" in cache) # True
```

### 答案 5：简化计算图与自动微分

```python
class Tensor:
    """支持自动微分的简化张量类。"""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None  # 反向传播函数
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            # d(out)/d(self) = other.data, d(out)/d(other) = self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Tensor(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-1 * other)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def backward(self):
        """拓扑排序后，从输出到输入反向传播梯度。"""
        topo = []
        visited = set()

        def build_topo(node):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data:.4f}, grad={self.grad:.4f})"


# 验证：z = x * y + x
# dz/dx = y + 1, dz/dy = x
x = Tensor(3.0)
y = Tensor(4.0)

z = x * y + x
print(f"z = {z.data}")   # z = 15.0

z.backward()
print(f"x.grad = {x.grad}")  # x.grad = 5.0 (= y + 1 = 4 + 1)
print(f"y.grad = {y.grad}")  # y.grad = 3.0 (= x = 3)

# 更复杂的例子：L = (x + y)^2 * z
a = Tensor(2.0)
b = Tensor(3.0)
c = Tensor(4.0)

L = (a + b) ** 2 * c
print(f"\nL = {L.data}")     # L = 100.0

L.backward()
# dL/da = 2*(a+b)*c = 2*5*4 = 40
# dL/db = 2*(a+b)*c = 40
# dL/dc = (a+b)^2 = 25
print(f"a.grad = {a.grad}")  # 40.0
print(f"b.grad = {b.grad}")  # 40.0
print(f"c.grad = {c.grad}")  # 25.0
```

---

*下一章：第6章 — 迭代器、生成器与函数式编程*
