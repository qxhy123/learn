# 13 — Stack（融合版）

> **难度**：★★★☆☆
> **题数**：5
> **核心套路**：括号匹配、逆波兰表达式求值、单调栈、辅助栈、路径/表达式栈
> **本文件**：覆盖 stack 5 题的算法套路总结 + 典型题精讲 + 自测

---

## 一例速记

> **括号匹配（20）**：遇左括号压栈，遇右括号弹栈验证配对，最终栈空则合法；O(n) 时间，O(n) 空间
> **路径简化（71）**：按 `/` 分割，用栈处理各部件：`..` 弹栈，`.` 忽略，普通名压栈，最后 `'/' + '/'.join(stack)` 拼接
> **逆波兰表达式 RPN（150）**：操作数压栈，遇运算符弹两个操作数计算后压回，最终栈顶即结果；O(n) 时间
> **最小栈（155）**：辅助栈同步记录"当前栈的最小值历史"，push/pop 时双栈联动，getMin() O(1)
> **基础计算器（224）**：处理 `+/−/()` 的带括号表达式求值；栈保存"遇到左括号时的 (result, sign)"，右括号时恢复并合并；单次扫描 O(n) 时间
> **AI 关联**：计算图的算子优先级调度 ≈ 逆波兰表达式；编译器/解释器的函数调用栈帧 ≈ 括号嵌套匹配；Transformer 层的残差连接可类比"栈式"状态保存

---

## 思维路径还原

> "看到 **20 Valid Parentheses**（括号合法性检查）→
> 典型栈匹配：遇 `(`, `[`, `{` 压栈；遇 `)`, `]`, `}` 弹栈验证是否配对。
> 边界：弹栈时栈为空（右括号多）→ False；遍历完栈非空（左括号多）→ False；否则 True。
> Python 用字典 `{')':'(', ']':'[', '}':'{'}` 简化配对查找。
>
> 看到 **71 Simplify Path**（UNIX 路径简化）→
> 按 `/` 分割得到各部件，栈处理规则：
> `'..'` → 若栈非空则 `pop()`（返回上一级）；
> `'.'` 或空串 → 忽略（当前目录或多余的斜杠）；
> 其他 → `push`（正常目录名）。
> 最终 `'/' + '/'.join(stack)` 即为简化路径。
>
> 看到 **150 Evaluate Reverse Polish Notation**（逆波兰表达式）→
> 顺序遍历 tokens：数字压栈；`+/-/*//` 弹出两个操作数 `b, a = pop(), pop()`（注意顺序，b 是后压的，是右操作数），计算 `a op b` 后压回。
> Python 整除向零取整用 `int(a / b)`（不能用 `a // b`，负数时方向不同）。
>
> 看到 **155 Min Stack**（支持 O(1) getMin 的栈）→
> 维护辅助栈 `min_stack`：push 时同时压入 `min(x, min_stack[-1])`（当前全局最小）；
> pop 时双栈同步弹；getMin 返回 `min_stack[-1]`。
> 初始化时 `min_stack` 预置一个 `inf`（防止首次 push 时无法比较）。
>
> 看到 **224 Basic Calculator**（带括号的 `+/−` 表达式求值）→
> 核心：括号里的表达式递归求值。用栈保存进入括号前的状态 `(result, sign)`：
> 遇 `(` → 把当前 `(result, sign)` 压栈，`result = 0, sign = 1`（重新开始子表达式）；
> 遇 `)` → 计算子表达式结果，弹出 `(prev_result, prev_sign)`，`result = prev_result + prev_sign * result`（将子结果以正确符号加回父结果）；
> 遇数字 → 读完整数，`result += sign * num`；
> 遇 `+` → `sign = 1`；遇 `-` → `sign = -1`。"

---

## 学习目标

- 掌握括号匹配的通用栈模板，能扩展到多种括号类型及嵌套
- 熟练用栈模拟路径操作（71），理解 `..`、`.`、空串三类分支
- 理解 RPN（逆波兰表达式）的计算过程，注意操作数弹出顺序（后弹的是左操作数）
- 掌握"辅助栈"设计模式（155），能推广到"支持 O(1) 最大值"的变体
- 掌握带括号表达式的栈求值（224），理解进入 `(` 时保存状态、遇 `)` 时合并的机制
- 理解单调栈（虽本 category 5 题未直接包含，但作为 stack 核心知识点补充）

---

## 抽象成方法（标准模板代码）

### 套路 1：括号匹配通用模板

适用题：20

```python
def isValid(s: str) -> bool:
    """20: 括号合法性检查。时间 O(n)，空间 O(n)。
    左括号压栈；右括号与栈顶匹配，不匹配或栈空则非法；遍历完栈必须为空。
    """
    matching = {')': '(', ']': '[', '}': '{'}
    stack: list[str] = []
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        else:
            # 遇到右括号：栈空 or 栈顶不配对 → 非法
            if not stack or stack[-1] != matching[ch]:
                return False
            stack.pop()
    return len(stack) == 0
```

> 泛化：若括号类型增多，只需扩展 `matching` 字典；若需找第一个非法位置，在 `return False` 前记录 `i`。

---

### 套路 2：路径栈（文件系统路径简化）

适用题：71

```python
def simplifyPath(path: str) -> str:
    """71: UNIX 路径简化。时间 O(n)，空间 O(n)。
    按 '/' 分割，栈处理各部件：'..' 弹栈，'.' 或空串忽略，其他压栈。
    """
    stack: list[str] = []
    for part in path.split('/'):
        if part == '..':
            if stack:
                stack.pop()
        elif part and part != '.':
            stack.append(part)
    return '/' + '/'.join(stack)
```

> 关键：`path.split('/')` 会在多个连续 `/` 处产生空串，`elif part and part != '.'` 自然过滤。

---

### 套路 3：逆波兰表达式求值（RPN）

适用题：150

```python
def evalRPN(tokens: list[str]) -> int:
    """150: 逆波兰表达式。时间 O(n)，空间 O(n)。
    操作数压栈；运算符弹两个操作数（后弹的 b 是左操作数，先弹的 a 是右操作数）
    ——注意：tokens 中 b 先于 a 出现，压栈顺序 b 先 a 后，所以 pop 得到 a（右），再 pop 得到 b（左）。
    整除向零取整用 int(b / a) 而非 b // a（Python 对负数 // 向下取整，与题意不符）。
    """
    stack: list[int] = []
    ops = {'+', '-', '*', '/'}
    for tok in tokens:
        if tok in ops:
            a = stack.pop()   # 右操作数（后压入的）
            b = stack.pop()   # 左操作数（先压入的）
            if tok == '+':
                stack.append(b + a)
            elif tok == '-':
                stack.append(b - a)
            elif tok == '*':
                stack.append(b * a)
            else:             # '/'，向零取整
                stack.append(int(b / a))
        else:
            stack.append(int(tok))
    return stack[0]
```

---

### 套路 4：辅助栈（Min Stack）

适用题：155

```python
class MinStack:
    """155: 支持 O(1) getMin 的栈。
    两个栈同步维护：main_stack 存元素，min_stack 存对应前缀最小值。
    push(x)：main_stack.append(x)；min_stack.append(min(x, min_stack[-1]))。
    pop()：两栈同步 pop。
    getMin()：min_stack[-1]。
    """
    def __init__(self):
        self.stack: list[int] = []
        self.min_stack: list[int] = [float('inf')]  # 哨兵，防止首次 push 无法比较

    def push(self, val: int) -> None:
        self.stack.append(val)
        self.min_stack.append(min(val, self.min_stack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]   # O(1)
```

> 关键：`min_stack[-1]` 存的是"从栈底到当前位置"的前缀最小值，而非全局历史最小（前缀最小随 pop 正确回退）。

---

### 套路 5：带括号表达式求值（Basic Calculator）

适用题：224

```python
def calculate(s: str) -> int:
    """224: 带括号的 +/- 表达式求值。时间 O(n)，空间 O(n)。
    栈保存进入括号前的 (result, sign)，遇 ')' 时弹出并合并。
    """
    stack: list[tuple[int, int]] = []  # (result, sign) 进入括号前的状态
    result = 0
    sign = 1   # 当前符号：+1 或 -1
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch.isdigit():
            num = 0
            while i < n and s[i].isdigit():
                num = num * 10 + int(s[i])
                i += 1
            result += sign * num
            continue          # i 已经由内层循环推进，不要再 i+=1
        elif ch == '+':
            sign = 1
        elif ch == '-':
            sign = -1
        elif ch == '(':
            # 保存当前状态，重置子表达式
            stack.append((result, sign))
            result = 0
            sign = 1
        elif ch == ')':
            # 子表达式结束，合并到父表达式
            prev_result, prev_sign = stack.pop()
            result = prev_result + prev_sign * result
        i += 1
    return result
```

> `prev_sign * result`：`prev_sign` 是**进入括号之前**外层的符号，例如 `3 - (2 + 1)` 中 `-` 就是 `prev_sign = -1`，子表达式结果 3 应以 `-3` 加回。

---

### 套路 6：单调栈通用模板（补充知识点）

虽然本 category 5 题未直接考查单调栈，但它是栈的核心扩展套路，在 739（Daily Temperatures）、84（Largest Rectangle）、85（Maximal Rectangle）等题大量使用。

```python
# 单调递增栈：找每个元素"右侧第一个更小的元素"
def next_smaller(nums: list[int]) -> list[int]:
    """结果 res[i] = 右侧第一个 < nums[i] 的下标，不存在则为 -1。时间 O(n)。"""
    n = len(nums)
    res = [-1] * n
    stack: list[int] = []   # 存下标，栈内对应值单调递增
    for i in range(n):
        # 当前元素比栈顶小 → 栈顶元素找到了"右侧第一个更小"
        while stack and nums[i] < nums[stack[-1]]:
            idx = stack.pop()
            res[idx] = i
        stack.append(i)
    return res


# 单调递减栈：找每个元素"右侧第一个更大的元素"（739 Daily Temperatures 变体）
def next_greater(nums: list[int]) -> list[int]:
    """结果 res[i] = 右侧第一个 > nums[i] 的下标，不存在则为 -1。时间 O(n)。"""
    n = len(nums)
    res = [-1] * n
    stack: list[int] = []   # 存下标，栈内对应值单调递减
    for i in range(n):
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            res[idx] = i
        stack.append(i)
    return res
```

> 记忆：**单调栈 = 维护"还没找到答案"的元素**，新元素到来时，凡是满足条件的都出栈（找到答案），其余压栈继续等待。每个元素最多入栈/出栈各一次，总 O(n)。

---

### 速查表

| 题目 | 核心结构 | 关键操作 | 时间 | 空间 |
|---|---|---|---|---|
| 20 Valid Parentheses | 单栈，存左括号 | 遇右括号弹栈验证 | O(n) | O(n) |
| 71 Simplify Path | 单栈，存路径部件 | `..` 弹栈，`.` 忽略，其他压栈 | O(n) | O(n) |
| 150 Evaluate RPN | 单栈，存操作数 | 运算符弹两个数计算后压回 | O(n) | O(n) |
| 155 Min Stack | 双栈（主栈 + 辅助最小栈） | push/pop 双栈联动 | O(1) 均摊 | O(n) |
| 224 Basic Calculator | 单栈，存 (result, sign) | `(` 压状态，`)` 弹状态合并 | O(n) | O(n) |

---

## 方法变形（3 类）

### 变形 1：括号类扩展

- **20**（合法性判断）→ **22 Generate Parentheses**（回溯生成合法括号，非本 category）：DFS 维护"未配对左括号数"，控制左括号数量 ≤ n，右括号数量 ≤ 已用左括号数。
- **20** 扩展：若括号有权重（如 `{}` 得 3 分），遍历时在匹配处累加分值；若需找最外层括号数量，维护深度计数而非用栈。
- **32 Longest Valid Parentheses**（非本 category）：栈存下标；入栈标记规则：左括号压下标，右括号若能匹配则弹栈，否则压自身下标作为"分隔符"；最长有效子串 = 相邻分隔符之间的区间长度。

### 变形 2：表达式类扩展

- **224**（`+/-/()`）→ **227 Basic Calculator II**（`+/-/*/÷`，无括号，非本 category）：遇数字时根据"当前运算符"决定压栈时的值；`*` 和 `/` 直接修改栈顶（高优先级立刻算），`+/-` 则直接压带符号的值，最终求和。
- **150 RPN**（后缀表达式）→ **中缀转后缀**（Shunting-Yard 算法）：用两个栈（操作符栈 + 输出栈），按优先级决定何时弹操作符到输出。
- **辅助栈的"最大值版"**：将 `min_stack` 中的 `min` 改为 `max`，即可实现 O(1) getMax 的栈。

### 变形 3：单调栈扩展（重点）

| 问题类型 | 单调栈方向 | 典型题 |
|---|---|---|
| 右侧第一个更大元素 | 单调递减栈 | 739 Daily Temperatures |
| 右侧第一个更小元素 | 单调递增栈 | 901 Online Stock Span（反向） |
| 柱状图最大矩形 | 单调递增栈（存柱高） | 84 Largest Rectangle in Histogram |
| 接雨水（栈法） | 单调递减栈（存高度） | 42 Trapping Rain Water（同 01-array-string 双指针法） |

> 单调栈的"方向"规则：栈内维持单调递增 → 用于寻找"更小值"（新更小元素入栈时能弹出未满足的大值）；单调递减 → 寻找"更大值"。

---

## 思考路标（条件反射）

1. 看到 **"括号合法性 / 配对"** → 左压栈，右弹栈验证；结尾栈必须为空
2. 看到 **"UNIX 路径 / 文件系统路径简化"** → 按 `/` 分割 + 栈：`..` 弹，`.` 忽略，普通名压栈
3. 看到 **"逆波兰 / 后缀表达式"** → 操作数压栈，运算符弹两个计算后压回，注意弹出顺序（后弹=左操作数）
4. 看到 **"O(1) 最小值 / 最大值 + 支持 push/pop"** → 辅助栈同步记录前缀 min/max
5. 看到 **"带括号的 +/- 表达式"** → 栈保存 (result, sign)：`(` 压，`)` 弹并合并；遇数字直接 `result += sign * num`
6. 看到 **"右侧第一个更大/更小"** → 单调栈：新元素触发弹栈（找到答案），未满足的继续等待
7. 看到 **"柱状图 / 接雨水"** → 单调栈（栈法）或双指针（数组法），时间均 O(n)
8. 看到 **"操作符优先级 / 表达式解析"** → Shunting-Yard 算法（操作符栈 + 输出栈），AI 中等价于 tokenizer 的优先级调度
9. 看到 **"函数调用 / 递归展开"** → 显式栈模拟递归：把"递归参数 + 返回地址"压栈，用循环代替递归；节省系统栈空间

---

## 易错点

1. **RPN 操作数弹出顺序**：`a = stack.pop()` 是**右**操作数（后压入），`b = stack.pop()` 是**左**操作数（先压入），计算 `b op a`。写成 `a op b` 对减法 `/` 除法结果错误（顺序敏感）。
2. **RPN 整除向零取整**：Python `//` 对负数向下取整（`-7 // 2 = -4`），LeetCode 要求向零（`-7 / 2 = -3`），应用 `int(b / a)` 或 `math.trunc(b / a)`。
3. **224 进入括号时符号重置**：进入 `(` 后 `result = 0, sign = 1`；若忘记重置 `sign = 1`，子表达式首个数字可能继承外层符号。
4. **224 `prev_sign` 的含义**：`prev_sign` 是括号**之前外层**的运算符，不是括号内首个运算符；例如 `1 - (2 - 3)` 中 `prev_sign = -1`，子结果 `-1` 要乘以 `prev_sign` 再加回。
5. **224 数字读取后 i 的移动**：内层 while 循环读完整数后 `i` 已指向非数字字符，此时用 `continue` 跳过末尾的 `i += 1`，否则 `i` 会多走一步，跳过当前字符。
6. **155 辅助栈初始化**：`min_stack = [float('inf')]` 作为哨兵，否则第一次 `push` 时 `min_stack[-1]` 报 IndexError；`pop` 时两栈必须同步，不能只弹 `stack`。
7. **71 路径末尾不加斜杠**：`'/' + '/'.join(stack)`，注意只有开头有 `/`，`join` 本身不在结尾加 `/`；根路径（`stack` 为空）结果正确是 `'/'`。
8. **20 空栈弹操作**：弹栈前必须检查 `not stack`，否则遇到纯右括号字符串（如 `"]"`）会 IndexError；Python `list.pop()` 在空列表上抛异常。

---

## 典型应用例题

### 例 1：155. Min Stack

**题目**：实现栈，支持 push、pop、top、getMin，所有操作均 O(1)。

**思路**：O(1) push/pop → 普通 list。O(1) getMin → 不能每次遍历，需要辅助栈保存历史最小值。辅助栈 `min_stack[i]` = 主栈前 i+1 个元素的最小值；pop 时同步弹，自然回退到弹前的最小值。

**解**：见模板代码"套路 4 辅助栈 MinStack"。

**分析**：所有操作均 $O(1)$；空间 $O(n)$（两个栈各存 n 个元素）。辅助栈和主栈等长是关键——这保证了 pop 后最小值正确回退。

---

### 例 2：224. Basic Calculator

**题目**：计算含 `+`、`-`、`(`、`)` 和空格的合法表达式字符串的值。

**思路**：遇到括号时，表达式在括号内"重新开始"；出括号时，把子结果以正确符号合并回父结果。用栈保存 `(result, sign)` 状态对——进入 `(` 时压，遇到 `)` 时弹。空格直接跳过，数字可能多位，需循环读完。

**解**：见模板代码"套路 5 带括号表达式求值"。

**正确性验证**：`1 + (2 - (3 + 4))`

- 初始：result=0, sign=1，stack=[]
- 读 `1`：result=1
- 读 `+`：sign=1
- 读 `(`：stack=[(1,1)]，result=0, sign=1
- 读 `2`：result=2
- 读 `-`：sign=-1
- 读 `(`：stack=[(1,1),(2,-1)]，result=0, sign=1
- 读 `3`：result=3
- 读 `+`：sign=1
- 读 `4`：result=7
- 读 `)`：弹 (2,-1)，result = 2 + (-1)*7 = -5
- 读 `)`：弹 (1,1)，result = 1 + 1*(-5) = -4

输出 $-4$，正确。

---

### 例 3：150. Evaluate Reverse Polish Notation

**题目**：给出逆波兰表达式（后缀表达式）的 tokens 数组，求其值。整除向零取整。

**思路**：RPN 的规律：数字压栈，运算符弹两个数字计算后压回；每个 token 处理一次，时间 O(n)。弹出顺序：第一次 pop 得到**右**操作数（后进先出），第二次 pop 得到**左**操作数。

**解**：见模板代码"套路 3 逆波兰表达式求值"。

**验证**：`tokens = ["2","1","+","3","*"]`
- 压 2，压 1，遇 `+`：弹 1（右），弹 2（左），压 3
- 压 3，遇 `*`：弹 3（右），弹 3（左），压 9

输出 9，正确（`(2+1)*3 = 9`）。

---

## 自测题

**自测 1**（20 Valid Parentheses）—— `s="()[]{}"` 返回 True；`s="(]"` 返回 False；`s="([)]"` 返回 False；`s="{[]}"` 返回 True。提示：`matching = {')':'(', ']':'[', '}':'{'}`，左括号压栈，右括号比对栈顶，结尾 `len(stack)==0`。参考 `solutions/stack/p020_valid_parentheses.py`。

**自测 2**（71 Simplify Path）—— `path="/home/"` 返回 `"/home"`；`path="/../"` 返回 `"/"`；`path="/home//foo/"` 返回 `"/home/foo"`。提示：`path.split('/')` 后栈处理，`'..'` 弹栈，`'.'` 和空串忽略，最后 `'/' + '/'.join(stack)`。参考 `solutions/stack/p071_simplify_path.py`。

**自测 3**（150 Evaluate RPN）—— `tokens=["2","1","+","3","*"]` 返回 9；`tokens=["4","13","5","/","+"]` 返回 6（`4+(13/5)=4+2=6`）；`tokens=["10","6","9","3","+","-11","*","/","*","17","+","5","+"]` 返回 22。提示：操作数压栈，运算符弹两个 `a=pop()`（右）、`b=pop()`（左），计算 `b op a`，整除用 `int(b/a)`。参考 `solutions/stack/p150_evaluate_reverse_polish_notation.py`。

**自测 4**（155 Min Stack）—— push(-2)、push(0)、push(-3)、getMin()=-3、pop()、top()=0、getMin()=-2。提示：双栈，`min_stack` 初始含哨兵 inf；push 时 `min_stack.append(min(val, min_stack[-1]))`；pop 时两栈同步弹；getMin 返回 `min_stack[-1]`。参考 `solutions/stack/p155_min_stack.py`。

**自测 5**（224 Basic Calculator）—— `s="1 + 1"` 返回 2；`s=" 2-1 + 2 "` 返回 3；`s="(1+(4+5+2)-3)+(6+8)"` 返回 23。提示：遇 `(` 压 (result, sign)，`result=0, sign=1`；遇 `)` 弹并 `result = prev + prev_sign * result`；遇数字读完整数累加；注意 `continue` 跳过多余的 `i+=1`。参考 `solutions/stack/p224_basic_calculator.py`。

---

## 题目全览（5 题）

| # | 题目 | 套路分类 | 难度 |
|---|---|---|---|
| 20 | Valid Parentheses | 括号匹配，单栈 | Easy |
| 71 | Simplify Path | 路径栈，按 `/` 分割 | Medium |
| 150 | Evaluate Reverse Polish Notation | RPN 求值，操作数栈 | Medium |
| 155 | Min Stack | 辅助栈同步最小值 | Medium |
| 224 | Basic Calculator | 带括号 +/- 表达式，状态栈 | Hard |

---

## 融合版说明

| 段 | 来源 | 价值 |
|---|---|---|
| 一例速记 | 本文件 | 5 题 5 种栈套路一览 + AI 场景关联 |
| 思维路径还原 | 本文件 | 5 道题的解题内心独白，含栈状态管理决策 |
| 抽象成方法 | 本文件 | 6 个标准模板（括号/路径/RPN/辅助栈/表达式/单调栈）+ 速查表 |
| 方法变形 | 本文件 | 3 类变体（括号扩展/表达式扩展/单调栈扩展）含对照表 |
| 思考路标 | 本文件 | 9 条题型识别条件反射，含 AI/编译器场景 |
| 易错点 | 本文件 | 8 条高频踩坑（RPN 弹出顺序/整除/符号重置/空栈弹操作等） |
| 典型应用例题 | solutions/ | 3 道精讲（155、224、150），含手动验证过程 |
| 自测题 | leetcode | 5 题带提示，链接 solutions 文件 |
| 题目全览 | 本文件 | 5 题完整列表，套路分类一览 |

---

> **跨 category 导航**：
> - 单调栈（本文套路 6）在 array_string 的接雨水（42）中有双指针替代实现 → 见 `01-array-string.md`
> - 递归 DFS 的显式栈改写（用 list 模拟调用栈）与此处栈求值思路一致 → 见 `07-binary-tree-dfs.md`
> - 带括号表达式（224）的"括号递归"结构与回溯的递归调用栈同理 → 见 `08-backtracking.md`
> - 编译器的词法分析（tokenizer）和语法分析（parser）都依赖栈；AI 中 ONNX/TorchScript 的算子图构建用 Shunting-Yard 调度优先级
