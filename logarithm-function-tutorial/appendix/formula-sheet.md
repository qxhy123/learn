# 公式速查表

## 1. 基本定义

| 公式 | 含义 |
|------|------|
| $\log_a x=y \iff a^y=x$ | 对数与指数互为反函数 |
| $a>0,\ a\neq1,\ x>0$ | 对数的基本条件 |
| $\ln x=\log_e x$ | 自然对数 |
| $\lg x=\log_{10}x$ | 常用对数 |

---

## 2. 对数运算律

| 公式 | 含义 |
|------|------|
| $\log_a(MN)=\log_aM+\log_aN$ | 乘法转加法 |
| $\log_a(M/N)=\log_aM-\log_aN$ | 除法转减法 |
| $\log_a(M^r)=r\log_aM$ | 幂可提到前面 |
| $\log_a x=\dfrac{\ln x}{\ln a}$ | 换底公式 |
| $a^{\log_a x}=x$ | 指数与对数互相抵消 |

---

## 3. 图像与性质

| 性质 | 结论 |
|------|------|
| 定义域 | $(0,+\infty)$ |
| 值域 | $\mathbb R$ |
| 关键点 | $(1,0)$、$(a,1)$ |
| 渐近线 | $x=0$ |
| 单调性 | $a>1$ 递增；$0<a<1$ 递减 |

---

## 4. 微积分

| 公式 | 结果 |
|------|------|
| $\dfrac{d}{dx}\ln x$ | $\dfrac1x$ |
| $\dfrac{d}{dx}\log_a x$ | $\dfrac1{x\ln a}$ |
| $\dfrac{d}{dx}\ln u$ | $\dfrac{u'}{u}$ |
| $\int\dfrac1x\,dx$ | $\ln|x|+C$ |
| $\ln(1+x)$ | $x-\dfrac{x^2}{2}+\dfrac{x^3}{3}-\cdots$ |

---

## 5. 增长比较

| 结论 | 含义 |
|------|------|
| $\ln x\to-\infty\ (x\to0^+)$ | 接近 0 时急剧下降 |
| $\ln x\to+\infty\ (x\to+\infty)$ | 无界增长 |
| $\ln x\ll x^\alpha$ | 对数慢于任意正幂 |
| $x^\alpha\ll a^x$ | 幂慢于指数 |

---

## 6. 概率、信息与计算

| 公式 | 含义 |
|------|------|
| $I(A)=-\log P(A)$ | 事件信息量 |
| $H(X)=-\sum p_i\log p_i$ | 熵 |
| $H(p,q)=-\sum p_i\log q_i$ | 交叉熵 |
| $\log\sum_i e^{x_i}=m+\log\sum_i e^{x_i-m}$ | `log-sum-exp` 稳定形式 |

---

## 7. 复对数

| 公式 | 含义 |
|------|------|
| $z=re^{i\theta}$ | 极形式 |
| $\log z=\ln r+i(\theta+2k\pi)$ | 复对数的一般形式 |
| $\mathrm{Log}\,z=\ln|z|+i\,\mathrm{Arg}(z)$ | 主值对数 |
