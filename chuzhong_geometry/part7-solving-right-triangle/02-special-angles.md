# 特殊角的三角函数值

## 一、为什么有"特殊角"

在直角三角形中，三个锐角 $30°$、$45°$、$60°$ 出现得最为频繁——它们来自两类"特殊直角三角形"：

- **$45°\text{-}45°\text{-}90°$ 三角形**（等腰直角三角形）：两直角边相等，三边之比为 $1 : 1 : \sqrt{2}$
- **$30°\text{-}60°\text{-}90°$ 三角形**（半个等边三角形）：三边之比为 $1 : \sqrt{3} : 2$

由 part7/01 的定义（正弦 = 对边/斜边，余弦 = 邻边/斜边，正切 = 对边/邻边），把上述边比直接代入，就得到这三个角度的精确三角函数值。这些值在中考中**必须脱口而出**，是解直角三角形的"乘法口诀表"。

![45-45-90 等腰直角三角形（1:1:√2）](../figures/svg/thm-special-right-45-45-90.svg)

![30-60-90 直角三角形（1:√3:2）](../figures/svg/thm-special-right-30-60-90.svg)

## 二、必记三角函数表

| 角度 | $\sin$ | $\cos$ | $\tan$ |
|---|---|---|---|
| $30°$ | $\dfrac{1}{2}$ | $\dfrac{\sqrt{3}}{2}$ | $\dfrac{\sqrt{3}}{3}$ |
| $45°$ | $\dfrac{\sqrt{2}}{2}$ | $\dfrac{\sqrt{2}}{2}$ | $1$ |
| $60°$ | $\dfrac{\sqrt{3}}{2}$ | $\dfrac{1}{2}$ | $\sqrt{3}$ |

**推导提示**：

- 取等腰直角三角形两直角边 $= 1$，斜边 $= \sqrt{2}$。则 $\sin 45° = \cos 45° = \dfrac{1}{\sqrt{2}} = \dfrac{\sqrt{2}}{2}$，$\tan 45° = \dfrac{1}{1} = 1$。
- 取边长为 $2$ 的等边三角形，作一条高，把它劈成两个 $30°\text{-}60°\text{-}90°$ 的直角三角形：斜边 $= 2$，短直角边（对 $30°$）$= 1$，长直角边（对 $60°$）$= \sqrt{3}$。立即读出 $\sin 30° = \dfrac{1}{2}$、$\cos 30° = \dfrac{\sqrt{3}}{2}$、$\tan 30° = \dfrac{1}{\sqrt{3}} = \dfrac{\sqrt{3}}{3}$，$\sin 60° = \dfrac{\sqrt{3}}{2}$ 等。

**忘记时怎么办**：画出对应的特殊直角三角形，标上 $1, \sqrt{3}, 2$ 或 $1, 1, \sqrt{2}$，再按定义直接读，比强记更稳。

## 三、记忆口诀

```
正弦 sin：1/2,  √2/2, √3/2     （分母都是 2，分子是 √1, √2, √3）
余弦 cos：√3/2, √2/2, 1/2      （正弦的倒序）
正切 tan：√3/3,  1,    √3       （首尾互为倒数，中间是 1）
              30°    45°    60°
```

理解要点：

- **$\sin$ 从 $30°$ 到 $60°$ 是递增的**（$\dfrac{1}{2} < \dfrac{\sqrt{2}}{2} < \dfrac{\sqrt{3}}{2}$）——符合 part7/01 中"$\sin$ 随锐角增大而增大"的规律。
- **$\cos$ 是 $\sin$ 的倒序**，因为余弦随锐角增大而减小。
- **$\tan 30°$ 与 $\tan 60°$ 互为倒数**：$\dfrac{\sqrt{3}}{3} \times \sqrt{3} = 1$，这正是 $\tan(90°-\alpha) = \dfrac{1}{\tan \alpha}$ 的体现。

## 四、互余关系再确认

由 part7/01 的互余恒等式 $\sin(90° - \alpha) = \cos \alpha$：

- $\sin 30° = \cos 60° = \dfrac{1}{2}$
- $\sin 60° = \cos 30° = \dfrac{\sqrt{3}}{2}$
- $\sin 45° = \cos 45° = \dfrac{\sqrt{2}}{2}$（$45°$ 自补，故 $\sin = \cos$）

并且 $\sin^2 \alpha + \cos^2 \alpha = 1$ 对每一行表都成立，可用来快速验算：
$$\left(\dfrac{1}{2}\right)^2 + \left(\dfrac{\sqrt{3}}{2}\right)^2 = \dfrac{1}{4} + \dfrac{3}{4} = 1.\ \checkmark$$

## 五、典型应用

**例 1（直接代值）** 计算 $\sin 30° + \cos 45° + \tan 60°$。

【思路】查表代入即可：
$$\sin 30° + \cos 45° + \tan 60° = \dfrac{1}{2} + \dfrac{\sqrt{2}}{2} + \sqrt{3} = \dfrac{1+\sqrt{2}}{2} + \sqrt{3}.$$

**例 2（化简比值）** 计算 $\dfrac{\sin 60°}{\cos 30°}$。

【思路】$\sin 60° = \dfrac{\sqrt{3}}{2}$，$\cos 30° = \dfrac{\sqrt{3}}{2}$，所以
$$\dfrac{\sin 60°}{\cos 30°} = \dfrac{\sqrt{3}/2}{\sqrt{3}/2} = 1.$$
本质：$\sin 60° = \cos 30°$（互余），所以比值必为 $1$。

**例 3（反求角度）** 已知 $\alpha$ 是锐角，$\sin \alpha = \dfrac{\sqrt{3}}{2}$，求 $\alpha$。

【思路】对照表格的 $\sin$ 一列：$\sin 60° = \dfrac{\sqrt{3}}{2}$，所以 $\alpha = 60°$。同理：

- $\sin \alpha = \dfrac{1}{2} \Rightarrow \alpha = 30°$
- $\cos \alpha = \dfrac{\sqrt{2}}{2} \Rightarrow \alpha = 45°$
- $\tan \alpha = \sqrt{3} \Rightarrow \alpha = 60°$
- $\tan \alpha = \dfrac{\sqrt{3}}{3} \Rightarrow \alpha = 30°$

## 六、易错点

1. **$\sin$ 与 $\cos$ 互调**：很多同学写出 $\sin 60° = \dfrac{1}{2}$ 是错的。记住"$\sin$ 在大角更大"——$60° > 30°$ 故 $\sin 60° > \sin 30°$，所以 $\sin 60° = \dfrac{\sqrt{3}}{2}$。
2. **$\tan 30°$ 与 $\tan 60°$ 互调**：$\tan 60° = \sqrt{3}$ 而不是 $\dfrac{\sqrt{3}}{3}$；后者是 $\tan 30°$。"角越大正切越大"，$\tan 60° > \tan 45° = 1 > \tan 30°$。
3. **分母有理化习惯**：$\tan 30°$ 写为 $\dfrac{1}{\sqrt{3}}$ 不算错但不规范，标准答案是 $\dfrac{\sqrt{3}}{3}$。
4. **$\sin 30° \ne \dfrac{\sqrt{1}}{2}$ 这种"形式记忆"陷阱**：虽然口诀里写"$\sqrt{1}, \sqrt{2}, \sqrt{3}$"，但 $\sqrt{1} = 1$，写成 $\dfrac{1}{2}$ 更简。
5. **混淆 $\tan$ 与 $\sin/\cos$ 的"分母为 $2$"**：$\tan$ 的特殊值分母不一定是 $2$（$\tan 45° = 1$、$\tan 60° = \sqrt{3}$），不要硬套。

## 七、自测题

1. 计算 $2\sin 30° - \cos 60° + \tan 45°$。
2. 计算 $\tan 30° \cdot \tan 60° + \sin^2 45°$。
3. 在 $\triangle ABC$ 中，$\angle C = 90°$，$\sin A = \dfrac{\sqrt{3}}{2}$，求 $\angle A$ 与 $\angle B$。
4. 已知锐角 $\alpha$ 满足 $\left(2\cos \alpha - \sqrt{3}\right)^2 + \left|\tan \alpha - \dfrac{\sqrt{3}}{3}\right| = 0$，求 $\alpha$。【思路】两非负量之和为 $0$ 各自为 $0$ → $\cos \alpha = \dfrac{\sqrt{3}}{2}$ 且 $\tan \alpha = \dfrac{\sqrt{3}}{3}$ → $\alpha = 30°$。
