import type { IntroCard } from '../../types'

// 学新知：第七部分（单元 19–21）各课的概念讲解卡片，按 lesson id 索引。
export const INTRO7: Record<string, IntroCard[]> = {
  // ============================== 单元 19 ==============================
  'u19-l1': [
    {
      title: '欧拉公式',
      body: '把指数 $e^{i\\theta}$ 同时拆出**实部余弦**与**虚部正弦**——一行公式把指数、旋转与三角统一起来。',
      formula: 'e^{i\\theta}=\\cos\\theta+i\\sin\\theta',
    },
    {
      title: '怎么来的',
      body: '把 $e^x$ 的幂级数代入 $x=i\\theta$，利用 $i^2=-1$ 让各项实虚交替，**分离实虚部**正好得到 $\\cos\\theta$ 与 $\\sin\\theta$ 的级数。',
      reveal: { q: '令 $\\theta=0$，$e^{i\\cdot 0}=?$', a: '$\\cos 0+i\\sin 0=1$' },
    },
    {
      title: '欧拉恒等式',
      body: '取 $\\theta=\\pi$：$e^{i\\pi}=\\cos\\pi+i\\sin\\pi=-1$，移项即得被誉为「最美公式」的恒等式，它把 $e,i,\\pi,1,0$ 五个常数连成一线。',
      formula: 'e^{i\\pi}+1=0',
    },
  ],
  'u19-l2': [
    {
      title: '模恒为 1',
      body: '$e^{i\\theta}$ 的模 $=\\sqrt{\\cos^2\\theta+\\sin^2\\theta}=1$，所以它**永远落在单位圆上**，$\\theta$ 只决定方向不改变长度。',
      formula: '|e^{i\\theta}|=1',
    },
    {
      title: '极形式',
      body: '一般复数写成 $z=r(\\cos\\theta+i\\sin\\theta)=re^{i\\theta}$：$r$ 是**模长**（离原点多远），$\\theta$ 是**辐角**（方向）。模 $r=\\sqrt{a^2+b^2}$。',
      reveal: { q: '$z=1+\\sqrt{3}\\,i$ 的模 $|z|=?$', a: '$\\sqrt{1+3}=2$' },
    },
    {
      title: '四个特殊点',
      body: '辐角取 $0,\\tfrac{\\pi}{2},\\pi,\\tfrac{3\\pi}{2}$ 时，$e^{i\\theta}$ 依次落在 $1,i,-1,-i$——单位圆与坐标轴的四个交点。',
    },
  ],
  'u19-l3': [
    {
      title: '乘法 = 旋转',
      body: '两个复数相乘，**模相乘、辐角相加**：$r_1e^{i\\theta_1}\\cdot r_2e^{i\\theta_2}=r_1r_2\\,e^{i(\\theta_1+\\theta_2)}$。乘 $e^{i\\theta}$ 相当于旋转 $\\theta$。',
      formula: 'e^{i\\alpha}\\cdot e^{i\\beta}=e^{i(\\alpha+\\beta)}',
    },
    {
      title: '除法相减',
      body: '除法反过来：**模相除、辐角相减**。$\\dfrac{r_1e^{i\\theta_1}}{r_2e^{i\\theta_2}}=\\dfrac{r_1}{r_2}e^{i(\\theta_1-\\theta_2)}$。',
      reveal: { q: '$2e^{i\\pi/4}\\div 3e^{i\\pi/6}$ 的辐角$=?$', a: '$\\tfrac{\\pi}{4}-\\tfrac{\\pi}{6}=\\dfrac{\\pi}{12}$' },
    },
    {
      title: '推出和角公式',
      body: '展开 $e^{i(\\alpha+\\beta)}=e^{i\\alpha}e^{i\\beta}$ 并比较实虚部，一步得到三角和角公式：实部给 $\\cos(\\alpha+\\beta)=\\cos\\alpha\\cos\\beta-\\sin\\alpha\\sin\\beta$。',
      formula: '\\cos\\theta=\\frac{e^{i\\theta}+e^{-i\\theta}}{2},\\quad \\sin\\theta=\\frac{e^{i\\theta}-e^{-i\\theta}}{2i}',
    },
  ],

  // ============================== 单元 20 ==============================
  'u20-l1': [
    {
      title: 'De Moivre 公式',
      body: '幂运算把**角度乘以 $n$**、模取 $n$ 次方。因为乘一次转一个 $\\theta$，连乘 $n$ 次总共转了 $n\\theta$。',
      formula: '(\\cos\\theta+i\\sin\\theta)^n=\\cos n\\theta+i\\sin n\\theta',
    },
    {
      title: '怎么用',
      body: '先把复数化成**极形式** $r(\\cos\\theta+i\\sin\\theta)$，再套公式：$[r(\\cos\\theta+i\\sin\\theta)]^n=r^n(\\cos n\\theta+i\\sin n\\theta)$，高次幂瞬间秒杀。',
      reveal: { q: '$\\left(\\cos\\tfrac{\\pi}{6}+i\\sin\\tfrac{\\pi}{6}\\right)^3=?$', a: '$\\cos\\tfrac{\\pi}{2}+i\\sin\\tfrac{\\pi}{2}=i$' },
    },
    {
      title: '看个例子',
      body: '算 $(1+i)^8$：先化 $1+i=\\sqrt2\\,e^{i\\pi/4}$，则模 $(\\sqrt2)^8=16$，角度 $8\\cdot\\tfrac{\\pi}{4}=2\\pi$ 转回正实轴，结果 $=16$。',
    },
  ],
  'u20-l2': [
    {
      title: '$n$ 次单位根',
      body: '方程 $z^n=1$ 在复数范围内**恰有 $n$ 个根**。由 $e^{in\\theta}=1$ 得辐角 $\\theta=\\dfrac{2\\pi k}{n}$，模全为 $1$。',
      formula: 'z_k=e^{2\\pi i k/n},\\quad k=0,1,\\dots,n-1',
    },
    {
      title: '正 $n$ 边形',
      body: '这 $n$ 个根都在单位圆上，相邻辐角间隔 $\\dfrac{2\\pi}{n}$，所以它们**等分圆周，构成正 $n$ 边形**的顶点。',
      reveal: { q: '$z^4=1$ 的四个根是？', a: '$1,\\ i,\\ -1,\\ -i$（正方形顶点）' },
    },
    {
      title: '根之和为零',
      body: '对称分布让向量相互抵消：所有 $n$ 次单位根之和为 $0$（$n\\ge2$）。如 $\\omega=e^{2\\pi i/3}$ 时 $1+\\omega+\\omega^2=0$。',
    },
  ],
  'u20-l3': [
    {
      title: '复数开方',
      body: '解 $w^n=z=re^{i\\theta}$：**模开 $n$ 次方，角度加整周期再除以 $n$**，共得 $n$ 个根，等角分布在同一个圆上。',
      formula: 'w_k=\\sqrt[n]{r}\\;e^{i(\\theta+2k\\pi)/n},\\quad k=0,\\dots,n-1',
    },
    {
      title: '为什么有 $n$ 个',
      body: '辐角加上 $2k\\pi$ 表示「同一方向转了 $k$ 整圈」，除以 $n$ 后这 $n$ 个角各不相同；它们**模长相同、间隔 $\\dfrac{2\\pi}{n}$**。',
      reveal: { q: '$w^3=-8$ 的实根 $=?$', a: '$-8=8e^{i\\pi}$，$w=2e^{i\\pi}=-2$' },
    },
    {
      title: '妙用',
      body: '单位根求和能算出三角值：由五次单位根实部之和 $1+2\\cos\\tfrac{2\\pi}{5}+2\\cos\\tfrac{4\\pi}{5}=0$，可得 $\\cos\\tfrac{2\\pi}{5}+\\cos\\tfrac{4\\pi}{5}=-\\dfrac12$。',
    },
  ],

  // ============================== 单元 21 ==============================
  'u21-l1': [
    {
      title: '频率分解',
      body: '傅里叶级数的核心：**任意规整的周期函数 = 不同频率正弦余弦的叠加**。三角函数在这里充当「频率基底」。',
      formula: 'f(x)=\\frac{a_0}{2}+\\sum_{n=1}^{\\infty}\\left(a_n\\cos nx+b_n\\sin nx\\right)',
    },
    {
      title: '正交性',
      body: '不同频率的 $\\sin nx,\\cos mx$ 在整周期上乘积积分为 $0$——**互不干扰**。正因如此才能逐个频率「投影」出系数。',
      reveal: { q: '$\\displaystyle\\int_{-\\pi}^{\\pi}\\cos x\\cos 2x\\,dx=?$', a: '积化和差后整周期积分 $=0$' },
    },
    {
      title: '系数的含义',
      body: '每个系数 $a_n,b_n$ 度量对应频率分量在原函数中所占的**权重**——哪些频率强、信号偏低频还是高频，一目了然。',
    },
  ],
  'u21-l2': [
    {
      title: '奇偶简化',
      body: '看函数的奇偶性能直接砍掉一半系数：**偶函数只含余弦项，奇函数只含正弦项**。',
      formula: 'f\\text{ 偶}\\Rightarrow b_n=0;\\qquad f\\text{ 奇}\\Rightarrow a_n=0',
    },
    {
      title: '为什么',
      body: '偶函数 $\\times\\sin$（奇）$=$ 奇，在对称区间积分为 $0$，故 $b_n=0$；奇函数 $\\times\\cos$（偶）$=$ 奇，积分为 $0$，故 $a_n=0$。',
      reveal: { q: '$f(x)=x^2$ 的级数含哪类项？', a: '偶函数 → 只含余弦项与常数项' },
    },
    {
      title: '快速判断',
      body: '$|x|$ 偶 → 只含余弦；$x$ 奇 → 只含正弦；$x\\cos x$ 奇×偶=奇 → 只含正弦。先判奇偶，能省一半计算。',
    },
  ],
  'u21-l3': [
    {
      title: '方波展开',
      body: '方波是奇函数，只含**奇次正弦谐波**（$\\sin x,\\sin 3x,\\sin 5x,\\dots$），偶次项系数全为零。',
      formula: 'f(x)=\\frac{4}{\\pi}\\left(\\sin x+\\frac{\\sin 3x}{3}+\\frac{\\sin 5x}{5}+\\cdots\\right)',
    },
    {
      title: '逐步逼近',
      body: '取的谐波项数越多，部分和越**接近方波**。但间断点处总有约 $9\\%$ 的固定过冲——这就是 **Gibbs 现象**，加再多项也不消失。',
      reveal: { q: '间断点处级数收敛到？', a: '左右极限的平均值 $\\dfrac{f(x^+)+f(x^-)}{2}$' },
    },
    {
      title: '意外的礼物',
      body: '把 $x=\\dfrac{\\pi}{2}$ 代入方波展开，得到著名的 **Leibniz 公式**，用无穷级数逼近 $\\pi$。',
      formula: '\\frac{\\pi}{4}=1-\\frac{1}{3}+\\frac{1}{5}-\\frac{1}{7}+\\cdots',
    },
  ],
}
