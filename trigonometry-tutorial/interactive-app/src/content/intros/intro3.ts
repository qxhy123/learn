import type { IntroCard } from '../../types'

// 学新知：第三部分（单元 7–9）各课的概念讲解卡片，按 lesson id 索引。
export const INTRO3: Record<string, IntroCard[]> = {
  // ===== u7 三角函数图象 =====
  'u7-l1': [
    {
      title: '正弦、余弦的波形',
      body: '把单位圆上的纵坐标随角度画出来，就得到 $y=\\sin x$ 的**波浪线**。它在 $1$ 与 $-1$ 之间上下摆动，每隔 $2\\pi$ 重复一次。',
      formula: 'y=\\sin x,\\qquad x\\in\\mathbb{R}',
    },
    {
      title: '周期与值域',
      body: '$\\sin x$ 与 $\\cos x$ 的最小正周期都是 $2\\pi$，值域都是 $[-1,1]$。它们形状相同，只是**起点不同**：正弦过原点，余弦从最高点 $(0,1)$ 出发。',
      formula: 'T=2\\pi,\\qquad y\\in[-1,1]',
    },
    {
      title: '差半个相位',
      body: '余弦其实就是正弦左移 $\\dfrac{\\pi}{2}$。记住这条关系，正余弦的图象就能互相转换。',
      reveal: { q: '$\\cos x$ 等于哪个正弦？', a: '$\\cos x=\\sin\\left(x+\\dfrac{\\pi}{2}\\right)$' },
    },
  ],
  'u7-l2': [
    {
      title: '从图象读单调性',
      body: '波形**上坡**的地方就是单调递增，**下坡**就是递减。$\\sin x$ 在 $\\left[-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right]$ 从谷爬到峰，是递增区间。',
    },
    {
      title: '峰、谷与对称',
      body: '正弦是**奇函数**，图象关于原点对称；余弦是**偶函数**，图象关于 $y$ 轴对称。峰顶、谷底所在的竖直线都是对称轴。',
      formula: '\\sin(-x)=-\\sin x,\\qquad \\cos(-x)=\\cos x',
    },
    {
      title: '看个例子',
      body: '抓住「过原点 + 有界 + 周期 $2\\pi$」这几个特征，就能一眼认出基本图象。',
      reveal: { q: '过原点、值域 $[-1,1]$、周期 $2\\pi$ 的是谁？', a: '$y=\\sin x$' },
    },
  ],
  'u7-l3': [
    {
      title: '正切是“切断”的',
      body: '$\\tan x=\\dfrac{\\sin x}{\\cos x}$。当 $\\cos x=0$ 时分母为零、函数无定义，图象在那里被**竖直渐近线**切成一段段分支。',
      formula: 'x=\\frac{\\pi}{2}+k\\pi\\ \\Rightarrow\\ \\text{渐近线}',
    },
    {
      title: '周期 π、无界、递增',
      body: '正切的最小正周期是 $\\pi$（不是 $2\\pi$）。它**无界**，值域是全体实数；在每两条渐近线之间都从 $-\\infty$ 单调递增到 $+\\infty$。',
      formula: '\\tan(x+\\pi)=\\tan x,\\qquad T=\\pi',
    },
    {
      title: '想一想',
      body: '渐近线出现在余弦的零点处。',
      reveal: { q: '离原点最近的正渐近线在哪？', a: '$x=\\dfrac{\\pi}{2}$（此处 $\\cos x=0$）' },
    },
  ],

  // ===== u8 平移·伸缩·相位 =====
  'u8-l1': [
    {
      title: '四要素登场',
      body: '一般正弦型函数 $y=A\\sin(\\omega x+\\varphi)+k$ 由**四个参数**控制：振幅 $A$、周期（由 $\\omega$ 定）、相位 $\\varphi$、中线 $k$。这一节先看 $A$ 和 $\\omega$。',
      formula: 'y=A\\sin(\\omega x+\\varphi)+k',
    },
    {
      title: '振幅 A 与周期',
      body: '$|A|$ 是**竖直拉伸**，决定上下摆动的幅度；$|\\omega|$ 越大、波被**水平压**得越紧，周期越小。周期用这条公式算：',
      formula: 'T=\\dfrac{2\\pi}{|\\omega|}',
    },
    {
      title: '由最值求振幅',
      body: '若已知最大值 $M$、最小值 $m$，振幅就是峰到中线的距离。',
      reveal: { q: '$M=4,\\ m=-2$ 时振幅 $A=?$', a: '$A=\\dfrac{M-m}{2}=\\dfrac{4-(-2)}{2}=3$' },
    },
  ],
  'u8-l2': [
    {
      title: '相位 φ 与平移',
      body: '$\\varphi$ 叫**相位**，但它本身不是平移量。要看真实的水平平移，得先把 $\\omega$ 提出来：$\\omega x+\\varphi=\\omega\\left(x+\\dfrac{\\varphi}{\\omega}\\right)$。',
      formula: '\\text{水平平移量}=-\\dfrac{\\varphi}{\\omega}',
    },
    {
      title: '中线 k 决定上下',
      body: '$k$ 把整条波**整体上下平移**，中线就是 $y=k$。于是最大值 $=k+|A|$，最小值 $=k-|A|$。',
      formula: 'y_{\\max}=k+|A|,\\qquad y_{\\min}=k-|A|',
    },
    {
      title: '易错点',
      body: '只有 $\\omega=1$ 时，平移量才恰好等于 $\\varphi$，否则一定要先提 $\\omega$。',
      reveal: { q: '$y=\\sin(2x-\\pi)$ 由 $\\sin 2x$ 右移多少？', a: '提 $\\omega$：$2\\left(x-\\dfrac{\\pi}{2}\\right)$，右移 $\\dfrac{\\pi}{2}$（不是 $\\pi$）' },
    },
  ],
  'u8-l3': [
    {
      title: '由图求式三步走',
      body: '看到一条正弦型曲线，按顺序定参数：① 由**高差**定 $A$；② 由**周期**定 $\\omega$；③ 代一个**特殊点**（峰、谷或零点）定 $\\varphi$。最后别忘了中线 $k$。',
      formula: 'A=\\dfrac{M-m}{2},\\quad \\omega=\\dfrac{2\\pi}{T},\\quad k=\\dfrac{M+m}{2}',
    },
    {
      title: '半周期的妙用',
      body: '相邻**最高点与最低点**水平相差**半个周期**；相邻两个最高点才相差一个完整周期。用它最快读出 $T$。',
      formula: '\\dfrac{T}{2}=x_{\\text{谷}}-x_{\\text{峰}}',
    },
    {
      title: '用峰值定 φ',
      body: '峰值处括号内的整体角等于 $\\dfrac{\\pi}{2}$，代入即可解出 $\\varphi$。',
      reveal: { q: '$A=3,\\omega=2$，峰在 $x=\\dfrac{\\pi}{6}$，求 $\\varphi$', a: '$2\\cdot\\dfrac{\\pi}{6}+\\varphi=\\dfrac{\\pi}{2}\\Rightarrow\\varphi=\\dfrac{\\pi}{6}$' },
    },
  ],

  // ===== u9 周期建模 =====
  'u9-l1': [
    {
      title: '把现实写成波',
      body: '潮汐、昼长、气温、振动……这些**重复、平滑、有明显周期**的现象，都能用 $y=A\\sin(\\omega x+\\varphi)+k$ 来建模。',
      formula: 'y=A\\sin(\\omega t+\\varphi)+k',
    },
    {
      title: '先定中线与振幅',
      body: '建模第一步：用最大值 $M$、最小值 $m$ 算出**中线 $k$**（平均水平）和**振幅 $A$**。它们只看最值，与起始时刻无关。',
      formula: 'k=\\dfrac{M+m}{2},\\qquad A=\\dfrac{M-m}{2}',
    },
    {
      title: '试一试',
      body: '气温最高 $18^\\circ$、最低 $6^\\circ$。',
      reveal: { q: '中线与振幅各是多少？', a: '$k=\\dfrac{18+6}{2}=12^\\circ$，$A=\\dfrac{18-6}{2}=6^\\circ$' },
    },
  ],
  'u9-l2': [
    {
      title: '由周期定 ω',
      body: '知道现象重复一次需要的时间 $T$（潮汐 $12$ 时、昼长 $365$ 天……），就能反推角频率。这是建模第二步。',
      formula: '\\omega=\\dfrac{2\\pi}{T}',
    },
    {
      title: '选 sin 还是 cos',
      body: '看 $t=0$ 时的**起始状态**：在**最高点**用 $+\\cos$，在**最低点**用 $-\\cos$，从中线**向上**出发用 $\\sin$。对号入座最省事。',
      formula: '\\cos 0=1,\\qquad \\sin 0=0',
    },
    {
      title: '看个例子',
      body: '潮汐周期 $12$ 时。',
      reveal: { q: '角频率 $\\omega=?$', a: '$\\omega=\\dfrac{2\\pi}{12}=\\dfrac{\\pi}{6}$' },
    },
  ],
  'u9-l3': [
    {
      title: '求关键时刻',
      body: '模型建好后，常要回答“何时最高 / 何时达到某高度”。方法：令括号内**整体角**等于特殊角（如 $\\dfrac{\\pi}{2}$ 取峰值），再解出 $t$。',
      formula: '\\omega t+\\varphi=\\dfrac{\\pi}{2}\\ \\Rightarrow\\ t',
    },
    {
      title: '求时长：化成不等式',
      body: '“水深不低于某值才能冲浪”这类问题，先把条件化成 $\\sin(\\cdots)\\ge c$ 的**三角不等式**，再数一个周期内满足的时长，乘以周期个数。',
      formula: 'A\\sin(\\cdots)+k\\ge h\\ \\Rightarrow\\ \\sin(\\cdots)\\ge\\dfrac{h-k}{A}',
    },
    {
      title: '试一试',
      body: '$y=3\\sin(\\cdots)+10$，要求 $y\\ge 11.5$。',
      reveal: { q: '化简后的条件是？', a: '$\\sin(\\cdots)\\ge\\dfrac{11.5-10}{3}=\\dfrac{1}{2}$' },
    },
  ],
}
