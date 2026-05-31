import type { IntroCard } from '../../types'

// 学新知：第四部分（单元 10–12）各课的概念讲解卡片，按 lesson id 索引。
// 每课 4–5 张：动机 → 概念/方法(+公式/图) → 例题精讲(steps) → 易错点(tip) → 互动(reveal)。
export const INTRO4: Record<string, IntroCard[]> = {
  // ============================== 单元 10 三角方程 ==============================
  'u10-l1': [
    {
      title: '为什么三角方程最容易漏解',
      body: '普通代数方程通常只有**有限个**解，而三角函数是**周期**的：水平线 $y=a$ 会和曲线交无穷多次，于是三角方程往往有**无限多族解**。\n\n所以求解必须分两层：先在**一个周期**里找出主值解，再利用周期性写出**完整通解**。漏掉第二步，答案就只是“看起来对”却不完整。',
      image: 'trig-p4-10-1.svg',
      imageCaption: '左：单位圆上 $\\sin x=\\tfrac12$ 的两个终边 $\\tfrac{\\pi}{6},\\tfrac{5\\pi}{6}$；右：正弦曲线与 $y=\\tfrac12$ 的无穷多交点，对应通解。',
    },
    {
      title: '正弦方程：两族解',
      body: '$\\sin x=a$（$|a|\\le1$）在单位圆上对应**纵坐标**为 $a$ 的两个点，分别落在第一、二象限终边方向上，所以一个周期里有**两个**解。\n\n设参考角为 $\\alpha$，则一族绕 $\\alpha$、另一族绕 $\\pi-\\alpha$，各自周期复制 $2\\pi$。',
      formula: '\\sin x=\\tfrac12\\ \\Rightarrow\\ x=\\tfrac{\\pi}{6}+2k\\pi\\ \\text{或}\\ x=\\tfrac{5\\pi}{6}+2k\\pi,\\ k\\in\\mathbb{Z}',
    },
    {
      title: '余弦与正切的通解形',
      body: '**余弦** $\\cos x=a$ 在单位圆上取**横坐标**，上下对称两支取同值，写成 $\\pm$：$x=\\pm\\alpha+2k\\pi$。\n\n**正切** $\\tan x=a$ 的周期只有 $\\pi$，每条终边方向都给同一斜率，所以只有**单族**解：$x=\\alpha+k\\pi$。\n\n当 $|a|>1$ 时，水平线超出 $[-1,1]$，正弦/余弦方程**无解**。',
      formula: '\\cos x=a\\Rightarrow x=\\pm\\alpha+2k\\pi;\\quad \\tan x=a\\Rightarrow x=\\alpha+k\\pi',
    },
    {
      title: '例题精讲：解 $2\\sin x-1=0$',
      body: '从混合常数项的方程，一路写到完整通解。',
      steps: [
        '移项整理成**最简方程**：$2\\sin x-1=0\\Rightarrow\\sin x=\\dfrac12$。',
        '在一个周期 $[0,2\\pi)$ 内定主值解：参考角 $\\dfrac{\\pi}{6}$，正弦为正在一、二象限，得 $x=\\dfrac{\\pi}{6}$ 与 $x=\\dfrac{5\\pi}{6}$。',
        '周期复制 $2\\pi$ 写通解：$x=\\dfrac{\\pi}{6}+2k\\pi$ 或 $x=\\dfrac{5\\pi}{6}+2k\\pi,\\ k\\in\\mathbb{Z}$。',
      ],
      tip: '正弦只写一族 $x=\\dfrac{\\pi}{6}+2k\\pi$ 是最常见的失分点——第二象限那族 $\\dfrac{5\\pi}{6}+2k\\pi$ 绝不能丢。',
    },
    {
      title: '互动：通解 vs 区间解',
      body: '题目若**限定区间**（如 $[0,2\\pi)$）只写区间内的解、**不加** $+2k\\pi$；若求“全体实数上的通解”才补周期项。两者别混用。',
      reveal: { q: '$\\cos x=\\tfrac12$ 在全体实数上的通解？', a: '$x=\\pm\\dfrac{\\pi}{3}+2k\\pi,\\ k\\in\\mathbb{Z}$（两支对称，用 $\\pm$）' },
    },
  ],
  'u10-l2': [
    {
      title: '动机：把三角“当成代数”',
      body: '很多方程不是直接给 $\\sin x=a$，而是 $\\sin x$ 反复出现的**多项式**，例如 $2\\sin^2 x-3\\sin x+1=0$。\n\n硬解很乱，但只要把整块 $\\sin x$ 看成一个字母 $t$，它立刻变成你早就会的**一元二次方程**——这就是换元法的威力。',
      formula: '2\\sin^2 x-3\\sin x+1=0\\ \\xrightarrow{\\,t=\\sin x\\,}\\ 2t^2-3t+1=0',
    },
    {
      title: '先化同名，再换元',
      body: '若式子里**同时**有 $\\sin x$ 和 $\\cos x$，不能直接换元。先用恒等式化成只含一种：\n\n- 平方关系 $\\cos^2 x=1-\\sin^2 x$；\n- 倍角 $\\cos 2x=1-2\\sin^2 x$（把 $\\cos 2x$ 直接变成 $\\sin x$）。\n\n选对“方向”能一步化净，避免又冒出新的 $\\cos x$。',
      formula: '\\cos 2x=1-2\\sin^2 x',
    },
    {
      title: '例题精讲：$\\cos 2x+3\\sin x-2=0$',
      body: '混合 $\\cos 2x$ 与 $\\sin x$，目标是压成关于 $\\sin x$ 的二次。',
      steps: [
        '化同名：用 $\\cos 2x=1-2\\sin^2 x$ 代入，得 $1-2\\sin^2 x+3\\sin x-2=0$。',
        '整理标准化：$2\\sin^2 x-3\\sin x+1=0$。',
        '换元 $t=\\sin x$：$(2t-1)(t-1)=0\\Rightarrow t=\\dfrac12$ 或 $t=1$，二者**都在** $[-1,1]$ 内，保留。',
        '逐个回代（在 $[0,2\\pi)$ 内）：$\\sin x=\\dfrac12\\Rightarrow x=\\dfrac{\\pi}{6},\\dfrac{5\\pi}{6}$；$\\sin x=1\\Rightarrow x=\\dfrac{\\pi}{2}$。解集 $\\left\\{\\dfrac{\\pi}{6},\\dfrac{\\pi}{2},\\dfrac{5\\pi}{6}\\right\\}$。',
      ],
      tip: '若选成 $\\cos 2x=2\\cos^2 x-1$ 反而引入 $\\cos x$，方程化不净。要顺着题里出现的函数选倍角“方向”。',
    },
    {
      title: '易错点：回代与筛解',
      body: '换元后必须做两件事：① 检查每个 $t$ 是否落在 $[-1,1]$，超出范围（如 $t=2$）**舍去**；② 保留的 $t$ 要**完整**展开成三角解集，每个 $t$ 可能再生两族解。\n\n只解出 $t$ 就停笔，是换元法最大的坑。',
      reveal: { q: '$2t^2-3t+1=0$ 解得哪些 $t$？需要舍弃吗？', a: '$t=\\tfrac12$ 或 $t=1$，都在 $[-1,1]$ 内，均保留；再各自解回 $x$' },
    },
  ],
  'u10-l3': [
    {
      title: '动机：线性组合也能合二为一',
      body: '面对 $a\\sin x+b\\cos x=c$，既有正弦又有余弦，似乎无从下手。\n\n但 $a\\sin x+b\\cos x$ 其实可以压成**单一正弦** $R\\sin(x+\\varphi)$——一旦合并，方程就退化成你已经会的“最简方程 $\\sin(\\cdot)=$ 常数”。这就是**辅助角公式**。',
      formula: 'a\\sin x+b\\cos x=R\\sin(x+\\varphi),\\quad R=\\sqrt{a^2+b^2}',
    },
    {
      title: '$R$ 与相位 $\\varphi$ 怎么来',
      body: '把右边展开 $R\\sin(x+\\varphi)=R\\cos\\varphi\\sin x+R\\sin\\varphi\\cos x$，与左边比较系数：\n\n$$R\\cos\\varphi=a,\\quad R\\sin\\varphi=b.$$\n\n平方相加得 $R=\\sqrt{a^2+b^2}$（恒正）；相位由 $\\tan\\varphi=\\dfrac{b}{a}$ 并结合 $a,b$ 的符号确定象限。',
      reveal: { q: '$\\sin x+\\cos x$ 的 $R$ 和合并式？', a: '$R=\\sqrt{1+1}=\\sqrt2$，即 $\\sqrt2\\sin\\!\\big(x+\\tfrac{\\pi}{4}\\big)$' },
    },
    {
      title: '例题精讲：$\\sin x-\\sqrt3\\cos x=1$',
      body: '用辅助角把左边压成单一正弦，再当最简方程解。',
      steps: [
        '算振幅：$R=\\sqrt{1^2+(\\sqrt3)^2}=\\sqrt4=2$。',
        '提取并配相位：$2\\big(\\tfrac12\\sin x-\\tfrac{\\sqrt3}{2}\\cos x\\big)=2\\sin\\!\\big(x-\\tfrac{\\pi}{3}\\big)$（因 $\\cos\\tfrac{\\pi}{3}=\\tfrac12,\\ \\sin\\tfrac{\\pi}{3}=\\tfrac{\\sqrt3}{2}$）。',
        '归约成最简方程：$2\\sin\\!\\big(x-\\tfrac{\\pi}{3}\\big)=1\\Rightarrow\\sin\\!\\big(x-\\tfrac{\\pi}{3}\\big)=\\tfrac12$。',
        '解内层角：$x-\\tfrac{\\pi}{3}=\\tfrac{\\pi}{6}+2k\\pi$ 或 $\\tfrac{5\\pi}{6}+2k\\pi$，得 $x=\\tfrac{\\pi}{2}+2k\\pi$ 或 $x=\\tfrac{7\\pi}{6}+2k\\pi,\\ k\\in\\mathbb{Z}$。',
      ],
      tip: '$b=-\\sqrt3<0$ 时相位取 $-\\dfrac{\\pi}{3}$；展开后逐项核对系数符号，是避免相位错误最稳的办法。',
    },
    {
      title: '易错点：变形别引入/丢失解',
      body: '两类危险变形要警惕：\n\n- **两边同除** $\\cos x$ 得 $\\tan x$，默认了 $\\cos x\\ne0$——可能丢掉 $\\cos x=0$ 处的解；\n- **两边平方**会引入**增根**，变形后必须把结果代回**原方程**逐一检验。\n\n口诀：除法先问分母是否为零，平方之后必须验根。',
      reveal: { q: '解 $\\sin x=\\cos x$ 两边除以 $\\cos x$ 安全吗？', a: '本题安全（$\\cos x=0$ 时 $\\sin x=\\pm1\\ne0$，不是解），但必须先确认这一点' },
    },
  ],

  // ============================== 单元 11 三角不等式 ==============================
  'u11-l1': [
    {
      title: '不等式 = 解出一段区间',
      body: '三角不等式本质在问：曲线在哪些 $x$ 处**落在水平线上方/下方**？方程求的是**交点**，不等式求的是交点之间那**一段**。\n\n所以图像法和单位圆法天然好用：区间结构、周期复制都一眼可见。框架仍是——先在一个周期里定区间，再写 $+2k\\pi$。',
      image: 'trig-p4-11-1.svg',
      imageCaption: '左：单位圆上 $\\sin x\\ge\\tfrac12$ 对应 $y=\\tfrac12$ 上方的弧 $\\tfrac{\\pi}{6}\\sim\\tfrac{5\\pi}{6}$；右：正弦曲线上 $y\\ge\\tfrac12$ 的阴影区间。',
    },
    {
      title: '端点：看不等号松紧',
      body: '先解对应**方程**得到边界点，再判区间方向，最后定端点开闭：\n\n- 含等号 $\\ge,\\le$ 取**闭区间**（端点也满足）；\n- 严格 $>,<$ 取**开区间**（端点不取）。',
      formula: '\\sin x\\ge\\tfrac12\\ \\Rightarrow\\ \\Big[\\tfrac{\\pi}{6}+2k\\pi,\\ \\tfrac{5\\pi}{6}+2k\\pi\\Big],\\ k\\in\\mathbb{Z}',
    },
    {
      title: '例题精讲：$2\\sin x-\\sqrt3>0$',
      body: '从带系数的不等式整理到全体实数解集。',
      steps: [
        '整理成最简：$2\\sin x-\\sqrt3>0\\Rightarrow\\sin x>\\dfrac{\\sqrt3}{2}$。',
        '一个周期内定边界：$\\sin x=\\dfrac{\\sqrt3}{2}$ 的交点为 $x=\\dfrac{\\pi}{3}$ 与 $\\dfrac{2\\pi}{3}$。',
        '判方向：二者之间正弦曲线在 $y=\\dfrac{\\sqrt3}{2}$ **上方**，故周期内解为 $\\big(\\dfrac{\\pi}{3},\\dfrac{2\\pi}{3}\\big)$（严格 $>$ 取开）。',
        '周期复制 $2\\pi$：$x\\in\\big(\\dfrac{\\pi}{3}+2k\\pi,\\ \\dfrac{2\\pi}{3}+2k\\pi\\big),\\ k\\in\\mathbb{Z}$。',
      ],
      tip: '$k\\in\\mathbb{Z}$ 是解集的一部分，绝不能省；省了就只剩一个周期的区间，答案不完整。',
    },
    {
      title: '余弦读左右',
      body: '余弦在单位圆上看**横坐标**：$\\cos x<0$ 即落在**左半圆**，对应 $\\big(\\dfrac{\\pi}{2},\\dfrac{3\\pi}{2}\\big)$；$\\cos x>0$ 在右半圆。\n\n含等号时取闭区间，注意 $\\cos x\\le-\\tfrac12$ 的两个边界 $\\tfrac{2\\pi}{3},\\tfrac{4\\pi}{3}$ 之间余弦更小（在水平线下方）。',
      reveal: { q: '$\\cos x\\le-\\tfrac12$ 在 $[0,2\\pi)$ 的解集？', a: '$\\Big[\\dfrac{2\\pi}{3},\\dfrac{4\\pi}{3}\\Big]$（含等号取闭区间）' },
    },
  ],
  'u11-l2': [
    {
      title: '正切：为什么和正余弦不一样',
      body: '正切的周期只有 $\\pi$（正余弦的一半），而且在 $x=\\dfrac{\\pi}{2}+k\\pi$ 处有**竖直渐近线**、函数无定义。\n\n所以正切不等式不能照搬“整周期图像块”，要**逐个单调区间**处理，并时刻守住无定义点。',
      formula: '\\tan x>1\\ \\Rightarrow\\ \\Big(\\tfrac{\\pi}{4}+k\\pi,\\ \\tfrac{\\pi}{2}+k\\pi\\Big),\\ k\\in\\mathbb{Z}',
    },
    {
      title: '守住渐近线',
      body: '在每个主值区间 $\\big(-\\dfrac{\\pi}{2}+k\\pi,\\dfrac{\\pi}{2}+k\\pi\\big)$ 内正切**严格递增**，可按单调性直接定方向。\n\n但端点 $x=\\pm\\dfrac{\\pi}{2}+k\\pi$ 是渐近线、正切**无定义**——这种端点**永远取开**，不论不等号是 $\\le$ 还是 $<$。',
    },
    {
      title: '例题精讲：$\\tan x\\le\\sqrt3$',
      body: '一个有“无定义端点”的典型不等式。',
      steps: [
        '在主值区间 $\\big(-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\big)$ 内讨论：该区间正切严格递增，$\\tan\\dfrac{\\pi}{3}=\\sqrt3$。',
        '由单调性，$\\tan x\\le\\sqrt3$ 对应 $-\\dfrac{\\pi}{2}<x\\le\\dfrac{\\pi}{3}$。',
        '定端点：左端 $-\\dfrac{\\pi}{2}$ 是渐近线取**开**；右端含 $\\le$ 取**闭**。',
        '复制周期 $\\pi$（不是 $2\\pi$）：$x\\in\\big(-\\dfrac{\\pi}{2}+k\\pi,\\ \\dfrac{\\pi}{3}+k\\pi\\big],\\ k\\in\\mathbb{Z}$。',
      ],
      tip: '正切复制周期加的是 $k\\pi$ 而非 $2k\\pi$——把正余弦的 $2\\pi$ 习惯搬过来是高频错误。',
    },
    {
      title: '互动：左开右闭怎么来',
      body: '同一个不等式里，端点开闭可以**不一样**：取决于该端点是“渐近线（无定义）”还是“真实边界（由不等号松紧定）”。',
      reveal: { q: '$\\tan x\\le\\sqrt3$ 的一段解为何左开右闭？', a: '左端 $-\\tfrac{\\pi}{2}$ 无定义→开；右端 $\\tfrac{\\pi}{3}$ 因含 $\\le$→闭' },
    },
  ],
  'u11-l3': [
    {
      title: '含参先看值域',
      body: '面对 $\\sin x\\ge a$，第一步**不是**求解，而是判断参数 $a$ 是否落在函数值域 $[-1,1]$ 内：\n\n- $a>1$：$\\sin x\\le1<a$，**无解**（空集）；\n- $a<-1$：$\\sin x\\ge-1>a$，**恒成立**（全体实数）；\n- $-1\\le a\\le1$：才需要找交点、写区间。\n\n值域分析是参数不等式的第一步，不是最后一步。',
    },
    {
      title: '化同名再求解',
      body: '含 $\\cos^2 x$、$\\sin^2 x$ 的不等式先用倍角**化同名 + 单一角**，再把内层角当整体处理。\n\n如 $2\\cos^2 x-1=\\cos 2x$，于是 $2\\cos^2 x-1\\ge0\\iff\\cos 2x\\ge0$；解出 $2x$ 的区间后**整体除以 $2$**，区间长度与周期都减半。',
      formula: '\\cos 2x\\ge0\\ \\Rightarrow\\ \\Big[-\\tfrac{\\pi}{4}+k\\pi,\\ \\tfrac{\\pi}{4}+k\\pi\\Big],\\ k\\in\\mathbb{Z}',
    },
    {
      title: '例题精讲：$2\\cos^2 x-1\\ge0$',
      body: '化同名后用整体代换求解。',
      steps: [
        '倍角化同名：$2\\cos^2 x-1=\\cos 2x$，原式即 $\\cos 2x\\ge0$。',
        '令 $u=2x$，余弦非负对应 $-\\dfrac{\\pi}{2}+2k\\pi\\le u\\le\\dfrac{\\pi}{2}+2k\\pi$。',
        '回代 $u=2x$ 并整体除以 $2$：$-\\dfrac{\\pi}{4}+k\\pi\\le x\\le\\dfrac{\\pi}{4}+k\\pi$。',
        '写解集：$x\\in\\big[-\\dfrac{\\pi}{4}+k\\pi,\\ \\dfrac{\\pi}{4}+k\\pi\\big],\\ k\\in\\mathbb{Z}$。',
      ],
      tip: '除以 $2$ 是对**整条不等式（含 $+2k\\pi$）**一起除：周期由 $2\\pi$ 缩成 $\\pi$，只缩一半 $k\\pi$，别忘改周期。',
    },
    {
      title: '单位圆读符号',
      body: '判断三角函数**符号**时单位圆最直接：正弦看**上下**、余弦看**左右**、正切看**象限奇偶**。\n\n例如 $\\sin x\\ge0$ 是上半圆、$\\cos x<0$ 是左半圆、$\\tan x>0$ 在第一、三象限（$x,y$ 同号）。',
      reveal: { q: '$\\tan x>0$ 对应哪些象限？为什么？', a: '第一、三象限：那里 $\\sin x,\\cos x$ 同号，比值为正' },
    },
  ],

  // ============================== 单元 12 参数·最值·综合变形 ==============================
  'u12-l1': [
    {
      title: '动机：先压缩，再读最值',
      body: '看到 $a\\sin x+b\\cos x$，很多人急着代特殊角、做分类讨论。但更高效的做法是先把它压成**单一正弦** $R\\sin(x+\\varphi)$。\n\n一旦写成这个形式，因 $\\sin(\\cdot)\\in[-1,1]$，**值域立刻就是 $[-R,R]$**，最大值 $R$、最小值 $-R$，最值问题瞬间变透明。',
      image: 'trig-p4-12-1.svg',
      imageCaption: '$3\\sin x+4\\cos x$ 的两个分量合成为单一正弦 $5\\sin(x+\\varphi)$，峰值 $R=5$、谷值 $-5$。',
    },
    {
      title: '$R$ 一锤定音',
      body: '值域只由振幅 $R=\\sqrt{a^2+b^2}$ 决定，与相位 $\\varphi$ **无关**——$\\varphi$ 只影响在哪个 $x$ 取到最值。\n\n所以求值域/最值时**只需算 $R$**，不必费力求 $\\varphi$。',
      formula: 'a\\sin x+b\\cos x\\in[-R,R],\\quad R=\\sqrt{a^2+b^2}',
    },
    {
      title: '例题精讲：$3\\sin x+4\\cos x$ 的最值',
      body: '辅助角公式最经典的应用。',
      steps: [
        '算振幅：$R=\\sqrt{3^2+4^2}=\\sqrt{25}=5$。',
        '合并：$3\\sin x+4\\cos x=5\\sin(x+\\varphi)$，其中 $\\cos\\varphi=\\tfrac35,\\ \\sin\\varphi=\\tfrac45$。',
        '由 $\\sin(x+\\varphi)\\in[-1,1]$ 读值域：$[-5,5]$，故最大值 $5$、最小值 $-5$。',
      ],
      tip: '$R$ 永远取**正**平方根：$\\sqrt{a^2+b^2}$ 里 $b$ 的符号（如 $-12$）平方后消失，别被负号带偏。',
    },
    {
      title: '有解条件与向量视角',
      body: '方程 $a\\sin x+b\\cos x=m$ **有解** $\\iff m$ 落入左边值域，即 $|m|\\le\\sqrt{a^2+b^2}$。\n\n几何上：$a\\sin x+b\\cos x$ 是向量 $(a,b)$ 的某种投影结构，向量模长 $R=\\sqrt{a^2+b^2}$ 限制了投影不超过 $R$——这就是辅助角公式背后的向量意义。',
      reveal: { q: '$5\\sin x-12\\cos x=m$ 何时有解？', a: '$R=\\sqrt{5^2+12^2}=13$，故 $|m|\\le13$ 时有解' },
    },
  ],
  'u12-l2': [
    {
      title: '动机：把“看似二次”标准化',
      body: '含 $\\sin^2 x$、$\\sin x\\cos x$ 的式子次数高、看着吓人，但它们本质上仍是一条正弦曲线。\n\n主线是：先用**降幂、倍角**把次数降到 $1$，化成 $A\\sin 2x+B\\cos 2x+C$；再用辅助角合并成单一正弦，值域就一目了然。',
      formula: '\\sin^2 x=\\tfrac{1-\\cos 2x}{2},\\quad \\sin x\\cos x=\\tfrac12\\sin 2x',
    },
    {
      title: '完全平方也是好工具',
      body: '识别结构能省下大量计算。例如完全平方加倍角：\n\n$$(\\sin x+\\cos x)^2=\\sin^2 x+\\cos^2 x+2\\sin x\\cos x=1+\\sin 2x.$$\n\n于是 $\\sin 2x\\in[-1,1]$ 直接给出值域 $[0,2]$，几乎不用算。',
      reveal: { q: '$(\\sin x-\\cos x)^2$ 化简？值域？', a: '$=1-\\sin 2x$，值域 $[0,2]$' },
    },
    {
      title: '例题精讲：$f(x)=\\sin^2 x+\\sqrt3\\sin x\\cos x$',
      body: '降幂 → 倍角 → 辅助角的完整链条。',
      steps: [
        '降幂与倍角：$\\sin^2 x=\\dfrac{1-\\cos 2x}{2}$，$\\sqrt3\\sin x\\cos x=\\dfrac{\\sqrt3}{2}\\sin 2x$。',
        '整理成 $A\\sin 2x+B\\cos 2x+C$：$f(x)=\\dfrac12+\\dfrac{\\sqrt3}{2}\\sin 2x-\\dfrac12\\cos 2x$。',
        '辅助角合并括号：$R=\\sqrt{(\\tfrac{\\sqrt3}{2})^2+(\\tfrac12)^2}=1$，得 $f(x)=\\dfrac12+\\sin\\!\\big(2x-\\dfrac{\\pi}{6}\\big)$。',
        '读最值：$\\sin(\\cdot)\\in[-1,1]$，故 $f_{\\max}=\\dfrac12+1=\\dfrac32$，$f_{\\min}=\\dfrac12-1=-\\dfrac12$。',
      ],
      tip: '常数项 $C=\\dfrac12$ 只把曲线整体平移，不改振幅；最值 $=C\\pm R$，别把 $C$ 漏在外面。',
    },
    {
      title: '易错点：降幂后别停手',
      body: '降幂只是第一步，化成 $A\\sin 2x+B\\cos 2x+C$ 后**必须继续辅助角合并**，否则两个三角项各自摆动、看不出值域。\n\n口诀：降幂 → 同角 → 合并 → 读值域，一步都不能少。',
      reveal: { q: '$f(x)=\\tfrac12+\\sin(2x-\\tfrac{\\pi}{6})$ 的最大值在何处取得？', a: '$2x-\\tfrac{\\pi}{6}=\\tfrac{\\pi}{2}$ 即 $x=\\tfrac{\\pi}{3}+k\\pi$ 处，$f=\\tfrac32$' },
    },
  ],
  'u12-l3': [
    {
      title: '动机：限定区间不能直接套值域',
      body: '求 $f(x)=A\\sin(\\omega x+\\varphi)$ 在某**限定区间**上的最值时，最大陷阱是直接套整体值域 $[-A,A]$。\n\n因为在限定区间里，内层角未必取遍一个周期——可能根本到不了正弦的峰或谷。必须先把**内层角的范围**算出来。',
    },
    {
      title: '方法：定内层角范围，再判最值',
      body: '令 $u=\\omega x+\\varphi$，由 $x$ 的区间推出 $u$ 的区间，再看 $u$ 的范围是否**盖住** $\\sin u$ 的峰 $\\dfrac{\\pi}{2}$（取 $+1$）或谷 $-\\dfrac{\\pi}{2}$（取 $-1$）：\n\n- 盖得住：在峰/谷处取到 $\\pm A$；\n- 盖不住：在**区间端点**比较取极值。',
    },
    {
      title: '例题精讲：$f(x)=2\\sin\\!\\big(2x-\\tfrac{\\pi}{6}\\big)$ 在 $[0,\\tfrac{\\pi}{2}]$',
      body: '内层角范围决定最值落点。',
      steps: [
        '定内层角范围：$x\\in[0,\\tfrac{\\pi}{2}]\\Rightarrow u=2x-\\tfrac{\\pi}{6}\\in\\big[-\\tfrac{\\pi}{6},\\ \\tfrac{5\\pi}{6}\\big]$。',
        '查峰：$u=\\tfrac{\\pi}{2}\\in[-\\tfrac{\\pi}{6},\\tfrac{5\\pi}{6}]$，能取到 $\\sin u=1$，故 $f_{\\max}=2\\cdot1=2$（在 $x=\\tfrac{\\pi}{3}$）。',
        '查谷：谷 $-\\tfrac{\\pi}{2}$ **不在**范围内，须比较端点——$\\sin(-\\tfrac{\\pi}{6})=-\\tfrac12$，$\\sin\\tfrac{5\\pi}{6}=\\tfrac12$，最小为 $-\\tfrac12$。',
        '得最值：$f_{\\min}=2\\cdot(-\\tfrac12)=-1$（在 $x=0$）。',
      ],
      tip: '能取到峰 $+1$ 不代表也能取到谷 $-1$——峰、谷要**分别**核对是否落在 $u$ 的范围内。',
    },
    {
      title: '含参化二次',
      body: '含参恒成立类如 $\\cos 2x+2a\\sin x+1\\ge0$，先化同名换元 $t=\\sin x\\in[-1,1]$，化成 $t^2-at-1\\le0$ 对一切 $t\\in[-1,1]$ 恒成立——这就是**闭区间上的二次恒成立**问题。\n\n开口向上的二次在闭区间最大值必在**端点**，只需 $g(-1)\\le0$ 且 $g(1)\\le0$ 夹逼出参数（这里得 $a=0$）。\n\n另一类联动换元：令 $u=\\sin x+\\cos x$，由平方关系可把 $\\sin x\\cos x$ 用 $u$ 表示，关键约束是 $u$ 的范围。',
      formula: 'u=\\sin x+\\cos x=\\sqrt2\\sin\\!\\big(x+\\tfrac{\\pi}{4}\\big)\\in[-\\sqrt2,\\sqrt2]',
    },
    {
      title: '互动：联动换元的关键约束',
      body: '$\\sin x\\pm\\cos x$ 与 $\\sin x\\cos x$ 联动题里，定出 $u$ 的范围 $[-\\sqrt2,\\sqrt2]$ 是命门——不可误放成全体实数，否则二次函数值域算错。',
      reveal: { q: '为什么 $u=\\sin x+\\cos x$ 的范围是 $[-\\sqrt2,\\sqrt2]$？', a: '辅助角 $u=\\sqrt2\\sin(x+\\tfrac{\\pi}{4})$，振幅 $\\sqrt2$，故 $u\\in[-\\sqrt2,\\sqrt2]$' },
    },
  ],
}
