import type { IntroCard } from '../../types'

// 学新知：第四部分（单元 10–12）各课的概念讲解卡片，按 lesson id 索引。
export const INTRO4: Record<string, IntroCard[]> = {
  // ============================== 单元 10 三角方程 ==============================
  'u10-l1': [
    {
      title: '主值解 ≠ 全部解',
      body: '三角函数是**周期**的，水平线 $y=a$ 会和曲线交无穷多次。先在一个周期里找解，再补上周期复制项 $+2k\\pi$（正切是 $+k\\pi$）。',
    },
    {
      title: '正弦：两族解',
      body: '$\\sin x=a$（$|a|\\le1$）在一个周期里有两个解，分别落在第一、二象限终边方向上。',
      formula: '\\sin x=\\tfrac12\\ \\Rightarrow\\ x=\\tfrac{\\pi}{6}+2k\\pi\\ \\text{或}\\ x=\\tfrac{5\\pi}{6}+2k\\pi,\\ k\\in\\mathbb{Z}',
    },
    {
      title: '余弦与正切的通解形',
      body: '**余弦**对称取值，写成 $\\pm$：$x=\\pm\\alpha+2k\\pi$。**正切**周期只有 $\\pi$，单族：$x=\\alpha+k\\pi$。当 $|a|>1$ 时正弦/余弦方程**无解**。',
      reveal: { q: '$\\cos x=\\tfrac12$ 的通解？', a: '$x=\\pm\\dfrac{\\pi}{3}+2k\\pi,\\ k\\in\\mathbb{Z}$' },
    },
  ],
  'u10-l2': [
    {
      title: '换元：把三角压成代数',
      body: '看到 $\\sin x$ 反复出现，就令 $t=\\sin x$，方程立刻变成熟悉的**二次方程**。',
      formula: '2\\sin^2 x-3\\sin x+1=0\\ \\xrightarrow{t=\\sin x}\\ 2t^2-3t+1=0',
    },
    {
      title: '先化同名再换元',
      body: '若同时含 $\\sin x$ 和 $\\cos x$，先用 $\\cos^2 x=1-\\sin^2 x$ 或倍角 $\\cos 2x=1-2\\sin^2 x$ **化成只含一种**，再换元。',
    },
    {
      title: '别忘了回代与筛解',
      body: '解出 $t$ 后必须满足 $t\\in[-1,1]$，否则舍去；保留的 $t$ 再解回 $x$ 的通解。',
      reveal: { q: '$2t^2-3t+1=0$ 解得？哪个 $t$ 要小心？', a: '$t=\\tfrac12$ 或 $t=1$，都在 $[-1,1]$ 内，均保留' },
    },
  ],
  'u10-l3': [
    {
      title: '辅助角：合二为一',
      body: '$a\\sin x+b\\cos x$ 可压成**单一正弦**，振幅 $R=\\sqrt{a^2+b^2}$，相位 $\\varphi$ 由 $\\tan\\varphi=\\tfrac{b}{a}$ 定。',
      formula: 'a\\sin x+b\\cos x=R\\sin(x+\\varphi),\\quad R=\\sqrt{a^2+b^2}',
    },
    {
      title: '化简后就是最简方程',
      body: '例如 $\\sin x-\\sqrt3\\cos x=2\\sin\\!\\big(x-\\tfrac{\\pi}{3}\\big)$，方程就归约成 $\\sin(\\cdot)=$ 常数，再套通解。',
      reveal: { q: '$\\sin x+\\cos x$ 的 $R$？', a: '$R=\\sqrt{1+1}=\\sqrt2$，即 $\\sqrt2\\sin\\!\\big(x+\\tfrac{\\pi}{4}\\big)$' },
    },
    {
      title: '变形要守规矩',
      body: '两边**同除** $\\cos x$ 得 $\\tan x$ 时须假设 $\\cos x\\ne0$；两边**平方**会引入增根，变形后必须代回原方程检验。',
    },
  ],

  // ============================== 单元 11 三角不等式 ==============================
  'u11-l1': [
    {
      title: '不等式 = 解出一段区间',
      body: '方程求的是**交点**，不等式求的是曲线在水平线**上方或下方**的那段 $x$。同样先在一个周期里定区间，再写 $+2k\\pi$。',
    },
    {
      title: '端点：看不等号松紧',
      body: '先解对应方程得边界点；含 $\\ge,\\le$ 取**闭区间**，严格 $>,<$ 取**开区间**。',
      formula: '\\sin x\\ge\\tfrac12\\ \\Rightarrow\\ \\Big[\\tfrac{\\pi}{6}+2k\\pi,\\ \\tfrac{5\\pi}{6}+2k\\pi\\Big],\\ k\\in\\mathbb{Z}',
    },
    {
      title: '余弦读左右',
      body: '$\\cos x<0$ 即落在单位圆左半，对应 $\\big(\\tfrac{\\pi}{2},\\tfrac{3\\pi}{2}\\big)$。',
      reveal: { q: '$\\cos x\\le-\\tfrac12$ 在 $[0,2\\pi)$ 的解集？', a: '$\\Big[\\dfrac{2\\pi}{3},\\dfrac{4\\pi}{3}\\Big]$' },
    },
  ],
  'u11-l2': [
    {
      title: '正切：周期是 $\\pi$',
      body: '正切周期只有 $\\pi$，是正弦余弦的一半，所以复制时加的是 $k\\pi$ 而非 $2k\\pi$。',
      formula: '\\tan x>1\\ \\Rightarrow\\ \\Big(\\tfrac{\\pi}{4}+k\\pi,\\ \\tfrac{\\pi}{2}+k\\pi\\Big),\\ k\\in\\mathbb{Z}',
    },
    {
      title: '守住渐近线',
      body: '在 $x=-\\tfrac{\\pi}{2}+k\\pi$ 处正切**无定义**（渐近线），这种端点永远取**开**，不论不等号松紧。',
    },
    {
      title: '逐单调区间处理',
      body: '正切在每个 $\\big(-\\tfrac{\\pi}{2}+k\\pi,\\tfrac{\\pi}{2}+k\\pi\\big)$ 内严格递增，按单调性定方向即可。',
      reveal: { q: '$\\tan x\\le\\sqrt3$ 的一段解？', a: '$\\Big(-\\dfrac{\\pi}{2}+k\\pi,\\ \\dfrac{\\pi}{3}+k\\pi\\Big]$：左开右闭' },
    },
  ],
  'u11-l3': [
    {
      title: '含参先看值域',
      body: '面对 $\\sin x\\ge a$，第一步不是求解，而是判断 $a$ 是否落在值域 $[-1,1]$ 内：$a>1$ 则**无解**，$a<-1$ 则**恒成立**。',
    },
    {
      title: '化同名再求解',
      body: '$2\\cos^2 x-1\\ge0$ 用倍角 $2\\cos^2 x-1=\\cos 2x$ 化成 $\\cos 2x\\ge0$，再把 $2x$ 当整体解、最后除以 $2$。',
      formula: '\\cos 2x\\ge0\\ \\Rightarrow\\ \\Big[-\\tfrac{\\pi}{4}+k\\pi,\\ \\tfrac{\\pi}{4}+k\\pi\\Big],\\ k\\in\\mathbb{Z}',
    },
    {
      title: '单位圆读符号',
      body: '正弦看上下、余弦看左右、正切看象限奇偶。',
      reveal: { q: '$\\tan x>0$ 对应哪些象限？', a: '第一、三象限（$x,y$ 同号）' },
    },
  ],

  // ============================== 单元 12 参数·最值·综合变形 ==============================
  'u12-l1': [
    {
      title: '辅助角直接读最值',
      body: '$a\\sin x+b\\cos x=R\\sin(x+\\varphi)$，因 $\\sin(\\cdot)\\in[-1,1]$，**值域就是 $[-R,R]$**，最大值 $R$、最小值 $-R$。',
      formula: 'R=\\sqrt{a^2+b^2}',
    },
    {
      title: '$R$ 一锤定音',
      body: '值域只由 $R$ 决定，与相位 $\\varphi$ 无关；$\\varphi$ 只影响在哪个 $x$ 取到最值。',
      reveal: { q: '$3\\sin x+4\\cos x$ 的最大值？', a: '$R=\\sqrt{3^2+4^2}=5$，最大值 $5$' },
    },
    {
      title: '有解条件',
      body: '方程 $a\\sin x+b\\cos x=m$ 有解 $\\iff m$ 落入值域。这也对应向量 $(a,b)$ 的模长 $R$。',
      formula: '|m|\\le\\sqrt{a^2+b^2}',
    },
  ],
  'u12-l2': [
    {
      title: '降幂：把二次压成一次',
      body: '含 $\\sin^2 x$、$\\sin x\\cos x$ 的式子先用降幂与倍角公式，把次数降到 $1$，化成 $A\\sin 2x+B\\cos 2x+C$。',
      formula: '\\sin^2 x=\\tfrac{1-\\cos 2x}{2},\\quad \\sin x\\cos x=\\tfrac12\\sin 2x',
    },
    {
      title: '再辅助角合并',
      body: '把 $A\\sin 2x+B\\cos 2x$ 用辅助角并成单一正弦，常数 $C$ 平移上下，值域就一目了然。',
      reveal: { q: '$\\sin^2 x+\\sqrt3\\sin x\\cos x$ 化简？', a: '$=\\dfrac12+\\sin\\!\\big(2x-\\dfrac{\\pi}{6}\\big)$' },
    },
    {
      title: '完全平方也好用',
      body: '$(\\sin x+\\cos x)^2=1+\\sin 2x$，于是值域为 $[0,2]$。识别结构能省下大量计算。',
    },
  ],
  'u12-l3': [
    {
      title: '限定区间：先定内层角',
      body: '求 $f(x)=A\\sin(\\omega x+\\varphi)$ 在某区间的最值，**不能直接套整体值域**。先把内层角 $u=\\omega x+\\varphi$ 的范围算出来。',
    },
    {
      title: '再按 $u$ 的范围判最值',
      body: '看 $u$ 的区间是否盖住 $\\sin u$ 的峰 $\\tfrac{\\pi}{2}$（取 $+1$）或谷；盖不到就在端点取极值。',
      reveal: { q: '$2\\sin u,\\ u\\in[-\\tfrac{\\pi}{6},\\tfrac{5\\pi}{6}]$ 的最值？', a: '含 $u=\\tfrac{\\pi}{2}$，最大 $2$；端点 $u=-\\tfrac{\\pi}{6}$ 给 $\\sin u=-\\tfrac12$，最小 $-1$' },
    },
    {
      title: '含参化二次',
      body: '恒成立类如 $\\cos 2x+2a\\sin x+1\\ge0$，换元 $t=\\sin x\\in[-1,1]$ 化成**闭区间上二次恒成立**，再用端点/顶点夹逼参数。',
      formula: 't=\\sin x+\\cos x=\\sqrt2\\sin\\!\\big(x+\\tfrac{\\pi}{4}\\big)\\in[-\\sqrt2,\\sqrt2]',
    },
  ],
}
