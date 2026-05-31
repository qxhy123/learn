import type { IntroCard } from '../../types'

// 学新知：第二部分（单元 4–6）各课的概念讲解卡片，按 lesson id 索引。
export const INTRO2: Record<string, IntroCard[]> = {
  // ============================== 单元 4 ==============================
  'u4-l1': [
    {
      title: '概念',
      body: '**平方关系**来自单位圆：终边交点 $(\\cos\\theta,\\sin\\theta)$ 满足 $x^2+y^2=1$，于是横纵坐标的平方和恒为 $1$。',
      formula: '\\sin^2\\theta+\\cos^2\\theta=1',
    },
    {
      title: '怎么用',
      body: '知道 $\\sin$ 与 $\\cos$ 之一，就能求另一个：移项 $\\cos^2\\theta=1-\\sin^2\\theta$，开方再**由象限定符号**。',
      reveal: { q: '$\\sin\\theta=\\dfrac35$（锐角），$\\cos\\theta=?$', a: '$\\cos\\theta=\\sqrt{1-9/25}=\\dfrac45$' },
    },
    {
      title: '易错点',
      body: '开平方只给**绝对值**！$\\sin^2\\theta=\\dfrac14$ 推不出 $\\sin\\theta=\\dfrac12$，还可能是 $-\\dfrac12$，正负号必须看象限。',
    },
  ],
  'u4-l2': [
    {
      title: '商数关系',
      body: '正切是“纵 ÷ 横”，把 $\\tan,\\cot$ 都拉回到 $\\sin,\\cos$ 上来理解。',
      formula: '\\tan\\theta=\\dfrac{\\sin\\theta}{\\cos\\theta},\\qquad \\cot\\theta=\\dfrac{\\cos\\theta}{\\sin\\theta}',
    },
    {
      title: '倒数关系',
      body: '另外三个函数是 $\\sin,\\cos,\\tan$ 的**倒数**：$\\sec\\leftrightarrow\\cos$、$\\csc\\leftrightarrow\\sin$、$\\cot\\leftrightarrow\\tan$。六个函数靠这两组关系连成一张网。',
      reveal: { q: '$\\sec\\theta=?$', a: '$\\dfrac{1}{\\cos\\theta}$' },
    },
    {
      title: '小技巧',
      body: '已知 $\\tan\\theta$ 求 $\\sin,\\cos$ 的**齐次式**时，分子分母同除 $\\cos\\theta$，整个式子就只剩 $\\tan\\theta$ 了。',
    },
  ],
  'u4-l3': [
    {
      title: '导出恒等式',
      body: '把平方关系两边分别除以 $\\cos^2\\theta$ 或 $\\sin^2\\theta$，立刻得到两条变体。它们不是新公式，都是平方关系的“换装”。',
      formula: '1+\\tan^2\\theta=\\sec^2\\theta,\\qquad 1+\\cot^2\\theta=\\csc^2\\theta',
    },
    {
      title: '化简思路',
      body: '看到 $1-\\sin^2\\theta$、$1+\\tan^2\\theta$ 这类组合，要能**模式识别**，一眼换成对应的平方形式。',
      reveal: { q: '$1+\\tan^2\\theta=?$', a: '$\\sec^2\\theta$' },
    },
    {
      title: '易错点',
      body: '证明恒等式时不能为了约分而忽略**定义域**：分母为零处不成立，随意约分可能得到错误结论。',
    },
  ],

  // ============================== 单元 5 ==============================
  'u5-l1': [
    {
      title: '和差公式',
      body: '和差公式是三角变换的“发动机”，把两个角的和或差展开成单角函数的组合。**正弦同号、余弦反号**是记忆口诀。',
      formula: '\\sin(\\alpha\\pm\\beta)=\\sin\\alpha\\cos\\beta\\pm\\cos\\alpha\\sin\\beta',
    },
    {
      title: '余弦版',
      body: '余弦展开时符号**相反**：$\\cos(\\alpha+\\beta)=\\cos\\alpha\\cos\\beta-\\sin\\alpha\\sin\\beta$，差角则号变正。',
      reveal: { q: '$\\cos(\\alpha-\\beta)=?$', a: '$\\cos\\alpha\\cos\\beta+\\sin\\alpha\\sin\\beta$' },
    },
    {
      title: '拆角凑特殊角',
      body: '把非特殊角拆成两个特殊角之和差，就能求精确值，例如 $75^\\circ=45^\\circ+30^\\circ$、$15^\\circ=45^\\circ-30^\\circ$。',
    },
  ],
  'u5-l2': [
    {
      title: '倍角从何而来',
      body: '在和角公式里令 $\\alpha=\\beta=x$，就得到**倍角公式**——它只是和角公式的特例。',
      formula: '\\sin 2x=2\\sin x\\cos x',
    },
    {
      title: '余弦倍角三件套',
      body: '$\\cos 2x$ 有三种等价写法，配合平方关系可以互推：$\\cos^2x-\\sin^2x=1-2\\sin^2x=2\\cos^2x-1$。',
      reveal: { q: '$\\cos 2x$ 用 $\\sin x$ 表示？', a: '$1-2\\sin^2x$' },
    },
    {
      title: '正切倍角',
      body: '正切倍角公式同样由和角推出，分母里那个减号别丢：$\\tan 2x=\\dfrac{2\\tan x}{1-\\tan^2x}$。',
    },
  ],
  'u5-l3': [
    {
      title: '半角公式',
      body: '把余弦倍角 $\\cos x=1-2\\sin^2\\dfrac{x}{2}=2\\cos^2\\dfrac{x}{2}-1$ **反解**，就得到半角的平方公式。',
      formula: '\\sin^2\\dfrac{x}{2}=\\dfrac{1-\\cos x}{2},\\quad \\cos^2\\dfrac{x}{2}=\\dfrac{1+\\cos x}{2}',
    },
    {
      title: '开方要带符号',
      body: '由平方式开方时，$\\sin\\dfrac{x}{2},\\cos\\dfrac{x}{2}$ 的正负取决于 $\\dfrac{x}{2}$ 所在象限，不能一律取正号。',
      reveal: { q: '半角开方时正负号由什么决定？', a: '$\\dfrac{x}{2}$ 所在象限' },
    },
    {
      title: '万能代换',
      body: '令 $t=\\tan\\dfrac{x}{2}$，可把 $\\sin x,\\cos x$ 统一写成关于 $t$ 的有理式，便于积分与化简：$\\sin x=\\dfrac{2t}{1+t^2},\\ \\cos x=\\dfrac{1-t^2}{1+t^2}$。',
    },
  ],

  // ============================== 单元 6 ==============================
  'u6-l1': [
    {
      title: '积化和差',
      body: '把两条和差公式相加或相减，就能把**两个函数的乘积**拆成和与差。代价是出现一个 $\\dfrac12$。',
      formula: '\\sin A\\cos B=\\dfrac12[\\sin(A+B)+\\sin(A-B)]',
    },
    {
      title: '余弦乘积',
      body: '同理 $\\cos A\\cos B=\\dfrac12[\\cos(A+B)+\\cos(A-B)]$，而 $\\sin A\\sin B$ 前面带负号，最易记错。',
      reveal: { q: '$\\cos A\\cos B=?$', a: '$\\dfrac12[\\cos(A+B)+\\cos(A-B)]$' },
    },
    {
      title: '用处',
      body: '遇到三角函数**乘积**求值或连乘抵消（如 $\\cos20^\\circ\\cos40^\\circ\\cos80^\\circ$），就该先想积化和差。',
    },
  ],
  'u6-l2': [
    {
      title: '和差化积',
      body: '反过来，把**和或差**写成乘积。设 $s=\\dfrac{A+B}{2},\\ d=\\dfrac{A-B}{2}$，正弦之和变成“正弦 × 余弦”。',
      formula: '\\sin A+\\sin B=2\\sin\\dfrac{A+B}{2}\\cos\\dfrac{A-B}{2}',
    },
    {
      title: '余弦那一组',
      body: '$\\cos A+\\cos B=2\\cos s\\cos d$，而 $\\cos A-\\cos B=-2\\sin s\\sin d$——前面的负号是符号陷阱。',
      reveal: { q: '$\\cos A-\\cos B=?$', a: '$-2\\sin\\dfrac{A+B}{2}\\sin\\dfrac{A-B}{2}$' },
    },
    {
      title: '拍频的来历',
      body: '两个相近频率叠加，如 $\\sin101t+\\sin99t=2\\sin100t\\cos t$，就被拆成“快速波形 × 慢变包络”，这正是**拍频**。',
    },
  ],
  'u6-l3': [
    {
      title: '辅助角公式',
      body: '$a\\sin x+b\\cos x$ 可以**合一**为单个正弦：振幅 $R=\\sqrt{a^2+b^2}$，相位 $\\varphi$ 由 $\\tan\\varphi=\\dfrac{b}{a}$ 确定。',
      formula: 'a\\sin x+b\\cos x=\\sqrt{a^2+b^2}\\,\\sin(x+\\varphi)',
    },
    {
      title: '求最值',
      body: '合一后正弦最大为 $1$，所以 $a\\sin x+b\\cos x$ 的最大值就是 $\\sqrt{a^2+b^2}$，最小值是它的相反数。',
      reveal: { q: '$3\\sin x+4\\cos x$ 的最大值？', a: '$\\sqrt{3^2+4^2}=5$' },
    },
    {
      title: '选公式的直觉',
      body: '看到**乘积**先想积化和差，看到**和差**先想和差化积，看到 $a\\sin x+b\\cos x$ 先想辅助角——本质都是在不同表示之间切换。',
    },
  ],
}
