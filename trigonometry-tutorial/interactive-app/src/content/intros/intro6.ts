import type { IntroCard } from '../../types'

// 学新知：第六部分（单元 16–18）各课的概念讲解卡片，按 lesson id 索引。
export const INTRO6: Record<string, IntroCard[]> = {
  // ============================== 单元 16 ==============================
  'u16-l1': [
    {
      title: '把两个波压成一个波',
      body: '形如 $a\\sin x+b\\cos x$ 的式子总能合成成**单个**正弦波 $R\\sin(x+\\varphi)$。$R$ 叫**振幅**，$\\varphi$ 叫**初相**。',
      formula: 'a\\sin x+b\\cos x=R\\sin(x+\\varphi)',
    },
    {
      title: '振幅怎么来',
      body: '展开 $R\\sin(x+\\varphi)=R\\cos\\varphi\\,\\sin x+R\\sin\\varphi\\,\\cos x$，对比系数得 $R\\cos\\varphi=a,\\ R\\sin\\varphi=b$。两式平方相加即得振幅。',
      formula: 'R=\\sqrt{a^2+b^2}',
    },
    {
      title: '几何解释',
      body: '把 $(a,b)$ 看成一个**向量**：它的长度就是振幅 $R$，它的辐角就是初相 $\\varphi$。',
      reveal: { q: '$3\\sin x+4\\cos x$ 的振幅 $R=?$', a: '$R=\\sqrt{3^2+4^2}=\\sqrt{25}=5$' },
    },
  ],
  'u16-l2': [
    {
      title: '先定 R，再定象限',
      body: '合成时先算 $R=\\sqrt{a^2+b^2}$，再由 $\\cos\\varphi=\\dfrac{a}{R},\\ \\sin\\varphi=\\dfrac{b}{R}$ 确定 $\\varphi$。',
      formula: '\\cos\\varphi=\\frac{a}{R},\\qquad \\sin\\varphi=\\frac{b}{R}',
    },
    {
      title: '两个符号缺一不可',
      body: '只看 $\\cos\\varphi$ 一个值会丢失象限信息。必须让 $\\cos\\varphi$ 与 $\\sin\\varphi$ 的**符号同时满足**，$\\varphi$ 才唯一。',
      reveal: {
        q: '$\\cos\\varphi=\\dfrac12,\\ \\sin\\varphi=-\\dfrac{\\sqrt3}{2}$，取 $\\varphi=?$',
        a: '余弦正、正弦负在第四象限，取 $\\varphi=-\\dfrac{\\pi}{3}$',
      },
    },
    {
      title: '看个例子',
      body: '对 $\\sqrt3\\sin t+\\cos t$：$R=\\sqrt{3+1}=2$，$\\cos\\varphi=\\dfrac{\\sqrt3}{2},\\ \\sin\\varphi=\\dfrac12$，于是 $\\varphi=\\dfrac{\\pi}{6}$。',
      formula: '\\sqrt3\\sin t+\\cos t=2\\sin\\left(t+\\frac{\\pi}{6}\\right)',
    },
  ],
  'u16-l3': [
    {
      title: '一眼读出最值',
      body: '合成成 $R\\sin(x+\\varphi)$ 后，因为 $\\sin\\in[-1,1]$，**最大值就是 $R$，最小值就是 $-R$**，值域为 $[-R,R]$。',
      formula: '\\max=R=\\sqrt{a^2+b^2},\\qquad \\min=-R',
    },
    {
      title: '周期看 ω',
      body: '$y=a\\sin\\omega x+b\\cos\\omega x$ 合成后仍是 $R\\sin(\\omega x+\\varphi)$，周期由 $\\omega$ 决定。',
      formula: 'T=\\frac{2\\pi}{\\omega}',
    },
    {
      title: '物理意义：简谐合成',
      body: '简谐运动 $x(t)=A\\sin(\\omega t+\\varphi)$ 里 $A$ 是振幅、$\\omega$ 是角频率、$\\varphi$ 是初相——与辅助角公式同构。**同频率**两简谐叠加，结果仍是同频率简谐振动。',
      reveal: { q: '$y=3\\sin x+4\\cos x$ 的最大值是？', a: '即振幅 $R=\\sqrt{9+16}=5$' },
    },
  ],

  // ============================== 单元 17 ==============================
  'u17-l1': [
    {
      title: '让三角函数吸收根号',
      body: '带根式的代数式，可以用单位圆恒等式把根号「化掉」。核心是两条：$1-\\sin^2\\theta=\\cos^2\\theta$ 和 $1+\\tan^2\\theta=\\sec^2\\theta$。',
      formula: '\\sqrt{a^2-x^2}\\ \\xrightarrow{\\,x=a\\sin\\theta\\,}\\ a\\cos\\theta',
    },
    {
      title: '三类经典代换',
      body: '看根式结构选代换：**减号配 $\\sin$，加号配 $\\tan$，$x^2-a^2$ 配 $\\sec$**。',
      formula: '\\begin{aligned}\\sqrt{a^2-x^2}&:\\ x=a\\sin\\theta\\\\ \\sqrt{a^2+x^2}&:\\ x=a\\tan\\theta\\\\ \\sqrt{x^2-a^2}&:\\ x=a\\sec\\theta\\end{aligned}',
    },
    {
      title: '看个例子',
      body: '加号结构用正切代换：$25+x^2$ 被 $1+\\tan^2\\theta=\\sec^2\\theta$ 化成完全平方。',
      reveal: { q: '设 $x=5\\tan\\theta$，$\\sqrt{25+x^2}=?$', a: '$\\sqrt{25\\sec^2\\theta}=5\\sec\\theta$' },
    },
  ],
  'u17-l2': [
    {
      title: '开方先得绝对值',
      body: '$\\sqrt{\\cos^2 t}$ 不能随手写成 $\\cos t$——开平方先得**绝对值** $|\\cos t|$，这是三角代换最易出错的一步。',
      formula: '\\sqrt{\\cos^2 t}=|\\cos t|',
    },
    {
      title: '用范围去绝对值',
      body: '$x=a\\sin t$ 常取 $t\\in\\left[-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right]$，正是因为该区间上 $\\cos t\\ge0$，可以放心去掉绝对值。',
      reveal: {
        q: '$x=2\\sin t,\\ t\\in\\left[-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right]$，$\\sqrt{4-x^2}=?$',
        a: '$2|\\cos t|=2\\cos t$（该范围内 $\\cos t\\ge0$）',
      },
    },
    {
      title: '别忘了限制范围',
      body: '若不限制参数 $t$，就**不能**把 $|\\cos t|$ 写成 $\\cos t$，否则会出符号错误。选代换时一并约定好 $t$ 的取值范围。',
    },
  ],
  'u17-l3': [
    {
      title: '根式问题变恒等式问题',
      body: '换元后 $dx$ 也要一起换。例如 $x=3\\sin t$ 时 $dx=3\\cos t\\,dt$，根号与微分里的因子常常**约掉**，积分立刻变简单。',
      formula: '\\int\\frac{dx}{\\sqrt{9-x^2}}=\\int\\frac{3\\cos t}{3\\cos t}\\,dt=\\int dt=t+C',
    },
    {
      title: '回代或换限',
      body: '不定积分要**回代**回原变量：由 $x=3\\sin t$ 得 $t=\\arcsin\\dfrac{x}{3}$。定积分则更省事——换元时把**上下限一起换掉**，省去回代。',
      formula: '\\int\\frac{dx}{\\sqrt{9-x^2}}=\\arcsin\\frac{x}{3}+C',
    },
    {
      title: '几何意义',
      body: '$x=a\\sin\\theta$ 相当于把 $x$ 看作直角三角形的对边、$a$ 看作斜边，则邻边恰是 $\\sqrt{a^2-x^2}=a\\cos\\theta$。',
      reveal: {
        q: '$\\displaystyle\\int_0^a\\sqrt{a^2-x^2}\\,dx=?$',
        a: '$\\dfrac{\\pi a^2}{4}$——半径 $a$ 的四分之一圆面积',
      },
    },
  ],

  // ============================== 单元 18 ==============================
  'u18-l1': [
    {
      title: '三倍角公式',
      body: '把 $\\cos 3x=\\cos(2x+x)$ 展开并用 $\\sin^2x=1-\\cos^2x$ 化简，就得到只含 $\\cos x$ 的三次式；正弦版形式对称。',
      formula: '\\begin{aligned}\\cos 3x&=4\\cos^3 x-3\\cos x\\\\ \\sin 3x&=3\\sin x-4\\sin^3 x\\end{aligned}',
    },
    {
      title: '它是一个三角多项式',
      body: '令 $u=\\cos x$，则 $\\cos 3x=4u^3-3u$ 是 $u$ 的**三次多项式**（即 Chebyshev 多项式 $T_3$）。一般地 $\\cos nx=T_n(\\cos x)$。',
      reveal: { q: '$8\\cos^3 x-6\\cos x=?$', a: '$2(4\\cos^3x-3\\cos x)=2\\cos 3x$' },
    },
    {
      title: '降幂公式',
      body: '反过来，由 $\\cos 2t=2\\cos^2 t-1$ 反解得**降幂公式**，把 $\\cos^2,\\sin^2$ 降成一次，便于积分。',
      formula: '\\cos^2 t=\\frac{1+\\cos 2t}{2},\\qquad \\sin^2 t=\\frac{1-\\cos 2t}{2}',
    },
  ],
  'u18-l2': [
    {
      title: '万能代换：三角变有理',
      body: '令 $t=\\tan\\dfrac{x}{2}$，则 $\\sin x,\\cos x,\\tan x$ 全都变成 $t$ 的**有理式**——含三角的积分/方程一键变有理。',
      formula: '\\sin x=\\frac{2t}{1+t^2},\\quad \\cos x=\\frac{1-t^2}{1+t^2},\\quad \\tan x=\\frac{2t}{1-t^2}',
    },
    {
      title: '微分也要换',
      body: '由 $t=\\tan\\dfrac{x}{2}$ 微分得到 $dx$ 的换元式，这样整个积分都落到有理函数上。',
      formula: 'dx=\\frac{2}{1+t^2}\\,dt',
    },
    {
      title: '当心 x=π',
      body: '$\\tan\\dfrac{\\pi}{2}$ 不存在，所以 $t=\\tan\\dfrac{x}{2}$ 在 $x=\\pi$ 处无定义，相当于预设 $x\\ne\\pi$。解题后要**单独验证** $x=\\pi$ 是否为解，以防漏解。',
      reveal: { q: '$\\cos x$ 的万能代换式是？', a: '$\\cos x=\\dfrac{1-t^2}{1+t^2}$' },
    },
  ],
  'u18-l3': [
    {
      title: '统一入口',
      body: '代入三条公式与 $dx$ 后，三角方程变成关于 $t$ 的**多项式方程**，三角积分变成**有理函数积分**——一个套路解一大类题。',
      formula: '\\int\\frac{dx}{2+\\cos x}=\\int\\frac{2}{3+t^2}\\,dt',
    },
    {
      title: '两种代换分工',
      body: '别把名字搞混：**三角代换**处理根式 $\\sqrt{\\cdot}$，**万能代换**处理同时含 $\\sin x,\\cos x$ 的有理式。对象不同，各司其职。',
      reveal: {
        q: '$\\sin x+\\cos x=1$ 代入后化成关于 $t$ 的什么方程？',
        a: '$2t=2t^2$，即 $t(t-1)=0$，故 $t=0$ 或 $t=1$',
      },
    },
    {
      title: '不是万灵药',
      body: '万能代换是统一入口，但**不总是最短路径**。很多题用图像、辅助角或恒等式会更快——先想清楚再动手。',
    },
  ],
}
