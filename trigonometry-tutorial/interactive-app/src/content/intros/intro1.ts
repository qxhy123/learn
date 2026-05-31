import type { IntroCard } from '../../types'

// 学新知：第一部分（单元 1–3）各课的概念讲解卡片，按 lesson id 索引。
export const INTRO1: Record<string, IntroCard[]> = {
  'u1-l1': [
    {
      title: '为什么有“弧度”',
      body: '量角有两套单位：**角度**（一圈 $360^\\circ$）和**弧度**。弧度直接用“弧长 ÷ 半径”定义，在高等数学里更自然。',
    },
    {
      title: '关键换算',
      body: '记住这一条，其余都能推：半圈 $=180^\\circ=\\pi$ 弧度。把 $\\pi$ 当作 $180^\\circ$ 代进去即可。',
      formula: '180^\\circ=\\pi\\ \\text{rad},\\qquad 1^\\circ=\\frac{\\pi}{180}',
    },
    {
      title: '看个例子',
      body: '换算时“按比例缩放”就行。',
      reveal: { q: '$90^\\circ$ 是多少弧度？', a: '$90^\\circ=\\dfrac{90}{180}\\pi=\\dfrac{\\pi}{2}$' },
    },
  ],
  'u1-l2': [
    {
      title: '单位圆',
      body: '**单位圆**就是半径为 $1$、圆心在原点的圆。三角函数都能放到它上面统一理解。',
    },
    {
      title: '终边交点的坐标',
      body: '角 $\\theta$ 的终边与单位圆交于点 $P$，那么 $P$ 的横坐标就是 $\\cos\\theta$，纵坐标就是 $\\sin\\theta$。',
      formula: 'P=(\\cos\\theta,\\ \\sin\\theta)',
    },
    {
      title: '由此而来',
      body: '因为 $P$ 在单位圆上，$x^2+y^2=1$，于是得到最重要的恒等式。',
      reveal: { q: '$\\sin^2\\theta+\\cos^2\\theta=?$', a: '$=1$（勾股恒等式）' },
    },
  ],
  'u1-l3': [
    {
      title: '弧长公式',
      body: '弧度制的好处：弧长 $=$ 半径 $\\times$ 圆心角，不用再乘换算因子。',
      formula: 'l=r\\theta',
    },
    {
      title: '扇形面积',
      body: '扇形面积按圆心角占整圆的比例缩放即可得到。',
      formula: 'S=\\tfrac12 r^2\\theta',
    },
    {
      title: '试一试',
      body: '直接代入公式。',
      reveal: { q: '半径 $2$、圆心角 $\\dfrac{\\pi}{2}$ 的弧长？', a: '$l=r\\theta=2\\times\\dfrac{\\pi}{2}=\\pi$' },
    },
  ],
  'u2-l1': [
    {
      title: '三角函数怎么来的',
      body: '在直角三角形里，正弦/余弦/正切是三条**边的比值**，记口诀 SOH-CAH-TOA。',
      formula: '\\sin=\\frac{\\text{对}}{\\text{斜}},\\quad \\cos=\\frac{\\text{邻}}{\\text{斜}},\\quad \\tan=\\frac{\\text{对}}{\\text{邻}}',
    },
    {
      title: '商数与倒数关系',
      body: '$\\tan\\theta=\\dfrac{\\sin\\theta}{\\cos\\theta}$；而 $\\sec,\\csc,\\cot$ 分别是 $\\cos,\\sin,\\tan$ 的**倒数**。',
    },
    {
      title: '看个例子',
      body: '已知两边比，用商数关系。',
      reveal: { q: '$\\sin\\theta=\\dfrac35,\\cos\\theta=\\dfrac45$，求 $\\tan\\theta$', a: '$\\tan\\theta=\\dfrac{3/5}{4/5}=\\dfrac34$' },
    },
  ],
  'u2-l2': [
    {
      title: '符号看象限',
      body: '三角函数的正负只取决于终边落在第几象限。一句口诀：**一全正、二正弦、三正切、四余弦**（谁为正）。',
    },
    {
      title: '怎么记',
      body: '点 $P=(\\cos\\theta,\\sin\\theta)$ 的横纵坐标符号，就决定了 $\\cos,\\sin$ 的符号；$\\tan=\\sin/\\cos$ 再据此定。',
      reveal: { q: '第三象限里 $\\tan\\theta$ 的符号？', a: '$\\sin<0,\\cos<0\\Rightarrow\\tan=\\dfrac{\\sin}{\\cos}>0$，为正' },
    },
  ],
  'u2-l3': [
    {
      title: '两步走',
      body: '已知一个三角函数值求其余：① 用**平方关系**求出配对的那个；② 用**象限**定正负号。',
      formula: '\\sin^2\\theta+\\cos^2\\theta=1',
    },
    {
      title: '小心符号',
      body: '只知道一个值、不知象限时，另一个值会有 $\\pm$ 两种可能，必须靠象限确定。',
      reveal: { q: '$\\cos\\theta=\\dfrac45$，$\\theta$ 是锐角，求 $\\sin\\theta$', a: '$\\sin\\theta=\\sqrt{1-16/25}=\\dfrac35$（锐角取正）' },
    },
  ],
  'u3-l1': [
    {
      title: '必背的特殊角',
      body: '$30^\\circ,45^\\circ,60^\\circ$ 的值是高频考点，建议记住下面这张表。',
      formula: '\\begin{array}{c|ccc}\\theta&30^\\circ&45^\\circ&60^\\circ\\\\\\hline \\sin&\\tfrac12&\\tfrac{\\sqrt2}{2}&\\tfrac{\\sqrt3}{2}\\\\ \\cos&\\tfrac{\\sqrt3}{2}&\\tfrac{\\sqrt2}{2}&\\tfrac12\\end{array}',
    },
    {
      title: '记忆窍门',
      body: '$\\sin$ 在 $30^\\circ\\to60^\\circ$ 依次是 $\\tfrac{\\sqrt1}{2},\\tfrac{\\sqrt2}{2},\\tfrac{\\sqrt3}{2}$（根号里 1,2,3 递增），$\\cos$ 正好倒过来。',
      reveal: { q: '$\\tan 60^\\circ=?$', a: '$\\dfrac{\\sin60^\\circ}{\\cos60^\\circ}=\\dfrac{\\sqrt3/2}{1/2}=\\sqrt3$' },
    },
  ],
  'u3-l2': [
    {
      title: '轴上的角',
      body: '$0^\\circ,90^\\circ,180^\\circ,270^\\circ$ 对应单位圆上的四个“正方向”点：$(1,0),(0,1),(-1,0),(0,-1)$。',
    },
    {
      title: '直接读坐标',
      body: '横坐标即 $\\cos$，纵坐标即 $\\sin$。注意 $\\cos90^\\circ=0$ 作分母，所以 $\\tan90^\\circ$ **无定义**。',
      reveal: { q: '$\\sin90^\\circ$ 与 $\\cos90^\\circ$？', a: '$\\sin90^\\circ=1,\\ \\cos90^\\circ=0$（点 $(0,1)$）' },
    },
  ],
  'u3-l3': [
    {
      title: '把任意角搬回锐角',
      body: '**诱导公式**让我们用参考角（锐角）算出任意角的值。常用：$\\sin(180^\\circ-\\theta)=\\sin\\theta$，$\\cos(-\\theta)=\\cos\\theta$。',
    },
    {
      title: '奇偶性',
      body: '正弦是奇函数 $\\sin(-\\theta)=-\\sin\\theta$；余弦是偶函数 $\\cos(-\\theta)=\\cos\\theta$。',
      reveal: { q: '用诱导公式求 $\\sin150^\\circ$', a: '$\\sin150^\\circ=\\sin(180^\\circ-30^\\circ)=\\sin30^\\circ=\\dfrac12$' },
    },
  ],
}
