import type { IntroCard } from '../../types'

// 学新知：第一部分（单元 1–3）各课的概念讲解。力求“讲清楚”：
// 动机 → 定义(配图) → 推导/直觉 → 例题精讲(分步) → 易错点 → 互动揭示。
export const INTRO1: Record<string, IntroCard[]> = {
  'u1-l1': [
    {
      title: '为什么要再学一套“弧度”',
      body:
        '我们从小用**角度**量角：一圈 $360^\\circ$。这个数来自古巴比伦的 60 进制，方便但有点“人为”。\n\n' +
        '进入高等数学后，**弧度制**会成为主角——因为它把“角”和“弧长”直接挂钩，让一大批公式变简洁（比如弧长 $l=r\\theta$、导数 $(\\sin x)\'=\\cos x$ 都依赖弧度）。',
    },
    {
      title: '弧度到底是什么',
      body:
        '在半径为 $r$ 的圆上，取一段**长度恰好等于半径**的弧，它所对的圆心角就规定为 **$1$ 弧度**。\n\n' +
        '所以弧度本质是一个**比值**：弧长除以半径，没有单位。',
      formula: '\\theta=\\dfrac{l}{r}\\quad(\\text{弧度})',
      image: 'trig-p1-01-1.svg',
      imageCaption: '单位圆（$r=1$）上，一圈特殊角的弧度与角度对照',
    },
    {
      title: '只需记住这一条换算',
      body:
        '半个圆的弧长是 $\\pi r$（半条周长），它对应的圆心角是 $180^\\circ$。\n\n' +
        '于是得到换算的“总钥匙”：',
      formula: '180^\\circ=\\pi\\ \\text{rad}\\;\\Longrightarrow\\;1^\\circ=\\dfrac{\\pi}{180},\\ \\ 1\\,\\text{rad}=\\dfrac{180^\\circ}{\\pi}',
      tip: '换算时不必背每个特殊角：把 $\\pi$ 直接当成 $180^\\circ$ 代进去，按比例缩放即可。',
    },
    {
      title: '例题精讲：$120^\\circ$ 化为弧度',
      body: '把“占半圈的比例”乘上 $\\pi$ 就行。',
      steps: [
        '$120^\\circ$ 占半圈 $180^\\circ$ 的比例：$\\dfrac{120}{180}=\\dfrac23$',
        '所以 $120^\\circ=\\dfrac23\\times\\pi=\\dfrac{2\\pi}{3}$',
      ],
    },
    {
      title: '反过来也一样',
      body: '弧度化角度，就把 $\\pi$ 换回 $180^\\circ$。先自己算算：',
      reveal: { q: '$\\dfrac{3\\pi}{4}$ 等于多少度？', a: '$\\dfrac{3\\pi}{4}=\\dfrac34\\times180^\\circ=135^\\circ$' },
    },
  ],

  'u1-l2': [
    {
      title: '把所有角“搬”到一个圆上',
      body:
        '三角函数原本是直角三角形里的边长比，但那样只能处理锐角。\n\n' +
        '**单位圆**（半径 $1$、圆心在原点）能把**任意角**统一起来：角的终边转到哪里，函数值就读哪里。',
    },
    {
      title: '核心：终边交点的坐标',
      body:
        '把角 $\\theta$ 的终边与单位圆的交点记作 $P$。规定：\n\n' +
        '$P$ 的**横坐标就是 $\\cos\\theta$**，**纵坐标就是 $\\sin\\theta$**。这就是 $\\sin,\\cos$ 的“单位圆定义”，对任意角都成立。',
      formula: 'P=(\\cos\\theta,\\ \\sin\\theta)',
      image: 'trig-p1-01-1.svg',
      imageCaption: '终边与单位圆交于 $P=(\\cos\\theta,\\sin\\theta)$',
    },
    {
      title: '一个“免费”得到的恒等式',
      body: '因为 $P$ 在单位圆上，它到原点的距离是 $1$，代入圆方程 $x^2+y^2=1$：',
      steps: [
        '$x=\\cos\\theta,\\ y=\\sin\\theta$',
        '代入 $x^2+y^2=1$ 得 $\\cos^2\\theta+\\sin^2\\theta=1$',
      ],
      formula: '\\sin^2\\theta+\\cos^2\\theta=1',
      tip: '这条**勾股恒等式**是后面无数变形的根，务必牢记它来自“单位圆上点的距离为 1”。',
    },
    {
      title: '读几个特殊位置',
      body: '终边指向四个正方向时，坐标一目了然。先想想：',
      reveal: { q: '$\\theta=\\dfrac{\\pi}{2}$（即 $90^\\circ$）时 $P$ 的坐标？', a: '$P=(\\cos90^\\circ,\\sin90^\\circ)=(0,1)$' },
    },
  ],

  'u1-l3': [
    {
      title: '弧度制让公式“去掉了系数”',
      body:
        '由弧度定义 $\\theta=\\dfrac{l}{r}$ 直接移项，就得到**弧长公式**——没有任何额外比例因子。',
      formula: 'l=r\\theta',
    },
    {
      title: '扇形面积',
      body:
        '扇形是整个圆的一部分，它占整圆的比例正好是 $\\dfrac{\\theta}{2\\pi}$。\n\n' +
        '用这个比例去乘整圆面积 $\\pi r^2$，化简即得：',
      steps: [
        '扇形面积 $=\\dfrac{\\theta}{2\\pi}\\times\\pi r^2$',
        '约分得 $S=\\dfrac12 r^2\\theta$',
      ],
      formula: 'S=\\dfrac12 r^2\\theta',
    },
    {
      title: '例题精讲',
      body: '半径 $3$、圆心角 $2$ 弧度的扇形，面积是多少？',
      steps: [
        '代入 $S=\\dfrac12 r^2\\theta$',
        '$S=\\dfrac12\\times3^2\\times2=\\dfrac12\\times9\\times2=9$',
      ],
      tip: '这两个公式**只在弧度制下成立**；若题目给的是角度，先换成弧度再代入。',
    },
    {
      title: '轮到你',
      body: '先自己算，再揭示。',
      reveal: { q: '半径 $2$、圆心角 $\\dfrac{\\pi}{2}$ 的弧长？', a: '$l=r\\theta=2\\times\\dfrac{\\pi}{2}=\\pi$' },
    },
  ],

  'u2-l1': [
    {
      title: '六个函数其实“同根生”',
      body:
        '三角函数不是六套互不相干的定义。$\\sin,\\cos$ 是“**主角**”（来自单位圆坐标或直角三角形边比），其余四个都由它们派生。',
    },
    {
      title: '直角三角形里的三个比',
      body:
        '在直角三角形中，相对于角 $\\theta$：\n\n' +
        '正弦 = 对边/斜边，余弦 = 邻边/斜边，正切 = 对边/邻边。口诀 **SOH-CAH-TOA**。',
      formula: '\\sin\\theta=\\dfrac{\\text{对}}{\\text{斜}},\\quad \\cos\\theta=\\dfrac{\\text{邻}}{\\text{斜}},\\quad \\tan\\theta=\\dfrac{\\text{对}}{\\text{邻}}',
      image: 'trig-p1-02-1.svg',
      imageCaption: '单位圆上，六个三角函数对应的线段',
    },
    {
      title: '另外三个：倒数与商数',
      body:
        '$\\tan$ 还能写成 $\\sin$ 比 $\\cos$（**商数关系**）；$\\sec,\\csc,\\cot$ 则分别是 $\\cos,\\sin,\\tan$ 的**倒数**。',
      formula: '\\tan\\theta=\\dfrac{\\sin\\theta}{\\cos\\theta},\\quad \\sec\\theta=\\dfrac{1}{\\cos\\theta},\\quad \\csc\\theta=\\dfrac{1}{\\sin\\theta},\\quad \\cot\\theta=\\dfrac{1}{\\tan\\theta}',
      tip: '记倒数别配错：$\\sec\\leftrightarrow\\cos$、$\\csc\\leftrightarrow\\sin$（首字母错位，容易搞反）。',
    },
    {
      title: '例题精讲',
      body: '已知 $\\sin\\theta=\\dfrac35,\\ \\cos\\theta=\\dfrac45$，求 $\\tan\\theta$。',
      steps: [
        '用商数关系 $\\tan\\theta=\\dfrac{\\sin\\theta}{\\cos\\theta}$',
        '$=\\dfrac{3/5}{4/5}=\\dfrac34$',
      ],
      reveal: { q: '那 $\\sec\\theta=?$', a: '$\\sec\\theta=\\dfrac{1}{\\cos\\theta}=\\dfrac{1}{4/5}=\\dfrac54$' },
    },
  ],

  'u2-l2': [
    {
      title: '符号只看“终边在哪个象限”',
      body:
        '三角函数的正负，完全由终边所在象限决定，因为它们就是单位圆上点的坐标（或其比值）。',
      image: 'trig-p1-03-1.svg',
      imageCaption: '四个象限里 $(\\cos\\theta,\\sin\\theta)$ 的符号',
    },
    {
      title: '一句口诀',
      body:
        '**“一全正、二正弦、三正切、四余弦”**——指每个象限里“为正”的是谁：\n\n' +
        '第一象限六个都正；第二象限只有 $\\sin$（和 $\\csc$）正；第三象限只有 $\\tan$（和 $\\cot$）正；第四象限只有 $\\cos$（和 $\\sec$）正。',
    },
    {
      title: '为什么第三象限 $\\tan>0$',
      body: '别死记，自己推：',
      steps: [
        '第三象限横纵坐标都为负：$\\cos\\theta<0,\\ \\sin\\theta<0$',
        '$\\tan\\theta=\\dfrac{\\sin\\theta}{\\cos\\theta}=\\dfrac{\\text{负}}{\\text{负}}>0$',
      ],
      tip: '“正切看同号”：$\\sin,\\cos$ 同号则 $\\tan>0$（一、三象限），异号则 $\\tan<0$（二、四象限）。',
    },
    {
      title: '判一判',
      body: '先想想：',
      reveal: { q: '$\\cos\\theta<0$ 且 $\\sin\\theta<0$，$\\theta$ 在第几象限？', a: '横纵都负 → 第三象限' },
    },
  ],

  'u2-l3': [
    {
      title: '“知一求多”的两步法',
      body:
        '已知一个三角函数值，求其余的，套路固定为两步：\n\n' +
        '① 用**平方关系**求出与它配对的那个（$\\sin\\leftrightarrow\\cos$）；② 用**象限**确定正负号。',
      formula: '\\sin^2\\theta+\\cos^2\\theta=1',
    },
    {
      title: '例题精讲',
      body: '已知 $\\cos\\theta=\\dfrac45$，且 $\\theta$ 为锐角，求 $\\sin\\theta$。',
      steps: [
        '由平方关系 $\\sin^2\\theta=1-\\cos^2\\theta=1-\\dfrac{16}{25}=\\dfrac{9}{25}$',
        '$\\sin\\theta=\\pm\\dfrac35$',
        '$\\theta$ 是锐角（第一象限）$\\Rightarrow \\sin\\theta>0$，取 $\\sin\\theta=\\dfrac35$',
      ],
    },
    {
      title: '最容易丢分的地方',
      body: '开平方会带来 $\\pm$ 两个候选，**必须靠象限筛掉一个**。',
      tip: '只给函数值、不给象限时，答案常常是 $\\pm$ 两解——别漏写另一种情况。',
      reveal: { q: '$\\cos\\theta=-\\dfrac{12}{13}$，$\\theta$ 在第三象限，$\\sin\\theta=?$', a: '$\\sin\\theta=-\\sqrt{1-\\tfrac{144}{169}}=-\\dfrac{5}{13}$（三象限取负）' },
    },
  ],

  'u3-l1': [
    {
      title: '为什么要背特殊角',
      body:
        '$30^\\circ,45^\\circ,60^\\circ$ 是考试与后续推导中出现最频繁的角。把它们的值记成“肌肉记忆”，能省下大量时间。',
    },
    {
      title: '一张表搞定',
      body: '把 $\\sin,\\cos$ 在三个特殊角的值列出来：',
      formula:
        '\\begin{array}{c|ccc}\\theta&30^\\circ&45^\\circ&60^\\circ\\\\\\hline \\sin&\\tfrac12&\\tfrac{\\sqrt2}{2}&\\tfrac{\\sqrt3}{2}\\\\[2pt] \\cos&\\tfrac{\\sqrt3}{2}&\\tfrac{\\sqrt2}{2}&\\tfrac12\\end{array}',
    },
    {
      title: '记忆窍门',
      body:
        '把 $\\sin$ 的三个值写成 $\\dfrac{\\sqrt1}{2},\\dfrac{\\sqrt2}{2},\\dfrac{\\sqrt3}{2}$——根号里 **1、2、3 递增**就好记！\n\n' +
        '$\\cos$ 正好把这串倒过来（3、2、1）。$\\tan=\\dfrac{\\sin}{\\cos}$ 现推即可。',
      tip: '不必把 $\\tan$ 也背下来：$\\tan60^\\circ=\\dfrac{\\sqrt3/2}{1/2}=\\sqrt3$，现场两步就能得到。',
    },
    {
      title: '现推一个',
      body: '试试不查表：',
      reveal: { q: '$\\tan 30^\\circ=?$', a: '$\\dfrac{\\sin30^\\circ}{\\cos30^\\circ}=\\dfrac{1/2}{\\sqrt3/2}=\\dfrac{1}{\\sqrt3}=\\dfrac{\\sqrt3}{3}$' },
    },
  ],

  'u3-l2': [
    {
      title: '坐标轴上的四个角',
      body:
        '$0^\\circ,90^\\circ,180^\\circ,270^\\circ$ 的终边正好指向坐标轴，对应单位圆上四个“整点”：$(1,0),(0,1),(-1,0),(0,-1)$。',
      image: 'trig-p1-01-1.svg',
      imageCaption: '单位圆上四个轴向位置',
    },
    {
      title: '直接读坐标',
      body:
        '横坐标即 $\\cos$，纵坐标即 $\\sin$。例如 $90^\\circ$ 处是 $(0,1)$，所以 $\\cos90^\\circ=0,\\ \\sin90^\\circ=1$。',
      tip: '$\\tan90^\\circ$ **无定义**！因为 $\\tan=\\dfrac{\\sin}{\\cos}$，而 $\\cos90^\\circ=0$ 不能作分母。',
    },
    {
      title: '小结一圈',
      body:
        '$\\sin$：$0\\to1\\to0\\to-1$（在 $0,90,180,270^\\circ$）；$\\cos$：$1\\to0\\to-1\\to0$。两者正好“错开 $90^\\circ$”。',
      reveal: { q: '$\\cos180^\\circ=?$', a: '点是 $(-1,0)$，故 $\\cos180^\\circ=-1$' },
    },
  ],

  'u3-l3': [
    {
      title: '诱导公式：把任意角搬回锐角',
      body:
        '任意角的三角值，都能通过对称关系，化到一个熟悉的**锐角（参考角）**上来算。这组对称关系就叫**诱导公式**。',
      image: 'trig-p1-03-1.svg',
      imageCaption: '关于 $x$ 轴 / $y$ 轴 / 原点的对称，决定了符号变化',
    },
    {
      title: '两类最常用',
      body:
        '**补角**（关于 $y$ 轴对称）：$\\sin(180^\\circ-\\theta)=\\sin\\theta$，$\\cos(180^\\circ-\\theta)=-\\cos\\theta$。\n\n' +
        '**奇偶性**（关于 $x$ 轴对称）：$\\sin(-\\theta)=-\\sin\\theta$（奇），$\\cos(-\\theta)=\\cos\\theta$（偶）。',
    },
    {
      title: '例题精讲：求 $\\sin150^\\circ$',
      body: '把 $150^\\circ$ 看成补角形式。',
      steps: [
        '$150^\\circ=180^\\circ-30^\\circ$',
        '由 $\\sin(180^\\circ-\\theta)=\\sin\\theta$，得 $\\sin150^\\circ=\\sin30^\\circ$',
        '$=\\dfrac12$',
      ],
      tip: '口诀“**奇变偶不变，符号看象限**”：函数名是否改变看 $\\dfrac{\\pi}{2}$ 的奇偶倍，符号看原角所在象限。',
    },
    {
      title: '试一试',
      body: '先自己化简：',
      reveal: { q: '$\\cos120^\\circ=?$', a: '$\\cos(180^\\circ-60^\\circ)=-\\cos60^\\circ=-\\dfrac12$' },
    },
  ],
}
