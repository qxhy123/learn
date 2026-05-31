import type { IntroCard } from '../../types'

// 学新知：第六部分（单元 16–18）各课的概念讲解卡片，按 lesson id 索引。
// 振幅·相位·谐波合成 / 三角换元 / 三角多项式与万能代换。
// TS 字符串内 LaTeX 反斜杠一律双写（\\sin \\cos \\tan \\frac \\sqrt \\theta \\varphi 等）。
export const INTRO6: Record<string, IntroCard[]> = {
  // ============================== 单元 16 ==============================
  'u16-l1': [
    {
      title: '动机：把两个波压成一个波',
      body: '弹簧的位移、交流电的电压、声音的波形……现实里的周期量常常写成 $a\\sin x+b\\cos x$ 这样**两项之和**的样子。表面上它像两个独立的波叠在一起，看不出振幅，也读不出从哪里出发。\n\n但只要 $a,b$ 都是常数，这个和**永远**可以重新打包成**单独一个**正弦波 $R\\sin(x+\\varphi)$。$R$ 叫**振幅**（波有多高），$\\varphi$ 叫**初相**（波从哪儿起步）。一旦压成这个形状，值域、最值、相位全都一目了然——这就是「辅助角公式」要做的事。',
      formula: 'a\\sin x+b\\cos x=R\\sin(x+\\varphi)',
      image: 'trig-p6-16-1.svg',
      imageCaption: '向量合成解释：把 $(a,b)$ 看成向量，模长就是振幅 $R$、辐角就是初相 $\\varphi$',
    },
    {
      title: '振幅 $R$ 从哪里来',
      body: '推导只需一步「展开 + 对比系数」。把右边 $R\\sin(x+\\varphi)$ 用和角公式拆开，再和左边逐项对齐：',
      steps: [
        '展开右边：$R\\sin(x+\\varphi)=R\\cos\\varphi\\,\\sin x+R\\sin\\varphi\\,\\cos x$',
        '对比左边 $a\\sin x+b\\cos x$ 的系数：$R\\cos\\varphi=a,\\ R\\sin\\varphi=b$',
        '两式平方相加：$R^2\\cos^2\\varphi+R^2\\sin^2\\varphi=a^2+b^2$',
        '左边 $=R^2(\\cos^2\\varphi+\\sin^2\\varphi)=R^2$，于是 $R=\\sqrt{a^2+b^2}$（取正根）',
      ],
      formula: 'R=\\sqrt{a^2+b^2}',
    },
    {
      title: '几何解释：振幅就是向量长度',
      body: '别把这条公式只当死记的口诀。把 $(a,b)$ 画成平面上的一个**向量**：\n\n它的**长度**恰好是振幅 $R=\\sqrt{a^2+b^2}$（勾股定理），它的**辐角**恰好是初相 $\\varphi$。于是「正弦项 + 余弦项」被理解成「一个向量的两个坐标分量」，合成不过是把分量还原成这个向量本身。',
      tip: '$R$ 一定取**正**根。如果算出来带负号，说明你把 $\\sin$ 与 $\\cos$ 的系数对应弄反了——回头检查 $R\\cos\\varphi=a$（不是 $b$）。',
    },
    {
      title: '例题精讲：求 $3\\sin x+4\\cos x$ 的振幅',
      body: '最经典的「3-4-5」型，体会一下公式怎么落地。',
      steps: [
        '识别系数：$a=3$（$\\sin$ 前），$b=4$（$\\cos$ 前）',
        '套振幅公式：$R=\\sqrt{a^2+b^2}=\\sqrt{3^2+4^2}$',
        '计算：$\\sqrt{9+16}=\\sqrt{25}=5$',
        '所以 $3\\sin x+4\\cos x=5\\sin(x+\\varphi)$，振幅是 $5$',
      ],
      tip: '$R$ 与 $\\sin,\\cos$ 的**顺序无关**：$3\\sin x+4\\cos x$ 和 $4\\sin x+3\\cos x$ 振幅都是 $5$，因为 $\\sqrt{a^2+b^2}$ 对调 $a,b$ 不变。',
    },
    {
      title: '动手想一想',
      body: '现在轮到你。先在心里算 $R=\\sqrt{a^2+b^2}$，再点开看答案。',
      reveal: {
        q: '$5\\sin x-12\\cos x$ 的振幅 $R=?$（注意系数带负号）',
        a: '$R=\\sqrt{5^2+(-12)^2}=\\sqrt{25+144}=\\sqrt{169}=13$。负号在平方后消失，所以振幅仍是正数 $13$。',
      },
    },
  ],
  'u16-l2': [
    {
      title: '动机：振幅好求，初相才是关键',
      body: '上节我们知道 $R=\\sqrt{a^2+b^2}$ 一步可得。可真正要把式子写完整，还差一个**初相** $\\varphi$。\n\n麻烦在于：$\\varphi$ 不能只靠一个三角值定下来。比如 $\\cos\\varphi=\\dfrac12$ 既可能是 $\\dfrac{\\pi}{3}$，也可能是 $-\\dfrac{\\pi}{3}$。要锁定唯一的 $\\varphi$，必须同时盯住 $\\cos\\varphi$ 和 $\\sin\\varphi$ 两个符号——这正是本节的核心。',
      formula: '\\cos\\varphi=\\frac{a}{R},\\qquad \\sin\\varphi=\\frac{b}{R}',
    },
    {
      title: '方法：用两个符号定象限',
      body: '由 $R\\cos\\varphi=a,\\ R\\sin\\varphi=b$ 得 $\\cos\\varphi=\\dfrac aR,\\ \\sin\\varphi=\\dfrac bR$。$\\cos\\varphi$ 的符号跟着 $a$ 走，$\\sin\\varphi$ 的符号跟着 $b$ 走，两者一起把 $\\varphi$ 锁进唯一象限：',
      steps: [
        '$a>0,\\,b>0$：两值皆正，$\\varphi$ 在**第一象限**',
        '$a<0,\\,b>0$：$\\cos\\varphi<0,\\sin\\varphi>0$，$\\varphi$ 在**第二象限**',
        '$a<0,\\,b<0$：两值皆负，$\\varphi$ 在**第三象限**',
        '$a>0,\\,b<0$：$\\cos\\varphi>0,\\sin\\varphi<0$，$\\varphi$ 在**第四象限**',
      ],
      tip: '只看 $\\cos\\varphi$（或只看 $\\tan\\varphi$）会丢掉象限信息！$\\tan\\varphi=\\dfrac ba$ 在相差 $\\pi$ 的两个角处相同，单凭它会把 $\\varphi$ 定错半圈。',
    },
    {
      title: '例题精讲：合成 $\\sqrt3\\sin t+\\cos t$',
      body: '把这个表达式化成 $R\\sin(t+\\varphi)$ 的完整形式。',
      steps: [
        '识别：$a=\\sqrt3,\\ b=1$，且都为正',
        '求振幅：$R=\\sqrt{(\\sqrt3)^2+1^2}=\\sqrt{3+1}=2$',
        '求初相分量：$\\cos\\varphi=\\dfrac{\\sqrt3}{2},\\ \\sin\\varphi=\\dfrac12$',
        '两值皆正 → 第一象限，标准角 $\\varphi=\\dfrac{\\pi}{6}$',
        '结论：$\\sqrt3\\sin t+\\cos t=2\\sin\\!\\left(t+\\dfrac{\\pi}{6}\\right)$',
      ],
      formula: '\\sqrt3\\sin t+\\cos t=2\\sin\\left(t+\\frac{\\pi}{6}\\right)',
    },
    {
      title: '易错点：第四象限取负角更自然',
      body: '当 $b<0$ 时，$\\sin\\varphi<0$，$\\varphi$ 落在第四象限。这时**取负角**（如 $-\\dfrac{\\pi}{3}$）往往比取 $\\dfrac{5\\pi}{3}$ 更简洁，写出的合成式也更好看。',
      tip: '别把 $\\cos\\varphi=\\dfrac12,\\ \\sin\\varphi=-\\dfrac{\\sqrt3}{2}$ 错认成 $\\dfrac{\\pi}{3}$——那是第一象限！正负号告诉你它在第四象限，应取 $\\varphi=-\\dfrac{\\pi}{3}$。',
    },
    {
      title: '动手想一想',
      body: '系数带负号时一定先定象限再取角。试试这道。',
      reveal: {
        q: '把 $\\sin x-\\sqrt3\\cos x$ 化成 $R\\sin(x+\\varphi)$，$R$ 与 $\\varphi$ 各是多少？',
        a: '$R=\\sqrt{1+3}=2$；$\\cos\\varphi=\\dfrac12>0,\\ \\sin\\varphi=-\\dfrac{\\sqrt3}{2}<0$ → 第四象限，取 $\\varphi=-\\dfrac{\\pi}{3}$。故 $=2\\sin\\!\\left(x-\\dfrac{\\pi}{3}\\right)$。',
      },
    },
  ],
  'u16-l3': [
    {
      title: '动机：合成之后，答案几乎是「读」出来的',
      body: '把 $a\\sin\\omega x+b\\cos\\omega x$ 压成 $R\\sin(\\omega x+\\varphi)$ 之后，许多原本要费劲求的量——最大值、最小值、值域、周期——都变成**直接读取**。\n\n这就是辅助角公式真正的威力：它不止是化简，更是把一个「看不透」的表达式翻译成「一眼看穿」的标准波形。',
      formula: 'a\\sin\\omega x+b\\cos\\omega x=R\\sin(\\omega x+\\varphi)',
    },
    {
      title: '方法：最值与周期一眼读出',
      body: '因为 $\\sin$ 的取值永远在 $[-1,1]$，把它乘上振幅 $R$ 后：\n\n最大值就是 $R$，最小值就是 $-R$，值域就是 $[-R,R]$。而周期只看 $x$ 前面的系数 $\\omega$，与 $R,\\varphi$ 都无关。',
      formula: '\\max=R,\\quad \\min=-R,\\quad T=\\frac{2\\pi}{\\omega}',
      tip: '别把**值域**和**振幅**混为一谈：振幅 $R$ 是「半高」，值域 $[-R,R]$ 的总跨度是 $2R$。说「最大值是 $R$」对，说「值域是 $R$」就错了。',
    },
    {
      title: '例题精讲：求 $y=3\\sin x+4\\cos x$ 的最值与周期',
      body: '把上节的合成结果直接拿来读。',
      steps: [
        '合成：$R=\\sqrt{3^2+4^2}=5$，得 $y=5\\sin(x+\\varphi)$',
        '最大值 $=R=5$（当 $\\sin(x+\\varphi)=1$ 时取得）',
        '最小值 $=-R=-5$（当 $\\sin(x+\\varphi)=-1$ 时取得）',
        '周期：$x$ 的系数 $\\omega=1$，故 $T=\\dfrac{2\\pi}{1}=2\\pi$',
      ],
      tip: '若题目限定 $x$ 在某区间（如 $[0,\\pi]$），最值不一定取到 $\\pm R$！要把 $x$ 的范围平移成相位 $x+\\varphi$ 的范围，再看 $\\sin$ 能否到达峰值 $\\pm1$，到不了就比较端点。',
    },
    {
      title: '物理意义：简谐叠加',
      body: '简谐运动写成 $x(t)=A\\sin(\\omega t+\\varphi)$——这里 $A$ 是振幅、$\\omega$ 是角频率、$\\varphi$ 是初相，和辅助角公式**完全同构**。\n\n所以辅助角公式不是「竞赛技巧」，而是周期现象建模的核心语法：**两个同频率的简谐振动叠加，结果仍是同频率的简谐振动**，频率不变，只是振幅和初相重新组合。',
    },
    {
      title: '动手想一想',
      body: '两个正交分量（相位差恰为 $\\dfrac{\\pi}{2}$）叠加，合振幅满足勾股关系。',
      reveal: {
        q: '$3\\sin\\omega t$ 与 $4\\cos\\omega t$ 叠加，合振动的振幅是多少？',
        a: '$\\cos\\omega t=\\sin\\!\\left(\\omega t+\\dfrac{\\pi}{2}\\right)$，两分量相位差 $\\dfrac{\\pi}{2}$（正交），合振幅 $R=\\sqrt{3^2+4^2}=5$。这正是勾股式 $R=\\sqrt{A_1^2+A_2^2}$。',
      },
    },
  ],

  // ============================== 单元 17 ==============================
  'u17-l1': [
    {
      title: '动机：让三角函数去「吃掉」根号',
      body: '碰到 $\\sqrt{a^2-x^2}$ 这种**根式**，直接积分或化简往往束手无策——根号下是个减法，开不出来。\n\n但单位圆上有两条恒等式：$1-\\sin^2\\theta=\\cos^2\\theta$ 和 $1+\\tan^2\\theta=\\sec^2\\theta$。它们能把「平方差 / 平方和」变成**完全平方**，于是根号顺势开出。换言之，我们用一个三角函数去替换 $x$，让恒等式替我们「吃掉」根号。这就是三角换元。',
      formula: '\\sqrt{a^2-x^2}\\ \\xrightarrow{\\,x=a\\sin\\theta\\,}\\ a\\cos\\theta',
      image: 'trig-p6-17-1.svg',
      imageCaption: '直角三角形解释 $x=a\\sin\\theta$：斜边 $a$、对边 $x$、邻边 $\\sqrt{a^2-x^2}=a\\cos\\theta$',
    },
    {
      title: '方法：三类经典代换',
      body: '看根号下的结构「对号入座」，记忆口诀是**减号配 $\\sin$，加号配 $\\tan$，$x^2-a^2$ 配 $\\sec$**：',
      steps: [
        '$\\sqrt{a^2-x^2}$ → 令 $x=a\\sin\\theta$，靠 $1-\\sin^2\\theta=\\cos^2\\theta$，得 $a\\cos\\theta$',
        '$\\sqrt{a^2+x^2}$ → 令 $x=a\\tan\\theta$，靠 $1+\\tan^2\\theta=\\sec^2\\theta$，得 $a\\sec\\theta$',
        '$\\sqrt{x^2-a^2}$ → 令 $x=a\\sec\\theta$，靠 $\\sec^2\\theta-1=\\tan^2\\theta$，得 $a\\tan\\theta$',
      ],
      formula: '\\begin{aligned}\\sqrt{a^2-x^2}&:\\ x=a\\sin\\theta\\\\ \\sqrt{a^2+x^2}&:\\ x=a\\tan\\theta\\\\ \\sqrt{x^2-a^2}&:\\ x=a\\sec\\theta\\end{aligned}',
    },
    {
      title: '例题精讲：化简 $\\sqrt{a^2-x^2}$',
      body: '逐步看「减号配 $\\sin$」如何把根号化掉。',
      steps: [
        '令 $x=a\\sin\\theta$，则 $x^2=a^2\\sin^2\\theta$',
        '代入：$\\sqrt{a^2-a^2\\sin^2\\theta}=\\sqrt{a^2(1-\\sin^2\\theta)}$',
        '用恒等式 $1-\\sin^2\\theta=\\cos^2\\theta$：$=\\sqrt{a^2\\cos^2\\theta}$',
        '开方：$=a\\,|\\cos\\theta|=a\\cos\\theta$（在 $\\cos\\theta\\ge0$ 的范围内，详见下节）',
      ],
      tip: '为什么不选 $x=a\\cos\\theta$？也行，但 $\\sin$ 求导更顺手、$\\arcsin$ 是常用主值，所以减号型默认配 $\\sin$。',
    },
    {
      title: '易错点：选错代换 = 根号化不掉',
      body: '三类结构对应三种代换，对错全在「正负号」。把 $\\sqrt{a^2+x^2}$（加号）误用 $x=a\\sin\\theta$，会得到 $a^2+a^2\\sin^2\\theta$，**没有**恒等式能把它变成完全平方，根号纹丝不动。',
      tip: '判断口诀：根号里**先**看是「常数减 $x^2$」「常数加 $x^2$」还是「$x^2$ 减常数」，**再**对号入座。顺序别记反——$\\sqrt{x^2-a^2}$（$x$ 在前）才配 $\\sec$。',
    },
    {
      title: '动手想一想',
      body: '加号结构该配哪种代换？算算看。',
      reveal: {
        q: '设 $x=5\\tan\\theta$，$\\sqrt{25+x^2}=?$',
        a: '$25+25\\tan^2\\theta=25(1+\\tan^2\\theta)=25\\sec^2\\theta$，开方得 $5\\sec\\theta$。加号 → 配 $\\tan$，根号被 $1+\\tan^2\\theta=\\sec^2\\theta$ 吸收。',
      },
    },
  ],
  'u17-l2': [
    {
      title: '动机：开方不是「脱帽子」那么简单',
      body: '上节我们顺手把 $\\sqrt{\\cos^2\\theta}$ 写成 $\\cos\\theta$。但这一步其实**藏着陷阱**。\n\n开平方的严格结果是**绝对值**：$\\sqrt{\\cos^2\\theta}=|\\cos\\theta|$，不是 $\\cos\\theta$。只有当 $\\cos\\theta\\ge0$ 时两者才相等。这是整个三角换元里最常丢分的地方——本节专门把这个坑填平。',
      formula: '\\sqrt{\\cos^2\\theta}=|\\cos\\theta|',
    },
    {
      title: '方法：先写绝对值，再用范围定号',
      body: '养成两步走的习惯，符号永不出错：',
      steps: [
        '第一步——开方先写绝对值：$\\sqrt{\\cos^2\\theta}=|\\cos\\theta|$，绝不直接去掉',
        '第二步——查代换时约定的 $\\theta$ 范围，判断 $\\cos\\theta$ 是正是负',
        '若该范围内 $\\cos\\theta\\ge0$，才把 $|\\cos\\theta|$ 写成 $\\cos\\theta$；若 $<0$ 则写 $-\\cos\\theta$',
      ],
      tip: '这就是为什么 $x=a\\sin\\theta$ 通常约定 $\\theta\\in\\left[-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right]$——在这个区间上 $\\cos\\theta\\ge0$，绝对值才能干净脱掉。',
    },
    {
      title: '例题精讲：$x=2\\sin t$ 时化简 $\\sqrt{4-x^2}$',
      body: '完整走一遍「先绝对值、再定号」。',
      steps: [
        '代入 $x=2\\sin t$：$\\sqrt{4-4\\sin^2t}=\\sqrt{4(1-\\sin^2t)}=\\sqrt{4\\cos^2t}$',
        '开方写绝对值：$=2|\\cos t|$（**不要**急着写 $2\\cos t$）',
        '查范围：约定 $t\\in\\left[-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right]$，此区间 $\\cos t\\ge0$',
        '去绝对值：$|\\cos t|=\\cos t$，故 $\\sqrt{4-x^2}=2\\cos t$',
      ],
      formula: '\\sqrt{4-x^2}=2|\\cos t|\\xrightarrow{\\,\\cos t\\ge0\\,}2\\cos t',
    },
    {
      title: '易错点：不限范围就乱脱绝对值',
      body: '若题目没约定 $t$ 的范围，或 $t$ 落到 $\\cos t<0$ 的区间，硬写 $|\\cos t|=\\cos t$ 就会得到**带错符号**的结果，整道题随之崩盘。',
      tip: '$\\sqrt{x^2}=|x|$ 这条铁律对所有变量都成立。换元时第一件事就是**白纸黑字写下 $\\theta$（或 $t$）的取值范围**，后面去绝对值才有依据。',
    },
    {
      title: '动手想一想',
      body: '$\\sec$ 代换里同样要先写绝对值。试试这道化简。',
      reveal: {
        q: '已知 $x=\\sec t,\\ t\\in\\left[0,\\dfrac{\\pi}{2}\\right)$，化简 $\\dfrac{\\sqrt{x^2-1}}{x}$。',
        a: '$\\sqrt{x^2-1}=\\sqrt{\\sec^2t-1}=|\\tan t|=\\tan t$（该区间 $\\tan t\\ge0$）。于是 $\\dfrac{\\tan t}{\\sec t}=\\dfrac{\\sin t/\\cos t}{1/\\cos t}=\\sin t$。',
      },
    },
  ],
  'u17-l3': [
    {
      title: '动机：根式积分的「万能钥匙」',
      body: '$\\displaystyle\\int\\frac{dx}{\\sqrt{9-x^2}}$ 这类带根号的积分，硬算无从下手。但换元之后，根号被三角恒等式化掉，分子分母里的因子常常**整片约掉**，积分瞬间变成最简单的 $\\int dt$。\n\n本节把换元、求 $dx$、积分、回代串成完整流程，你会看到根式问题如何被「翻译」成三角恒等式问题。',
      formula: '\\int\\frac{dx}{\\sqrt{9-x^2}}\\xrightarrow{\\,x=3\\sin t\\,}\\int dt=t+C',
    },
    {
      title: '方法：换元四步走',
      body: '换元积分不止换 $x$，还要换 $dx$、换被积式、最后回代或换限：',
      steps: [
        '① 选代换：识别根式结构选 $\\sin/\\tan/\\sec$，写下 $\\theta$ 范围',
        '② 求微分：如 $x=3\\sin t\\Rightarrow dx=3\\cos t\\,dt$（**别漏了 $dx$**）',
        '③ 代入化简：根号与微分里的因子约掉，化成纯三角积分',
        '④ 收尾：不定积分**回代**回 $x$；定积分则**换上下限**，省去回代',
      ],
      tip: '最常见的丢分点是**忘记换 $dx$**：只把 $x$ 换成 $3\\sin t$、却把 $dx$ 原样留着，结果必错。',
    },
    {
      title: '例题精讲：$\\displaystyle\\int\\frac{dx}{\\sqrt{9-x^2}}$',
      body: '完整走一遍四步流程。',
      steps: [
        '识别 $\\sqrt{9-x^2}$（减号型），令 $x=3\\sin t$，则 $dx=3\\cos t\\,dt$',
        '根号化简：$\\sqrt{9-9\\sin^2t}=3\\cos t$（取 $t\\in\\left[-\\dfrac{\\pi}{2},\\dfrac{\\pi}{2}\\right]$）',
        '代入：$\\displaystyle\\int\\frac{3\\cos t}{3\\cos t}\\,dt=\\int dt=t+C$，因子整片约掉',
        '回代：由 $x=3\\sin t$ 得 $t=\\arcsin\\dfrac{x}{3}$，故积分 $=\\arcsin\\dfrac{x}{3}+C$',
      ],
      formula: '\\int\\frac{dx}{\\sqrt{9-x^2}}=\\arcsin\\frac{x}{3}+C',
    },
    {
      title: '几何意义：根式就是三角形的一条边',
      body: '$x=a\\sin\\theta$ 相当于在直角三角形里把 $x$ 当**对边**、$a$ 当**斜边**，于是 $\\sin\\theta=\\dfrac xa$。由勾股定理，**邻边**恰是 $\\sqrt{a^2-x^2}=a\\cos\\theta$。\n\n根号不是被「消去」，而是变成了三角形的一条边。回代时画一个这样的三角形，比硬套反函数更不容易出错。',
      tip: '定积分换限可省回代：$x=0\\Rightarrow t=0$，$x=a\\Rightarrow t=\\dfrac{\\pi}{2}$。换限后直接代数值，既简洁又避免回代写错。',
    },
    {
      title: '动手想一想',
      body: '换元后这个定积分有个漂亮的几何答案。',
      reveal: {
        q: '$\\displaystyle\\int_0^a\\sqrt{a^2-x^2}\\,dx=?$ 它代表什么图形的面积？',
        a: '令 $x=a\\sin t$ 算得 $\\dfrac{\\pi a^2}{4}$。因为 $y=\\sqrt{a^2-x^2}$ 是上半圆，$x$ 从 $0$ 到 $a$ 正好扫出半径 $a$ 的**四分之一圆面积** $\\dfrac14\\pi a^2$。',
      },
    },
  ],

  // ============================== 单元 18 ==============================
  'u18-l1': [
    {
      title: '动机：把 $\\cos 3x$ 拆成 $\\cos x$ 的多项式',
      body: '我们已经会二倍角，那么三倍角呢？$\\cos 3x$ 能不能只用 $\\cos x$ 表示？\n\n答案是能，而且结果格外漂亮：$\\cos 3x=4\\cos^3 x-3\\cos x$，是 $\\cos x$ 的一个**三次多项式**。这类「把多倍角化成单角多项式」的恒等式，是连接三角与代数的又一座桥，也是后面解高次三角方程、做降幂积分的基础。',
      formula: '\\cos 3x=4\\cos^3 x-3\\cos x,\\qquad \\sin 3x=3\\sin x-4\\sin^3 x',
      image: 'trig-p6-18-1.svg',
      imageCaption: '$y=\\cos 3x$ 与 $y=4\\cos^3 x-3\\cos x$ 两条曲线完全重合，可视化恒等式 $T_3(\\cos x)=\\cos 3x$',
    },
    {
      title: '方法：从二倍角推三倍角',
      body: '三倍角不用硬背，把 $3x$ 拆成 $2x+x$ 现推即可：',
      steps: [
        '$\\cos 3x=\\cos(2x+x)=\\cos 2x\\cos x-\\sin 2x\\sin x$',
        '代入 $\\cos 2x=2\\cos^2x-1$，$\\sin 2x=2\\sin x\\cos x$',
        '$=(2\\cos^2x-1)\\cos x-2\\sin^2x\\cos x$',
        '把 $\\sin^2x=1-\\cos^2x$ 代入第二项并合并',
        '$=2\\cos^3x-\\cos x-2(1-\\cos^2x)\\cos x=4\\cos^3x-3\\cos x$',
      ],
      tip: '$\\sin 3x=3\\sin x-4\\sin^3x$ 与 $\\cos 3x$ 形式对称但**系数符号相反**（$\\cos$ 是 $4u^3-3u$，$\\sin$ 是 $3u-4u^3$），别张冠李戴。',
    },
    {
      title: '它是 Chebyshev 多项式',
      body: '令 $u=\\cos x$，则 $\\cos 3x=4u^3-3u$ 干脆就是 $u$ 的三次多项式，记作 $T_3(u)$。\n\n一般地，$\\cos(nx)$ **总能**写成 $\\cos x$ 的 $n$ 次多项式 $T_n(\\cos x)$，这族多项式叫 **Chebyshev 多项式**。于是「求 $\\cos nx$」就等价于「把 $T_n$ 在 $u=\\cos x$ 处求值」，三角问题被翻译成多项式问题。',
      reveal: {
        q: '$8\\cos^3 x-6\\cos x$ 能化简成什么？',
        a: '提取系数 $2$：$8\\cos^3x-6\\cos x=2(4\\cos^3x-3\\cos x)=2\\cos 3x$。识破这一步，三次三角式立刻塌缩成一次的 $\\cos 3x$。',
      },
    },
    {
      title: '反过来用：降幂公式',
      body: '三倍角是「升幂」（把 $\\cos^3$ 表成 $\\cos 3x$），有时我们要反着来——把高次幂**降**成一次，便于积分。\n\n由 $\\cos 2t=2\\cos^2t-1$ 反解即得降幂公式。$\\cos^2,\\sin^2$ 一旦降成关于 $\\cos 2t$ 的一次式，就能逐项直接积分。',
      formula: '\\cos^2 t=\\frac{1+\\cos 2t}{2},\\qquad \\sin^2 t=\\frac{1-\\cos 2t}{2}',
      tip: '升幂还是降幂，看你要解方程还是要积分：解 $\\cos 3x=\\tfrac12$ 这类用升幂识别；算 $\\int\\cos^2t\\,dt$ 这类用降幂铺平。',
    },
  ],
  'u18-l2': [
    {
      title: '动机：把整个三角世界变成有理式',
      body: '当一个式子**同时**含 $\\sin x$ 和 $\\cos x$，还是它们的分式组合时，倍角、半角、和差都不一定凑效。有没有一招通吃？\n\n有。令 $t=\\tan\\dfrac{x}{2}$，神奇的事发生了：$\\sin x,\\cos x,\\tan x$ **全部**变成 $t$ 的有理式。于是含三角的方程变成多项式方程、含三角的积分变成有理函数积分——这就是「万能代换」（Weierstrass 代换）。',
      formula: 't=\\tan\\frac{x}{2}',
    },
    {
      title: '方法：三条核心公式',
      body: '把 $\\sin x,\\cos x$ 用半角写开，再统一成 $t$。三条公式务必背熟（分母都是 $1+t^2$）：',
      steps: [
        '$\\sin x=\\dfrac{2t}{1+t^2}$（分子 $2t$）',
        '$\\cos x=\\dfrac{1-t^2}{1+t^2}$（分子 $1-t^2$）',
        '$\\tan x=\\dfrac{\\sin x}{\\cos x}=\\dfrac{2t}{1-t^2}$（注意分母是 $1-t^2$，不是 $1+t^2$）',
      ],
      formula: '\\sin x=\\frac{2t}{1+t^2},\\quad \\cos x=\\frac{1-t^2}{1+t^2},\\quad \\tan x=\\frac{2t}{1-t^2}',
    },
    {
      title: '例题精讲：别忘了换 $dx$',
      body: '做积分时，$dx$ 也得用 $t$ 表示，否则积分根本算不下去。',
      steps: [
        '由 $t=\\tan\\dfrac{x}{2}$，对 $x$ 微分：$dt=\\dfrac12\\sec^2\\dfrac{x}{2}\\,dx$',
        '而 $\\sec^2\\dfrac{x}{2}=1+\\tan^2\\dfrac{x}{2}=1+t^2$',
        '于是 $dt=\\dfrac{1+t^2}{2}\\,dx$，反解得 $dx=\\dfrac{2}{1+t^2}\\,dt$',
      ],
      formula: 'dx=\\frac{2}{1+t^2}\\,dt',
      tip: '$\\sin,\\cos,dx$ 三者代换后分母都带 $1+t^2$，做题时它们常常成片**约掉**，这是万能代换化简的典型节奏。',
    },
    {
      title: '易错点：$x=\\pi$ 处代换失效',
      body: '$t=\\tan\\dfrac{x}{2}$ 在 $\\dfrac{x}{2}=\\dfrac{\\pi}{2}$（即 $x=\\pi$）处**无定义**——$\\tan\\dfrac{\\pi}{2}$ 不存在。\n\n这等于在解题时**偷偷预设了 $x\\ne\\pi$**。所以解三角方程用完万能代换后，必须**单独回头验证 $x=\\pi$**（以及 $x=\\pi+2k\\pi$）是不是漏掉的解。',
      tip: '判断 $x=\\pi$ 是否为解，直接把它代回**原方程**检验，而不是代入 $t$ 的方程（因为那里根本取不到）。',
    },
    {
      title: '动手想一想',
      body: '三条公式里 $\\cos x$ 的分子最容易记错。确认一下。',
      reveal: {
        q: '$\\cos x$ 的万能代换式是？分子是 $2t$ 还是 $1-t^2$？',
        a: '$\\cos x=\\dfrac{1-t^2}{1+t^2}$，分子是 $1-t^2$。（分子 $2t$ 的是 $\\sin x$。）记忆：$\\sin$ 取偶函数 $2t$、$\\cos$ 取 $1-t^2$，分母统一 $1+t^2$。',
      },
    },
  ],
  'u18-l3': [
    {
      title: '动机：一个套路解一大类题',
      body: '万能代换的真正价值，是把两类看似不同的难题**收进同一个入口**：\n\n含 $\\sin x,\\cos x$ 的**三角方程** → 代入后变成关于 $t$ 的**多项式方程**；含 $\\sin x,\\cos x$ 的**三角积分** → 代入后变成**有理函数积分**。学会这一招，一大片题目都有了统一打法。',
      formula: '\\int\\frac{dx}{2+\\cos x}\\xrightarrow{\\,t=\\tan\\frac x2\\,}\\int\\frac{2}{3+t^2}\\,dt',
    },
    {
      title: '例题精讲：用万能代换解 $\\sin x+\\cos x=1$',
      body: '看三角方程如何塌缩成简单的代数方程。',
      steps: [
        '代入：$\\dfrac{2t}{1+t^2}+\\dfrac{1-t^2}{1+t^2}=1$',
        '两边乘 $1+t^2$：$2t+1-t^2=1+t^2$',
        '整理：$2t=2t^2$，即 $t(t-1)=0$，得 $t=0$ 或 $t=1$',
        '回代 $t=0$：$\\tan\\dfrac{x}{2}=0\\Rightarrow x=2k\\pi$',
        '回代 $t=1$：$\\tan\\dfrac{x}{2}=1\\Rightarrow \\dfrac{x}{2}=\\dfrac{\\pi}{4}+k\\pi\\Rightarrow x=\\dfrac{\\pi}{2}+2k\\pi$',
        '验 $x=\\pi$（代换的盲点）：$\\sin\\pi+\\cos\\pi=-1\\ne1$，不是解，无遗漏',
      ],
      tip: '解完务必回到原方程验 $x=\\pi$！这是万能代换最容易漏解的地方。',
    },
    {
      title: '例题精讲：积分 $\\displaystyle\\int\\frac{dx}{2+\\cos x}$',
      body: '体会三角分式如何精确化为有理函数。',
      steps: [
        '代入 $\\cos x=\\dfrac{1-t^2}{1+t^2}$，$dx=\\dfrac{2}{1+t^2}\\,dt$',
        '化分母：$2+\\cos x=\\dfrac{2(1+t^2)+(1-t^2)}{1+t^2}=\\dfrac{3+t^2}{1+t^2}$',
        '整体代入，$1+t^2$ 约掉：$\\displaystyle\\int\\frac{1+t^2}{3+t^2}\\cdot\\frac{2}{1+t^2}\\,dt=\\int\\frac{2}{3+t^2}\\,dt$',
        '套公式 $\\displaystyle\\int\\frac{dt}{t^2+b^2}=\\frac1b\\arctan\\frac tb$（$b=\\sqrt3$）：$=\\dfrac{2}{\\sqrt3}\\arctan\\dfrac{\\tan\\frac x2}{\\sqrt3}+C$',
      ],
      formula: '\\int\\frac{dx}{2+\\cos x}=\\frac{2}{\\sqrt3}\\arctan\\frac{\\tan\\frac x2}{\\sqrt3}+C',
    },
    {
      title: '易错点：两种代换分工不同',
      body: '别把名字搞混——它们都叫「代换」，对象却完全不同：\n\n**三角代换**（第17章）处理**根式** $\\sqrt{\\cdot}$，用 $x=a\\sin\\theta$ 一类；**万能代换**（本章）处理**同时含 $\\sin x,\\cos x$ 的有理式**，用 $t=\\tan\\dfrac{x}{2}$。看到根号找前者，看到三角分式找后者。',
      tip: '万能代换是「统一入口」，但**不总是最短路径**。$3\\sin x+4\\cos x=5$ 用辅助角一眼看出「右边正是振幅，只一个解」；硬上万能代换反而要解 $(3t-1)^2=0$ 绕远路。先想清楚再动手。',
    },
    {
      title: '动手想一想',
      body: '验证「振幅 = 右端」时方程恰好相切只有一解。',
      reveal: {
        q: '$3\\sin x+4\\cos x=5$ 在 $[0,2\\pi)$ 上有几个解？为什么？',
        a: '右端 $5$ 恰等于振幅 $\\sqrt{3^2+4^2}=5$，即函数的**最大值**，所以图像与直线 $y=5$ 相切，只有 **1 个**解。用万能代换会得到重根 $t=\\dfrac13$（即 $(3t-1)^2=0$），对应同一个解 $x=2\\arctan\\dfrac13$。',
      },
    },
  ],
}
