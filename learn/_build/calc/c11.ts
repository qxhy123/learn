import type { Lesson } from '../../src/types'

export const LESSONS: Lesson[] = [
  // ─────────────────────────────────────────────────────────
  // 关 c11-a  乘积法则与商法则
  // 6 intro 卡 + 7 题
  // ─────────────────────────────────────────────────────────
  {
    id: 'c11-a',
    title: '乘积法则与商法则',
    subtitle: '为什么 $(fg)^{\\prime} \\neq f^{\\prime}g^{\\prime}$？逐步推导并应用',
    intro: [
      // 卡 1：动机——直接相乘求导为什么错
      {
        title: '直觉陷阱：导数可以"分别乘"吗？',
        body:
          '设 $f(x)=x^2$，$g(x)=x^3$，则 $h(x)=x^5$。\n\n' +
          '直接计算：$h^{\\prime}(x)=5x^4$。\n\n' +
          '如果按"分别求导再相乘"：$f^{\\prime}(x)=2x$，$g^{\\prime}(x)=3x^2$，' +
          '相乘得 $6x^3$——与 $5x^4$ **完全不同**！\n\n' +
          '**结论**：$(fg)^{\\prime}\\neq f^{\\prime}\\cdot g^{\\prime}$。正确法则需要推导。',
        tip: '乘积法则是微积分中最容易写错的公式之一，必须熟记正确形式。',
      },
      // 卡 2：乘积法则推导
      {
        title: '乘积法则的推导（"加一减一"技巧）',
        body:
          '从定义出发，设 $h=fg$：\n\n' +
          '$h^{\\prime}(x)=\\lim_{\\Delta x\\to 0}\\dfrac{f(x+\\Delta x)g(x+\\Delta x)-f(x)g(x)}{\\Delta x}$\n\n' +
          '**关键一步**——在分子中加减同一项 $f(x)g(x+\\Delta x)$，把差拆成两部分：\n\n' +
          '$= \\lim_{\\Delta x\\to 0}\\left[\\dfrac{f(x+\\Delta x)-f(x)}{\\Delta x}\\cdot g(x+\\Delta x)' +
          '+f(x)\\cdot\\dfrac{g(x+\\Delta x)-g(x)}{\\Delta x}\\right]$\n\n' +
          '当 $\\Delta x\\to 0$：第一个差商趋向 $f^{\\prime}(x)$，$g(x+\\Delta x)$ 趋向 $g(x)$（$g$ 可导必连续），' +
          '第二个差商趋向 $g^{\\prime}(x)$。两部分加起来就是乘积法则。',
        formula: '(fg)^{\\prime} = f^{\\prime}g + fg^{\\prime}',
        tip: '口诀：**导前乘后** + **前乘导后**——$f$ 求导配上 $g$ 原值，再加 $f$ 原值配 $g$ 求导。',
      },
      // 卡 3：乘积法则例题（具体数字）
      {
        title: '例题精讲：$y = x^2 e^x$',
        body:
          '求 $y=x^2 e^x$ 的导数。\n\n' +
          '**识别**：两个函数相乘，令 $f=x^2$，$g=e^x$。\n\n' +
          '**分别求导**：$f^{\\prime}=2x$，$g^{\\prime}=e^x$（指数函数导数等于自身）。\n\n' +
          '**代入乘积法则**：$y^{\\prime}=f^{\\prime}g+fg^{\\prime}=2x\\cdot e^x+x^2\\cdot e^x$。\n\n' +
          '**提公因子化简**：$y^{\\prime}=e^x(2x+x^2)=x(x+2)e^x$。',
        steps: [
          '令 $f=x^2$，$g=e^x$',
          '求 $f^{\\prime}=2x$，$g^{\\prime}=e^x$',
          '代入：$y^{\\prime}=2x\\cdot e^x + x^2\\cdot e^x$',
          '提取 $e^x$：$y^{\\prime}=(x^2+2x)e^x$',
        ],
        reveal: {
          q: '用乘积法则求 $y=x\\sin x$ 的导数，结果是？',
          a: '$f=x$，$f^{\\prime}=1$；$g=\\sin x$，$g^{\\prime}=\\cos x$。$y^{\\prime}=\\sin x + x\\cos x$。',
        },
      },
      // 卡 4：商法则推导
      {
        title: '商法则的推导（同款"加一减一"）',
        body:
          '设 $h=f/g$（$g\\neq 0$），从定义出发通分后，分子加减 $f(x)g(x)$：\n\n' +
          '$\\dfrac{f(x+\\Delta x)g(x)-f(x)g(x+\\Delta x)}{g(x+\\Delta x)g(x)}$\n\n' +
          '分子拆开：$[f(x+\\Delta x)-f(x)]g(x) - f(x)[g(x+\\Delta x)-g(x)]$\n\n' +
          '除以 $\\Delta x$ 并取极限，$g(x+\\Delta x)\\to g(x)$，分母趋向 $[g(x)]^2$，得到商法则。\n\n' +
          '**记忆口诀**：分子"**先正后负**"——导分子乘分母，减分子乘导分母，全除分母平方。**顺序不可颠倒！**',
        formula: '\\left(\\dfrac{f}{g}\\right)^{\\prime} = \\dfrac{f^{\\prime}g - fg^{\\prime}}{g^2}',
        tip: '易错：分子是 $f^{\\prime}g-fg^{\\prime}$，不是 $fg^{\\prime}-f^{\\prime}g$，一旦颠倒符号完全相反。',
      },
      // 卡 5：商法则例题 tan x
      {
        title: '例题精讲：$y = \\tan x$ 的导数',
        body:
          '求 $y=\\tan x=\\dfrac{\\sin x}{\\cos x}$ 的导数。\n\n' +
          '**令** $f=\\sin x$，$g=\\cos x$；$f^{\\prime}=\\cos x$，$g^{\\prime}=-\\sin x$。\n\n' +
          '**代入商法则**：\n\n' +
          '$y^{\\prime}=\\dfrac{\\cos x\\cdot\\cos x - \\sin x\\cdot(-\\sin x)}{\\cos^2 x}' +
          '=\\dfrac{\\cos^2 x+\\sin^2 x}{\\cos^2 x}=\\dfrac{1}{\\cos^2 x}=\\sec^2 x$\n\n' +
          '**关键步骤**：分子用了 $\\cos^2 x+\\sin^2 x=1$ 化简。\n\n' +
          '结果：$(\\tan x)^{\\prime}=\\sec^2 x$，定义域 $x\\neq k\\pi+\\dfrac{\\pi}{2}$。',
        steps: [
          '$f=\\sin x$，$g=\\cos x$；$f^{\\prime}=\\cos x$，$g^{\\prime}=-\\sin x$',
          '分子：$\\cos^2 x - \\sin x\\cdot(-\\sin x) = \\cos^2 x + \\sin^2 x = 1$',
          '分母：$\\cos^2 x$',
          '结果：$y^{\\prime}=\\dfrac{1}{\\cos^2 x}=\\sec^2 x$',
        ],
      },
      // 卡 6：易错总结 + 速查
      {
        title: '易错点汇总与速查',
        body:
          '**错误 1**：$(x^2 e^x)^{\\prime}=2x\\cdot e^x$——漏掉了 $x^2\\cdot e^x$ 这一项。\n\n' +
          '**错误 2**：$(f/g)^{\\prime}=f^{\\prime}/g^{\\prime}$——商法则不是"上下分别求导"，分母里有平方和减法。\n\n' +
          '**错误 3**：商法则分子写成 $fg^{\\prime}-f^{\\prime}g$——符号颠倒，结果完全错误，牢记"先正后负"。\n\n' +
          '**速查**：乘积 $(fg)^{\\prime}=f^{\\prime}g+fg^{\\prime}$；' +
          '商 $(f/g)^{\\prime}=(f^{\\prime}g-fg^{\\prime})/g^2$；' +
          '特例 $(1/g)^{\\prime}=-g^{\\prime}/g^2$。',
        tip: '三函数乘积：$(fgh)^{\\prime}=f^{\\prime}gh+fg^{\\prime}h+fgh^{\\prime}$（每次只对一个求导，其余保持原样，三项相加）。',
        reveal: {
          q: '求 $y=\\dfrac{e^x}{1+x^2}$ 在 $x=0$ 处的导数值。',
          a: '$f=e^x$，$f^{\\prime}=e^x$；$g=1+x^2$，$g^{\\prime}=2x$。' +
            '$y^{\\prime}=\\dfrac{e^x(1+x^2)-e^x\\cdot 2x}{(1+x^2)^2}=\\dfrac{e^x(1-x)^2}{(1+x^2)^2}$。' +
            '代 $x=0$：$y^{\\prime}(0)=\\dfrac{1\\cdot 1}{1}=1$。',
        },
      },
    ],
    questions: [
      // 题 1：judge — 乘积法则是否正确
      {
        id: 'c11-a-q1',
        type: 'judge',
        prompt: '$(x^3\\sin x)^{\\prime} = 3x^2\\sin x + x^3\\cos x$，这个结论正确吗？',
        answer: true,
        explain:
          '令 $f=x^3$，$f^{\\prime}=3x^2$；$g=\\sin x$，$g^{\\prime}=\\cos x$。' +
          '由乘积法则：$(x^3\\sin x)^{\\prime}=3x^2\\sin x+x^3\\cos x$。✓',
      },
      // 题 2：choice — 乘积法则选项
      {
        id: 'c11-a-q2',
        type: 'choice',
        prompt: '以下哪个是乘积法则的正确表达式？',
        options: [
          '$(fg)^{\\prime} = f^{\\prime} \\cdot g^{\\prime}$',
          '$(fg)^{\\prime} = f^{\\prime}g + fg^{\\prime}$',
          '$(fg)^{\\prime} = f^{\\prime}g - fg^{\\prime}$',
          '$(fg)^{\\prime} = (f+g)^{\\prime}$',
        ],
        answer: 1,
        explain:
          '乘积法则：$(fg)^{\\prime}=f^{\\prime}g+fg^{\\prime}$。第一项是"导 $f$ 乘 $g$"，第二项是"$f$ 乘导 $g$"，两项相加。',
      },
      // 题 3：input — 具体乘积求导
      {
        id: 'c11-a-q3',
        type: 'input',
        prompt:
          '求 $y=x^2 e^x$ 的导数，将结果写成 $(x^2+ax)e^x$ 的形式，$a$ 的值是多少？',
        accept: ['2'],
        explain:
          '$(x^2 e^x)^{\\prime}=2x\\cdot e^x+x^2\\cdot e^x=(x^2+2x)e^x$，故 $a=2$。',
      },
      // 题 4：choice — 商法则分子顺序
      {
        id: 'c11-a-q4',
        type: 'choice',
        prompt: '商法则 $(f/g)^{\\prime}$ 的分子是：',
        options: [
          '$fg^{\\prime}-f^{\\prime}g$',
          '$f^{\\prime}g^{\\prime}$',
          '$f^{\\prime}g - fg^{\\prime}$',
          '$f^{\\prime}g + fg^{\\prime}$',
        ],
        answer: 2,
        explain:
          '商法则分子是 $f^{\\prime}g-fg^{\\prime}$——"**先正后负**"：导分子乘分母，减分子乘导分母。' +
          '顺序颠倒则差一个负号，完全错误。',
      },
      // 题 5：input — tan x 导数
      {
        id: 'c11-a-q5',
        type: 'input',
        prompt: '$(\\tan x)^{\\prime}$ 等于（用 sec 表示，填写 LaTeX 形式如 sec^2 x）：',
        accept: ['\\sec^2 x', 'sec^2 x', '1/cos^2 x', '\\frac{1}{\\cos^2 x}'],
        explain:
          '$(\\tan x)^{\\prime}=\\left(\\dfrac{\\sin x}{\\cos x}\\right)^{\\prime}' +
          '=\\dfrac{\\cos^2 x+\\sin^2 x}{\\cos^2 x}=\\dfrac{1}{\\cos^2 x}=\\sec^2 x$。',
      },
      // 题 6：judge — 三函数乘积
      {
        id: 'c11-a-q6',
        type: 'judge',
        prompt: '$(xyz)^{\\prime} = x^{\\prime}yz + xy^{\\prime}z + xyz^{\\prime}$，正确吗？',
        answer: true,
        explain:
          '三函数乘积法则（两次应用二函数乘积法则推导）：' +
          '$(fgh)^{\\prime}=f^{\\prime}gh+fg^{\\prime}h+fgh^{\\prime}$。' +
          '每次只对一个函数求导，另外两个保持原样，三项相加。✓',
      },
      // 题 7：match — 函数与导数配对
      {
        id: 'c11-a-q7',
        type: 'match',
        prompt: '将下列函数与其导数正确配对：',
        left: [
          '$x\\ln x$',
          '$e^x \\sin x$',
          '$\\dfrac{x}{e^x}$',
          '$x^2\\cos x$',
        ],
        right: [
          '$\\ln x + 1$',
          '$e^x(\\sin x+\\cos x)$',
          '$\\dfrac{1-x}{e^x}$',
          '$2x\\cos x - x^2\\sin x$',
        ],
        explain:
          '$(x\\ln x)^{\\prime}=\\ln x+x\\cdot\\frac{1}{x}=\\ln x+1$；' +
          '$(e^x\\sin x)^{\\prime}=e^x\\sin x+e^x\\cos x=e^x(\\sin x+\\cos x)$；' +
          '$(x/e^x)^{\\prime}=\\frac{e^x-xe^x}{e^{2x}}=\\frac{1-x}{e^x}$；' +
          '$(x^2\\cos x)^{\\prime}=2x\\cos x+x^2(-\\sin x)=2x\\cos x-x^2\\sin x$。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────
  // 关 c11-b  链式法则——复合函数求导
  // 7 intro 卡 + 7 题
  // ─────────────────────────────────────────────────────────
  {
    id: 'c11-b',
    title: '链式法则：剥洋葱求导',
    subtitle: '复合函数从外到内逐层相乘，绝不漏层',
    intro: [
      // 卡 1：什么是复合函数
      {
        title: '复合函数的结构识别',
        body:
          '复合函数 $y=f(g(x))$ 的特征：**把一个函数的输出喂给另一个函数**。\n\n' +
          '**例子**：$y=\\sin(x^2)$——先计算 $x^2$（内层），再计算 $\\sin$（外层）。\n\n' +
          '识别层数：$y=\\sqrt{e^x+1}$ 有两层：内层 $e^x+1$，外层 $\\sqrt{\\cdot}$。' +
          '$y=\\ln(\\sin(x^2))$ 有三层：最内 $x^2$，中层 $\\sin$，最外 $\\ln$。\n\n' +
          '**关键问题**：如果直接对 $y=\\sin(x^2)$ 求导，写成 $\\cos(x^2)$ 就漏掉了内层 $(x^2)^{\\prime}=2x$——' +
          '链式法则就是补上这一乘。',
        tip: '判断是否需要链式法则：看括号内是否有 $x$ 的非平凡函数。若括号里是 $x$ 本身，直接求导即可。',
      },
      // 卡 2：链式法则正式陈述
      {
        title: '链式法则：正式表达',
        body:
          '设 $y=f(u)$，$u=g(x)$，若 $g$ 在 $x$ 处可导、$f$ 在 $u=g(x)$ 处可导，则复合函数 $y=f(g(x))$ 可导，导数为：\n\n' +
          '$\\dfrac{dy}{dx}=\\dfrac{dy}{du}\\cdot\\dfrac{du}{dx}=f^{\\prime}(g(x))\\cdot g^{\\prime}(x)$\n\n' +
          '**Leibniz 记号的妙处**：$\\dfrac{dy}{dx}=\\dfrac{dy}{du}\\cdot\\dfrac{du}{dx}$ 形式上像分数约分——但这只是记号便利，本质是极限定理。\n\n' +
          '**口诀**："从**外**向**内**逐层求导相乘"。先对最外层函数求导（内部不动），再乘内层对 $x$ 的导数。',
        formula: '\\dfrac{dy}{dx} = f^{\\prime}(g(x))\\cdot g^{\\prime}(x)',
      },
      // 卡 3：两层链式例题
      {
        title: '两层链式：$y = \\sin(x^2)$',
        body:
          '求 $y=\\sin(x^2)$ 的导数。\n\n' +
          '**识别**：外层 $\\sin(\\cdot)$，内层 $u=x^2$。\n\n' +
          '**外层导数**：$(\\sin u)^{\\prime}=\\cos u=\\cos(x^2)$（内部 $x^2$ 暂时不动）。\n\n' +
          '**内层导数**：$(x^2)^{\\prime}=2x$。\n\n' +
          '**乘起来**：$y^{\\prime}=\\cos(x^2)\\cdot 2x=2x\\cos(x^2)$。\n\n' +
          '**验证直觉**：$x=1$ 时，$y=\\sin 1$，$y^{\\prime}=2\\cos 1\\approx 2\\times 0.540=1.080$——用数值差商 $\\frac{\\sin(1.001^2)-\\sin(1^2)}{0.001}\\approx 1.080$ 吻合。',
        steps: [
          '令外层 $f(u)=\\sin u$，内层 $g(x)=x^2$',
          '外层导：$f^{\\prime}(u)=\\cos u$，代入内层：$\\cos(x^2)$',
          '内层导：$g^{\\prime}(x)=2x$',
          '链式相乘：$y^{\\prime}=\\cos(x^2)\\cdot 2x=2x\\cos(x^2)$',
        ],
      },
      // 卡 4：三层链式例题
      {
        title: '三层链式："剥洋葱"',
        body:
          '求 $y=\\ln(\\cos(e^x))$ 的导数。\n\n' +
          '**三层**：最外 $\\ln(\\cdot)$，中层 $\\cos(\\cdot)$，最内 $e^x$。\n\n' +
          '**第 1 层**（最外 $\\ln u$，$u=\\cos(e^x)$）：$(\\ln u)^{\\prime}=1/u$，故写 $\\dfrac{1}{\\cos(e^x)}$。\n\n' +
          '**第 2 层**（中层 $\\cos v$，$v=e^x$）：$(\\cos v)^{\\prime}=-\\sin v=-\\sin(e^x)$。\n\n' +
          '**第 3 层**（最内 $e^x$）：$(e^x)^{\\prime}=e^x$。\n\n' +
          '**三层相乘**：$y^{\\prime}=\\dfrac{1}{\\cos(e^x)}\\cdot(-\\sin(e^x))\\cdot e^x=-e^x\\tan(e^x)$。',
        steps: [
          '第 1 层：$\\dfrac{1}{\\cos(e^x)}$',
          '第 2 层：$-\\sin(e^x)$',
          '第 3 层：$e^x$',
          '相乘化简：$y^{\\prime}=-e^x\\cdot\\dfrac{\\sin(e^x)}{\\cos(e^x)}=-e^x\\tan(e^x)$',
        ],
        tip: '每完成一层，问自己："内层还是 $x$ 本身吗？"若不是，继续向内再乘一次导数。',
      },
      // 卡 5：链式 + 乘积综合
      {
        title: '链式与乘积法则的组合',
        body:
          '实际题目常常同时用链式和乘积法则。\n\n' +
          '**例**：求 $y=x^2 e^{\\sin x}$ 的导数。\n\n' +
          '整体是 $x^2$ 与 $e^{\\sin x}$ 的**乘积**——先用乘积法则：\n\n' +
          '$y^{\\prime}=(x^2)^{\\prime}\\cdot e^{\\sin x}+x^2\\cdot(e^{\\sin x})^{\\prime}$\n\n' +
          '$=2x\\cdot e^{\\sin x}+x^2\\cdot e^{\\sin x}\\cdot\\cos x$\n\n' +
          '其中 $(e^{\\sin x})^{\\prime}$ 用链式：外层 $e^u$，内层 $\\sin x$，得 $e^{\\sin x}\\cdot\\cos x$。\n\n' +
          '最终：$y^{\\prime}=e^{\\sin x}(2x+x^2\\cos x)=xe^{\\sin x}(2+x\\cos x)$。',
        reveal: {
          q: '求 $y=\\sqrt{\\ln x}$（$x>0$）的导数。',
          a: '外层 $(\\cdot)^{1/2}$，内层 $\\ln x$。$y^{\\prime}=\\dfrac{1}{2}(\\ln x)^{-1/2}\\cdot\\dfrac{1}{x}=\\dfrac{1}{2x\\sqrt{\\ln x}}$。',
        },
      },
      // 卡 6：常用复合公式速查
      {
        title: '常用复合公式速查',
        body:
          '设 $u=u(x)$ 是任意可导函数，以下公式由链式法则直接给出：\n\n' +
          '$(u^n)^{\\prime}=nu^{n-1}\\cdot u^{\\prime}$ （幂函数链式）\n\n' +
          '$(e^u)^{\\prime}=e^u\\cdot u^{\\prime}$\n\n' +
          '$(\\ln u)^{\\prime}=\\dfrac{u^{\\prime}}{u}$（$u>0$）\n\n' +
          '$(\\sin u)^{\\prime}=\\cos u\\cdot u^{\\prime}$\n\n' +
          '$(\\cos u)^{\\prime}=-\\sin u\\cdot u^{\\prime}$\n\n' +
          '**使用方法**：把括号里的 $u$ 当作整体，按基本公式求外层导数，再在末尾乘以 $u^{\\prime}$。',
        formula: '[f(u)]^{\\prime} = f^{\\prime}(u)\\cdot u^{\\prime}',
        tip: '$(\\sin(3x+1))^{\\prime}=\\cos(3x+1)\\cdot 3=3\\cos(3x+1)$——内层 $u=3x+1$，$u^{\\prime}=3$，别忘了这个"3"！',
      },
      // 卡 7：易错点 + 引入思路还原
      {
        title: '易错点与思维路径还原',
        body:
          '**最高频错误**：求导到第一层就停。例如 $y=e^{x^2}$，只写 $e^{x^2}$ 而漏了 $\\cdot 2x$，正确结果是 $2xe^{x^2}$。\n\n' +
          '**思维路径还原**（以 $y=\\sin(\\ln(x^2+1))$ 为例）：\n\n' +
          '"我看到三层：最外 $\\sin$，中层 $\\ln$，最内 $x^2+1$。"\n\n' +
          '"第 1 层：$\\cos(\\ln(x^2+1))$。内层是 $\\ln(x^2+1)$，不是 $x$，继续剥。"\n\n' +
          '"第 2 层：$\\dfrac{1}{x^2+1}$。内层是 $x^2+1$，还不是 $x$，继续。"\n\n' +
          '"第 3 层：$2x$。是 $x$ 了，停。三层相乘：$y^{\\prime}=\\dfrac{2x\\cos(\\ln(x^2+1))}{x^2+1}$。"',
        tip: '每写完一层就问："我正在对哪个变量求导？内层还是 $x$ 本身吗？"这个问题能避免 90% 的链式遗漏。',
      },
    ],
    questions: [
      // 题 1：choice — 识别链式层数
      {
        id: 'c11-b-q1',
        type: 'choice',
        prompt: '函数 $y=e^{\\sin(x^2)}$ 一共有几层复合？',
        options: ['1 层', '2 层', '3 层', '4 层'],
        answer: 2,
        explain:
          '$y=e^{\\sin(x^2)}$：最外层 $e^{(\\cdot)}$，中层 $\\sin(\\cdot)$，最内层 $x^2$，共 **3 层**复合。' +
          '用链式法则求导需要三个因子相乘。',
      },
      // 题 2：input — 两层链式结果
      {
        id: 'c11-b-q2',
        type: 'input',
        prompt:
          '求 $y=e^{3x}$ 的导数。写出系数：$y^{\\prime}=\\_\\_e^{3x}$，空格处填数字。',
        accept: ['3'],
        explain:
          '令 $u=3x$，$y=e^u$。外层导：$e^u$；内层导：$3$。' +
          '链式：$y^{\\prime}=e^{3x}\\cdot 3=3e^{3x}$，系数为 $3$。',
      },
      // 题 3：judge — 链式法则应用
      {
        id: 'c11-b-q3',
        type: 'judge',
        prompt: '$(\\cos(2x))^{\\prime} = -\\sin(2x)$，正确吗？',
        answer: false,
        explain:
          '$(\\cos(2x))^{\\prime}=-\\sin(2x)\\cdot(2x)^{\\prime}=-\\sin(2x)\\cdot 2=-2\\sin(2x)$。' +
          '漏掉内层导数 $2$，所以 $-\\sin(2x)$ 是**错的**，正确答案是 $-2\\sin(2x)$。',
      },
      // 题 4：choice — 三层链式
      {
        id: 'c11-b-q4',
        type: 'choice',
        prompt: '求 $y=\\ln(\\sin(2x))$ 的导数，正确结果是？',
        options: [
          '$\\dfrac{1}{\\sin(2x)}$',
          '$\\dfrac{\\cos(2x)}{\\sin(2x)}$',
          '$\\dfrac{2\\cos(2x)}{\\sin(2x)}$',
          '$2\\cos(2x)\\cdot\\ln(\\sin(2x))$',
        ],
        answer: 2,
        explain:
          '三层：外层 $\\ln u$（导 $1/u$），中层 $\\sin v$（导 $\\cos v$），内层 $2x$（导 $2$）。' +
          '$y^{\\prime}=\\dfrac{1}{\\sin(2x)}\\cdot\\cos(2x)\\cdot 2=\\dfrac{2\\cos(2x)}{\\sin(2x)}=2\\cot(2x)$。',
      },
      // 题 5：input — 链式结合幂次
      {
        id: 'c11-b-q5',
        type: 'input',
        prompt:
          '求 $y=(x^2+1)^5$ 的导数，结果形如 $a\\cdot x(x^2+1)^4$，$a$ 的值是多少？',
        accept: ['10'],
        explain:
          '令 $u=x^2+1$，$y=u^5$。外层导：$5u^4$；内层导：$(x^2+1)^{\\prime}=2x$。' +
          '$y^{\\prime}=5(x^2+1)^4\\cdot 2x=10x(x^2+1)^4$，故 $a=10$。',
      },
      // 题 6：judge — 链式与乘积组合
      {
        id: 'c11-b-q6',
        type: 'judge',
        prompt: '$(xe^{x^2})^{\\prime} = e^{x^2}+2x^2 e^{x^2}$，正确吗？',
        answer: true,
        explain:
          '乘积法则：$(x)^{\\prime}\\cdot e^{x^2}+x\\cdot(e^{x^2})^{\\prime}=e^{x^2}+x\\cdot e^{x^2}\\cdot 2x=e^{x^2}+2x^2 e^{x^2}=e^{x^2}(1+2x^2)$。✓',
      },
      // 题 7：match — 函数与链式导数配对
      {
        id: 'c11-b-q7',
        type: 'match',
        prompt: '将下列复合函数与其导数正确配对：',
        left: [
          '$\\sin(3x)$',
          '$e^{-x^2}$',
          '$\\sqrt{1+x^2}$',
          '$\\ln(x^2+1)$',
        ],
        right: [
          '$3\\cos(3x)$',
          '$-2xe^{-x^2}$',
          '$\\dfrac{x}{\\sqrt{1+x^2}}$',
          '$\\dfrac{2x}{x^2+1}$',
        ],
        explain:
          '$(\\sin(3x))^{\\prime}=\\cos(3x)\\cdot 3=3\\cos(3x)$；' +
          '$(e^{-x^2})^{\\prime}=e^{-x^2}\\cdot(-2x)=-2xe^{-x^2}$；' +
          '$(\\sqrt{1+x^2})^{\\prime}=\\dfrac{1}{2\\sqrt{1+x^2}}\\cdot 2x=\\dfrac{x}{\\sqrt{1+x^2}}$；' +
          '$(\\ln(x^2+1))^{\\prime}=\\dfrac{2x}{x^2+1}$。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────
  // 关 c11-c  反函数法则与对数求导法
  // 6 intro 卡 + 7 题
  // ─────────────────────────────────────────────────────────
  {
    id: 'c11-c',
    title: '反函数法则与对数求导法',
    subtitle: '反三角函数求导 + 幂指函数 $x^x$ 的处理策略',
    intro: [
      // 卡 1：反函数法则的直觉
      {
        title: '反函数法则：变化率取倒数',
        body:
          '设 $y=f(x)$ 严格单调且可导，反函数 $x=f^{-1}(y)$ 也可导，且：\n\n' +
          '$[f^{-1}]^{\\prime}(y)=\\dfrac{1}{f^{\\prime}(x)}$（等价地：$\\dfrac{dx}{dy}=\\dfrac{1}{\\,\\dfrac{dy}{dx}\\,}$）\n\n' +
          '**直觉**：若 $y$ 关于 $x$ 的变化率是 $f^{\\prime}(x)=3$，那么 $x$ 关于 $y$ 的变化率自然是 $1/3$。\n\n' +
          '**为什么需要 $f^{\\prime}(x)\\neq 0$？** 若 $f^{\\prime}(x_0)=0$，切线水平，反函数在该点切线竖直，导数无限大，无意义。\n\n' +
          '**链式推导（一行）**：在 $f^{-1}(f(x))=x$ 两边对 $x$ 求导，链式得 $[f^{-1}]^{\\prime}(f(x))\\cdot f^{\\prime}(x)=1$，解出即得。',
        formula: '[f^{-1}]^{\\prime}(y) = \\dfrac{1}{f^{\\prime}(x)}',
        tip: '条件 $f^{\\prime}(x)\\neq 0$ 不可少——反函数求导法则的分母必须非零。',
      },
      // 卡 2：arcsin 的导数推导
      {
        title: '应用：推导 $(\\arcsin x)^{\\prime}$',
        body:
          '设 $y=\\arcsin x$，则 $x=\\sin y$，$y\\in(-\\pi/2,\\,\\pi/2)$。\n\n' +
          '由反函数法则：$(\\arcsin x)^{\\prime}=\\dfrac{1}{(\\sin y)^{\\prime}}=\\dfrac{1}{\\cos y}$。\n\n' +
          '**处理 $\\cos y$**：在 $y\\in(-\\pi/2,\\pi/2)$ 上，$\\cos y>0$，故 $\\cos y=\\sqrt{1-\\sin^2 y}=\\sqrt{1-x^2}$。\n\n' +
          '**最终结果**：$(\\arcsin x)^{\\prime}=\\dfrac{1}{\\sqrt{1-x^2}}$，定义域 $|x|<1$。\n\n' +
          '**类比**：$(\\arccos x)^{\\prime}=-\\dfrac{1}{\\sqrt{1-x^2}}$，两者差一个负号——因为 $\\arcsin x+\\arccos x=\\dfrac{\\pi}{2}$，两边求导右边为 $0$。',
        steps: [
          '令 $y=\\arcsin x$，则 $x=\\sin y$，$y\\in(-\\pi/2,\\pi/2)$',
          '反函数法则：$(\\arcsin x)^{\\prime}=1/(\\sin y)^{\\prime}=1/\\cos y$',
          '区间内 $\\cos y>0$，故 $\\cos y=\\sqrt{1-\\sin^2 y}=\\sqrt{1-x^2}$',
          '结果：$(\\arcsin x)^{\\prime}=\\dfrac{1}{\\sqrt{1-x^2}}$',
        ],
      },
      // 卡 3：arctan 的导数推导
      {
        title: '应用：推导 $(\\arctan x)^{\\prime}$',
        body:
          '设 $y=\\arctan x$，则 $x=\\tan y$，$y\\in(-\\pi/2,\\pi/2)$。\n\n' +
          '由反函数法则：$(\\arctan x)^{\\prime}=\\dfrac{1}{(\\tan y)^{\\prime}}=\\dfrac{1}{\\sec^2 y}$。\n\n' +
          '**处理 $\\sec^2 y$**：$\\sec^2 y=1+\\tan^2 y=1+x^2$（恒等式 $1+\\tan^2=\\sec^2$）。\n\n' +
          '**最终结果**：$(\\arctan x)^{\\prime}=\\dfrac{1}{1+x^2}$，定义域为全体实数 $\\mathbb{R}$。\n\n' +
          '**速查表**：$(\\arcsin x)^{\\prime}=\\dfrac{1}{\\sqrt{1-x^2}}$；$(\\arccos x)^{\\prime}=-\\dfrac{1}{\\sqrt{1-x^2}}$；' +
          '$(\\arctan x)^{\\prime}=\\dfrac{1}{1+x^2}$；$(\\text{arccot}\\,x)^{\\prime}=-\\dfrac{1}{1+x^2}$。',
        formula: '(\\arctan x)^{\\prime} = \\dfrac{1}{1+x^2}',
        reveal: {
          q: '求 $y=\\arctan\\dfrac{1}{x}$（$x>0$）的导数。',
          a: '令 $u=1/x$，用链式：$(\\arctan u)^{\\prime}\\cdot u^{\\prime}=\\dfrac{1}{1+u^2}\\cdot\\left(-\\dfrac{1}{x^2}\\right)=\\dfrac{1}{1+1/x^2}\\cdot\\left(-\\dfrac{1}{x^2}\\right)=\\dfrac{x^2}{x^2+1}\\cdot\\left(-\\dfrac{1}{x^2}\\right)=-\\dfrac{1}{1+x^2}$。',
        },
      },
      // 卡 4：为什么需要对数求导法
      {
        title: '为什么 $x^x$ 直接求导行不通？',
        body:
          '对 $y=x^x$（$x>0$），尝试用普通幂法则：\n\n' +
          '**幂法则** $(x^n)^{\\prime}=nx^{n-1}$ 要求 $n$ 为**常数**——但这里指数也是 $x$，不适用。\n\n' +
          '**指数法则** $(a^x)^{\\prime}=a^x\\ln a$ 要求底数 $a$ 为**常数**——但这里底数也是 $x$，不适用。\n\n' +
          '**指数和底数都含 $x$** 时，两种法则均失效。解决方案：两边取 $\\ln$，把幂转为乘积，再用已有法则。\n\n' +
          '**通用策略**：任何 $y=f(x)^{g(x)}$ 形式，都写成 $y=e^{g(x)\\ln f(x)}$，或直接两边取 $\\ln$ 再求导。',
        tip: '出现 $f^g$（底数和指数都含 $x$）→ **立刻**想到对数求导法，这是条件反射！',
      },
      // 卡 5：对数求导法步骤与例题
      {
        title: '对数求导法：$y = x^x$ 的完整推导',
        body:
          '求 $y=x^x$（$x>0$）的导数。\n\n' +
          '**第一步**：两边取 $\\ln$：$\\ln y=x\\ln x$。\n\n' +
          '**第二步**：两边对 $x$ 求导（左边用链式：$(\\ln y)^{\\prime}=y^{\\prime}/y$）：\n\n' +
          '$\\dfrac{y^{\\prime}}{y}=(\\ln x)^{\\prime}\\cdot x+\\ln x\\cdot 1=\\dfrac{1}{x}\\cdot x+\\ln x=1+\\ln x$\n\n' +
          '**第三步**：两边乘以 $y=x^x$：$y^{\\prime}=x^x(1+\\ln x)$。\n\n' +
          '**验证**：$x=1$ 时，$y=1$，$y^{\\prime}=1\\cdot(1+0)=1$——用差商数值近似验证正确。',
        steps: [
          '取对数：$\\ln y = x\\ln x$',
          '两边对 $x$ 求导：$\\dfrac{y^{\\prime}}{y}=1+\\ln x$（乘积法则）',
          '乘回 $y$：$y^{\\prime}=x^x(1+\\ln x)$',
        ],
        formula: '(x^x)^{\\prime} = x^x(1 + \\ln x)',
      },
      // 卡 6：对数求导法处理多因子积 + 易错总结
      {
        title: '对数求导法处理多因子积',
        body:
          '对含多个因子的乘积求导，对数法能大幅简化。\n\n' +
          '**例**：$y=\\dfrac{\\sqrt{x+1}\\cdot(x-2)^3}{(x^2+1)^2}$（$x>2$）。\n\n' +
          '取对数：$\\ln y=\\dfrac{1}{2}\\ln(x+1)+3\\ln(x-2)-2\\ln(x^2+1)$。\n\n' +
          '两边对 $x$ 求导：\n\n' +
          '$\\dfrac{y^{\\prime}}{y}=\\dfrac{1}{2(x+1)}+\\dfrac{3}{x-2}-\\dfrac{4x}{x^2+1}$\n\n' +
          '再乘回 $y$ 即得完整结果。\n\n' +
          '**总结三种场景**：① 普通乘积/商 → 乘积/商法则；② $f^g$（两个都含 $x$）→ 对数求导；③ 多因子积 → 对数求导更简洁。',
        tip: '对数求导后，最后一步**一定要乘回 $y$**——$(y^{\\prime}/y)\\cdot y=y^{\\prime}$，这步常被遗忘。',
        reveal: {
          q: '用对数求导法求 $(\\sin x)^x$（$0<x<\\pi$）的导数。',
          a: '$\\ln y=x\\ln(\\sin x)$；$\\dfrac{y^{\\prime}}{y}=\\ln(\\sin x)+x\\cdot\\dfrac{\\cos x}{\\sin x}=\\ln(\\sin x)+x\\cot x$；' +
            '$y^{\\prime}=(\\sin x)^x(\\ln(\\sin x)+x\\cot x)$。',
        },
      },
    ],
    questions: [
      // 题 1：choice — 反函数法则
      {
        id: 'c11-c-q1',
        type: 'choice',
        prompt: '$(\\arcsin x)^{\\prime}$ 等于：',
        options: [
          '$\\dfrac{1}{\\sqrt{1+x^2}}$',
          '$-\\dfrac{1}{\\sqrt{1-x^2}}$',
          '$\\dfrac{1}{\\sqrt{1-x^2}}$',
          '$\\dfrac{1}{1+x^2}$',
        ],
        answer: 2,
        explain:
          '设 $y=\\arcsin x$，$x=\\sin y$，$(\\arcsin x)^{\\prime}=1/\\cos y=1/\\sqrt{1-\\sin^2 y}=1/\\sqrt{1-x^2}$。',
      },
      // 题 2：judge — arccos 与 arcsin 的关系
      {
        id: 'c11-c-q2',
        type: 'judge',
        prompt: '$(\\arccos x)^{\\prime} = \\dfrac{1}{\\sqrt{1-x^2}}$，正确吗？',
        answer: false,
        explain:
          '$(\\arccos x)^{\\prime}=-\\dfrac{1}{\\sqrt{1-x^2}}$，**差一个负号**。' +
          '因为 $\\arcsin x+\\arccos x=\\pi/2$，两边求导得 $(\\arcsin x)^{\\prime}+(\\arccos x)^{\\prime}=0$，故两者互为相反数。',
      },
      // 题 3：input — arctan 导数值
      {
        id: 'c11-c-q3',
        type: 'input',
        prompt:
          '求 $(\\arctan x)^{\\prime}$ 在 $x=1$ 处的值。答案化简为分数（填分子，分母为 2）。',
        accept: ['1/2', '0.5'],
        explain:
          '$(\\arctan x)^{\\prime}=\\dfrac{1}{1+x^2}$。代 $x=1$：$\\dfrac{1}{1+1}=\\dfrac{1}{2}$。',
      },
      // 题 4：choice — 对数求导法的适用场景
      {
        id: 'c11-c-q4',
        type: 'choice',
        prompt: '下列哪种函数**最需要**用对数求导法？',
        options: [
          '$y = x^5$',
          '$y = 3^x$',
          '$y = x^{\\sin x}$',
          '$y = \\sin x \\cdot e^x$',
        ],
        answer: 2,
        explain:
          '$y=x^{\\sin x}$：底数 $x$ 和指数 $\\sin x$ 都含 $x$，普通幂法则和指数法则均失效，必须用对数求导法。' +
          '其余三个用基本法则或乘积法则即可处理。',
      },
      // 题 5：input — x^x 的导数验证
      {
        id: 'c11-c-q5',
        type: 'input',
        prompt:
          '$(x^x)^{\\prime}=x^x(1+\\ln x)$。在 $x=e$ 处，$(e^e)^{\\prime}=e^e\\cdot c$，$c$ 等于多少？（填整数）',
        accept: ['2'],
        explain:
          '代 $x=e$：$1+\\ln e=1+1=2$。故 $(e^e)^{\\prime}=e^e\\cdot 2=2e^e$，$c=2$。',
      },
      // 题 6：judge — 对数求导步骤
      {
        id: 'c11-c-q6',
        type: 'judge',
        prompt:
          '求 $y=(1+x)^{\\sin x}$ 时，对数求导得 $\\dfrac{y^{\\prime}}{y}=\\cos x\\ln(1+x)+\\dfrac{\\sin x}{1+x}$，正确吗？',
        answer: true,
        explain:
          '$\\ln y=\\sin x\\cdot\\ln(1+x)$。两边对 $x$ 求导（乘积法则）：' +
          '$\\dfrac{y^{\\prime}}{y}=\\cos x\\cdot\\ln(1+x)+\\sin x\\cdot\\dfrac{1}{1+x}$。✓',
      },
      // 题 7：match — 反三角函数与导数配对
      {
        id: 'c11-c-q7',
        type: 'match',
        prompt: '将反三角函数与其导数正确配对：',
        left: [
          '$\\arcsin x$',
          '$\\arccos x$',
          '$\\arctan x$',
          '$\\text{arccot}\\,x$',
        ],
        right: [
          '$\\dfrac{1}{\\sqrt{1-x^2}}$',
          '$-\\dfrac{1}{\\sqrt{1-x^2}}$',
          '$\\dfrac{1}{1+x^2}$',
          '$-\\dfrac{1}{1+x^2}$',
        ],
        explain:
          '反函数法则推导：$(\\arcsin x)^{\\prime}=1/\\sqrt{1-x^2}$；' +
          '$(\\arccos x)^{\\prime}=-1/\\sqrt{1-x^2}$（与 arcsin 差负号）；' +
          '$(\\arctan x)^{\\prime}=1/(1+x^2)$；' +
          '$(\\text{arccot}\\,x)^{\\prime}=-1/(1+x^2)$（与 arctan 差负号）。',
      },
    ],
  },
]
