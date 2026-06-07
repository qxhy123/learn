import type { Lesson } from '../../src/types'

export const LESSONS: Lesson[] = [
  // ─────────────────────────────────────────────────────────────────
  // 关 c6-a  齐次与线性
  // ─────────────────────────────────────────────────────────────────
  {
    id: 'c6-a',
    title: '齐次与线性',
    subtitle: '从"缩放不变"到叠加原理',
    intro: [
      // 卡 1：动机——为什么要区分这些概念
      {
        title: '为什么要区分"齐次""线性"？',
        body:
          '初中说 $y=2x+3$ 是"线性函数"；微积分说 $y^{\\prime}+p(x)y=g(x)$ 是"线性方程"；' +
          '深度学习又把 $y=Wx+b$ 叫"线性层"。\n\n' +
          '这三处"线性"含义**完全不同**！如果不分清，后面学 ODE、矩阵、神经网络结构时会反复混淆。\n\n' +
          '本关先搞清两对概念：**齐次 vs 非齐次**（有没有"常数项"），**线性 vs 非线性**（满不满足叠加原理）。',
        tip: '判定口诀：先算 $f(\\mathbf{0})$，再验 $f(\\alpha x+\\beta y)=\\alpha f(x)+\\beta f(y)$。',
      },
      // 卡 2：齐次函数——定义与直觉
      {
        title: '齐次函数：整体缩放，输出同步',
        body:
          '**定义**：若对任意 $\\alpha>0$ 都有 $f(\\alpha\\mathbf{x})=\\alpha^k f(\\mathbf{x})$，' +
          '则称 $f$ 是 **$k$ 次齐次函数**，$k=1$ 时简称齐次。\n\n' +
          '**直觉**：把输入整体放大 $\\alpha$ 倍，输出恰好放大 $\\alpha^k$ 倍——像一个"均匀响应"的系统。\n\n' +
          '**具体例子**\n\n' +
          '$f(x,y)=3x+2y$：$f(2x,2y)=6x+4y=2f(x,y)$，$k=1$，一次齐次。\n\n' +
          '$f(x,y)=x^2+xy+y^2$：$f(2x,2y)=4x^2+4xy+4y^2=4f(x,y)=2^2 f(x,y)$，$k=2$，二次齐次。\n\n' +
          '$f(x,y)=x+y+1$：$f(0,0)=1\\neq0$，含常数项，**非齐次**。',
        formula: 'f(\\alpha\\mathbf{x})=\\alpha^k f(\\mathbf{x})',
        tip: '关键：齐次函数必须有 $f(\\mathbf{0})=0$。只要含非零常数项，即非齐次。',
      },
      // 卡 3：线性映射——定义（可加性 + 齐次性）
      {
        title: '线性映射：两条公理合一',
        body:
          '**定义**：映射 $T:V\\to W$ 称为线性映射，若对所有 $\\mathbf{x},\\mathbf{y}\\in V$、所有标量 $\\alpha,\\beta$：\n\n' +
          '$T(\\alpha\\mathbf{x}+\\beta\\mathbf{y})=\\alpha T(\\mathbf{x})+\\beta T(\\mathbf{y})$\n\n' +
          '这等价于两条之合：\n\n' +
          '**可加性**：$T(\\mathbf{x}+\\mathbf{y})=T(\\mathbf{x})+T(\\mathbf{y})$\n\n' +
          '**（一次）齐次性**：$T(\\alpha\\mathbf{x})=\\alpha T(\\mathbf{x})$\n\n' +
          '合在一起称为**叠加原理**：输入的线性组合，映射到输出的相同线性组合。\n\n' +
          '**立即推论**：取 $\\alpha=\\beta=0$ 得 $T(\\mathbf{0})=\\mathbf{0}$。**线性映射必须"过原点"**——这是最快的排除条件。',
        formula: 'T(\\alpha\\mathbf{x}+\\beta\\mathbf{y})=\\alpha T(\\mathbf{x})+\\beta T(\\mathbf{y})',
      },
      // 卡 4：分步推导——验证 f(x,y)=3x-2y 是线性映射
      {
        title: '例题精讲：逐步验证线性映射',
        body:
          '**判断 $f(x,y)=3x-2y$ 是否是线性映射。**\n\n' +
          '设 $(x_1,y_1),(x_2,y_2)\\in\\mathbb{R}^2$，$\\alpha,\\beta\\in\\mathbb{R}$。',
        steps: [
          '**计算 $f(\\mathbf{0})$**：$f(0,0)=3\\cdot0-2\\cdot0=0$ ✓（必要条件满足，继续验证）',
          '**验证可加性**：$f((x_1+x_2),(y_1+y_2))=3(x_1+x_2)-2(y_1+y_2)=(3x_1-2y_1)+(3x_2-2y_2)=f(x_1,y_1)+f(x_2,y_2)$ ✓',
          '**验证齐次性**：$f(\\alpha x_1,\\alpha y_1)=3\\alpha x_1-2\\alpha y_1=\\alpha(3x_1-2y_1)=\\alpha f(x_1,y_1)$ ✓',
          '**结论**：两条公理均成立，$f(x,y)=3x-2y$ 是**线性映射**。',
        ],
        tip: '验证线性三步走：① $f(0)=0$；② 可加性；③ 齐次性。全过才算线性。',
      },
      // 卡 5：反例——f(x)=x+1 为何非线性
      {
        title: '反例：$f(x)=x+1$ 不是线性映射',
        body:
          '初中常说 $y=x+1$ 是"线性函数"，但按线性代数的严格定义它**不是**线性映射。\n\n' +
          '**快速验证**：$f(0)=0+1=1\\neq0$。线性映射必须过原点，这里 $f(0)\\neq0$，直接排除。\n\n' +
          '**再验可加性**：$f(x_1+x_2)=(x_1+x_2)+1$，而 $f(x_1)+f(x_2)=(x_1+1)+(x_2+1)=x_1+x_2+2$。两者相差 $1$，不相等。\n\n' +
          '**结论**：$f(x)=x+1$ 不是线性映射，它是**仿射映射**（下一关讲解）。',
        tip: '中学"线性函数"（图像为直线）≠ 线性代数"线性映射"（必须过原点）。不要混用！',
      },
      // 卡 6：方程的齐次性——ODE 场景
      {
        title: '方程的齐次性：右端有没有"强迫项"',
        body:
          '对线性算子 $L$，方程 $L[y]=g(x)$ 中：\n\n' +
          '若 $g(x)\\equiv0$，叫**齐次方程**；否则 $g(x)$ 叫**非齐次项**或**强迫项**，方程叫**非齐次方程**。\n\n' +
          '**判定诀窍**：齐次方程一定有零解 $y\\equiv0$。把 $y=0$ 代入，若方程成立，则大概率是齐次的。\n\n' +
          '**例子对比**\n\n' +
          '$y^{\\prime\\prime}+ 3y^{\\prime}+2y=0$ → 右端 $=0$，**齐次线性 ODE**。\n\n' +
          '$y^{\\prime\\prime}+ 3y^{\\prime}+2y=\\sin x$ → 右端 $\\neq0$，**非齐次线性 ODE**。\n\n' +
          '$Ax=\\mathbf{0}$ → 齐次线性方程组，零解 $x=\\mathbf{0}$ 总存在。\n\n' +
          '$Ax=b$（$b\\neq\\mathbf{0}$）→ 非齐次，零解一般不存在。',
        tip: '口诀：代入 $y=0$（或 $x=\\mathbf{0}$）看方程是否成立——成立则为齐次方程。',
      },
      // 卡 7：线性 ODE 的判断：系数是 x 的函数不影响线性
      {
        title: '易错点：系数是 $x$ 的函数，不影响对 $y$ 的线性性',
        body:
          '线性微分方程的形式是 $a_n(x)y^{(n)}+\\cdots+a_1(x)y^{\\prime}+a_0(x)y=g(x)$。\n\n' +
          '**关键**：系数 $a_i(x)$ 可以是 $x$ 的任意函数（如 $x^2$、$\\sin x$），这不会破坏方程对 $y$ 及其导数的线性性——我们只要求 $y,y^{\\prime},y^{\\prime\\prime},\\dots$ 以**一次幂**出现，且不相互相乘。\n\n' +
          '**对比四个例子**\n\n' +
          '$y^{\\prime\\prime}+ x^2 y=0$：系数 $x^2$ 是 $x$ 的函数，但 $y$ 一次幂，**齐次线性** ODE。\n\n' +
          '$y^{\\prime\\prime}+ y^2=0$：含 $y^2$，**非线性** ODE。\n\n' +
          '$(\\sin x)y^{\\prime}+y=e^x$：$y,y^{\\prime}$ 各一次幂，右端 $e^x\\neq0$，**非齐次线性** ODE。\n\n' +
          '$yy^{\\prime}=1$：含 $y\\cdot y^{\\prime}$，**非线性** ODE。',
        reveal: {
          q: '$y^{\\prime\\prime} - (\\cos x)y^{\\prime} + x^3 y = 0$ 是线性的吗？是齐次的吗？',
          a: '系数 $\\cos x$ 和 $x^3$ 是 $x$ 的函数，$y,y^{\\prime},y^{\\prime\\prime}$ 均以一次幂出现，故**线性**。右端为 $0$，故**齐次**。这是齐次线性 ODE。',
        },
      },
      // 卡 8：小结与条件反射
      {
        title: '小结与条件反射',
        body:
          '**齐次函数**：$f(\\alpha\\mathbf{x})=\\alpha^k f(\\mathbf{x})$；含常数项即非齐次；必有 $f(\\mathbf{0})=0$。\n\n' +
          '**线性映射**：可加性 + 一次齐次性 = 叠加原理；必须过原点；矩阵可以完整表示。\n\n' +
          '**线性方程**：未知量及其导数以一次形式出现，系数为 $x$ 的函数没关系。\n\n' +
          '**齐次方程**：右端为 $0$；必有零解；解集构成向量子空间。\n\n' +
          '条件反射：看到 $f(0)\\neq0$ → 立即排除线性；看到 $y^2$、$yy^{\\prime}$ → 非线性；看到右端为 $0$ → 齐次方程。',
        tip: '线性映射 = 过原点 + 叠加原理。缺一不可。',
      },
    ],
    questions: [
      // 题 1：choice — f(0) 快速判断
      {
        id: 'c6-a-q1',
        type: 'choice',
        prompt:
          '映射 $f(x)=3x+5$，要判断它是否为线性映射，最快的排除方法是？',
        options: [
          '验证 $f(x+y)=f(x)+f(y)$，计算较复杂',
          '计算 $f(0)=5\\neq0$，直接排除线性',
          '画出图像，观察是否为直线',
          '计算 $f(1)$ 和 $f(-1)$，看是否互为相反数',
        ],
        answer: 1,
        explain:
          '线性映射必须满足 $T(\\mathbf{0})=\\mathbf{0}$。这里 $f(0)=3\\cdot0+5=5\\neq0$，直接排除，无需再验证其他条件。这是最快的判断方法。',
      },
      // 题 2：judge — f(x,y)=2x-3y 是线性映射
      {
        id: 'c6-a-q2',
        type: 'judge',
        prompt: '$f(x,y)=2x-3y$ 是线性映射。',
        answer: true,
        explain:
          '$f(0,0)=0$ ✓；$f(x_1+x_2,y_1+y_2)=2(x_1+x_2)-3(y_1+y_2)=f(x_1,y_1)+f(x_2,y_2)$ ✓；$f(\\alpha x,\\alpha y)=\\alpha(2x-3y)=\\alpha f(x,y)$ ✓。两条公理均满足，是线性映射。',
      },
      // 题 3：choice — 识别线性 ODE
      {
        id: 'c6-a-q3',
        type: 'choice',
        prompt:
          '下列哪个方程是**齐次线性** ODE？',
        options: [
          '$y^{\\prime\\prime}+y^2=0$',
          '$y^{\\prime\\prime}+ x^2 y=0$',
          '$y^{\\prime\\prime}+ 3y^{\\prime}+2y=\\sin x$',
          '$yy^{\\prime}=1$',
        ],
        answer: 1,
        explain:
          '$y^{\\prime\\prime}+ x^2 y=0$：系数 $x^2$ 是 $x$ 的函数，$y,y^{\\prime\\prime}$ 均一次幂，无非线性项（关于 $y$），右端 $=0$，故为**齐次线性** ODE。选项 A 含 $y^2$（非线性），C 右端 $\\sin x\\neq0$（非齐次），D 含 $yy^{\\prime}$（非线性）。',
      },
      // 题 4：input — 验证齐次次数
      {
        id: 'c6-a-q4',
        type: 'input',
        prompt:
          '$f(x,y)=x^2+xy+y^2$，计算 $f(3x,3y)/f(x,y)$（其中 $f(x,y)\\neq0$），结果是多少？（即 $3$ 的几次幂）',
        accept: ['9', '9.0', '3^2'],
        placeholder: '输入整数',
        explain:
          '$f(3x,3y)=(3x)^2+(3x)(3y)+(3y)^2=9x^2+9xy+9y^2=9f(x,y)$，故 $f(3x,3y)/f(x,y)=9=3^2$。说明 $f$ 是 $2$ 次齐次函数。',
      },
      // 题 5：judge — 系数为 x 的函数不影响线性
      {
        id: 'c6-a-q5',
        type: 'judge',
        prompt:
          'ODE $y^{\\prime\\prime} + (\\sin x)y^{\\prime} + x^3 y = e^x$ 中，因为系数含 $\\sin x$ 和 $x^3$，所以它是非线性方程。',
        answer: false,
        explain:
          '线性性只要求 **$y$** 及其导数以一次幂出现，且不出现 $y\\cdot y^{\\prime}$ 等非线性项。系数 $\\sin x$、$x^3$ 是关于自变量 $x$ 的函数，**不影响**方程对 $y$ 的线性性。该方程是**非齐次线性** ODE（右端 $e^x\\neq0$）。',
      },
      // 题 6：match — 方程类型
      {
        id: 'c6-a-q6',
        type: 'match',
        prompt: '将每个方程与其正确的类型对应。',
        left: [
          '$y^{\\prime\\prime}+ 4y=0$',
          '$y^{\\prime\\prime}+ y^2=0$',
          '$y^{\\prime\\prime}+ 4y=\\cos x$',
        ],
        right: [
          '齐次线性 ODE',
          '非线性 ODE',
          '非齐次线性 ODE',
        ],
      },
      // 题 7：choice — 齐次方程必有零解
      {
        id: 'c6-a-q7',
        type: 'choice',
        prompt:
          '对于线性齐次方程 $Ax=\\mathbf{0}$，下列说法正确的是？',
        options: [
          '方程可能没有解',
          '$x=\\mathbf{0}$ 一定是解',
          '方程有且只有一个解 $x=\\mathbf{0}$',
          '解集是所有非零向量的集合',
        ],
        answer: 1,
        explain:
          '把 $x=\\mathbf{0}$ 代入：$A\\cdot\\mathbf{0}=\\mathbf{0}$ 恒成立。所以零解 $x=\\mathbf{0}$ **一定**是线性齐次方程的解。是否还有非零解，取决于 $A$ 的秩（奇异性）。',
      },
      // 题 8：input — 验证可加性失败
      {
        id: 'c6-a-q8',
        type: 'input',
        prompt:
          '$f(x)=x+1$，计算 $f(1)+f(2)$ 的值是多少？',
        accept: ['5'],
        placeholder: '输入整数',
        explain:
          '$f(1)+f(2)=(1+1)+(2+1)=2+3=5$。而 $f(1+2)=f(3)=3+1=4\\neq5$，可加性失败，证明 $f(x)=x+1$ 不是线性映射。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────
  // 关 c6-b  仿射与非线性
  // ─────────────────────────────────────────────────────────────────
  {
    id: 'c6-b',
    title: '仿射与非线性',
    subtitle: '线性 + 平移，以及激活函数的角色',
    intro: [
      // 卡 1：动机——线性 vs 仿射的混淆
      {
        title: '为什么要专门讲"仿射"？',
        body:
          '神经网络里的"全连接层"被叫做 Linear Layer（线性层），' +
          '代码写的是 $y=Wx+b$。但按线性代数的严格定义，当 $b\\neq\\mathbf{0}$ 时，' +
          '这是**仿射变换**，不是线性变换。\n\n' +
          '理解这一区别，会帮你搞清：\n\n' +
          '① 为什么纯堆仿射层没有用（多层仍等于单层）；\n\n' +
          '② 为什么激活函数（ReLU、Sigmoid 等）是网络表达力的真正来源；\n\n' +
          '③ 为什么 Conv + BatchNorm 时可以省掉卷积偏置 $b$。',
      },
      // 卡 2：仿射映射的定义
      {
        title: '仿射映射：线性映射 + 常数平移',
        body:
          '**定义**：映射 $f:V\\to W$ 称为**仿射映射**，若存在线性映射 $L$ 和常向量 $\\mathbf{b}\\in W$，使得\n\n' +
          '$f(\\mathbf{x})=L(\\mathbf{x})+\\mathbf{b}$\n\n' +
          '**等价刻画**：$f$ 仿射 $\\iff$ $f(\\mathbf{x})-f(\\mathbf{0})$ 是线性映射。\n\n' +
          '**几何含义**：仿射映射把直线映为直线，但不一定过原点——它是线性变换加了一个"平移"。\n\n' +
          '**包含关系**：线性映射是 $\\mathbf{b}=\\mathbf{0}$ 的特殊仿射映射。因此：**线性 $\\subset$ 仿射 $\\subset$ 所有映射**。',
        formula: 'f(\\mathbf{x})=A\\mathbf{x}+\\mathbf{b}',
        tip: '判断仿射：先算 $f(\\mathbf{0})$；若 $\\neq\\mathbf{0}$，再看 $g(\\mathbf{x})=f(\\mathbf{x})-f(\\mathbf{0})$ 是否线性。若 $g$ 线性，则 $f$ 仿射。',
      },
      // 卡 3：仿射只保持"系数和为 1"的组合
      {
        title: '仿射映射保持"系数和为 1"的组合',
        body:
          '线性映射保持**所有**线性组合；仿射映射只保持**系数和为 1** 的组合（称为仿射组合）。\n\n' +
          '**证明**：设 $\\sum_i\\alpha_i=1$，则\n\n' +
          '$f(\\sum_i\\alpha_i\\mathbf{x}_i)=A\\sum_i\\alpha_i\\mathbf{x}_i+\\mathbf{b}=\\sum_i\\alpha_iA\\mathbf{x}_i+(\\sum_i\\alpha_i)\\mathbf{b}=\\sum_i\\alpha_i(A\\mathbf{x}_i+\\mathbf{b})=\\sum_i\\alpha_if(\\mathbf{x}_i)$。\n\n' +
          '若 $\\sum_i\\alpha_i\\neq1$，多出的 $(\\sum_i\\alpha_i-1)\\mathbf{b}\\neq\\mathbf{0}$，等式不成立。\n\n' +
          '**实例**：直线参数化 $(1-t)\\mathbf{x}_0+t\\mathbf{x}_1$（系数和 $=1$），仿射映射把这段直线映为直线——这就是"仿射映射保持直线"的代数原因。',
        formula: 'f\\!\\left(\\sum_i\\alpha_i\\mathbf{x}_i\\right)=\\sum_i\\alpha_i f(\\mathbf{x}_i),\\quad\\sum_i\\alpha_i=1',
      },
      // 卡 4：三步判定流程（例题精讲）
      {
        title: '例题精讲：三步判定映射类型',
        body:
          '**判断以下三个映射各属于哪类：**\n\n' +
          '(a) $f(x,y)=2x-3y$；(b) $f(x,y)=2x-3y+1$；(c) $f(x,y)=xy$。',
        steps: [
          '**第一步：算 $f(\\mathbf{0})$**。(a) $f(0,0)=0$；(b) $f(0,0)=1\\neq0$；(c) $f(0,0)=0$。',
          '**第二步（$f(\\mathbf{0})=0$ 时）验可加性/齐次性**。(a) 可验证满足叠加原理，**线性**。(c) $f(\\alpha x,\\alpha y)=\\alpha^2 xy\\neq\\alpha f(x,y)$（除非 $\\alpha^2=\\alpha$），**非线性**（二次齐次但不是一次线性）。',
          '**第三步（$f(\\mathbf{0})\\neq0$ 时）看 $g=f-f(\\mathbf{0})$ 是否线性**。(b) $g(x,y)=f(x,y)-1=2x-3y$，与(a)相同，**线性**。故(b)**仿射但非线性**。',
          '**结论**：(a) 线性；(b) 仿射但非线性；(c) 非线性（且非仿射）。',
        ],
        tip: '三步走：① $f(0)\\neq0$？排除线性；② 验叠加原理；③ 看 $f-f(0)$ 是否线性。',
      },
      // 卡 5：神经网络视角——仿射层与激活函数
      {
        title: '神经网络：仿射层 + 激活 = 非线性',
        body:
          '标准全连接层（"线性层"）的计算：$\\mathbf{z}=W\\mathbf{x}+\\mathbf{b}$。\n\n' +
          '当 $\\mathbf{b}\\neq\\mathbf{0}$ 时，这是**仿射变换**，不是线性变换。\n\n' +
          '加上激活函数：$\\mathbf{a}=\\sigma(\\mathbf{z})=\\sigma(W\\mathbf{x}+\\mathbf{b})$。\n\n' +
          '$\\sigma$ 如 ReLU $=\\max(0,x)$、Sigmoid $=\\dfrac{1}{1+e^{-x}}$、Tanh，都是非线性逐元素函数。\n\n' +
          '**整体**：仿射 + 非线性激活 $\\Rightarrow$ 非线性映射。这才是神经网络获得强大表达力的原因。\n\n' +
          '**若去掉激活函数**：多层叠加 $W_2(W_1\\mathbf{x}+\\mathbf{b}_1)+\\mathbf{b}_2=(W_2W_1)\\mathbf{x}+(W_2\\mathbf{b}_1+\\mathbf{b}_2)=W^{\\prime}\\mathbf{x}+\\mathbf{b}^{\\prime}$，仍是单层仿射——深度没有意义。',
        formula: '\\mathbf{y}=W_2(W_1\\mathbf{x}+\\mathbf{b}_1)+\\mathbf{b}_2=W^{\\prime}\\mathbf{x}+\\mathbf{b}^{\\prime}',
        tip: '无激活的多层仿射网络，等价于单层仿射。激活函数是打破这一等价的唯一手段。',
      },
      // 卡 6：ReLU 是非线性的——详细论证
      {
        title: 'ReLU 为什么是非线性？分步论证',
        body:
          '**$f(x)=\\max(0,x)$（ReLU）的类型分析**',
        steps: [
          '**步骤一：算 $f(0)$**。$f(0)=\\max(0,0)=0$，满足过原点的必要条件。',
          '**步骤二：验可加性**。取 $x_1=-1,x_2=2$：$f(-1+2)=f(1)=1$；$f(-1)+f(2)=0+2=2$。$1\\neq2$，**可加性失败**。',
          '**步骤三：结论**。ReLU 不满足可加性（在折点 $x=0$ 处折断了叠加），故为**非线性**映射。',
          '**补充：仿射吗？** 仿射要求 $f(x)-f(0)=f(x)$ 是线性。但步骤二已证 $f$ 非线性（且 $f(0)=0$），故 ReLU 也**不是仿射**映射。',
        ],
        tip: 'ReLU 是分段线性，但整体非线性——折点打破了叠加原理。',
      },
      // 卡 7：易错点——"仿射"不等于"线性"；"非线性"的范围
      {
        title: '易错点汇总与快速自测',
        body:
          '**易错点 1**：$y=kx+b$（$b\\neq0$）是中学的"线性函数"，但严格说是**仿射**函数，不是线性映射。\n\n' +
          '**易错点 2**：神经网络的"线性层"（`nn.Linear`）其实是**仿射变换**。只有 `bias=False` 才是严格的线性映射。\n\n' +
          '**易错点 3**："非线性"是"仿射"的补集——**仿射通常不被称为非线性**。凡不能写成 $A\\mathbf{x}+\\mathbf{b}$ 的才叫非线性。\n\n' +
          '**易错点 4**：$f(x,y)=x^2$ 是 2 次齐次（$f(\\alpha x,\\alpha y)=\\alpha^2 x^2=\\alpha^2 f$），但**不满足一次线性**（$f(2x)=4x^2\\neq2x^2=2f(x)$）。\n\n' +
          '**概念层级（记牢）**：线性 $\\subset$ 仿射 $\\subset$ 所有映射；非线性 $=$ 仿射的补集。',
        reveal: {
          q: 'PyTorch 中 `nn.Linear(4, 3)` 默认带偏置。这是线性变换还是仿射变换？若设 `bias=False`，结论如何？',
          a: '默认带偏置 $b\\neq\\mathbf{0}$，是**仿射变换**，不是线性变换。设 `bias=False` 后偏置为 $\\mathbf{0}$，退化为**线性变换** $y=Wx$。',
        },
      },
    ],
    questions: [
      // 题 1：choice — 仿射的定义
      {
        id: 'c6-b-q1',
        type: 'choice',
        prompt:
          '映射 $f(x)=5x-2$ 属于哪种类型？',
        options: [
          '线性映射',
          '仿射映射（非线性）',
          '非线性且非仿射',
          '二次齐次函数',
        ],
        answer: 1,
        explain:
          '$f(0)=5\\cdot0-2=-2\\neq0$，排除线性。$g(x)=f(x)-f(0)=f(x)+2=5x$ 是线性映射，故 $f$ 是**仿射映射但非线性**。',
      },
      // 题 2：judge — 多层无激活网络等价于单层仿射
      {
        id: 'c6-b-q2',
        type: 'judge',
        prompt:
          '两层无激活网络 $y=W_2(W_1 x+b_1)+b_2$ 可以表示为单层仿射变换 $y=W^{\\prime}x+b^{\\prime}$。',
        answer: true,
        explain:
          '展开：$y=W_2W_1 x+W_2 b_1+b_2$。令 $W^{\\prime}=W_2W_1$，$b^{\\prime}=W_2 b_1+b_2$，即为单层仿射 $W^{\\prime}x+b^{\\prime}$。无论堆多少层，无激活的仿射网络都等价于单层仿射，无法增加表达能力。',
      },
      // 题 3：input — 两层仿射合并
      {
        id: 'c6-b-q3',
        type: 'input',
        prompt:
          '一维情形：$y=w_2(w_1 x+b_1)+b_2$，其中 $w_1=2,b_1=3,w_2=4,b_2=1$。等价单层 $y=w^{\\prime}x+b^{\\prime}$ 中，$b^{\\prime}$ 等于多少？',
        accept: ['13'],
        placeholder: '输入整数',
        explain:
          '$b^{\\prime}=w_2 b_1+b_2=4\\times3+1=12+1=13$。等价形式为 $y=8x+13$。这说明两层仿射合并后只是单层仿射，参数量减少但表达能力不变。',
      },
      // 题 4：choice — ReLU 的类型
      {
        id: 'c6-b-q4',
        type: 'choice',
        prompt:
          '$f(x)=\\max(0,x)$（ReLU），判断其映射类型。',
        options: [
          '线性映射',
          '仿射映射（非线性）',
          '非线性且非仿射',
          '二次齐次函数',
        ],
        answer: 2,
        explain:
          '$f(0)=0$，但验可加性：$f(-1+2)=f(1)=1$，$f(-1)+f(2)=0+2=2$，$1\\neq2$，可加性失败。ReLU 不是线性也不是仿射，属于**非线性且非仿射**映射。',
      },
      // 题 5：judge — 仿射保持仿射组合
      {
        id: 'c6-b-q5',
        type: 'judge',
        prompt:
          '仿射映射 $f(\\mathbf{x})=A\\mathbf{x}+\\mathbf{b}$ 保持系数和为 $2$ 的线性组合：$f(2\\mathbf{x}_1)=2f(\\mathbf{x}_1)$。',
        answer: false,
        explain:
          '$f(2\\mathbf{x}_1)=A(2\\mathbf{x}_1)+\\mathbf{b}=2A\\mathbf{x}_1+\\mathbf{b}$，而 $2f(\\mathbf{x}_1)=2(A\\mathbf{x}_1+\\mathbf{b})=2A\\mathbf{x}_1+2\\mathbf{b}$。两者相差 $\\mathbf{b}$，当 $\\mathbf{b}\\neq\\mathbf{0}$ 时不相等。仿射映射只保持**系数和为 $1$** 的组合。',
      },
      // 题 6：match — 映射类型对应例子
      {
        id: 'c6-b-q6',
        type: 'match',
        prompt: '将映射与其正确类型对应。',
        left: [
          '$f(x)=3x$',
          '$f(x)=3x+5$',
          '$f(x)=x^2$',
        ],
        right: [
          '线性映射',
          '仿射映射（非线性）',
          '非线性且非仿射',
        ],
      },
      // 题 7：choice — Conv + BN 省掉偏置
      {
        id: 'c6-b-q7',
        type: 'choice',
        prompt:
          '卷积层紧跟 BatchNorm 时，常设 `bias=False`（去掉偏置 $b$）。从仿射变换的角度，下列哪个理由最准确？',
        options: [
          '去掉偏置可以加速计算，但会损失模型精度',
          'BatchNorm 先减批均值，偏置 $b$ 被均值吸收而失效；BN 末端 $\\beta$ 已提供平移自由度，$b$ 冗余',
          '卷积层必须是纯线性变换，不允许偏置',
          '偏置会导致 BN 统计量不稳定，引发梯度爆炸',
        ],
        answer: 1,
        explain:
          'BN 对每个通道减去批均值 $\\mu$，而卷积层的偏置 $b$ 是常数，会被 $\\mu$ 吸收并消去——所以 $b$ 对 BN 后的输出毫无贡献。同时 BN 末端的可学习参数 $\\beta$ 已经提供了等价的平移自由度，故卷积偏置是冗余参数，去掉不损失表达能力。',
      },
      // 题 8：input — 验证仿射保持中点（系数和为 1）
      {
        id: 'c6-b-q8',
        type: 'input',
        prompt:
          '设 $f(x)=2x+3$，$x_1=1,x_2=5$，中点 $m=\\frac{x_1+x_2}{2}=3$。计算 $f(m)$ 的值。',
        accept: ['9'],
        placeholder: '输入整数',
        explain:
          '$f(3)=2\\cdot3+3=9$。同时 $\\frac{f(1)+f(5)}{2}=\\frac{5+13}{2}=9$，与 $f(m)$ 相等——这正说明仿射映射保持中点（系数各 $\\frac{1}{2}$，和为 $1$）。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────
  // 关 c6-c  ODE 解结构：齐次通解 + 特解
  // ─────────────────────────────────────────────────────────────────
  {
    id: 'c6-c',
    title: 'ODE 解结构',
    subtitle: '叠加原理 · 齐次通解 + 特解 · 线性方程组对应',
    intro: [
      // 卡 1：动机——为什么先解齐次再找特解
      {
        title: '为什么 ODE 求解要先解齐次方程？',
        body:
          '解一阶线性 ODE $y^{\\prime}+y=e^x$ 的标准流程，教材都会说：\n\n' +
          '第一步，解齐次方程 $y^{\\prime}+y=0$；第二步，找一个特解；第三步，两者相加。\n\n' +
          '但为什么是这三步？背后的原理是**叠加原理**——线性算子保证了"齐次通解 + 特解"能覆盖所有解。\n\n' +
          '理解这一点，ODE、线性方程组、泛函分析中的"通解结构定理"就是同一个定理的不同面貌。',
      },
      // 卡 2：叠加原理
      {
        title: '叠加原理：线性算子的核心性质',
        body:
          '设 $L$ 是线性算子（例如 $L[y]=y^{\\prime}+py^{\\prime}$ 或 $L[\\mathbf{x}]=A\\mathbf{x}$）。\n\n' +
          '**齐次叠加**：若 $L[y_1]=0$ 且 $L[y_2]=0$，则对任意常数 $c_1,c_2$，\n\n' +
          '$L[c_1y_1+c_2y_2]=c_1L[y_1]+c_2L[y_2]=0$\n\n' +
          '即齐次方程的解集是**向量子空间**（对线性组合封闭）。\n\n' +
          '**非齐次叠加**：若 $L[y_p]=g$ 且 $L[y_h]=0$，则\n\n' +
          '$L[y_h+y_p]=L[y_h]+L[y_p]=0+g=g$\n\n' +
          '即"齐次解 + 特解"仍满足非齐次方程。',
        formula: 'L[y_h+y_p]=g,\\quad L[y_h]=0,\\;L[y_p]=g',
      },
      // 卡 3：通解结构定理
      {
        title: '通解结构定理：全部解 = 齐次通解 + 任一特解',
        body:
          '**定理**：线性方程 $L[y]=g$ 的全部解为\n\n' +
          '$y=y_h+y_p$\n\n' +
          '其中 $y_h$ 取遍齐次方程 $L[y]=0$ 的所有解，$y_p$ 是非齐次方程的**任一**特解。\n\n' +
          '**为什么"全部"？** 设 $\\tilde{y}$ 是 $L[y]=g$ 的任意解，则 $L[\\tilde{y}-y_p]=g-g=0$，' +
          '故 $\\tilde{y}-y_p$ 是齐次解，即 $\\tilde{y}=y_h+y_p$ 的某个实例。所以没有遗漏。\n\n' +
          '**应用流程**：① 先解 $L[y]=0$ 得 $y_h$；② 用待定系数或参数变易法找 $y_p$；③ 写出 $y=y_h+y_p$；④ 代入初值条件定常数。',
        formula: 'y=y_h+y_p',
      },
      // 卡 4：例题——y' + y = e^x
      {
        title: '例题精讲：$y^{\\prime}+y=e^x$ 的通解',
        body:
          '这是章节思维路径还原的原题，我们完整走一遍。',
        steps: [
          '**确认类型**：$y^{\\prime}$ 和 $y$ 各一次幂，关于 $y$ 线性；右端 $e^x\\neq0$，**非齐次线性 ODE**。',
          '**解齐次方程** $y^{\\prime}+y=0$：分离变量 $\\dfrac{dy}{y}=-dx$，积分得 $\\ln|y|=-x+C_0$，故 $y_h=Ce^{-x}$（$C$ 为任意常数）。',
          '**设特解** $y_p=Ae^x$（右端 $e^x$ 形式提示），代入：$(Ae^x)^{\\prime}+Ae^x=Ae^x+Ae^x=2Ae^x=e^x$，得 $A=\\dfrac{1}{2}$，故 $y_p=\\dfrac{1}{2}e^x$。',
          '**写通解**：$y=y_h+y_p=Ce^{-x}+\\dfrac{1}{2}e^x$（$C$ 为任意常数）。',
          '**验证**：$y^{\\prime}=-Ce^{-x}+\\dfrac{1}{2}e^x$，$y^{\\prime}+y=(-Ce^{-x}+\\dfrac{1}{2}e^x)+(Ce^{-x}+\\dfrac{1}{2}e^x)=e^x$ ✓。',
        ],
        formula: 'y=Ce^{-x}+\\frac{1}{2}e^{x}',
      },
      // 卡 5：线性方程组的对应——特解 + 零空间
      {
        title: '线性方程组对应：$Ax=b$ 的解结构',
        body:
          '通解结构定理不只适用于 ODE，对线性方程组 $A\\mathbf{x}=\\mathbf{b}$ 同样成立。\n\n' +
          '若 $A\\mathbf{x}_p=\\mathbf{b}$（特解），$\\mathbf{v}\\in\\ker A$（$A\\mathbf{v}=\\mathbf{0}$），则\n\n' +
          '$A(\\mathbf{x}_p+\\mathbf{v})=A\\mathbf{x}_p+A\\mathbf{v}=\\mathbf{b}+\\mathbf{0}=\\mathbf{b}$ ✓\n\n' +
          '故全部解为 $\\mathbf{x}=\\mathbf{x}_p+\\mathbf{v}$，$\\mathbf{v}\\in\\ker A$。\n\n' +
          '**几何含义**：齐次方程 $A\\mathbf{x}=\\mathbf{0}$ 的解集（零空间）是过原点的子空间；' +
          '非齐次方程的解集是该子空间沿 $\\mathbf{x}_p$ 平移后的**仿射子空间**——这就是"仿射集"的由来。',
        formula: '\\mathbf{x}=\\mathbf{x}_p+\\mathbf{v},\\quad\\mathbf{v}\\in\\ker A',
        tip: '关键联系：ODE "齐次通解 + 特解" = 矩阵论 "零空间 + 特解"，本质是同一定理。',
      },
      // 卡 6：待定系数法提示——如何猜特解形式
      {
        title: '怎么猜特解？待定系数法要点',
        body:
          '对常系数线性 ODE，特解形式与右端 $g(x)$ 的结构有关：\n\n' +
          '**右端是多项式** $x^n$：设 $y_p=a_nx^n+\\cdots+a_0$（同次多项式）。\n\n' +
          '**右端是指数** $e^{\\lambda x}$：设 $y_p=Ae^{\\lambda x}$（除非 $\\lambda$ 是齐次方程的特征根，则乘 $x$）。\n\n' +
          '**右端是正弦/余弦** $\\sin(\\omega x)$ 或 $\\cos(\\omega x)$：设 $y_p=A\\sin(\\omega x)+B\\cos(\\omega x)$。\n\n' +
          '**例**：$y^{\\prime}+2y=4x$。右端是 $1$ 次多项式，设 $y_p=ax+b$，代入：$(ax+b)^{\\prime}+2(ax+b)=a+2ax+2b=4x+0$，对比系数得 $2a=4\\Rightarrow a=2$，$a+2b=0\\Rightarrow b=-1$，故 $y_p=2x-1$。',
        reveal: {
          q: '$y^{\\prime}+3y=6$，应如何设特解 $y_p$？代入后求出 $y_p$。',
          a: '右端为常数（$0$ 次多项式），设 $y_p=A$（常数）。代入：$0+3A=6$，得 $A=2$，故 $y_p=2$。',
        },
      },
      // 卡 7：解结构汇总与易错点
      {
        title: '解结构汇总与易错点',
        body:
          '**易错点 1**：特解 $y_p$ 只需找**一个**，不用是最一般的形式。\n\n' +
          '**易错点 2**：齐次通解里的任意常数（$C$）是在代入初值**之后**才确定的——通解本身不确定 $C$。\n\n' +
          '**易错点 3**：非齐次方程的解集**不是**子空间（因为不包含 $\\mathbf{0}$，零解不存在）；它是仿射子空间（过特解的平行平面/平行线）。\n\n' +
          '**易错点 4**：叠加原理只适用于**线性** ODE。非线性方程（如 $y^{\\prime}=y^2$）没有"通解 = 齐次通解 + 特解"的结构。\n\n' +
          '**小结**：叠加原理 $\\Rightarrow$ 通解结构定理 $\\Rightarrow$ "先解齐次再找特解"的标准流程——这是贯穿微分方程、线性代数的核心逻辑。',
        tip: '非线性 ODE 没有通解结构定理——每道题都要重新想思路，没有万能套路。',
      },
    ],
    questions: [
      // 题 1：choice — 叠加原理
      {
        id: 'c6-c-q1',
        type: 'choice',
        prompt:
          '已知 $y_1,y_2$ 均满足 $y^{\\prime\\prime}+ 4y=0$，则 $3y_1-2y_2$ 满足哪个方程？',
        options: [
          '$y^{\\prime\\prime}+ 4y=3-2=1$',
          '$y^{\\prime\\prime}+ 4y=0$',
          '$y^{\\prime\\prime}+ 4y=5$',
          '无法确定',
        ],
        answer: 1,
        explain:
          '由叠加原理，线性算子 $L[y]=y^{\\prime\\prime}+ 4y$ 满足 $L[3y_1-2y_2]=3L[y_1]-2L[y_2]=3\\cdot0-2\\cdot0=0$。故 $3y_1-2y_2$ 也满足 $y^{\\prime\\prime}+ 4y=0$（齐次叠加）。',
      },
      // 题 2：input — 特解待定系数
      {
        id: 'c6-c-q2',
        type: 'input',
        prompt:
          '$y^{\\prime}+2y=4x$，设特解 $y_p=ax+b$，代入后 $a=2,b=-1$，通解为 $y=Ce^{-2x}+y_p$。则 $y_p$ 在 $x=1$ 时的值是多少？',
        accept: ['1', '1.0'],
        placeholder: '输入整数',
        explain:
          '$y_p=2x-1$，代入 $x=1$：$y_p(1)=2\\cdot1-1=1$。通解为 $y=Ce^{-2x}+2x-1$（$C$ 为任意常数）。',
      },
      // 题 3：judge — 非齐次方程解集是向量子空间
      {
        id: 'c6-c-q3',
        type: 'judge',
        prompt:
          '非齐次线性 ODE $L[y]=g$（$g\\neq0$）的解集是向量子空间。',
        answer: false,
        explain:
          '非齐次方程通常没有零解（代入 $y=0$ 得 $L[0]=0\\neq g$），所以解集不包含零元素，**不是向量子空间**。它是仿射子空间：$\\{y_p+y_h:L[y_h]=0\\}$，是齐次解空间的平移。',
      },
      // 题 4：choice — y' + y = e^x 的通解
      {
        id: 'c6-c-q4',
        type: 'choice',
        prompt:
          '求 $y^{\\prime}+y=e^x$ 的通解（$C$ 为任意常数）。',
        options: [
          '$y=Ce^{x}+\\dfrac{1}{2}e^x$',
          '$y=Ce^{-x}+\\dfrac{1}{2}e^x$',
          '$y=Ce^{-x}+2e^x$',
          '$y=Ce^{x}-\\dfrac{1}{2}e^x$',
        ],
        answer: 1,
        explain:
          '齐次方程 $y^{\\prime}+y=0$ 的通解为 $y_h=Ce^{-x}$。设特解 $y_p=Ae^x$，代入：$2Ae^x=e^x$，$A=\\dfrac{1}{2}$，故 $y_p=\\dfrac{1}{2}e^x$。通解：$y=Ce^{-x}+\\dfrac{1}{2}e^x$。',
      },
      // 题 5：input — 初值问题
      {
        id: 'c6-c-q5',
        type: 'input',
        prompt:
          '$y^{\\prime}+y=e^x$，通解为 $y=Ce^{-x}+\\dfrac{1}{2}e^x$，初始条件 $y(0)=1$，求 $C$。',
        accept: ['1/2', '0.5'],
        placeholder: '输入分数或小数',
        explain:
          '代入 $x=0$，$y(0)=C\\cdot1+\\dfrac{1}{2}\\cdot1=C+\\dfrac{1}{2}=1$，故 $C=\\dfrac{1}{2}$。特解（满足初值的解）为 $y=\\dfrac{1}{2}e^{-x}+\\dfrac{1}{2}e^x=\\cosh x$。',
      },
      // 题 6：match — 解结构与方程类型
      {
        id: 'c6-c-q6',
        type: 'match',
        prompt: '将解的结构与对应的方程类型对应。',
        left: [
          '解集为向量子空间（过原点）',
          '解集为仿射子空间（过特解）',
          '叠加原理失效',
        ],
        right: [
          '线性**齐次**方程 $L[y]=0$',
          '线性**非齐次**方程 $L[y]=g$（$g\\neq0$）',
          '**非线性**方程',
        ],
      },
      // 题 7：judge — 线性方程组的通解结构
      {
        id: 'c6-c-q7',
        type: 'judge',
        prompt:
          '线性方程组 $Ax=b$ 的通解为 $x=x_p+v$，其中 $x_p$ 是任一特解，$v\\in\\ker A$（即 $Av=\\mathbf{0}$）。',
        answer: true,
        explain:
          '由通解结构定理：$A(x_p+v)=Ax_p+Av=b+\\mathbf{0}=b$，所以 $x_p+v$ 仍是解。反过来，任何解 $\\tilde{x}$ 满足 $A(\\tilde{x}-x_p)=\\mathbf{0}$，故 $\\tilde{x}-x_p\\in\\ker A$，即 $\\tilde{x}=x_p+(\\tilde{x}-x_p)$，符合结构。',
      },
      // 题 8：choice — 特解设法
      {
        id: 'c6-c-q8',
        type: 'choice',
        prompt:
          '$y^{\\prime}+ 3y=6$（右端为常数），用待定系数法设特解 $y_p$，最合适的形式是？',
        options: [
          '$y_p=Ax+B$（一次多项式）',
          '$y_p=A$（常数）',
          '$y_p=Ae^{3x}$（指数）',
          '$y_p=A\\sin x+B\\cos x$（三角）',
        ],
        answer: 1,
        explain:
          '右端 $g(x)=6$ 是 $0$ 次多项式（常数），待定系数法对应设 $y_p=A$（同次多项式）。代入：$0+3A=6$，$A=2$，故 $y_p=2$。',
      },
    ],
  },
]
