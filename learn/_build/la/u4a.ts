export const LESSONS = [
  // ─────────────────────────────────────────────────────────────────
  // 关卡 1：向量空间的公理与例子
  // ─────────────────────────────────────────────────────────────────
  {
    id: 'u4-l1',
    title: '向量空间的公理与例子',
    subtitle: '八条公理统一 $\\mathbb{R}^n$、矩阵与多项式',
    intro: [
      {
        title: '为什么需要向量空间？',
        body: '列向量、矩阵、多项式、连续函数——看似毫无关联，却都满足同一套"加法与数乘"规则。数学家将这套共同规则提炼成**向量空间**的定义：证明一次，所有满足公理的结构全部受益。\n\n直觉上，向量空间就是一个"对加法和数乘封闭"的集合——两个元素相加还在里面，乘以任意标量还在里面。',
        tip: '向量空间的关键不是"向量是什么形状"，而是"运算满足什么规则"。',
      },
      {
        title: '向量空间的正式定义',
        body: '设 $V$ 是非空集合，$\\mathbb{F}$ 是数域（通常取 $\\mathbb{R}$）。若 $V$ 上定义了向量加法 $V \\times V \\to V$ 和标量乘法 $\\mathbb{F} \\times V \\to V$，并满足下面八条公理，则称 $(V, +, \\cdot)$ 为 $\\mathbb{F}$ 上的**向量空间**。',
        formula: 'V \\neq \\emptyset, \\quad V \\text{ 对加法与标量乘法封闭，满足八条公理}',
        tip: '$V$ 的元素统称"向量"，不限于几何箭头——矩阵、多项式都可以是"向量"。',
      },
      {
        title: '加法五条公理（A1–A5）',
        body: '**A1 加法封闭**：$\\mathbf{u} + \\mathbf{v} \\in V$\n**A2 交换律**：$\\mathbf{u} + \\mathbf{v} = \\mathbf{v} + \\mathbf{u}$\n**A3 结合律**：$(\\mathbf{u} + \\mathbf{v}) + \\mathbf{w} = \\mathbf{u} + (\\mathbf{v} + \\mathbf{w})$\n**A4 零向量**：存在 $\\mathbf{0} \\in V$ 使 $\\mathbf{v} + \\mathbf{0} = \\mathbf{v}$\n**A5 加法逆元**：对每个 $\\mathbf{v}$ 存在 $-\\mathbf{v} \\in V$ 使 $\\mathbf{v} + (-\\mathbf{v}) = \\mathbf{0}$',
        tip: 'A2–A5 说明 $V$ 关于加法构成**交换群**。A1 封闭性最容易被遗漏。',
      },
      {
        title: '标量乘法四条公理（S1–S4）',
        body: '**S1 数乘封闭**：$c\\mathbf{v} \\in V$\n**S2 数乘结合律**：$(cd)\\mathbf{v} = c(d\\mathbf{v})$\n**S3 单位元**：$1 \\cdot \\mathbf{v} = \\mathbf{v}$\n**S4a 向量分配律**：$c(\\mathbf{u} + \\mathbf{v}) = c\\mathbf{u} + c\\mathbf{v}$\n**S4b 标量分配律**：$(c + d)\\mathbf{v} = c\\mathbf{v} + d\\mathbf{v}$',
        tip: 'S1–S4 保证标量乘法与加法"相容"。S1 和 A1 合称**封闭性**，是验证向量空间时最先要查的。',
      },
      {
        title: '典型例子一览',
        body: '以下集合在通常的加法和数乘下均构成向量空间：\n\n- $\\mathbb{R}^n$：$n$ 维实列向量，零向量为 $(0,\\ldots,0)^T$\n- $\\mathbb{R}^{m \\times n}$：全体 $m \\times n$ 实矩阵，零向量为全零矩阵 $O$\n- $\\mathcal{P}_n$：次数不超过 $n$ 的实多项式，零向量为零多项式\n- $C[a,b]$：闭区间上的连续函数，零向量为零函数 $f(x) = 0$',
        tip: '$C[a,b]$ 是**无穷维**向量空间；$\\mathcal{P}_n$ 维数为 $n+1$；$\\mathbb{R}^{m \\times n}$ 维数为 $mn$。',
      },
      {
        title: '如何证明一个集合不是向量空间',
        body: '只需找到**一条公理失效**的具体反例，无需验证其余。\n\n**优先检查顺序**：\n1. 零向量是否在集合中（A4）\n2. 取 $c = -1$ 验证 S1（数乘封闭）\n3. 取两个端点元素相加验证 A1（加法封闭）\n\n**例**：$S = \\{(x,y) \\mid x^2 + y^2 = 1\\}$（单位圆），取 $(1,0) \\in S$，$2 \\times (1,0) = (2,0)$，但 $4 \\neq 1$，$(2,0) \\notin S$。S1 失效，否定。',
        reveal: { q: '集合 $\\{(x,y) \\mid x + y = 1\\}$ 是向量空间吗？', a: '不是。零向量 $(0,0)$ 满足 $0 + 0 = 0 \\neq 1$，不在集合中，A4 失效。也可用 A1：$(1,0) + (0,1) = (1,1)$，$1+1=2 \\neq 1$，不在集合中。' },
      },
      {
        title: '验证向量空间的标准四步',
        body: '面对"验证 $V$ 是向量空间"的题目，按以下步骤逐一检查：\n\n**步骤1** 写清集合 $V$ 的元素形式和运算定义。\n**步骤2** 验证封闭性（A1, S1）：取任意元素证明和与数乘结果仍在 $V$ 中。\n**步骤3** 找出零向量（A4）并验证加法逆元（A5）。\n**步骤4** 其余公理（A2, A3, S2, S3, S4）通常由数域性质直接继承，说明即可。',
        tip: '自定义运算时零向量不一定是"各分量为 0"——先用 $\\mathbf{v} + \\mathbf{0} = \\mathbf{v}$ 倒推零向量的具体形式。',
      },
    ],
    questions: [
      {
        id: 'u4-l1-q1',
        type: 'judge',
        prompt: '集合 $Q = \\{(x, y) \\in \\mathbb{R}^2 \\mid x \\geq 0, y \\geq 0\\}$（第一象限含坐标轴）在通常的向量加法和数乘下构成向量空间。',
        answer: false,
        explain: '取 $(1,1) \\in Q$，令 $c = -1$，则 $(-1)(1,1) = (-1,-1)$，但 $-1 < 0$，不在 $Q$ 中。数乘封闭性（S1）失效，$Q$ 不是向量空间。',
      },
      {
        id: 'u4-l1-q2',
        type: 'choice',
        prompt: '以下哪个集合在通常的加法和数乘下**不构成**向量空间？',
        options: [
          '$\\mathbb{R}^{3 \\times 2}$（全体 $3 \\times 2$ 实矩阵）',
          '$\\mathcal{P}_3$（次数不超过 3 的实多项式）',
          '$\\{(x,y) \\in \\mathbb{R}^2 \\mid x^2 + y^2 \\leq 1\\}$（单位闭圆盘）',
          '$C[0,1]$（$[0,1]$ 上的连续函数）',
        ],
        answer: 2,
        explain: '单位闭圆盘不满足数乘封闭性（S1）：取 $(1,0)$，$2 \\cdot (1,0) = (2,0)$，$|\\!(2,0)\\!| = 2 > 1$，不在圆盘内。其余三个均是向量空间。',
      },
      {
        id: 'u4-l1-q3',
        type: 'input',
        prompt: '$\\mathcal{P}_2$ 中多项式 $p(x) = 5 - 3x + x^2$ 的加法逆元 $-p(x)$ 的常数项是多少？（填整数）',
        accept: ['-5'],
        explain: '加法逆元是将每个系数取反：$-p(x) = -5 + 3x - x^2$。常数项为 $-5$。验证：$p(x) + (-p(x)) = 0$，正确。',
      },
      {
        id: 'u4-l1-q4',
        type: 'match',
        prompt: '将下列向量空间与其维数配对。',
        left: [
          '$\\mathbb{R}^5$',
          '$\\mathbb{R}^{2 \\times 3}$',
          '$\\mathcal{P}_4$',
          '$\\{\\mathbf{0}\\}$',
        ],
        right: [
          '维数 $0$',
          '维数 $5$',
          '维数 $5$（$n+1 = 4+1$）',
          '维数 $6$（$2 \\times 3$）',
        ],
        explain: '$\\mathbb{R}^5$ 维数 5；$\\mathbb{R}^{2\\times 3}$ 维数 $2 \\times 3 = 6$；$\\mathcal{P}_4$ 维数 $4+1=5$；$\\{\\mathbf{0}\\}$ 只含零向量，维数为 0。',
      },
      {
        id: 'u4-l1-q5',
        type: 'judge',
        prompt: '在多项式空间 $\\mathcal{P}_2$ 中，零向量是常数多项式 $p(x) = 1$。',
        answer: false,
        explain: '零向量必须满足 $p(x) + \\mathbf{0}(x) = p(x)$ 对所有 $p \\in \\mathcal{P}_2$ 成立，故零向量是零多项式 $\\mathbf{0}(x) = 0$（所有系数均为零），不是 $p(x) = 1$。',
      },
      {
        id: 'u4-l1-q6',
        type: 'choice',
        prompt: '验证向量空间时，下列哪条公理**最容易被遗漏**，也是最常见的出错点？',
        options: [
          'A2 加法交换律',
          'A1/S1 封闭性',
          'S3 单位元 $1 \\cdot \\mathbf{v} = \\mathbf{v}$',
          'A3 加法结合律',
        ],
        answer: 1,
        explain: '封闭性（A1 加法封闭、S1 数乘封闭）是最容易被遗漏的步骤。许多同学只验证内部的代数性质，忘记先确认运算结果仍属于集合 $V$。',
      },
      {
        id: 'u4-l1-q7',
        type: 'input',
        prompt: '在 $\\mathbb{R}^2$ 上定义非标准运算：$(x_1,y_1) \\oplus (x_2,y_2) = (x_1+x_2,\\ y_1+y_2+1)$，$c \\odot (x,y) = (cx,\\ cy+c-1)$。该向量空间的零向量第二分量是多少？（填整数）',
        accept: ['-1'],
        explain: '设零向量为 $(e_1, e_2)$，需 $(x,y) \\oplus (e_1,e_2) = (x,y)$，即 $(x+e_1,\\ y+e_2+1) = (x,y)$，得 $e_1 = 0$，$e_2 + 1 = 0$，$e_2 = -1$。零向量为 $(0,-1)$。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────
  // 关卡 2：子空间的判定
  // ─────────────────────────────────────────────────────────────────
  {
    id: 'u4-l2',
    title: '子空间的判定',
    subtitle: '三条件快速判定向量空间中的向量空间',
    intro: [
      {
        title: '什么是子空间？',
        body: '向量空间 $V$ 的非空子集 $W$ 如果**本身也构成向量空间**（在相同的运算下），则称 $W$ 是 $V$ 的一个**子空间**。\n\n关键简化：因为 $W \\subseteq V$，加法的交换律、结合律、数乘结合律等公理自动从 $V$ 继承，**无需重新验证**。只需额外检查三个条件即可。',
        tip: '子空间是"住在大向量空间里的小向量空间"。$\\{\\mathbf{0}\\}$ 和 $V$ 本身都是 $V$ 的子空间，称为**平凡子空间**。',
      },
      {
        title: '子空间判定三条件',
        body: '设 $W$ 是向量空间 $V$ 的非空子集，$W$ 是子空间当且仅当满足：\n\n**条件 1（零向量）**：$\\mathbf{0} \\in W$\n**条件 2（加法封闭）**：$\\mathbf{u}, \\mathbf{v} \\in W \\Rightarrow \\mathbf{u} + \\mathbf{v} \\in W$\n**条件 3（数乘封闭）**：$\\mathbf{v} \\in W,\\ c \\in \\mathbb{R} \\Rightarrow c\\mathbf{v} \\in W$',
        formula: '\\mathbf{0} \\in W, \\quad W \\text{ 对加法封闭}, \\quad W \\text{ 对数乘封闭}',
        tip: '三个条件缺一不可。实际上条件 2+3 可合并为"对线性组合封闭"：$c_1\\mathbf{u} + c_2\\mathbf{v} \\in W$。',
      },
      {
        title: '条件 1：零向量必须在 $W$ 中',
        body: '子空间必须过原点。若零向量 $\\mathbf{0} \\notin W$，则 $W$ **必定不是子空间**，立刻否定，无需继续验证。\n\n**典型否定案例**：$W = \\{(x,y) \\mid x + y = 1\\}$，因为 $(0,0)$ 满足 $0+0=0 \\neq 1$，零向量不在 $W$ 中，故 $W$ 不是子空间。\n\n**推论**：所有"$A\\mathbf{x} = \\mathbf{b}$（$\\mathbf{b} \\neq \\mathbf{0}$）的解集"都不是子空间（零向量不是解）。',
        tip: '"不过原点"是否定子空间最快的方法。',
      },
      {
        title: '条件 2+3：对加法和数乘封闭',
        body: '封闭性验证的通用写法：\n\n取任意 $\\mathbf{u}, \\mathbf{v} \\in W$（写出它们的一般形式），令 $c \\in \\mathbb{R}$，然后：\n- 计算 $\\mathbf{u} + \\mathbf{v}$，验证结果满足 $W$ 的定义条件\n- 计算 $c\\mathbf{v}$，验证结果满足 $W$ 的定义条件',
        steps: [
          '**例**：$W = \\{(x,y) \\mid y = 3x\\}$，取 $\\mathbf{u} = (a,3a)$，$\\mathbf{v} = (b,3b)$。',
          '$\\mathbf{u}+\\mathbf{v} = (a+b,\\ 3a+3b) = (a+b,\\ 3(a+b)) \\in W$。加法封闭。',
          '$c\\mathbf{u} = (ca,\\ 3ca) = (ca,\\ 3(ca)) \\in W$。数乘封闭。',
          '零向量 $(0,0)$ 满足 $0 = 3 \\times 0 \\in W$。三条件全满足，$W$ 是子空间。',
        ],
      },
      {
        title: '常见子空间举例',
        body: '以下均为 $\\mathbb{R}^n$（或相应空间）的子空间：\n\n- **零子空间** $\\{\\mathbf{0}\\}$：仅含零向量\n- **$\\mathbb{R}^3$ 中过原点的直线**：$W = \\{t\\mathbf{v} \\mid t \\in \\mathbb{R}\\}$（一个非零向量的所有数倍）\n- **$\\mathbb{R}^3$ 中过原点的平面**：$W = \\{(x,y,z) \\mid ax+by+cz=0\\}$\n- **齐次方程组的解集** $\\{\\mathbf{x} \\mid A\\mathbf{x}=\\mathbf{0}\\}$：即矩阵 $A$ 的**零空间**\n- **$\\mathcal{P}_n$ 中常数项为 0 的多项式**：$W = \\{a_1x+\\cdots+a_nx^n\\}$',
        tip: '齐次方程组（右端为 $\\mathbf{0}$）的解集一定是子空间；非齐次方程组（$\\mathbf{b} \\neq \\mathbf{0}$）的解集一定不是。',
      },
      {
        title: '否定子空间的快捷路径',
        body: '证明 $W$ **不是**子空间，找到一个具体反例使某条件失效即可：\n\n1. 若 $\\mathbf{0} \\notin W$ → 直接否定\n2. 若找到 $\\mathbf{u}, \\mathbf{v} \\in W$ 使 $\\mathbf{u}+\\mathbf{v} \\notin W$ → 否定\n3. 若找到 $\\mathbf{v} \\in W$，$c \\in \\mathbb{R}$ 使 $c\\mathbf{v} \\notin W$ → 否定（常取 $c=-1$）\n\n**例**：$W = \\{(x,y) \\mid xy \\geq 0\\}$（第一、三象限及坐标轴），取 $\\mathbf{u}=(1,0)$，$\\mathbf{v}=(0,1)$，均在 $W$ 中，但 $\\mathbf{u}+\\mathbf{v}=(1,1)$ 也在 $W$ 中……再取 $\\mathbf{u}=(1,1)$，$c=-1$，$c\\mathbf{u}=(-1,-1) \\in W$……其实对于这个集合需要更仔细验证——但 $\\mathbf{u}=(1,0)$，$\\mathbf{v}=(0,1)$ 均满足 $xy=0\\geq 0$，$\\mathbf{u}+\\mathbf{v}=(1,1)$，$xy=1>0$，在 $W$ 中；$\\mathbf{u}=(1,-1)$……$xy=-1<0$，不在。',
        tip: '最优先验证零向量；其次对典型边界元素取 $c=-1$ 测试数乘封闭。',
      },
      {
        title: '子空间与线性方程组的关联',
        body: '子空间与线性方程组有深刻联系：\n\n- 齐次方程组 $A\\mathbf{x}=\\mathbf{0}$ 的解集是 $\\mathbb{R}^n$ 的子空间（称**零空间** $\\operatorname{Null}(A)$）\n- 矩阵 $A$ 的列向量的所有线性组合构成**列空间** $\\operatorname{Col}(A)$，是 $\\mathbb{R}^m$ 的子空间\n\n这两个子空间是线性代数最重要的子空间，后续章节将深入研究。',
        reveal: { q: '为什么非齐次方程组 $A\\mathbf{x}=\\mathbf{b}$（$\\mathbf{b}\\neq\\mathbf{0}$）的解集不是子空间？', a: '因为零向量 $\\mathbf{x}=\\mathbf{0}$ 不满足 $A\\mathbf{0}=\\mathbf{0}\\neq\\mathbf{b}$，故 $\\mathbf{0}$ 不在解集中，条件 1 失效。' },
      },
    ],
    questions: [
      {
        id: 'u4-l2-q1',
        type: 'judge',
        prompt: '$W = \\{(x,y,z) \\in \\mathbb{R}^3 \\mid 2x - y + z = 0\\}$ 是 $\\mathbb{R}^3$ 的子空间。',
        answer: true,
        explain: '验证三条件：(1) $(0,0,0)$：$2(0)-0+0=0$，零向量在 $W$ 中。(2) 加法封闭：取 $(a,b,c),(d,e,f) \\in W$，则 $2a-b+c=0$，$2d-e+f=0$，相加得 $2(a+d)-(b+e)+(c+f)=0$，和在 $W$ 中。(3) 数乘封闭：$k(a,b,c)$ 满足 $2(ka)-kb+kc=k(2a-b+c)=0$。三条件全满足，$W$ 是子空间。',
      },
      {
        id: 'u4-l2-q2',
        type: 'choice',
        prompt: '以下哪个集合**不是** $\\mathbb{R}^2$ 的子空间？',
        options: [
          '$W_1 = \\{(x,y) \\mid y = 0\\}$（$x$ 轴）',
          '$W_2 = \\{(x,y) \\mid x + y = 0\\}$（过原点的直线）',
          '$W_3 = \\{(x,y) \\mid x + y = 1\\}$（不过原点的直线）',
          '$W_4 = \\{(0,0)\\}$（零子空间）',
        ],
        answer: 2,
        explain: '$W_3$ 不过原点：$(0,0)$ 满足 $0+0=0\\neq 1$，零向量不在 $W_3$ 中，条件 1 失效，不是子空间。$W_1$、$W_2$、$W_4$ 均过原点且对加法数乘封闭，都是子空间。',
      },
      {
        id: 'u4-l2-q3',
        type: 'input',
        prompt: '齐次方程组 $A\\mathbf{x}=\\mathbf{0}$ 的解集是 $\\mathbb{R}^n$ 的子空间，其专有名称（两字）叫什么？（填"零空间"）',
        accept: ['零空间', 'null space', 'Null Space'],
        explain: '齐次方程组 $A\\mathbf{x}=\\mathbf{0}$ 的解集称为矩阵 $A$ 的**零空间**（Null Space），记作 $\\operatorname{Null}(A)$。它是 $\\mathbb{R}^n$ 的子空间，因为零向量是解，且对加法和数乘封闭。',
      },
      {
        id: 'u4-l2-q4',
        type: 'judge',
        prompt: '非空集合 $W \\subseteq V$ 是子空间，当且仅当对所有 $\\mathbf{u},\\mathbf{v}\\in W$ 和 $c,d\\in\\mathbb{R}$，都有 $c\\mathbf{u}+d\\mathbf{v}\\in W$。',
        answer: true,
        explain: '这是子空间三条件的等价合并形式——"对线性组合封闭"。取 $d=0$ 得数乘封闭（S1）；取 $c=d=1$ 得加法封闭（A1）；取 $\\mathbf{u}=\\mathbf{v}$，$c=1$，$d=-1$ 得 $\\mathbf{0}=c\\mathbf{u}+d\\mathbf{u}=\\mathbf{0}\\in W$（零向量条件）。三条件等价于"对线性组合封闭"。',
      },
      {
        id: 'u4-l2-q5',
        type: 'match',
        prompt: '将下列集合与其作为子空间的状态配对。',
        left: [
          '$\\{\\mathbf{x}\\in\\mathbb{R}^3 \\mid A\\mathbf{x}=\\mathbf{0}\\}$',
          '$\\{\\mathbf{x}\\in\\mathbb{R}^3 \\mid A\\mathbf{x}=\\mathbf{b}\\}$，$\\mathbf{b}\\neq\\mathbf{0}$',
          '$\\mathbb{R}^3$ 中过原点的任意直线',
          '$\\mathbb{R}^3$ 中不过原点的平面',
        ],
        right: [
          '是子空间（零空间）',
          '是子空间（一维）',
          '不是子空间（零向量不在其中）',
          '不是子空间（零向量不满足 $A\\mathbf{x}=\\mathbf{b}$）',
        ],
        explain: '齐次方程组解集是子空间（零空间）；非齐次方程组解集因 $\\mathbf{0}$ 不满足 $A\\mathbf{0}=\\mathbf{b}\\neq\\mathbf{0}$ 而不是子空间。过原点的直线对数乘封闭（$t\\mathbf{v}$ 仍在直线上）故是子空间；不过原点的平面缺少零向量，不是子空间。',
      },
      {
        id: 'u4-l2-q6',
        type: 'choice',
        prompt: '集合 $W = \\{p(x) \\in \\mathcal{P}_2 \\mid p(0) = 0\\}$ 是 $\\mathcal{P}_2$ 的子空间吗？',
        options: [
          '是，三条判定条件全部满足',
          '不是，零多项式 $p(x)=0$ 不满足 $p(0)=0$',
          '不是，加法不封闭',
          '不是，数乘封闭失败',
        ],
        answer: 0,
        explain: '(1) 零多项式满足 $p(0)=0$，零向量在 $W$ 中。(2) 若 $p(0)=0$，$q(0)=0$，则 $(p+q)(0)=0+0=0$，加法封闭。(3) $(cp)(0)=c\\cdot p(0)=0$，数乘封闭。三条件满足，$W$ 是子空间。',
      },
      {
        id: 'u4-l2-q7',
        type: 'judge',
        prompt: '$\\mathbb{R}^3$ 中两个子空间的**并集** $W_1 \\cup W_2$ 一定是子空间。',
        answer: false,
        explain: '一般情况下并集不是子空间。取 $W_1 = \\{(x,0,0)\\}$（$x$ 轴），$W_2 = \\{(0,y,0)\\}$（$y$ 轴），$(1,0,0) \\in W_1 \\subseteq W_1\\cup W_2$，$(0,1,0) \\in W_2 \\subseteq W_1\\cup W_2$，但 $(1,0,0)+(0,1,0)=(1,1,0)$ 既不在 $W_1$ 也不在 $W_2$ 中，故 $W_1\\cup W_2$ 对加法不封闭，不是子空间。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────
  // 关卡 3：线性相关与无关
  // ─────────────────────────────────────────────────────────────────
  {
    id: 'u4-l3',
    title: '线性相关与无关',
    subtitle: '线性组合、张成与冗余向量的判别',
    intro: [
      {
        title: '线性组合',
        body: '设 $\\mathbf{v}_1, \\ldots, \\mathbf{v}_k \\in \\mathbb{R}^n$，$c_1, \\ldots, c_k \\in \\mathbb{R}$，则称\n$\\mathbf{w} = c_1\\mathbf{v}_1 + c_2\\mathbf{v}_2 + \\cdots + c_k\\mathbf{v}_k$\n为 $\\mathbf{v}_1, \\ldots, \\mathbf{v}_k$ 的一个**线性组合**。\n\n**例**：$\\mathbf{v}_1 = (1,2)^T$，$\\mathbf{v}_2 = (3,1)^T$，取 $c_1=2$，$c_2=-1$，得线性组合 $2(1,2)^T + (-1)(3,1)^T = (-1,3)^T$。',
        tip: '线性组合是向量空间最基本的操作：按比例缩放再相加。系数 $c_i$ 可以是任意实数（包括负数和零）。',
      },
      {
        title: '张成（Span）',
        body: '向量组 $\\{\\mathbf{v}_1, \\ldots, \\mathbf{v}_k\\}$ 的所有线性组合构成的集合，称为这组向量的**张成**：',
        formula: '\\operatorname{span}\\{\\mathbf{v}_1,\\ldots,\\mathbf{v}_k\\} = \\{c_1\\mathbf{v}_1+\\cdots+c_k\\mathbf{v}_k \\mid c_i\\in\\mathbb{R}\\}',
        steps: [
          '张成是一个**子空间**（对线性组合封闭），是包含 $\\{\\mathbf{v}_1,\\ldots,\\mathbf{v}_k\\}$ 的最小子空间。',
          '**一个非零向量**的张成：过原点的直线。',
          '**两个线性无关向量**的张成：过原点的平面。',
          '**$n$ 个线性无关的 $n$ 维向量**的张成：整个 $\\mathbb{R}^n$。',
        ],
        tip: '向向量组中添加已在张成中的向量，不会扩大张成空间——只有线性无关的新向量才能扩大维度。',
      },
      {
        title: '线性相关与线性无关的定义',
        body: '考虑齐次方程 $c_1\\mathbf{v}_1 + c_2\\mathbf{v}_2 + \\cdots + c_k\\mathbf{v}_k = \\mathbf{0}$：\n\n- **线性无关**：唯一解是全零解 $c_1=c_2=\\cdots=c_k=0$，即每个向量都提供独立方向，没有冗余。\n- **线性相关**：存在不全为零的系数使等式成立，即至少有一个向量可被其余向量线性表示（"冗余"）。',
        formula: 'c_1\\mathbf{v}_1+\\cdots+c_k\\mathbf{v}_k=\\mathbf{0} \\;\\Rightarrow\\; c_i=0 \\;\\Leftrightarrow\\; \\text{线性无关}',
        tip: '零解总是存在的（令所有 $c_i=0$），判断线性无关要证明**没有非零解**。',
      },
      {
        title: '判定线性相关性：行化简法',
        body: '**最通用的方法**：将向量排为矩阵的列，对矩阵做行化简，数主元数量。',
        steps: [
          '**步骤1**：构造矩阵 $A = [\\mathbf{v}_1 \\mid \\mathbf{v}_2 \\mid \\cdots \\mid \\mathbf{v}_k]$（向量作为列）。',
          '**步骤2**：行化简得到行阶梯形（REF），数主元数量 $r = \\operatorname{rank}(A)$。',
          '**步骤3**：若 $r = k$（列满秩，无自由列）→ 线性无关；若 $r < k$（有自由列）→ 线性相关。',
          '**例**：$\\mathbf{v}_1=(1,2,1)^T$，$\\mathbf{v}_2=(2,5,3)^T$，$\\mathbf{v}_3=(0,1,1)^T$ 化简后秩为 2，$2 < 3$，线性相关，$\\mathbf{v}_3 = \\mathbf{v}_2 - 2\\mathbf{v}_1$。',
        ],
        tip: '主元所在的列是"独立列"；自由列对应的向量可被前面的主元列线性表示。',
      },
      {
        title: '判定线性相关性：行列式与快速规则',
        body: '**行列式法**（仅限 $k = n$，即向量个数等于维数的方阵）：$\\det(A) \\neq 0 \\Rightarrow$ 线性无关；$\\det(A) = 0 \\Rightarrow$ 线性相关。\n\n**快速否定规则**（无需计算）：\n- 向量组含**零向量** → 必线性相关\n- 两个向量**成比例** → 必线性相关\n- 向量个数 $k > n$（超过空间维数）→ 必线性相关',
        tip: '方阵判定首选行列式——计算更快；非方阵只能用行化简。',
      },
      {
        title: '判断向量是否在张成中',
        body: '判断 $\\mathbf{b}$ 是否属于 $\\operatorname{span}\\{\\mathbf{v}_1,\\ldots,\\mathbf{v}_k\\}$，等价于判断方程组 $A\\mathbf{x}=\\mathbf{b}$ 是否有解：',
        steps: [
          '构造增广矩阵 $[A \\mid \\mathbf{b}]$ 并行化简。',
          '若行化简后**无矛盾行**（没有 $0=c$，$c\\neq 0$ 的行）→ 有解 → $\\mathbf{b}\\in\\operatorname{span}\\{\\mathbf{v}_1,\\ldots,\\mathbf{v}_k\\}$。',
          '若出现矛盾行 → 无解 → $\\mathbf{b}\\notin\\operatorname{span}$。',
          '**例**：$\\mathbf{u}_1=(1,1,0)^T$，$\\mathbf{u}_2=(2,0,1)^T$，$\\mathbf{b}=(5,1,2)^T$，增广矩阵化简后无矛盾，有解 $\\mathbf{b}=\\mathbf{u}_1+2\\mathbf{u}_2$。',
        ],
        reveal: { q: '若向量组线性相关，添加一个新向量 $\\mathbf{w}$ 之后，整体一定还是线性相关的吗？', a: '是的。若 $\\{\\mathbf{v}_1,\\ldots,\\mathbf{v}_k\\}$ 线性相关，则存在不全为零的系数使 $c_1\\mathbf{v}_1+\\cdots+c_k\\mathbf{v}_k=\\mathbf{0}$，令 $c_{k+1}=0$，得 $c_1\\mathbf{v}_1+\\cdots+c_k\\mathbf{v}_k+0\\cdot\\mathbf{w}=\\mathbf{0}$，系数仍不全为零，整体线性相关。' },
      },
      {
        title: '线性相关性的核心直觉',
        body: '**线性无关**：向量组是"高效的"——每个向量都在扩大张成空间，没有冗余信息。就像坐标轴：$x$、$y$、$z$ 方向各自独立，缺少任何一个都无法描述 $\\mathbb{R}^3$。\n\n**线性相关**：向量组是"冗余的"——至少有一个向量可被其余向量的线性组合替代，去掉它不缩小张成空间。就像 $x$、$y$、$x+y$ 三个方向：第三个由前两个完全决定。\n\n**AI 联系**：神经网络的权重矩阵若列线性相关（秩 $< n$），说明存在冗余神经元——LoRA 等低秩近似方法正是利用这点压缩参数。',
        tip: '$\\operatorname{rank}(A) = k$ 等价于列线性无关；$\\operatorname{rank}(A) < k$ 等价于列线性相关。',
      },
    ],
    questions: [
      {
        id: 'u4-l3-q1',
        type: 'judge',
        prompt: '向量组 $\\{\\mathbf{v}_1, \\mathbf{v}_2, \\mathbf{v}_3\\}$ 在 $\\mathbb{R}^2$ 中，则该向量组必定线性相关。',
        answer: true,
        explain: '$\\mathbb{R}^2$ 的维数为 2，而向量个数 $k=3 > 2$。由"维数界"性质：$\\mathbb{R}^n$ 中任意 $n+1$ 个或更多向量必线性相关。因此该向量组必定线性相关，无需计算。',
      },
      {
        id: 'u4-l3-q2',
        type: 'choice',
        prompt: '设 $A = [\\mathbf{v}_1 \\mid \\mathbf{v}_2 \\mid \\mathbf{v}_3]$ 是 $4 \\times 3$ 矩阵，行化简后有 2 个主元。关于这三个向量，正确的结论是？',
        options: [
          '三向量线性无关，因为有 2 个主元',
          '三向量线性相关，$\\operatorname{rank}(A) = 2 < 3$，存在自由列',
          '三向量线性相关，因为 $4 > 3$',
          '无法从已知信息判断',
        ],
        answer: 1,
        explain: '$\\operatorname{rank}(A) = 2 < 3 = k$（列数），存在 $3 - 2 = 1$ 个自由列，故三向量**线性相关**。正确选项 B。选项 C 中 $4 > 3$ 只是行数大于列数，并不直接决定线性相关性（列满秩时仍可能线性无关）。',
      },
      {
        id: 'u4-l3-q3',
        type: 'input',
        prompt: '判断 $\\mathbf{v}_1 = (1,2,-2)^T$ 与 $\\mathbf{v}_2 = (-2,-4,4)^T$ 是否线性相关。两者之间的比例系数（$\\mathbf{v}_2$ 等于 $\\mathbf{v}_1$ 的多少倍）是多少？（填整数）',
        accept: ['-2'],
        explain: '$\\mathbf{v}_2 = (-2,-4,4)^T = -2 \\times (1,2,-2)^T = -2\\mathbf{v}_1$。两向量成比例，线性相关。比例系数为 $-2$。验证：$1 \\cdot \\mathbf{v}_1 + \\frac{1}{2} \\cdot \\mathbf{v}_2 = \\mathbf{0}$，非零系数存在，线性相关。',
      },
      {
        id: 'u4-l3-q4',
        type: 'match',
        prompt: '将下列条件与其对应的结论配对。',
        left: [
          '向量组含零向量',
          '$\\operatorname{rank}(A) = k$（$k$ 为列数）',
          '$\\det(A) = 0$（方阵）',
          '$k > n$（向量个数超过维数）',
        ],
        right: [
          '必线性相关（维数界）',
          '必线性相关（含零向量）',
          '线性无关（列满秩）',
          '线性相关（行列式为零）',
        ],
        explain: '含零向量时取该向量系数为 1、其余为 0 即得非零解，必相关；列满秩意味着 $A\\mathbf{x}=\\mathbf{0}$ 只有零解，线性无关；方阵行列式为零等价于列线性相关；$k > n$ 由维数界定理直接得到线性相关。',
      },
      {
        id: 'u4-l3-q5',
        type: 'judge',
        prompt: '$\\mathbf{b} = (3, 1, 4)^T$ 属于 $\\operatorname{span}\\{(1,0,1)^T,\\ (2,1,1)^T\\}$。',
        answer: false,
        explain: '设 $c_1(1,0,1)^T + c_2(2,1,1)^T = (3,1,4)^T$，得方程组：$c_1+2c_2=3$，$c_2=1$，$c_1+c_2=4$。由第二式 $c_2=1$，代入第一式 $c_1=1$；代入第三式 $1+1=2\\neq 4$，矛盾。方程组无解，故 $\\mathbf{b}\\notin\\operatorname{span}\\{\\mathbf{v}_1,\\mathbf{v}_2\\}$。',
      },
      {
        id: 'u4-l3-q6',
        type: 'choice',
        prompt: '若 $\\mathbf{v}_1, \\mathbf{v}_2$ 线性无关，以下哪个向量组一定也线性无关？',
        options: [
          '$\\{\\mathbf{v}_1, \\mathbf{v}_2, \\mathbf{v}_1 + \\mathbf{v}_2\\}$',
          '$\\{\\mathbf{v}_1, \\mathbf{v}_2, 2\\mathbf{v}_1\\}$',
          '$\\{\\mathbf{v}_1 + \\mathbf{v}_2,\\ \\mathbf{v}_1 - \\mathbf{v}_2\\}$',
          '$\\{\\mathbf{v}_1, \\mathbf{v}_2, \\mathbf{0}\\}$',
        ],
        answer: 2,
        explain: '选项 C：设 $a(\\mathbf{v}_1+\\mathbf{v}_2)+b(\\mathbf{v}_1-\\mathbf{v}_2)=\\mathbf{0}$，整理得 $(a+b)\\mathbf{v}_1+(a-b)\\mathbf{v}_2=\\mathbf{0}$。由 $\\mathbf{v}_1,\\mathbf{v}_2$ 线性无关，得 $a+b=0$ 且 $a-b=0$，解得 $a=b=0$，故线性无关。\n\n选项 A：$1\\cdot\\mathbf{v}_1+1\\cdot\\mathbf{v}_2+(-1)(\\mathbf{v}_1+\\mathbf{v}_2)=\\mathbf{0}$，线性相关。选项 B：$2\\mathbf{v}_1$ 与 $\\mathbf{v}_1$ 成比例，线性相关。选项 D：含零向量，必线性相关。',
      },
      {
        id: 'u4-l3-q7',
        type: 'input',
        prompt: '矩阵 $A$ 的三个列向量行化简后得到 3 个主元（共 3 列），则这三个向量线性无关，$\\operatorname{rank}(A)$ 等于多少？（填整数）',
        accept: ['3'],
        explain: '3 个列向量各有一个主元，说明每列都是主元列，无自由列，$\\operatorname{rank}(A) = 3 = k$（列数），故三向量线性无关。',
      },
    ],
  },
]
