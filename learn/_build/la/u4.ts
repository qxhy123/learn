import type { Unit } from '../../src/types'

export const UNIT: Unit = {
  id: 'u4',
  title: '向量空间',
  color: '#ff9600',
  icon: '🧮',
  blurb: '从 $\\mathbb{R}^n$ 到多项式、函数——只要满足 8 条公理，就是向量空间。掌握线性无关、基与维数、子空间，为理解一切线性结构打下最坚实的根基。',
  lessons: [
    // =====================================================================
    // u4-l1  向量空间与八条公理
    // =====================================================================
    {
      id: 'u4-l1',
      title: '向量空间',
      subtitle: '8 条公理统一 $\\mathbb{R}^n$、矩阵、多项式……',
      intro: [
        {
          title: '为什么需要"向量空间"？',
          body: '我们已经学过 $\\mathbb{R}^2$、$\\mathbb{R}^3$ 中的向量加法和数乘。\n\n观察一个有趣现象：\n\n- 两个向量 $\\begin{pmatrix}1\\\\2\\end{pmatrix} + \\begin{pmatrix}3\\\\4\\end{pmatrix} = \\begin{pmatrix}4\\\\6\\end{pmatrix}$\n- 两个多项式 $(1+2x) + (3+4x) = 4+6x$\n- 两个连续函数 $\\sin x + \\cos x$ 还是连续函数\n\n这三件事用的是**完全相同的结构规则**。数学家的聪明做法：把这些共同规则抽象成**向量空间**，证明一次定理，所有满足规则的对象都能用！',
          tip: '向量空间的"向量"不必是箭头——矩阵、多项式、函数都可以是"向量"，只要它们的运算满足 8 条公理。',
        },
        {
          title: '向量空间的定义',
          body: '设 $V$ 是非空集合，$\\mathbb{F}$ 是数域（取 $\\mathbb{R}$）。若 $V$ 上定义了向量加法和标量乘法，且满足以下 **8 条公理**，则 $(V, +, \\cdot)$ 是 $\\mathbb{F}$ 上的**向量空间**。\n\n**加法 5 条（A1–A5）**\n- A1 加法封闭：$\\mathbf{u}+\\mathbf{v} \\in V$\n- A2 交换律：$\\mathbf{u}+\\mathbf{v}=\\mathbf{v}+\\mathbf{u}$\n- A3 结合律：$(\\mathbf{u}+\\mathbf{v})+\\mathbf{w}=\\mathbf{u}+(\\mathbf{v}+\\mathbf{w})$\n- A4 零向量：存在 $\\mathbf{0}\\in V$，使 $\\mathbf{v}+\\mathbf{0}=\\mathbf{v}$\n- A5 加法逆元：存在 $-\\mathbf{v}\\in V$，使 $\\mathbf{v}+(-\\mathbf{v})=\\mathbf{0}$\n\n**数乘 4 条（S1–S4）**\n- S1 数乘封闭：$c\\mathbf{v}\\in V$\n- S2 数乘结合律：$(cd)\\mathbf{v}=c(d\\mathbf{v})$\n- S3 单位元：$1\\cdot\\mathbf{v}=\\mathbf{v}$\n- S4 两分配律：$c(\\mathbf{u}+\\mathbf{v})=c\\mathbf{u}+c\\mathbf{v}$；$(c+d)\\mathbf{v}=c\\mathbf{v}+d\\mathbf{v}$',
          tip: '**最容易遗漏的**是 A1（加法封闭）和 S1（数乘封闭）——运算结果必须还在集合 $V$ 里，不能"逃出去"。',
        },
        {
          title: '常见向量空间举例',
          body: '**例 1：$\\mathbb{R}^n$**——$n$ 维列向量，按分量做加法和数乘。零向量是 $(0,0,\\ldots,0)^T$。这是最经典的向量空间。\n\n**例 2：$\\mathbb{R}^{m\\times n}$**——所有 $m\\times n$ 实矩阵，矩阵加法 + 数乘。零向量是全零矩阵 $O$。\n\n**例 3：$\\mathcal{P}_n$**——次数 $\\leq n$ 的实系数多项式。加法按系数逐项相加，数乘按系数缩放。零向量是零多项式 $0$。\n\n**例 4：$C[a,b]$**——$[a,b]$ 上所有连续函数。逐点加法和数乘，"两个连续函数之和仍连续"保证了 A1。这是无穷维向量空间。',
          formula: '\\mathbb{R}^n,\\; \\mathbb{R}^{m\\times n},\\; \\mathcal{P}_n,\\; C[a,b] \\text{ 都是向量空间}',
        },
        {
          title: '验证向量空间：标准 4 步',
          body: '验证集合 $V$ 是向量空间，按以下 4 步进行：\n\n**第 1 步**：写清楚 $V$ 的元素形式，以及加法和数乘的定义。\n\n**第 2 步**：验证**封闭性**（A1、S1）——取任意元素，证明加法结果和数乘结果还在 $V$ 中。\n\n**第 3 步**：找出**零向量**（A4），再写出**加法逆元**（A5）并验证其在 $V$ 中。\n\n**第 4 步**：其余公理（A2、A3、S2、S3、S4）通常由数域的性质直接继承，说明即可。',
          tip: '**证明不是向量空间**只需找**一个反例**，指出哪条公理不满足，无需验证其余。优先检查：零向量不在集合里、负数乘时逃出集合。',
        },
        {
          title: '例题 1：验证向量空间',
          body: '**题目**：$V = \\{(x,y)^T \\in \\mathbb{R}^2 \\mid 2x - y = 0\\}$，用 $\\mathbb{R}^2$ 通常的加法和数乘，验证 $V$ 是否为向量空间。',
          steps: [
            '封闭性：设 $\\mathbf{u}=(a,2a)^T$，$\\mathbf{v}=(b,2b)^T \\in V$。加法：$(a+b, 2a+2b)^T = (a+b, 2(a+b))^T \\in V$；数乘：$c\\mathbf{u}=(ca,2ca)^T \\in V$。封闭。',
            '零向量：$(0,0)^T$ 满足 $2\\cdot 0 - 0 = 0$，故 $(0,0)^T \\in V$，A4 成立。',
            '加法逆元：$(a,2a)^T$ 的逆元为 $(-a,-2a)^T = (-a, 2(-a))^T \\in V$，A5 成立。',
            '其余公理（A2、A3、S2–S4）：直接继承自 $\\mathbb{R}^2$，成立。',
            '结论：$V$ **是**向量空间（$\\mathbb{R}^2$ 中过原点的一条直线）。',
          ],
        },
        {
          title: '例题 2：找反例否定向量空间',
          body: '**题目**：$S = \\{(x,y)^T \\in \\mathbb{R}^2 \\mid x\\geq 0, y\\geq 0\\}$（第一象限含坐标轴），是否为向量空间？',
          steps: [
            '取 $\\mathbf{v}=(1,1)^T \\in S$，令标量 $c = -1$。',
            '计算：$(-1)\\cdot(1,1)^T = (-1,-1)^T$。',
            '检查：$-1 < 0$，故 $(-1,-1)^T \\notin S$。',
            '结论：**数乘封闭性（S1）不满足**，$S$ 不是向量空间。',
          ],
          tip: '快速否定技巧：对第一象限/正实数类集合，令 $c=-1$ 做数乘，几乎必定出界，立即否定。',
        },
        {
          title: '易错点与 AI 联系',
          body: '**易错点 1**：零向量不一定是"每分量为 0 的向量"。自定义运算时，需用 $\\mathbf{v}+\\mathbf{0}=\\mathbf{v}$ 倒推零向量。\n\n**易错点 2**：不过原点的平面（如 $x+y=1$）必定不是向量空间，因为 $(0,0)^T$ 不满足 $0+0=1$，零向量不在其中。\n\n**AI 联系**：深度学习中的**潜在空间**（Latent Space）就是向量空间——VAE 的"微笑方向"向量加法、Word2Vec 的 king-man+woman≈queen，这些"语义算术"之所以合法，正是因为潜在空间 $\\mathbb{R}^k$ 满足向量空间公理（A1 加法封闭、S1 数乘封闭保证结果还在空间里）。',
          reveal: {
            q: '正实数集合 $\\mathbb{R}^+$，定义 $u\\oplus v = uv$（乘法当加法），$c\\odot v = v^c$（幂运算当数乘），它是向量空间吗？',
            a: '**是**！零向量（满足 $v\\oplus\\mathbf{0}=v$ 的元素）是 $1$（因为 $v\\cdot 1=v$）；逆元是 $1/v$（因为 $v\\cdot(1/v)=1$）；单位元：$1\\odot v=v^1=v$。所有 8 条公理均满足——$\\mathbb{R}^+$ 通过对数同构于 $\\mathbb{R}$，是名副其实的向量空间！这是最反直觉的经典例子。',
          },
        },
      ],
      questions: [
        {
          id: 'u4-l1-q1',
          type: 'choice',
          prompt: '验证集合是否为向量空间时，以下哪条公理**最容易被遗漏**？',
          options: [
            'A2 加法交换律',
            'A1 加法封闭性与 S1 数乘封闭性',
            'S3 单位元 $1\\cdot\\mathbf{v}=\\mathbf{v}$',
            'A3 加法结合律',
          ],
          answer: 1,
          explain: '封闭性（A1 和 S1）要求运算结果还在集合 $V$ 内，是最常被遗漏的步骤。其他公理往往由数域性质自动继承。',
        },
        {
          id: 'u4-l1-q2',
          type: 'judge',
          prompt: '集合 $B = \\{(x,y)^T \\in \\mathbb{R}^2 \\mid x+y=1\\}$（不过原点的直线）在通常加法和数乘下构成向量空间。',
          answer: false,
          explain: '零向量 $(0,0)^T$ 满足 $0+0=0\\neq 1$，故 $(0,0)^T\\notin B$，A4（零向量存在）不满足。直接否定。',
        },
        {
          id: 'u4-l1-q3',
          type: 'choice',
          prompt: '$\\mathcal{P}_2 = \\{a+bx+cx^2 \\mid a,b,c\\in\\mathbb{R}\\}$ 中，多项式 $p(x)=3-2x+x^2$ 的加法逆元是？',
          options: [
            '$3-2x+x^2$',
            '$-3+2x-x^2$',
            '$0$（零多项式）',
            '$3+2x-x^2$',
          ],
          answer: 1,
          explain: '加法逆元满足 $p+(-p)=0$，故对每个系数取反：$-3+2x-x^2$。验证：$(3-2x+x^2)+(-3+2x-x^2)=0$。',
        },
        {
          id: 'u4-l1-q4',
          type: 'choice',
          prompt: '单位球面 $S = \\{(x,y,z)^T \\mid x^2+y^2+z^2=1\\}$ 不是向量空间，违反了哪条公理？',
          options: [
            'A2 加法交换律',
            'A4 零向量存在',
            'S1 数乘封闭性',
            'A5 加法逆元存在',
          ],
          answer: 2,
          explain: '取 $(1,0,0)^T\\in S$，令 $c=2$，则 $2(1,0,0)^T=(2,0,0)^T$，但 $4+0+0=4\\neq 1$，不在 $S$ 中。S1 数乘封闭性不满足。（注意 A4 也不满足，因为 $(0,0,0)^T$ 不在 $S$ 上，但 S1 是最直接的反例。）',
        },
        {
          id: 'u4-l1-q5',
          type: 'input',
          prompt: '$\\mathcal{P}_n$（次数 $\\leq n$ 的实多项式）空间的零向量是什么？请用汉字或数字填写（如"零多项式"或"0"）。',
          accept: ['零多项式', '0', '零向量是零多项式'],
          placeholder: '零多项式',
          explain: '$\\mathcal{P}_n$ 的零向量是零多项式：所有系数均为 0 的多项式，即恒等于 0 的多项式。',
        },
        {
          id: 'u4-l1-q6',
          type: 'judge',
          prompt: '矩阵空间 $\\mathbb{R}^{m\\times n}$（所有 $m\\times n$ 实矩阵）在矩阵加法和数乘下构成向量空间，其维数为 $m+n$。',
          answer: false,
          explain: '维数应为 $mn$（共 $mn$ 个独立元素/基矩阵 $E_{ij}$），不是 $m+n$。例如 $\\mathbb{R}^{2\\times 3}$ 的维数是 $6$，不是 $5$。',
        },
        {
          id: 'u4-l1-q7',
          type: 'match',
          prompt: '将各向量空间与其**维数**配对：',
          left: [
            '$\\mathbb{R}^5$',
            '$\\mathbb{R}^{2\\times 3}$（$2\\times 3$ 矩阵）',
            '$\\mathcal{P}_3$（次数 $\\leq 3$ 多项式）',
            '$\\{\\mathbf{0}\\}$（零空间）',
          ],
          right: ['$5$', '$6$', '$4$', '$0$'],
          explain: '$\\mathbb{R}^n$ 维数 $n$；$\\mathbb{R}^{m\\times n}$ 维数 $mn=6$；$\\mathcal{P}_3$ 的基是 $\\{1,x,x^2,x^3\\}$，维数 $4$；零空间维数 $0$。',
        },
        {
          id: 'u4-l1-q8',
          type: 'choice',
          prompt: '设 $V = \\{(x,y)^T\\in\\mathbb{R}^2 \\mid y=3x\\}$，使用通常加法和数乘。下列关于 $V$ 的说法中，正确的是？',
          options: [
            '$V$ 不是向量空间，因为 $V$ 只含有限多个向量',
            '$V$ 是向量空间，零向量为 $(0,0)^T$，逆元为 $(-a,-3a)^T$',
            '$V$ 不是向量空间，数乘封闭性不满足',
            '$V$ 是向量空间，但没有加法逆元',
          ],
          answer: 1,
          explain: '封闭性：$(a,3a)+(b,3b)=(a+b,3(a+b))\\in V$；$c(a,3a)=(ca,3ca)\\in V$。零向量 $(0,0)^T$，逆元 $(-a,-3a)^T$。其余继承自 $\\mathbb{R}^2$。$V$ 是向量空间。',
        },
        {
          id: 'u4-l1-q9',
          type: 'judge',
          prompt: '在 VAE 的潜在空间 $\\mathbb{R}^k$ 中，对两个潜在向量 $\\mathbf{z}_A$ 和 $\\mathbf{z}_B$ 做线性插值 $(1-t)\\mathbf{z}_A + t\\mathbf{z}_B$（$t\\in[0,1]$），结果一定还在 $\\mathbb{R}^k$ 中。',
          answer: true,
          explain: '$(1-t)\\mathbf{z}_A$ 和 $t\\mathbf{z}_B$ 分别由 S1（数乘封闭）保证在 $\\mathbb{R}^k$ 中；两者之和由 A1（加法封闭）保证仍在 $\\mathbb{R}^k$ 中。正是向量空间公理赋予了这个操作的合法性。',
        },
      ],
    },

    // =====================================================================
    // u4-l2  线性相关与线性无关
    // =====================================================================
    {
      id: 'u4-l2',
      title: '线性相关与线性无关',
      subtitle: '冗余 vs 独立——行化简看主元',
      intro: [
        {
          title: '什么是线性组合？',
          body: '设 $\\mathbf{v}_1, \\ldots, \\mathbf{v}_k \\in \\mathbb{R}^n$，以及标量 $c_1, \\ldots, c_k \\in \\mathbb{R}$，则\n$\\mathbf{w} = c_1\\mathbf{v}_1 + c_2\\mathbf{v}_2 + \\cdots + c_k\\mathbf{v}_k$\n叫做这些向量的**线性组合**。\n\n**几何直觉**：在 $\\mathbb{R}^2$ 中，如果 $\\mathbf{v}_1$ 和 $\\mathbf{v}_2$ 不共线，调整 $c_1,c_2$ 可以到达平面上任意一点；如果共线，只能在那条直线上移动。',
          formula: '\\mathbf{w} = c_1\\mathbf{v}_1 + c_2\\mathbf{v}_2 + \\cdots + c_k\\mathbf{v}_k',
        },
        {
          title: '线性无关与线性相关的定义',
          body: '考虑齐次方程\n$c_1\\mathbf{v}_1 + c_2\\mathbf{v}_2 + \\cdots + c_k\\mathbf{v}_k = \\mathbf{0}$\n\n- **线性无关**：上式**只有零解**（$c_1=c_2=\\cdots=c_k=0$）——每个向量都提供独立方向，没有冗余。\n- **线性相关**：上式**存在非零解**——至少有一个向量能被其余向量线性表示，存在冗余。\n\n**等价表述**：$\\mathbf{v}_1,\\ldots,\\mathbf{v}_k$ 线性相关 $\\Leftrightarrow$ 至少一个 $\\mathbf{v}_i$ 是其余向量的线性组合。',
          tip: '零向量必定导致线性相关（零向量的系数取任意非零值都成立）；两向量成比例也必定线性相关。',
        },
        {
          title: '判断方法：行化简法（最通用）',
          body: '将 $\\mathbf{v}_1, \\ldots, \\mathbf{v}_k$ 排成矩阵 $A = [\\mathbf{v}_1 \\mid \\cdots \\mid \\mathbf{v}_k]$ 的列，行化简得 REF：\n\n- 每列都有主元（$\\operatorname{rank}(A)=k$）$\\Rightarrow$ **线性无关**\n- 存在无主元的自由列（$\\operatorname{rank}(A)<k$）$\\Rightarrow$ **线性相关**\n\n**方阵快速法**：当 $k=n$ 时，构成方阵 $A$，计算行列式：\n- $\\det(A)\\neq 0 \\Rightarrow$ 线性无关\n- $\\det(A)=0 \\Rightarrow$ 线性相关\n\n**维数界**：$\\mathbb{R}^n$ 中任意 $n+1$ 个向量必线性相关，无需计算，直接给结论。',
          formula: '\\operatorname{rank}(A)=k \\Leftrightarrow \\text{线性无关}',
        },
        {
          title: '张成（Span）',
          body: '$\\operatorname{span}\\{\\mathbf{v}_1,\\ldots,\\mathbf{v}_k\\}$ 是所有线性组合的集合，是包含这组向量的**最小子空间**。\n\n**关键事实**：向向量组添加已在张成内的向量，张成空间不变；只有添加"新方向"（线性无关向量）才能扩大张成。\n\n**判断 $\\mathbf{b} \\in \\operatorname{span}\\{\\mathbf{v}_1,\\ldots,\\mathbf{v}_k\\}$**：对增广矩阵 $[A|\\mathbf{b}]$ 行化简，若无矛盾行（如 $0=c,c\\neq0$）则有解，$\\mathbf{b}$ 在 Span 中。',
          tip: '$\\operatorname{span}$ 的维数等于 $\\operatorname{rank}([\\mathbf{v}_1|\\cdots|\\mathbf{v}_k])$，而非向量个数 $k$（线性相关时两者不同）。',
        },
        {
          title: '例题 1：行化简判断线性相关性',
          body: '**题目**：判断 $\\mathbf{v}_1=(1,2,1)^T$，$\\mathbf{v}_2=(2,5,3)^T$，$\\mathbf{v}_3=(0,1,1)^T$ 的线性相关性。',
          steps: [
            '构造矩阵 $A=[\\mathbf{v}_1|\\mathbf{v}_2|\\mathbf{v}_3]=\\begin{pmatrix}1&2&0\\\\2&5&1\\\\1&3&1\\end{pmatrix}$',
            '$R_2\\leftarrow R_2-2R_1$，$R_3\\leftarrow R_3-R_1$：$\\begin{pmatrix}1&2&0\\\\0&1&1\\\\0&1&1\\end{pmatrix}$',
            '$R_3\\leftarrow R_3-R_2$：$\\begin{pmatrix}1&2&0\\\\0&1&1\\\\0&0&0\\end{pmatrix}$',
            '$\\operatorname{rank}(A)=2<3$（列数），第 3 列是自由列，**线性相关**。',
            '令 $c_3=1$，回代得 $c_2=-1$，$c_1=2$，即 $2\\mathbf{v}_1-\\mathbf{v}_2+\\mathbf{v}_3=\\mathbf{0}$，亦即 $\\mathbf{v}_3=\\mathbf{v}_2-2\\mathbf{v}_1$。',
          ],
        },
        {
          title: '例题 2：判断向量是否在张成中',
          body: '**题目**：$\\mathbf{u}_1=(1,1,0)^T$，$\\mathbf{u}_2=(2,0,1)^T$，$\\mathbf{b}=(5,1,2)^T$。$\\mathbf{b}$ 是否在 $\\operatorname{span}\\{\\mathbf{u}_1,\\mathbf{u}_2\\}$ 中？',
          steps: [
            '对增广矩阵 $[A|\\mathbf{b}]=\\begin{pmatrix}1&2&5\\\\1&0&1\\\\0&1&2\\end{pmatrix}$ 行化简。',
            '$R_2\\leftarrow R_2-R_1$：$\\begin{pmatrix}1&2&5\\\\0&-2&-4\\\\0&1&2\\end{pmatrix}$',
            '$R_2\\leftarrow -\\frac{1}{2}R_2$，$R_3\\leftarrow R_3-R_2$：$\\begin{pmatrix}1&2&5\\\\0&1&2\\\\0&0&0\\end{pmatrix}$',
            '无矛盾行，方程组有解。回代：$c_2=2$，$c_1=5-4=1$。',
            '故 $\\mathbf{b}=\\mathbf{u}_1+2\\mathbf{u}_2=(1,1,0)^T+(4,0,2)^T=(5,1,2)^T$。$\\mathbf{b}\\in\\operatorname{span}\\{\\mathbf{u}_1,\\mathbf{u}_2\\}$。',
          ],
          tip: '**AI 联系**：神经网络权重矩阵的列若线性相关，意味着存在冗余神经元——LoRA 和低秩分解正是利用这一点压缩模型参数。',
        },
        {
          title: '易错点与条件反射',
          body: '**易错点 1**：线性相关不等于所有向量都能被其余表示——只需**至少一个**被其余表示即可。\n\n**易错点 2**：行化简后，取**主元所在的列**（看哪列有主元），而非直接看行。第 $j$ 列是自由列，$\\mathbf{v}_j$ 才可被之前主元列表示。\n\n**易错点 3**：$\\operatorname{span}$ 的维数是矩阵的秩，不是向量个数。\n\n**条件反射**：\n- 看到"含零向量"→ 立刻说线性相关\n- 看到"两向量成比例"→ 立刻说线性相关\n- 看到"$k>n$"（向量个数超过维数）→ 立刻说线性相关\n- 看到方阵 → 算行列式，非零则无关',
          reveal: {
            q: '若 $\\mathbf{v}_1,\\mathbf{v}_2$ 线性无关，那么 $\\mathbf{v}_1+\\mathbf{v}_2$ 和 $\\mathbf{v}_1-\\mathbf{v}_2$ 是否线性无关？',
            a: '**是**！设 $a(\\mathbf{v}_1+\\mathbf{v}_2)+b(\\mathbf{v}_1-\\mathbf{v}_2)=\\mathbf{0}$，整理得 $(a+b)\\mathbf{v}_1+(a-b)\\mathbf{v}_2=\\mathbf{0}$。由 $\\mathbf{v}_1,\\mathbf{v}_2$ 线性无关，得 $a+b=0$ 且 $a-b=0$，解得 $a=b=0$。故 $\\mathbf{v}_1\\pm\\mathbf{v}_2$ 线性无关。',
          },
        },
      ],
      questions: [
        {
          id: 'u4-l2-q1',
          type: 'judge',
          prompt: '$\\mathbb{R}^3$ 中任意 4 个向量必定线性相关，无论这 4 个向量是什么。',
          answer: true,
          explain: '$\\mathbb{R}^3$ 的维数是 3，向量个数 $4>3$，由"维数界"（$\\mathbb{R}^n$ 中任意 $n+1$ 个向量必线性相关）直接得出，无需任何计算。',
        },
        {
          id: 'u4-l2-q2',
          type: 'choice',
          prompt: '向量组 $\\mathbf{v}_1=(1,0)^T$，$\\mathbf{v}_2=(0,1)^T$，$\\mathbf{v}_3=(2,3)^T$ 的线性相关性如何？',
          options: [
            '线性无关，因为三个向量方向各不相同',
            '线性相关，$\\mathbf{v}_3=2\\mathbf{v}_1+3\\mathbf{v}_2$',
            '线性相关，$\\mathbf{v}_1=2\\mathbf{v}_3-3\\mathbf{v}_2$',
            '无法判断，需要计算行列式',
          ],
          answer: 1,
          explain: '设 $c_1\\mathbf{v}_1+c_2\\mathbf{v}_2+c_3\\mathbf{v}_3=\\mathbf{0}$，得 $c_1+2c_3=0,c_2+3c_3=0$。取 $c_3=1$ 得非零解 $(c_1,c_2,c_3)=(-2,-3,1)$，故线性相关，且 $\\mathbf{v}_3=2\\mathbf{v}_1+3\\mathbf{v}_2$。（$\\mathbb{R}^2$ 中 3 个向量也必然线性相关。）',
        },
        {
          id: 'u4-l2-q3',
          type: 'choice',
          prompt: '对矩阵 $A=[\\mathbf{v}_1|\\mathbf{v}_2|\\mathbf{v}_3]$ 行化简后得\n$\\begin{pmatrix}1&0&2\\\\0&1&-1\\\\0&0&0\\end{pmatrix}$\n则 $\\mathbf{v}_3$ 与 $\\mathbf{v}_1,\\mathbf{v}_2$ 的关系是？',
          options: [
            '$\\mathbf{v}_3 = 2\\mathbf{v}_1 + (-1)\\mathbf{v}_2$（线性相关）',
            '$\\mathbf{v}_3 = 2\\mathbf{v}_1 - \\mathbf{v}_2$（线性相关）',
            '$\\mathbf{v}_3 = -2\\mathbf{v}_1 + \\mathbf{v}_2$（线性相关）',
            '$\\mathbf{v}_1,\\mathbf{v}_2,\\mathbf{v}_3$ 线性无关',
          ],
          answer: 1,
          explain: 'RREF 第 3 列是自由列，令 $c_3=1$，回代：$c_1+2(1)=0\\Rightarrow c_1=-2$；$c_2-1=0\\Rightarrow c_2=1$。故 $-2\\mathbf{v}_1+\\mathbf{v}_2+\\mathbf{v}_3=\\mathbf{0}$，即 $\\mathbf{v}_3=2\\mathbf{v}_1-\\mathbf{v}_2$。',
        },
        {
          id: 'u4-l2-q4',
          type: 'judge',
          prompt: '向量 $\\mathbf{b}=(1,2,3)^T$ 是否一定在 $\\operatorname{span}\\{(1,0,0)^T,(0,1,0)^T,(0,0,1)^T\\}$ 中？（标准基的张成）',
          answer: true,
          explain: '标准基 $\\{e_1,e_2,e_3\\}$ 的张成是整个 $\\mathbb{R}^3$，任何向量都在其中。$\\mathbf{b}=1\\cdot e_1+2\\cdot e_2+3\\cdot e_3$。',
        },
        {
          id: 'u4-l2-q5',
          type: 'choice',
          prompt: '用行列式判断 $\\mathbf{a}_1=(1,0,2)^T$，$\\mathbf{a}_2=(0,1,-1)^T$，$\\mathbf{a}_3=(3,2,4)^T$ 的线性相关性：\n$\\det\\begin{pmatrix}1&0&3\\\\0&1&2\\\\2&-1&4\\end{pmatrix}$\n的值是？',
          options: ['$6$', '$0$', '$-6$', '$1$'],
          answer: 1,
          explain: '按第一列展开：$1\\cdot(1\\cdot4-2\\cdot(-1))-0+2\\cdot(0\\cdot2-1\\cdot3)=1\\cdot(4+2)+2\\cdot(0-3)=6-6=0$。行列式为 $0$，三向量线性相关（$\\mathbf{a}_3=3\\mathbf{a}_1+2\\mathbf{a}_2$）。',
        },
        {
          id: 'u4-l2-q6',
          type: 'input',
          prompt: '矩阵 $A=[\\mathbf{v}_1|\\mathbf{v}_2|\\mathbf{v}_3]$ 行化简后有 2 个主元，则 $\\operatorname{rank}(A)=$ 多少？（填数字）',
          accept: ['2'],
          placeholder: '2',
          explain: '行化简后主元的个数即为矩阵的秩。2 个主元，秩为 2，且 $2<3$（列数），故三列向量线性相关。',
        },
        {
          id: 'u4-l2-q7',
          type: 'judge',
          prompt: '设 $\\mathbf{v}_1,\\mathbf{v}_2,\\mathbf{v}_3\\in\\mathbb{R}^4$ 线性无关，则向量组 $\\{\\mathbf{v}_1,\\mathbf{v}_2,\\mathbf{v}_3,\\mathbf{v}_1+\\mathbf{v}_2-\\mathbf{v}_3\\}$ 必定线性相关。',
          answer: true,
          explain: '设 $c_1\\mathbf{v}_1+c_2\\mathbf{v}_2+c_3\\mathbf{v}_3+c_4(\\mathbf{v}_1+\\mathbf{v}_2-\\mathbf{v}_3)=\\mathbf{0}$，整理得 $(c_1+c_4)\\mathbf{v}_1+(c_2+c_4)\\mathbf{v}_2+(c_3-c_4)\\mathbf{v}_3=\\mathbf{0}$。由无关性得 $c_1+c_4=c_2+c_4=0$，$c_3-c_4=0$，取 $c_4=1$ 得非零解。第四个向量已在前三者的张成中，故整体线性相关。',
        },
        {
          id: 'u4-l2-q8',
          type: 'match',
          prompt: '将各情形与线性相关性结论配对：',
          left: [
            '含零向量',
            '两向量成比例：$\\mathbf{v}_2=3\\mathbf{v}_1$',
            '$\\mathbb{R}^3$ 中 3 个向量，行列式非零',
            '$\\mathbb{R}^3$ 中 4 个向量',
          ],
          right: [
            '必线性相关（零向量系数任取非零值）',
            '必线性相关（$1\\cdot\\mathbf{v}_2+(-3)\\cdot\\mathbf{v}_1=\\mathbf{0}$）',
            '线性无关',
            '必线性相关（维数界）',
          ],
          explain: '三个规则：①含零向量必相关；②成比例必相关；③行列式非零的 $n\\times n$ 方阵列向量线性无关；④向量个数超过维数必相关。',
        },
        {
          id: 'u4-l2-q9',
          type: 'choice',
          prompt: '某神经网络权重矩阵 $W\\in\\mathbb{R}^{3\\times 3}$ 为\n$W=\\begin{pmatrix}1&2&3\\\\2&4&6\\\\1&2&3\\end{pmatrix}$\n$\\operatorname{rank}(W)$ 是多少？',
          options: ['$3$', '$2$', '$1$', '$0$'],
          answer: 2,
          explain: '三行均成比例：行 2 = 2 × 行 1，行 3 = 行 1。仅有 1 个线性独立方向，故 $\\operatorname{rank}(W)=1$。三个神经元完全冗余——实际上只有 1 个独立神经元在工作。',
        },
      ],
    },

    // =====================================================================
    // u4-l3  基与维数
    // =====================================================================
    {
      id: 'u4-l3',
      title: '基与维数',
      subtitle: '坐标系骨架与空间自由度',
      intro: [
        {
          title: '基的定义',
          body: '向量空间 $V$ 的一个**基**（Basis）是满足以下**两个条件**的向量组 $\\mathcal{B}=\\{\\mathbf{b}_1,\\ldots,\\mathbf{b}_n\\}$：\n\n1. **线性无关**：$c_1\\mathbf{b}_1+\\cdots+c_n\\mathbf{b}_n=\\mathbf{0}$ 只有零解。\n2. **张成整个空间**：$V=\\operatorname{span}\\{\\mathbf{b}_1,\\ldots,\\mathbf{b}_n\\}$，即 $V$ 中每个向量都能写成 $\\mathbf{b}_i$ 的线性组合。\n\n**两个条件缺一不可**：只有张成但有冗余（线性相关），或只有无关但覆盖不了整个空间——都不能叫基。',
          tip: '在 $\\mathbb{R}^n$ 中，$n$ 个向量构成基 $\\Leftrightarrow$ 它们线性无关（因为 $n$ 个无关向量自动张成 $\\mathbb{R}^n$）$\\Leftrightarrow$ 方阵行列式非零 $\\Leftrightarrow$ 矩阵可逆。',
        },
        {
          title: '标准基与基的非唯一性',
          body: '**$\\mathbb{R}^n$ 的标准基**：\n$e_1=\\begin{pmatrix}1\\\\0\\\\\\vdots\\\\0\\end{pmatrix},\\quad e_2=\\begin{pmatrix}0\\\\1\\\\\\vdots\\\\0\\end{pmatrix},\\quad\\ldots,\\quad e_n=\\begin{pmatrix}0\\\\0\\\\\\vdots\\\\1\\end{pmatrix}$\n\n$\\mathbb{R}^2$ 的另一个合法的基：$\\mathcal{B}_2=\\{(1,1)^T,(1,-1)^T\\}$——同样线性无关且张成 $\\mathbb{R}^2$。\n\n**关键事实**：同一向量空间可有无数不同的基，但任意两个基包含**相同数量**的向量——这个共同数字定义了维数。',
          formula: '\\mathcal{P}_n \\text{ 的标准基：} \\{1, x, x^2, \\ldots, x^n\\},\\quad \\dim(\\mathcal{P}_n)=n+1',
        },
        {
          title: '维数',
          body: '向量空间 $V$ 的**维数**（Dimension）$\\dim(V)$ 定义为任意一个基中向量的个数。\n\n**为什么良定义？**——由"替换引理"可证明：任意两个基的元素个数相同（Steinitz 交换引理），所以不管挑哪个基，数出来的维数相同。\n\n**常见维数**：\n\n| 空间 | 维数 |\n|------|------|\n| $\\mathbb{R}^n$ | $n$ |\n| $\\mathbb{R}^{m\\times n}$（矩阵） | $mn$ |\n| $\\mathcal{P}_n$（多项式） | $n+1$ |\n| $\\{\\mathbf{0}\\}$（零空间） | $0$ |\n| $C[a,b]$（连续函数） | $\\infty$ |\n\n**重要推论**：若 $\\dim(V)=n$，则 $V$ 中任意 $n+1$ 个向量必线性相关；任意 $n$ 个线性无关向量自动构成基。',
        },
        {
          title: '秩-零化度定理',
          body: '设 $A$ 是 $m\\times n$ 矩阵，$\\operatorname{rank}(A)$ 是其秩（行化简后主元个数），$\\operatorname{nullity}(A)=\\dim(\\operatorname{Null}(A))$ 是零空间维数，则：',
          formula: '\\operatorname{rank}(A) + \\operatorname{nullity}(A) = n',
          tip: '这里 $n$ 是**列数**（不是行数）！秩 = 独立方向数，零化度 = 被压缩为零的方向数，两者合计恰好填满输入空间 $\\mathbb{R}^n$。',
        },
        {
          title: '求坐标',
          body: '设 $\\mathcal{B}=\\{\\mathbf{b}_1,\\ldots,\\mathbf{b}_n\\}$ 是 $V$ 的基，$\\mathbf{x}\\in V$，则存在**唯一**的系数 $c_1,\\ldots,c_n$ 使得\n$\\mathbf{x}=c_1\\mathbf{b}_1+\\cdots+c_n\\mathbf{b}_n$\n向量 $[\\mathbf{x}]_{\\mathcal{B}}=(c_1,\\ldots,c_n)^T$ 叫做 $\\mathbf{x}$ 在基 $\\mathcal{B}$ 下的**坐标向量**。\n\n**求坐标的标准方法**：对增广矩阵 $[\\mathbf{b}_1|\\cdots|\\mathbf{b}_n|\\mathbf{x}]$ 行化简到 RREF，右列即为坐标向量。',
          tip: '坐标是描述工具，向量是客观存在——换一个基，坐标改变，但向量本身不变。PCA 就是找到最"紧凑"的基，让数据在新坐标下大部分信息浓缩在前几个坐标里。',
        },
        {
          title: '例题：验证基并求坐标',
          body: '**题目**：设 $\\mathbf{b}_1=(1,1,0)^T$，$\\mathbf{b}_2=(1,0,1)^T$，$\\mathbf{b}_3=(0,1,1)^T$。\n(1) 证明 $\\mathcal{B}$ 是 $\\mathbb{R}^3$ 的基；(2) 求 $\\mathbf{x}=(2,1,3)^T$ 在 $\\mathcal{B}$ 下的坐标。',
          steps: [
            '(1) 计算行列式：$\\det\\begin{pmatrix}1&1&0\\\\1&0&1\\\\0&1&1\\end{pmatrix}=1(0-1)-1(1-0)+0=-1-1=-2\\neq 0$。',
            '行列式非零，故三向量线性无关，在 $\\mathbb{R}^3$ 中构成基。',
            '(2) 增广矩阵行化简：$\\begin{pmatrix}1&1&0&2\\\\1&0&1&1\\\\0&1&1&3\\end{pmatrix}\\xrightarrow{R_2-R_1}\\begin{pmatrix}1&1&0&2\\\\0&-1&1&-1\\\\0&1&1&3\\end{pmatrix}\\xrightarrow{R_3+R_2}\\begin{pmatrix}1&1&0&2\\\\0&-1&1&-1\\\\0&0&2&2\\end{pmatrix}$',
            '回代：$2c_3=2\\Rightarrow c_3=1$；$-c_2+1=-1\\Rightarrow c_2=2$；$c_1+2=2\\Rightarrow c_1=0$。',
            '验证：$0\\cdot\\mathbf{b}_1+2\\cdot\\mathbf{b}_2+1\\cdot\\mathbf{b}_3=(0,0,0)^T+(2,0,2)^T+(0,1,1)^T=(2,1,3)^T$ ✓',
            '故 $[\\mathbf{x}]_{\\mathcal{B}}=(0,2,1)^T$。',
          ],
        },
        {
          title: '易错点与 PCA 联系',
          body: '**易错点 1**：求列空间基时，取**原矩阵**的主元列，不是行阶梯形的列！行化简只用来定位主元位置。\n\n**易错点 2**："基的大小唯一"$\\neq$"基唯一"，同一空间可有无数基，但它们的向量个数（维数）相同。\n\n**易错点 3**：秩-零化度定理中 $n$ 是**列数**，不是行数。\n\n**PCA 联系**：主成分分析的本质是找协方差矩阵的前 $k$ 个特征向量（正交基），把数据坐标变换到新基下——信息浓缩在前几个坐标里，后面的坐标接近 0 就可以安全丢弃，实现降维。',
          reveal: {
            q: '$A$ 是 $4\\times 6$ 矩阵，$\\operatorname{rank}(A)=3$。零空间的维数是多少？零空间的基有几个向量？',
            a: '由秩-零化度定理（$n=6$ 列）：$\\operatorname{nullity}(A)=6-3=3$。所以零空间维数是 $3$，其基含 **3 个**向量。直觉：输入空间 $\\mathbb{R}^6$ 中有 3 个方向被 $A$ "有效映射"，另外 3 个方向被压缩为零。',
          },
        },
      ],
      questions: [
        {
          id: 'u4-l3-q1',
          type: 'choice',
          prompt: '向量组 $\\{(1,0,-1)^T,(2,1,0)^T,(0,1,2)^T\\}$ 是否构成 $\\mathbb{R}^3$ 的基？计算行列式：\n$\\det\\begin{pmatrix}1&2&0\\\\0&1&1\\\\-1&0&2\\end{pmatrix}$',
          options: [
            '行列式 $= 4\\neq 0$，构成基',
            '行列式 $= 0$，不构成基（线性相关）',
            '行列式 $= -4\\neq 0$，构成基',
            '行列式 $= 2\\neq 0$，构成基',
          ],
          answer: 0,
          explain: '按第一列展开：$1\\cdot(1\\cdot2-1\\cdot0)-0+(-1)\\cdot(-1)\\cdot(2\\cdot1-0\\cdot1)=1\\cdot2+1\\cdot2=4\\neq 0$。行列式非零，三向量线性无关，构成 $\\mathbb{R}^3$ 的基。',
        },
        {
          id: 'u4-l3-q2',
          type: 'input',
          prompt: '向量 $\\mathbf{x}=(7,11)^T$ 在基 $\\mathcal{B}=\\{\\mathbf{b}_1=(1,2)^T,\\mathbf{b}_2=(3,5)^T\\}$ 下的坐标 $c_1$ 是多少？（填整数）',
          accept: ['-2', '−2'],
          placeholder: '-2',
          explain: '解 $c_1(1,2)^T+c_2(3,5)^T=(7,11)^T$，增广矩阵行化简得 $c_1=-2,c_2=3$。验证：$-2(1,2)^T+3(3,5)^T=(-2,-4)^T+(9,15)^T=(7,11)^T$ ✓。',
        },
        {
          id: 'u4-l3-q3',
          type: 'judge',
          prompt: '$A$ 是 $3\\times 5$ 矩阵，$\\operatorname{rank}(A)=3$，则 $\\operatorname{nullity}(A)=2$。',
          answer: true,
          explain: '由秩-零化度定理（$n=5$ 列）：$\\operatorname{nullity}(A)=5-3=2$。零空间有 2 个基向量，方程 $A\\mathbf{x}=\\mathbf{0}$ 的通解含 2 个自由变量。',
        },
        {
          id: 'u4-l3-q4',
          type: 'choice',
          prompt: '矩阵 $A=\\begin{pmatrix}1&2&1\\\\2&4&2\\\\3&6&3\\end{pmatrix}$ 的 $\\operatorname{nullity}(A)$ 是多少？',
          options: ['$0$', '$1$', '$2$', '$3$'],
          answer: 2,
          explain: '三行均成比例（行 2=2行 1，行 3=3行 1），$\\operatorname{rank}(A)=1$。由秩-零化度（$n=3$）：$\\operatorname{nullity}(A)=3-1=2$。',
        },
        {
          id: 'u4-l3-q5',
          type: 'choice',
          prompt: '$\\mathcal{P}_2$（次数 $\\leq 2$ 的实多项式）的维数是多少？',
          options: ['$2$', '$3$', '$4$', '无穷'],
          answer: 1,
          explain: '$\\mathcal{P}_2$ 的标准基是 $\\{1, x, x^2\\}$，共 3 个向量，故 $\\dim(\\mathcal{P}_2)=3=2+1$。一般地，$\\dim(\\mathcal{P}_n)=n+1$。',
        },
        {
          id: 'u4-l3-q6',
          type: 'judge',
          prompt: '向量 $\\mathbf{x}=(2,1,3)^T$ 在基 $\\mathcal{B}=\\{(1,1,0)^T,(1,0,1)^T,(0,1,1)^T\\}$ 下的坐标向量为 $(0,2,1)^T$。',
          answer: true,
          explain: '验证：$0\\cdot(1,1,0)^T+2\\cdot(1,0,1)^T+1\\cdot(0,1,1)^T=(0,0,0)^T+(2,0,2)^T+(0,1,1)^T=(2,1,3)^T$ ✓。坐标向量 $(0,2,1)^T$ 正确。',
        },
        {
          id: 'u4-l3-q7',
          type: 'match',
          prompt: '将各矩阵情形与 $\\operatorname{nullity}$ 配对（$A$ 均为 $m\\times n$ 矩阵）：',
          left: [
            '$3\\times 4$ 矩阵，$\\operatorname{rank}=3$',
            '$2\\times 5$ 矩阵，$\\operatorname{rank}=2$',
            '$4\\times 4$ 方阵，$\\operatorname{rank}=4$（满秩）',
            '$3\\times 3$ 矩阵，$\\operatorname{rank}=1$',
          ],
          right: ['$1$', '$3$', '$0$', '$2$'],
          explain: '由秩-零化度定理 $\\operatorname{nullity}=n-\\operatorname{rank}$（$n$ 为列数）：$4-3=1$；$5-2=3$；$4-4=0$；$3-1=2$。',
        },
        {
          id: 'u4-l3-q8',
          type: 'choice',
          prompt: '求矩阵 $A=\\begin{pmatrix}1&2&3&0\\\\2&4&7&-1\\\\3&6&10&-1\\end{pmatrix}$ 的列空间的基，应取原矩阵的哪些列？（行化简后主元在第 1、3 列。）',
          options: [
            '行阶梯形的第 1、3 列',
            '原矩阵 $A$ 的第 1、3 列：$(1,2,3)^T$ 和 $(3,7,10)^T$',
            '原矩阵 $A$ 的第 1、2 列',
            '行阶梯形的非零行',
          ],
          answer: 1,
          explain: '列空间的基**必须取原矩阵的主元列**，不是行阶梯形的列。主元在第 1、3 列，故基为原矩阵的第 1、3 列：$(1,2,3)^T$ 和 $(3,7,10)^T$。',
        },
        {
          id: 'u4-l3-q9',
          type: 'judge',
          prompt: 'PCA 降维的数学本质，是找协方差矩阵的前 $k$ 个特征向量作为新基，然后把数据坐标变换到新基下，并丢弃后面贡献小的坐标。',
          answer: true,
          explain: 'PCA 的核心：协方差矩阵对称正定，其特征向量构成正交基；按特征值从大到小取前 $k$ 个，组成 $k$ 维子空间；数据在此子空间的坐标保留了最多方差。这正是"选最优基，重新坐标化"的过程。',
        },
      ],
    },

    // =====================================================================
    // u4-l4  子空间与四大基本子空间
    // =====================================================================
    {
      id: 'u4-l4',
      title: '子空间',
      subtitle: '含零向量 + 两个封闭性——矩阵的四大子空间',
      intro: [
        {
          title: '子空间的定义',
          body: '设 $V$ 是向量空间，$W\\subseteq V$ 是非空子集。若 $W$ 满足以下三个条件，则称 $W$ 是 $V$ 的**子空间（subspace）**：\n\n1. **包含零向量**：$\\mathbf{0}\\in W$\n2. **加法封闭**：$\\mathbf{u},\\mathbf{v}\\in W \\Rightarrow \\mathbf{u}+\\mathbf{v}\\in W$\n3. **数乘封闭**：$\\mathbf{v}\\in W,\\, c\\in\\mathbb{R} \\Rightarrow c\\mathbf{v}\\in W$\n\n条件 2 和 3 合在一起等价于**对线性组合封闭**：$a\\mathbf{u}+b\\mathbf{v}\\in W$。\n\n**两个平凡子空间**：$\\{\\mathbf{0}\\}$（最小）和 $V$ 自身（最大），始终是子空间。\n\n**张成集是最小子空间**：$\\operatorname{span}\\{\\mathbf{v}_1,\\ldots,\\mathbf{v}_k\\}$ 一定是子空间，且是包含这些向量的最小子空间。',
          tip: '验证子空间**三步法**：①零向量在吗？②两元素之和还在吗？③任意标量倍还在吗？只要有一步失败就不是子空间。',
        },
        {
          title: '子空间的具体验证',
          body: '**例 1（是子空间）**：$W=\\{(x,y,0)^T\\mid x,y\\in\\mathbb{R}\\}$（$xy$ 平面）。\n- 零向量：$(0,0,0)^T\\in W$ ✓\n- 加法：$(x_1,y_1,0)^T+(x_2,y_2,0)^T=(x_1+x_2,y_1+y_2,0)^T\\in W$ ✓\n- 数乘：$c(x,y,0)^T=(cx,cy,0)^T\\in W$ ✓\n\n**例 2（不是子空间）**：$W_2=\\{(x,y,z)^T\\mid x+y+z=1\\}$（不过原点的平面）。\n- 零向量：$0+0+0=0\\neq 1$，$(0,0,0)^T\\notin W_2$ ✗\n\n**快速否定**：若集合不含零向量，直接否定——不过原点的平面、仿射子空间（$A\\mathbf{x}=\\mathbf{b},\\mathbf{b}\\neq\\mathbf{0}$）都不是子空间。',
        },
        {
          title: '矩阵的四大基本子空间',
          body: '设 $A$ 是 $m\\times n$ 矩阵，秩为 $r$。四大基本子空间：\n\n| 子空间 | 定义 | 所在空间 | 维数 |\n|--------|------|----------|------|\n| 列空间 $\\operatorname{Col}(A)$ | $A$ 的列向量的所有线性组合 | $\\mathbb{R}^m$ | $r$ |\n| 行空间 $\\operatorname{Row}(A)$ | $A$ 的行向量的所有线性组合 | $\\mathbb{R}^n$ | $r$ |\n| 零空间 $\\operatorname{Null}(A)$ | $A\\mathbf{x}=\\mathbf{0}$ 的解集 | $\\mathbb{R}^n$ | $n-r$ |\n| 左零空间 $\\operatorname{Null}(A^T)$ | $A^T\\mathbf{y}=\\mathbf{0}$ 的解集 | $\\mathbb{R}^m$ | $m-r$ |\n\n**几何含义**：$A\\mathbf{x}=\\mathbf{b}$ 有解 $\\Leftrightarrow$ $\\mathbf{b}\\in\\operatorname{Col}(A)$；零空间描述了被 $A$ "压缩"成零的所有输入方向。',
          formula: '\\operatorname{Col}(A)\\perp\\operatorname{Null}(A^T),\\quad \\operatorname{Row}(A)\\perp\\operatorname{Null}(A)',
        },
        {
          title: '秩-零化度与解的结构',
          body: '**秩-零化度定理**：$\\operatorname{rank}(A)+\\operatorname{nullity}(A)=n$（列数）。\n\n**解的结构定理**：若 $A\\mathbf{x}=\\mathbf{b}$ 有解，则通解为\n$\\mathbf{x}=\\mathbf{x}_p+\\mathbf{x}_h,\\quad \\mathbf{x}_h\\in\\operatorname{Null}(A)$\n其中 $\\mathbf{x}_p$ 是任一特解，$\\mathbf{x}_h$ 是零空间中的任意向量。\n\n**零空间是子空间**（验证）：\n- 零向量：$A\\mathbf{0}=\\mathbf{0}$ ✓\n- 加法封闭：$A\\mathbf{u}=A\\mathbf{v}=\\mathbf{0}\\Rightarrow A(\\mathbf{u}+\\mathbf{v})=\\mathbf{0}$ ✓\n- 数乘封闭：$A\\mathbf{v}=\\mathbf{0}\\Rightarrow A(c\\mathbf{v})=c\\mathbf{0}=\\mathbf{0}$ ✓',
          tip: '通解 = 特解 + 零空间。零空间是子空间（过原点），通解集是把零空间"平移"到特解处的仿射集（不是子空间，除非 $\\mathbf{x}_p=\\mathbf{0}$）。',
        },
        {
          title: '子空间的交与和',
          body: '设 $U,W$ 是 $V$ 的子空间。\n\n- **交** $U\\cap W$ 是子空间（两者共有元素，含零向量，封闭性验证简单）。\n- **并** $U\\cup W$ 一般**不是**子空间（各取一个元素相加，结果未必在并中）。\n- **和** $U+W=\\{\\mathbf{u}+\\mathbf{w}\\mid\\mathbf{u}\\in U,\\mathbf{w}\\in W\\}$ 是子空间，是同时包含 $U$ 和 $W$ 的最小子空间。\n\n**维数公式**（子空间版容斥原理）：\n$\\dim(U+W)=\\dim(U)+\\dim(W)-\\dim(U\\cap W)$\n\n**直和**：若 $U\\cap W=\\{\\mathbf{0}\\}$，则 $U+W=U\\oplus W$（直和），且 $\\dim(U\\oplus W)=\\dim(U)+\\dim(W)$。',
          formula: '\\mathbb{R}^n = \\operatorname{Row}(A) \\oplus \\operatorname{Null}(A)',
        },
        {
          title: '例题：四大子空间计算',
          body: '**题目**：设 $A=\\begin{pmatrix}1&0&2\\\\0&1&-1\\\\2&1&3\\end{pmatrix}$，求四大子空间的维数。',
          steps: [
            '行化简：$\\begin{pmatrix}1&0&2\\\\0&1&-1\\\\2&1&3\\end{pmatrix}\\xrightarrow{R_3-2R_1}\\begin{pmatrix}1&0&2\\\\0&1&-1\\\\0&1&-1\\end{pmatrix}\\xrightarrow{R_3-R_2}\\begin{pmatrix}1&0&2\\\\0&1&-1\\\\0&0&0\\end{pmatrix}$',
            '$\\operatorname{rank}(A)=2$（两个主元，在第 1、2 列）。',
            '$\\operatorname{Col}(A)$：取原矩阵第 1、2 列，$\\dim=2$；所在 $\\mathbb{R}^3$。',
            '$\\operatorname{Row}(A)$：取非零行（行阶梯形的非零行），$\\dim=2$；所在 $\\mathbb{R}^3$（列数）。',
            '$\\operatorname{Null}(A)$：$\\operatorname{nullity}=3-2=1$，自由变量 $x_3=t$，回代得 $x_2=t,x_1=-2t$，基向量 $(-2,1,1)^T$。',
            '$\\operatorname{Null}(A^T)$：$\\dim=m-r=3-2=1$。',
          ],
          tip: 'LoRA 联系：LoRA 微调大模型时，将权重更新量 $\\Delta W=BA$（$B\\in\\mathbb{R}^{d\\times r}$，$A\\in\\mathbb{R}^{r\\times d}$，$r\\ll d$）的列空间限制在 $r$ 维子空间内——正是本章子空间思想的直接应用，用极少参数刻画任务迁移所需的更新方向。',
        },
        {
          title: '易错点总结',
          body: '**易错点 1**：两子空间的**并**一般不是子空间（$x$ 轴和 $y$ 轴都是 $\\mathbb{R}^2$ 的子空间，但它们的并不是——$(1,0)+(0,1)=(1,1)$ 不在并中）。\n\n**易错点 2**：不含零向量的集合直接否定（A4 不满足）。不过原点的平面 $x+y+z=1$ 不是子空间。\n\n**易错点 3**：四大子空间的所在空间：列空间和左零空间在 $\\mathbb{R}^m$（行数），行空间和零空间在 $\\mathbb{R}^n$（列数）。\n\n**易错点 4**：通解集不是子空间（除非 $\\mathbf{b}=\\mathbf{0}$）——特解不过原点，平移后的仿射集不含零向量。',
          reveal: {
            q: '若 $V$ 和 $W$ 都是 $\\mathbb{R}^5$ 的三维子空间，则 $\\dim(V\\cap W)$ 的最小可能值是多少？',
            a: '**最小为 1**。由维数公式：$\\dim(V+W)=\\dim(V)+\\dim(W)-\\dim(V\\cap W)=6-\\dim(V\\cap W)$。又 $V+W\\subseteq\\mathbb{R}^5$，故 $\\dim(V+W)\\leq 5$，即 $6-\\dim(V\\cap W)\\leq 5$，得 $\\dim(V\\cap W)\\geq 1$。所以最小值是 **1**（此时 $V+W=\\mathbb{R}^5$，两三维子空间恰好只共享一条直线）。',
          },
        },
      ],
      questions: [
        {
          id: 'u4-l4-q1',
          type: 'judge',
          prompt: '集合 $W_1 = \\{(x,y,z)^T \\in \\mathbb{R}^3 \\mid 2x-y+z=0\\}$（过原点的平面）是 $\\mathbb{R}^3$ 的子空间。',
          answer: true,
          explain: '①零向量：$2(0)-0+0=0$ ✓；②加法封闭：若 $2x_1-y_1+z_1=0$ 且 $2x_2-y_2+z_2=0$，则 $2(x_1+x_2)-(y_1+y_2)+(z_1+z_2)=0$ ✓；③数乘封闭：$2(cx)-cy+cz=c(2x-y+z)=0$ ✓。三条满足，是子空间。',
        },
        {
          id: 'u4-l4-q2',
          type: 'choice',
          prompt: '设 $A$ 是 $5\\times 7$ 矩阵，$\\operatorname{rank}(A)=3$。下列维数中，**错误**的是哪一项？',
          options: [
            '$\\dim(\\operatorname{Col}(A))=3$',
            '$\\dim(\\operatorname{Null}(A))=4$',
            '$\\dim(\\operatorname{Row}(A))=3$',
            '$\\dim(\\operatorname{Null}(A^T))=3$',
          ],
          answer: 3,
          explain: '$\\operatorname{Null}(A^T)$ 所在空间是 $\\mathbb{R}^m=\\mathbb{R}^5$，维数为 $m-r=5-3=2$，不是 $3$。其余三项均正确：$\\dim(\\operatorname{Col})=\\dim(\\operatorname{Row})=r=3$；$\\dim(\\operatorname{Null})=n-r=7-3=4$。',
        },
        {
          id: 'u4-l4-q3',
          type: 'judge',
          prompt: '两个子空间的**并** $U\\cup W$ 也一定是子空间。',
          answer: false,
          explain: '反例：$U=\\{(x,0)^T\\}$（$x$ 轴）和 $W=\\{(0,y)^T\\}$（$y$ 轴）都是 $\\mathbb{R}^2$ 的子空间，但 $(1,0)^T+(0,1)^T=(1,1)^T\\notin U\\cup W$，加法不封闭。并集一般不是子空间；**和** $U+W$ 才是子空间。',
        },
        {
          id: 'u4-l4-q4',
          type: 'choice',
          prompt: '设 $A=\\begin{pmatrix}1&2&1\\\\1&2&2\\end{pmatrix}$，$\\mathbf{b}=\\begin{pmatrix}3\\\\4\\end{pmatrix}$。方程 $A\\mathbf{x}=\\mathbf{b}$ 的通解结构是？',
          options: [
            '唯一解',
            '特解 $\\mathbf{x}_p=(2,0,1)^T$ 加上 $\\operatorname{Null}(A)$ 的任意向量（无穷多解）',
            '无解',
            '特解 $\\mathbf{x}_p=(1,1,1)^T$ 加上 $\\operatorname{Null}(A)$ 的任意向量',
          ],
          answer: 1,
          explain: '$A$ 是 $2\\times 3$ 矩阵，行化简后 $\\operatorname{rank}(A)=2$，$\\operatorname{nullity}(A)=1$，有自由变量。特解（令 $x_2=0$）为 $(2,0,1)^T$，零空间基向量为 $(-2,1,0)^T$。通解：$(2,0,1)^T+t(-2,1,0)^T$，$t\\in\\mathbb{R}$。',
        },
        {
          id: 'u4-l4-q5',
          type: 'input',
          prompt: '矩阵 $A=\\begin{pmatrix}1&2&3\\\\2&4&6\\\\1&1&2\\end{pmatrix}$ 的 $\\operatorname{rank}(A)=2$。则 $\\operatorname{Null}(A)$ 的维数是多少？（填数字）',
          accept: ['1'],
          placeholder: '1',
          explain: '由秩-零化度定理（$n=3$ 列）：$\\operatorname{nullity}(A)=3-2=1$。零空间是 $\\mathbb{R}^3$ 中的一条过原点的直线（一维子空间）。',
        },
        {
          id: 'u4-l4-q6',
          type: 'choice',
          prompt: '$A\\mathbf{x}=\\mathbf{b}$ 对**所有** $\\mathbf{b}\\in\\mathbb{R}^m$ 都有解，当且仅当？',
          options: [
            '$\\operatorname{Null}(A)=\\{\\mathbf{0}\\}$（零空间只含零向量）',
            '$\\operatorname{Col}(A)=\\mathbb{R}^m$（列空间等于整个 $\\mathbb{R}^m$）',
            '$\\operatorname{rank}(A)=n$（列满秩）',
            '$A$ 是方阵',
          ],
          answer: 1,
          explain: '$A\\mathbf{x}=\\mathbf{b}$ 有解 $\\Leftrightarrow$ $\\mathbf{b}\\in\\operatorname{Col}(A)$。对所有 $\\mathbf{b}$ 都有解 $\\Leftrightarrow$ $\\operatorname{Col}(A)=\\mathbb{R}^m$ $\\Leftrightarrow$ $\\operatorname{rank}(A)=m$（行满秩）。注意是行满秩，不是列满秩。',
        },
        {
          id: 'u4-l4-q7',
          type: 'judge',
          prompt: '若 $A\\mathbf{x}=\\mathbf{b}$ 有解，则其解集（通解集合）是 $\\mathbb{R}^n$ 的一个子空间。',
          answer: false,
          explain: '通解集 = 特解 + 零空间，即把零空间平移到特解处。若 $\\mathbf{b}\\neq\\mathbf{0}$，特解不过原点，通解集不含零向量，**不是子空间**。只有当 $\\mathbf{b}=\\mathbf{0}$ 时，通解集等于零空间，才是子空间。',
        },
        {
          id: 'u4-l4-q8',
          type: 'match',
          prompt: '将各集合与"是否为 $\\mathbb{R}^3$ 的子空间"配对：',
          left: [
            '$W_1=\\{(x,y,z)^T\\mid x=y=z\\}$',
            '$W_2=\\{(x,y,z)^T\\mid x+y+z=1\\}$',
            '$W_3=\\{(x,y,z)^T\\mid x\\geq 0\\}$',
            '$W_4=\\operatorname{span}\\{(1,0,0)^T,(0,1,0)^T\\}$',
          ],
          right: ['是子空间（过原点的直线）', '不是（不含零向量）', '不是（$c=-1$ 时数乘不封闭）', '是子空间（$xy$ 平面）'],
          explain: '$W_1$：过原点的对角线，线性组合封闭，是子空间；$W_2$：不含零向量否定；$W_3$：取 $c=-1$ 数乘逃出，否定；$W_4$：张成集是子空间（$xy$ 平面）。',
        },
        {
          id: 'u4-l4-q9',
          type: 'choice',
          prompt: 'LoRA 微调大模型时，将权重更新量参数化为 $\\Delta W=BA$（$B\\in\\mathbb{R}^{512\\times 4}$，$A\\in\\mathbb{R}^{4\\times 512}$）。$\\Delta W$ 的列空间维数最大是多少？',
          options: ['$512$', '$4$', '$256$', '$1$'],
          answer: 1,
          explain: '$\\operatorname{rank}(\\Delta W)=\\operatorname{rank}(BA)\\leq\\min(\\operatorname{rank}(B),\\operatorname{rank}(A))\\leq 4$（因 $B$ 最多 4 列，$A$ 最多 4 行）。列空间维数最大为 $4$——LoRA 把权重更新限制在 4 维子空间中，大幅压缩参数。',
        },
      ],
    },
  ],
}
