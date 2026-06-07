export const UNIT = {
  id: 'u2',
  title: '线性方程组',
  color: '#1cb0f6',
  icon: '🔢',
  blurb:
    '从 $A\\mathbf{x}=\\mathbf{b}$ 的几何直觉，到高斯消元的系统算法，再到逆矩阵的理论与工程——七关带你全面掌握线性方程组的建模、求解与结构分析。',
  lessons: [
    // ─────────────────────────────────────────
    // 关 1  几何直觉与解的三种情形
    // ─────────────────────────────────────────
    {
      id: 'u2-l1',
      title: '几何直觉与解的三种情形',
      subtitle: '方程组·增广矩阵·列视角·三种解',
      intro: [
        {
          title: '为什么要学线性方程组？',
          body:
            '现实中大量问题——电路分析、图像处理、机器学习——都可归结为"用若干个线性约束求若干个未知量"。\n\n' +
            '当未知量超过 3 个，手工逐一代入已经不可行。矩阵语言让我们用一行 $A\\mathbf{x}=\\mathbf{b}$ 描述整个方程组，并用系统化算法求解。\n\n' +
            '**线性方程**：每个未知量都以一次幂出现，没有 $x^2$、$xy$、$\\sin x$ 等非线性项。',
          tip: '非线性方程（如 $x^2+y=1$）不在本章讨论范围，线性方程组专指一次方程构成的组合。',
        },
        {
          title: '系数矩阵与增广矩阵',
          body:
            '对方程组\n\n$\\begin{cases}x_1+2x_2=5\\\\2x_1-x_2=0\\\\x_1+x_2=3\\end{cases}$\n\n' +
            '把系数按位置排列得**系数矩阵** $A$；在右侧拼上常数列得**增广矩阵** $[A\\mid\\mathbf{b}]$：',
          formula:
            '[A\\mid\\mathbf{b}]=\\left(\\begin{array}{cc|c}1&2&5\\\\2&-1&0\\\\1&1&3\\end{array}\\right)',
          tip: '竖线右侧是常数列 $\\mathbf{b}$，大小为 $m\\times(n+1)$。增广矩阵把一切信息都装进了一张表。',
        },
        {
          title: '矩阵方程与列视角',
          body:
            '把未知量排成列向量 $\\mathbf{x}=(x_1,\\dots,x_n)^T$，整个方程组等价于\n\n' +
            '$A\\mathbf{x}=\\mathbf{b}$\n\n' +
            '**列视角**：记 $A$ 的第 $j$ 列为 $\\mathbf{a}_j$，则\n\n' +
            '$x_1\\mathbf{a}_1+x_2\\mathbf{a}_2+\\cdots+x_n\\mathbf{a}_n=\\mathbf{b}$\n\n' +
            '求解方程组等价于：**能否把 $\\mathbf{b}$ 写成 $A$ 各列的线性组合？**',
          tip: '列视角把"求解方程组"与"列空间"直接联系起来，是后续章节的理论基础。',
        },
        {
          title: '解只有三种情形——反直觉事实',
          body:
            '线性方程组 $A\\mathbf{x}=\\mathbf{b}$ 的解集只有三种可能，**不可能恰好有两个解**。\n\n' +
            '两个方程解两个未知数，直觉上应有唯一解——但实际上三种情形都可能发生：\n\n' +
            '对 $\\begin{cases}x+y=3\\\\2x+2y=6\\end{cases}$（A）、$\\begin{cases}x+y=3\\\\2x+2y=7\\end{cases}$（B）、$\\begin{cases}x+y=3\\\\x-y=1\\end{cases}$（C）：\n\n' +
            '(A) $R_2-2R_1$ 得 $\\left(\\begin{array}{cc|c}1&1&3\\\\0&0&0\\end{array}\\right)$ → **无穷多解**；\n\n' +
            '(B) 同操作得 $\\left(\\begin{array}{cc|c}1&1&3\\\\0&0&1\\end{array}\\right)$，出现 $0=1$ → **无解**；\n\n' +
            '(C) $R_2-R_1$ 得两个主元 → **唯一解** $x=2,y=1$。',
          tip: '(A)(B) 左边系数完全相同，只因常数项不同就从无穷多解变成无解——必须看增广矩阵的秩，不能只看系数矩阵 $A$。',
        },
        {
          title: '为什么不能有恰好两个解？',
          body:
            '若 $\\mathbf{x}_1,\\mathbf{x}_2$ 都是 $A\\mathbf{x}=\\mathbf{b}$ 的解，则\n\n' +
            '$A(\\mathbf{x}_1-\\mathbf{x}_2)=\\mathbf{0}$\n\n' +
            '即 $\\mathbf{x}_1-\\mathbf{x}_2$ 在核（零空间）$\\ker(A)$ 中。\n\n' +
            '- 若 $\\ker(A)=\\{\\mathbf{0}\\}$：$\\mathbf{x}_1=\\mathbf{x}_2$，解唯一。\n\n' +
            '- 若 $\\ker(A)$ 非零：它是向量空间，含无穷多方向，解也无穷多。\n\n' +
            '两者之间没有中间状态——永远不会有"恰好两个解"。',
          formula:
            '\\begin{cases}\\text{无解} & \\text{rank}(A)<\\text{rank}([A\\mid\\mathbf{b}])\\\\\\text{唯一解} & \\text{rank}(A)=\\text{rank}([A\\mid\\mathbf{b}])=n\\\\\\text{无穷多解} & \\text{rank}(A)=\\text{rank}([A\\mid\\mathbf{b}])<n\\end{cases}',
        },
        {
          title: '几何解释',
          body:
            '以二元方程组为例，每个方程对应平面上的一条直线：\n\n' +
            '- **唯一解**：两直线相交于一点\n\n' +
            '- **无解**：两直线平行（不相交）\n\n' +
            '- **无穷多解**：两直线重合\n\n' +
            '三元方程组（每个方程对应空间中的一个平面）同理：三个平面可能交于一点（唯一解）、没有公共点（无解）、交于一条线乃至一个平面（无穷多解）。\n\n' +
            '有解的充要条件：$\\text{rank}(A)=\\text{rank}([A\\mid\\mathbf{b}])$，即增加常数列不增加新约束。',
          tip: '几何视角帮助建立直觉，但高维情形（$n>3$）已无法可视化，必须回到代数工具——秩的比较。',
        },
      ],
      questions: [
        {
          id: 'u2-l1-q1',
          type: 'choice',
          prompt:
            '方程组 $\\begin{cases}x+y=3\\\\2x+2y=7\\end{cases}$ 的解的情况是？',
          options: ['唯一解', '无穷多解', '无解', '两个解'],
          answer: 2,
          explain:
            '$R_2-2R_1$ 得 $[0\\;0\\mid 1]$，出现 $0=1$ 矛盾行，无解。',
        },
        {
          id: 'u2-l1-q2',
          type: 'judge',
          prompt:
            '一个线性方程组可以恰好有两个解。',
          answer: false,
          explain:
            '若有两个解 $\\mathbf{x}_1,\\mathbf{x}_2$，则 $\\mathbf{x}_1-\\mathbf{x}_2\\in\\ker(A)$，核非零则解无穷多；核为零则解唯一。两者之间无中间状态。',
        },
        {
          id: 'u2-l1-q3',
          type: 'choice',
          prompt:
            '方程组 $A\\mathbf{x}=\\mathbf{b}$ 有解的充要条件是？',
          options: [
            '$\\text{rank}(A)=n$',
            '$\\text{rank}(A)=\\text{rank}([A\\mid\\mathbf{b}])$',
            '$\\text{rank}(A)<\\text{rank}([A\\mid\\mathbf{b}])$',
            '$\\det(A)\\ne 0$',
          ],
          answer: 1,
          explain:
            '有解充要条件：增广矩阵与系数矩阵的秩相等，即增加常数列不引入新矛盾。',
        },
        {
          id: 'u2-l1-q4',
          type: 'match',
          prompt: '将下列条件与解的情形对应。',
          left: [
            '$\\text{rank}(A)<\\text{rank}([A\\mid\\mathbf{b}])$',
            '$\\text{rank}(A)=\\text{rank}([A\\mid\\mathbf{b}])=n$',
            '$\\text{rank}(A)=\\text{rank}([A\\mid\\mathbf{b}])<n$',
          ],
          right: ['唯一解', '无穷多解', '无解'],
          answer: [2, 0, 1],
          explain:
            '无解对应矛盾行（秩增大）；唯一解对应秩等于未知量数；无穷多解对应秩小于未知量数。',
        },
        {
          id: 'u2-l1-q5',
          type: 'choice',
          prompt:
            '增广矩阵 $\\left(\\begin{array}{cc|c}1&2&3\\\\0&0&5\\end{array}\\right)$ 对应的方程组：',
          options: ['有唯一解', '有无穷多解', '无解', '需要更多信息才能判断'],
          answer: 2,
          explain:
            '第二行 $[0\\;0\\mid 5]$ 对应 $0=5$，是矛盾行，方程组无解。',
        },
        {
          id: 'u2-l1-q6',
          type: 'input',
          prompt:
            '方程组 $\\begin{cases}x_1+2x_2=4\\\\2x_1+4x_2=8\\end{cases}$ 的解集是一条直线，其中自由变量有多少个？',
          accept: ['1'],
          placeholder: '输入整数',
          explain:
            '$R_2-2R_1$ 后第二行全零，只有 1 个独立方程、2 个未知量，自由变量 $2-1=1$ 个。',
        },
      ],
    },

    // ─────────────────────────────────────────
    // 关 2  秩、零空间与解结构
    // ─────────────────────────────────────────
    {
      id: 'u2-l2',
      title: '秩、零空间与解结构',
      subtitle: '零空间·秩-零化度·通解 = 特解 + 齐次通解',
      intro: [
        {
          title: '齐次方程组与零空间',
          body:
            '当 $\\mathbf{b}=\\mathbf{0}$ 时，$A\\mathbf{x}=\\mathbf{0}$ 称为**齐次线性方程组**，它永远有零解 $\\mathbf{x}=\\mathbf{0}$（平凡解）。\n\n' +
            '$A\\mathbf{x}=\\mathbf{0}$ 的所有解构成 $A$ 的**零空间**（核）$\\ker(A)$。\n\n' +
            '**为什么是向量空间？** 若 $A\\mathbf{u}=\\mathbf{0}$ 且 $A\\mathbf{v}=\\mathbf{0}$，则对任意 $c,d$：\n\n' +
            '$A(c\\mathbf{u}+d\\mathbf{v})=cA\\mathbf{u}+dA\\mathbf{v}=\\mathbf{0}$\n\n' +
            '线性组合封闭，故是向量空间。',
          tip: '对比非齐次方程组：$A\\mathbf{x}=\\mathbf{b}$（$\\mathbf{b}\\ne\\mathbf{0}$）的解集不含 $\\mathbf{0}$，不是向量空间。',
        },
        {
          title: '非平凡解存在的条件',
          body:
            '齐次方程组 $A\\mathbf{x}=\\mathbf{0}$ 必有零解，但非平凡解（$\\mathbf{x}\\ne\\mathbf{0}$）不一定存在。\n\n' +
            '**非平凡解存在 $\\Leftrightarrow$ $\\text{rank}(A)<n$（有自由变量）**\n\n' +
            '直觉：若变量多于独立约束，系统"过于自由"，在满足所有约束的同时还有额外自由度，必然有非零解。\n\n' +
            '特别地，若 $m<n$（方程数 $<$ 未知量数），则 $\\text{rank}(A)\\le m<n$，齐次方程组必有非平凡解。',
          formula: '\\ker(A)\\ne\\{\\mathbf{0}\\}\\Leftrightarrow\\text{rank}(A)<n',
        },
        {
          title: '秩-零化度定理',
          body:
            '设 $A$ 是 $m\\times n$ 矩阵，定义**零化度**（nullity）为 $\\ker(A)$ 的维数，则：\n\n' +
            '$\\text{rank}(A)+\\text{nullity}(A)=n$\n\n' +
            '即主元数 + 自由变量数 = 未知量总数。\n\n' +
            '**例**：$A$ 是 $3\\times 5$ 矩阵，$\\text{rank}(A)=2$，则 $\\dim(\\ker(A))=5-2=3$，零空间是 $\\mathbb{R}^5$ 中的三维子空间。',
          formula: '\\dim(\\ker(A))=n-\\text{rank}(A)',
          tip: '记住：主元列对应基本变量（有约束），非主元列对应自由变量（自由取值）。',
        },
        {
          title: '非齐次通解 = 特解 + 齐次通解',
          body:
            '设 $\\mathbf{x}_p$ 是 $A\\mathbf{x}=\\mathbf{b}$ 的一个特解（$A\\mathbf{x}_p=\\mathbf{b}$），则完整解为：\n\n' +
            '$\\mathbf{x}=\\mathbf{x}_p+\\mathbf{x}_h$，$A\\mathbf{x}_h=\\mathbf{0}$\n\n' +
            '**两个方向的验证**：\n\n' +
            '- 每个 $\\mathbf{x}_p+\\mathbf{x}_h$ 都是解：$A(\\mathbf{x}_p+\\mathbf{x}_h)=\\mathbf{b}+\\mathbf{0}=\\mathbf{b}$。\n\n' +
            '- 每个解都形如此：若 $\\mathbf{x}$ 是解，令 $\\mathbf{x}_h=\\mathbf{x}-\\mathbf{x}_p$，则 $A\\mathbf{x}_h=\\mathbf{0}$。',
          tip: '几何上：齐次解集是过原点的子空间，非齐次解集是它平移了 $\\mathbf{x}_p$ 的仿射子空间（不过原点）。',
        },
        {
          title: '求通解的标准流程',
          body:
            '以方程组 $\\begin{cases}x_1+2x_2-x_3=1\\\\2x_1+4x_2-2x_3=2\\\\x_1+2x_2+x_3=3\\end{cases}$ 为例：\n\n' +
            '增广矩阵行变换得 $\\left(\\begin{array}{ccc|c}1&2&-1&1\\\\0&0&2&2\\\\0&0&0&0\\end{array}\\right)$\n\n' +
            '主元列：1（$x_1$）和 3（$x_3$）；$x_2$ 是自由变量。$\\text{rank}=2<3=n$ → 无穷多解。\n\n' +
            '回代：$x_3=1$；$x_1=1+x_3-2x_2=2-2x_2$。令 $x_2=t$：\n\n' +
            '$\\mathbf{x}=\\underbrace{\\begin{pmatrix}2\\\\0\\\\1\\end{pmatrix}}_{\\text{特解}}+t\\underbrace{\\begin{pmatrix}-2\\\\1\\\\0\\end{pmatrix}}_{\\text{齐次通解}}$',
          tip: '求特解：令自由变量 $=0$ 回代；求齐次基础解向量：令该自由变量 $=1$、其余自由变量 $=0$ 后回代。',
        },
        {
          title: 'AI 关联：过参数化与欠定系统',
          body:
            '神经网络中某层 $W\\in\\mathbb{R}^{m\\times n}$，给定输出 $\\mathbf{y}=W\\mathbf{x}$，逆问题是 $W\\mathbf{x}=\\mathbf{y}$。\n\n' +
            '当 $m<n$（参数多于样本，**过参数化**）：方程组欠定，有无穷多解，梯度下降隐式选择极小范数解——这正是大模型的常态。\n\n' +
            '**代码体现**：\n\n' +
            '`rank_A = torch.linalg.matrix_rank(A)`\n\n' +
            '`n_free = A.shape[1] - rank_A  # 自由变量个数`\n\n' +
            '零空间的维数 $= n-\\text{rank}(A)$，可通过 SVD 的零奇异值对应的右奇异向量求出。',
          tip: '理解零空间和欠定系统，是理解正则化（L1/L2）和最小范数解的理论基础。',
        },
        {
          title: '易错点总结',
          body:
            '**易错 1**：把 $m<n$（方程数少于未知量数）误解为"一定有无穷多解"——实际上可能无解（出现矛盾行）。\n\n' +
            '**易错 2**：非齐次通解只写特解，忘了叠加齐次通解。\n\n' +
            '**易错 3**：秩只看系数矩阵，忘了对比 $\\text{rank}([A\\mid\\mathbf{b}])$。\n\n' +
            '**条件反射**：\n\n' +
            '- 看到"齐次方程组" → 必有零解；非平凡解要求 $\\text{rank}(A)<n$\n\n' +
            '- 看到"零空间" → 维数 $=n-\\text{rank}(A)$，是向量空间\n\n' +
            '- 看到"非齐次通解" → 特解 + 齐次通解，缺一不可',
        },
      ],
      questions: [
        {
          id: 'u2-l2-q1',
          type: 'judge',
          prompt:
            '齐次线性方程组 $A\\mathbf{x}=\\mathbf{0}$ 一定有解。',
          answer: true,
          explain:
            '$\\mathbf{x}=\\mathbf{0}$（零向量）总满足 $A\\mathbf{0}=\\mathbf{0}$，是平凡解。',
        },
        {
          id: 'u2-l2-q2',
          type: 'input',
          prompt:
            '若 $A$ 是 $3\\times 5$ 矩阵且 $\\text{rank}(A)=3$，则 $\\ker(A)$ 的维数是多少？',
          accept: ['2'],
          placeholder: '输入整数',
          explain:
            '秩-零化度定理：$\\dim(\\ker(A))=n-\\text{rank}(A)=5-3=2$。',
        },
        {
          id: 'u2-l2-q3',
          type: 'choice',
          prompt:
            '设 $A$ 是 $m\\times n$ 矩阵，$\\mathbf{b}\\ne\\mathbf{0}$，非齐次方程组 $A\\mathbf{x}=\\mathbf{b}$ 的解集是：',
          options: [
            '向量空间',
            '特解加零空间构成的仿射子空间',
            '只含一个元素的集合',
            '空集',
          ],
          answer: 1,
          explain:
            '非齐次解集 $=\\mathbf{x}_p+\\ker(A)$，是零空间平移特解后的仿射子空间，不过原点，故不是向量空间。',
        },
        {
          id: 'u2-l2-q4',
          type: 'choice',
          prompt:
            '对 $n$ 元线性方程组 $A\\mathbf{x}=\\mathbf{b}$，下列哪个条件保证无穷多解？',
          options: [
            '$\\text{rank}(A)=n$',
            '$m<n$（方程数少于未知量数）',
            '$\\text{rank}(A)=\\text{rank}([A\\mid\\mathbf{b}])<n$',
            '$\\text{rank}(A)<\\text{rank}([A\\mid\\mathbf{b}])$',
          ],
          answer: 2,
          explain:
            '无穷多解：有解（两个秩相等）且自由变量存在（$\\text{rank}<n$）。$m<n$ 本身不保证有解。',
        },
        {
          id: 'u2-l2-q5',
          type: 'input',
          prompt:
            '方程组消元后得到 REF，主元数 $r=2$，未知量数 $n=5$，则自由变量有多少个？',
          accept: ['3'],
          placeholder: '输入整数',
          explain:
            '自由变量数 $=n-r=5-2=3$，对应秩-零化度定理中 $\\dim(\\ker(A))=3$。',
        },
        {
          id: 'u2-l2-q6',
          type: 'judge',
          prompt:
            '若 $\\mathbf{x}_1$ 和 $\\mathbf{x}_2$ 都是 $A\\mathbf{x}=\\mathbf{b}$ 的解，则 $\\mathbf{x}_1+\\mathbf{x}_2$ 也是解。',
          answer: false,
          explain:
            '$A(\\mathbf{x}_1+\\mathbf{x}_2)=\\mathbf{b}+\\mathbf{b}=2\\mathbf{b}\\ne\\mathbf{b}$（$\\mathbf{b}\\ne\\mathbf{0}$），故不是解。正确结论是 $\\mathbf{x}_1-\\mathbf{x}_2\\in\\ker(A)$。',
        },
        {
          id: 'u2-l2-q7',
          type: 'match',
          prompt: '将下列描述与正确的解结构对应。',
          left: [
            '$A\\mathbf{x}=\\mathbf{0}$ 的解集',
            '$A\\mathbf{x}=\\mathbf{b}$（有解）的解集',
            '$A$ 满秩（$\\text{rank}=n$）时 $A\\mathbf{x}=\\mathbf{b}$ 的解集',
          ],
          right: ['单个特解（唯一解）', '零空间（向量空间）', '特解 + 零空间（仿射子空间）'],
          answer: [1, 2, 0],
          explain:
            '齐次解集是向量空间（零空间）；非齐次有解时是仿射子空间；满秩时零空间为零，仿射子空间退化为单点。',
        },
      ],
    },

    // ─────────────────────────────────────────
    // 关 3  消元步骤与初等行变换
    // ─────────────────────────────────────────
    {
      id: 'u2-l3',
      title: '消元步骤与初等行变换',
      subtitle: '三种行变换·初等矩阵·REF·高斯消元完整演示',
      intro: [
        {
          title: '三种初等行变换',
          body:
            '对增广矩阵施行以下操作，**不改变方程组的解集**（每种操作均可逆）：\n\n' +
            '1. **交换两行**：$R_i\\leftrightarrow R_j$\n\n' +
            '2. **数乘非零倍**：$R_i\\leftarrow cR_i$（$c\\ne 0$）\n\n' +
            '3. **倍加**：$R_j\\leftarrow R_j+cR_i$\n\n' +
            '两个矩阵若可经初等行变换相互转化，称为**行等价**，它们对应的方程组有相同解集。',
          tip: '三种变换都有对应的逆操作：交换↔再次交换；$cR_i$↔$\\frac{1}{c}R_i$；倍加↔倍加的逆。',
        },
        {
          title: '初等矩阵',
          body:
            '每种初等行变换对应一个**初等矩阵** $E$（对单位矩阵施行同样操作得到），且 $E$ 可逆。\n\n' +
            '对矩阵 $A$ 施行某初等行变换，等价于 $E\\cdot A$。\n\n' +
            '**例**：$3\\times 3$ 矩阵的 $R_3\\leftarrow R_3-2R_1$ 对应：',
          formula:
            'E=\\begin{pmatrix}1&0&0\\\\0&1&0\\\\-2&0&1\\end{pmatrix}',
          tip: '初等矩阵的逆仍是初等矩阵，对应逆变换（倍加的逆：将 $-c$ 替换为 $+c$）。',
        },
        {
          title: '行阶梯形（REF）的定义',
          body:
            '矩阵 $A$ 是**行阶梯形**（REF），若：\n\n' +
            '1. 所有全零行（若有）在矩阵底部。\n\n' +
            '2. 每个非零行的**主元**（该行最左非零元素）严格位于上一行主元的右侧。\n\n' +
            '例：$\\begin{pmatrix}2&1&-1&8\\\\0&3&2&5\\\\0&0&-1&3\\end{pmatrix}$ 是 REF，三个主元分别在第 1、2、3 列。\n\n' +
            '**主元数 = 秩 $r$；非主元列对应自由变量，个数 $= n-r$。**',
          formula: 'r=\\text{rank}(A)=\\text{主元个数}',
        },
        {
          title: '高斯消元完整演示（3×3 唯一解）',
          body:
            '求解 $\\begin{cases}x_1+2x_2+x_3=4\\\\2x_1+5x_2+2x_3=9\\\\x_1+3x_2+2x_3=6\\end{cases}$\n\n' +
            '**初始**：$\\left(\\begin{array}{ccc|c}1&2&1&4\\\\2&5&2&9\\\\1&3&2&6\\end{array}\\right)$\n\n' +
            '**$R_2-2R_1$，$R_3-R_1$**：$\\left(\\begin{array}{ccc|c}1&2&1&4\\\\0&1&0&1\\\\0&1&1&2\\end{array}\\right)$\n\n' +
            '**$R_3-R_2$**：$\\left(\\begin{array}{ccc|c}1&2&1&4\\\\0&1&0&1\\\\0&0&1&1\\end{array}\\right)$\n\n' +
            '三个主元，无矛盾行 → **唯一解**。回代：$x_3=1$，$x_2=1$，$x_1=4-2-1=1$。',
          tip: '每完成一列消元后，检查该列主元下方是否全为零——这是判断消元正确性的最快方法。',
        },
        {
          title: '高斯消元演示（含矛盾行：无解）',
          body:
            '求解 $\\begin{cases}2x_1+x_2-x_3=8\\\\-3x_1-x_2+2x_3=-11\\\\-2x_1+x_2+2x_3=-3\\end{cases}$\n\n' +
            '对增广矩阵消元（$R_2+\\frac{3}{2}R_1$，$R_3+R_1$，再 $R_3-4R_2$）：\n\n' +
            '$\\left(\\begin{array}{ccc|c}2&1&-1&8\\\\0&\\frac{1}{2}&\\frac{1}{2}&1\\\\0&0&-1&1\\end{array}\\right)$\n\n' +
            '三个主元均存在，无矛盾行 → **唯一解**：回代得 $x_3=-1,x_2=3,x_1=2$。\n\n' +
            '若某步出现 $[0\\;0\\;0\\mid c]$（$c\\ne 0$）即矛盾行 → **立刻停止，无解**。',
          tip: '区分全零行 $[0\\cdots 0\\mid 0]$（无矛盾）和矛盾行 $[0\\cdots 0\\mid c]$（$c\\ne 0$，无解）。这是消元中最重要的判断节点。',
        },
        {
          title: '高斯消元标准 4 步',
          body:
            '**步骤 1**：写增广矩阵 $[A\\mid\\mathbf{b}]$\n\n' +
            '**步骤 2**：前向消元，化为 REF\n\n' +
            '**步骤 3**：判断解的类型：\n\n' +
            '- 出现矛盾行 $[0\\cdots 0\\mid c]$（$c\\ne 0$）→ 无解，停止\n\n' +
            '- 主元数 $r=n$ → 唯一解\n\n' +
            '- 主元数 $r<n$ → 无穷多解，有 $n-r$ 个自由变量\n\n' +
            '**步骤 4**：回代（或继续化 RREF）读出解',
          tip: '永远先做步骤 3 的判断——看到矛盾行就停手，不要继续消元浪费时间。',
        },
      ],
      questions: [
        {
          id: 'u2-l3-q1',
          type: 'choice',
          prompt:
            '对矩阵 $A=\\begin{pmatrix}1&2&3\\\\2&5&4\\\\0&1&2\\end{pmatrix}$ 施行 $R_2\\leftarrow R_2-2R_1$，第 2 行变为？',
          options: [
            '$\\begin{pmatrix}0&1&-2\\end{pmatrix}$',
            '$\\begin{pmatrix}0&1&2\\end{pmatrix}$',
            '$\\begin{pmatrix}4&9&11\\end{pmatrix}$',
            '$\\begin{pmatrix}0&9&-2\\end{pmatrix}$',
          ],
          answer: 0,
          explain:
            '$(2-2\\times 1,5-2\\times 2,4-2\\times 3)=(0,1,-2)$，选项 A。',
        },
        {
          id: 'u2-l3-q2',
          type: 'judge',
          prompt:
            '初等行变换不改变矩阵的解集，因此任意两个行等价的矩阵对应方程组的解相同。',
          answer: true,
          explain:
            '每种初等行变换均可逆，因此行等价的矩阵对应完全相同的解集——这是消元法的理论基础。',
        },
        {
          id: 'u2-l3-q3',
          type: 'choice',
          prompt:
            '高斯消元后 REF 含矛盾行 $[0\\;0\\;0\\mid 3]$，方程组：',
          options: ['唯一解', '无穷多解', '无解', '需要看主元数'],
          answer: 2,
          explain:
            '矛盾行 $0=3$ 意味着方程组不相容，无论主元数多少都无解。',
        },
        {
          id: 'u2-l3-q4',
          type: 'input',
          prompt:
            '对 4 元方程组，高斯消元后 REF 中主元数 $r=3$，有几个自由变量？',
          accept: ['1'],
          placeholder: '输入整数',
          explain:
            '自由变量数 $=n-r=4-3=1$。',
        },
        {
          id: 'u2-l3-q5',
          type: 'match',
          prompt: '将消元操作与名称对应。',
          left: [
            '$R_i\\leftrightarrow R_j$',
            '$R_i\\leftarrow cR_i$',
            '$R_j\\leftarrow R_j+cR_i$',
          ],
          right: ['倍加', '数乘', '交换'],
          answer: [2, 1, 0],
          explain:
            '交换行：$R_i\\leftrightarrow R_j$；数乘：$cR_i$；倍加：$R_j+cR_i$。',
        },
        {
          id: 'u2-l3-q6',
          type: 'input',
          prompt:
            '求解 $\\begin{cases}2x_1+x_2-x_3=8\\\\-3x_1-x_2+2x_3=-11\\\\-2x_1+x_2+2x_3=-3\\end{cases}$，$x_1=$？',
          accept: ['2'],
          placeholder: '输入数字',
          explain:
            '高斯消元后回代得 $x_3=-1$，$x_2=3$，$x_1=2$。',
        },
      ],
    },

    // ─────────────────────────────────────────
    // 关 4  RREF 与自由变量参数化
    // ─────────────────────────────────────────
    {
      id: 'u2-l4',
      title: 'RREF 与自由变量参数化',
      subtitle: 'RREF·Gauss-Jordan·无穷多解·LU 分解',
      intro: [
        {
          title: '简化行阶梯形（RREF）的定义',
          body:
            'RREF 是 REF 的进一步简化，额外要求：\n\n' +
            '1. 每个主元为 **1**（首一）。\n\n' +
            '2. 每个主元所在列的其他元素**全为 0**（向上也消干净）。\n\n' +
            '**唯一性定理**：任何矩阵的 RREF 唯一，不随消元顺序改变。\n\n' +
            '例：$\\left(\\begin{array}{ccc|c}1&0&0&2\\\\0&1&0&3\\\\0&0&1&-1\\end{array}\\right)$ 是 RREF，直接读出 $x_1=2,x_2=3,x_3=-1$。',
          tip: 'RREF 的唯一性是因为解集唯一确定，不依赖于消元顺序。',
        },
        {
          title: 'Gauss-Jordan 消元法',
          body:
            '在高斯消元（前向消元→REF）的基础上，继续**向上消元**+**主元归一**，得到 RREF——这就是 **Gauss-Jordan 消元法**。\n\n' +
            '**步骤**：\n\n' +
            '1. 前向消元，化为 REF\n\n' +
            '2. 从最后一个主元开始向上消，使主元列其余元素全为 0\n\n' +
            '3. 各主元行除以主元值，使主元变为 1\n\n' +
            '4. 直接从右端列读出解，无需手工回代',
          tip: '高斯消元 + 回代 $\\approx$ Gauss-Jordan 直接读解，后者在人工计算中步骤更多，但逻辑更清晰。',
        },
        {
          title: '无穷多解：自由变量参数化',
          body:
            '当 REF 中主元数 $r<n$ 时，有 $n-r$ 个自由变量，需参数化写出通解。\n\n' +
            '**例**：消元后得 REF\n\n' +
            '$\\left(\\begin{array}{ccc|c}1&2&-1&1\\\\0&0&2&2\\\\0&0&0&0\\end{array}\\right)$\n\n' +
            '主元列：1（$x_1$）和 3（$x_3$）；$x_2$ 是自由变量，令 $x_2=t$。\n\n' +
            '由第 2 行：$x_3=1$；代入第 1 行：$x_1=1+x_3-2x_2=2-2t$。\n\n' +
            '$\\mathbf{x}=\\begin{pmatrix}2\\\\0\\\\1\\end{pmatrix}+t\\begin{pmatrix}-2\\\\1\\\\0\\end{pmatrix},\\quad t\\in\\mathbb{R}$',
          tip: '基础解向量的求法：令该自由变量 $=1$、其余自由变量 $=0$ 后回代，得一个基础解向量。',
        },
        {
          title: '多个自由变量的参数化',
          body:
            '设消元后 REF 为 $\\left(\\begin{array}{cccc|c}1&2&0&1&3\\\\0&0&1&-1&2\\\\0&0&0&0&0\\end{array}\\right)$\n\n' +
            '主元列：1（$x_1$）和 3（$x_3$）；自由变量：$x_2=s$，$x_4=t$。\n\n' +
            '由第 2 行：$x_3=2+t$；代入第 1 行：$x_1=3-2s-t$。\n\n' +
            '$\\mathbf{x}=\\begin{pmatrix}3\\\\0\\\\2\\\\0\\end{pmatrix}+s\\begin{pmatrix}-2\\\\1\\\\0\\\\0\\end{pmatrix}+t\\begin{pmatrix}-1\\\\0\\\\1\\\\1\\end{pmatrix}$\n\n' +
            '有几个自由变量，通解就有几个基础解向量。',
          reveal: {
            q: '上例中，自由变量有几个？$\\ker(A)$ 的维数是多少？',
            a: '自由变量 2 个（$x_2,x_4$）；$\\dim(\\ker(A))=2$。',
          },
        },
        {
          title: 'LU 分解：分解一次，多次求解',
          body:
            '高斯消元记录每步消元乘数 $m_{ij}=a_{ij}/a_{ii}$，可把矩阵 $A$ 分解为\n\n' +
            '$A=LU$\n\n' +
            '其中 $L$ 是单位下三角矩阵（对角线为 1，乘数填入下方），$U$ 是行阶梯形（上三角）。\n\n' +
            '**优势**：对相同 $A$、不同 $\\mathbf{b}$ 多次求解时，只需分解一次（$O(n^3)$），每次求解两步：\n\n' +
            '- 前代 $L\\mathbf{y}=\\mathbf{b}$（$O(n^2)$）\n\n' +
            '- 回代 $U\\mathbf{x}=\\mathbf{y}$（$O(n^2)$）',
          formula: 'A=LU,\\quad L=\\begin{pmatrix}1&0&0\\\\m_{21}&1&0\\\\m_{31}&m_{32}&1\\end{pmatrix},\\quad U\\text{ 为上三角}',
          tip: '实际工程（numpy/torch 底层）使用带部分选主元的 $PA=LU$，保证数值稳定性。',
        },
        {
          title: '易错点总结',
          body:
            '**易错 1**：把全零行 $[0\\cdots 0\\mid 0]$ 与矛盾行 $[0\\cdots 0\\mid c]$（$c\\ne 0$）混淆——前者无矛盾，后者无解。\n\n' +
            '**易错 2**：LU 中 $L$ 的乘数填写正负号搞反——消元 $R_j\\leftarrow R_j-m_{ij}R_i$，$L_{ji}=m_{ij}$（直接放原值）。\n\n' +
            '**易错 3**：行交换时忘记同步增广列（常数项那列也要换）。\n\n' +
            '**条件反射**：\n\n' +
            '- 看到"不回代直接读解" → 需要 RREF，用 Gauss-Jordan\n\n' +
            '- 看到"同一 $A$ 多个右端项" → LU 分解，分解一次复用\n\n' +
            '- 高斯消元时间复杂度：$O(n^3)$；LU 后每次求解：$O(n^2)$',
        },
      ],
      questions: [
        {
          id: 'u2-l4-q1',
          type: 'judge',
          prompt: '任何矩阵的简化行阶梯形（RREF）是唯一的，不随消元顺序改变。',
          answer: true,
          explain:
            'RREF 完全由方程组的解集决定，解集不依赖于消元顺序，故 RREF 也唯一确定。',
        },
        {
          id: 'u2-l4-q2',
          type: 'choice',
          prompt:
            '高斯消元法的时间复杂度是？',
          options: ['$O(n)$', '$O(n^2)$', '$O(n^3)$', '$O(n!)$'],
          answer: 2,
          explain:
            '高斯消元约需 $\\frac{2n^3}{3}$ 次乘加运算，时间复杂度为 $O(n^3)$。',
        },
        {
          id: 'u2-l4-q3',
          type: 'judge',
          prompt: 'LU 分解中，矩阵 $L$ 的主对角线元素全为 1。',
          answer: true,
          explain:
            '$L$ 是**单位**下三角矩阵，对角线为 1，下方填消元乘数。',
        },
        {
          id: 'u2-l4-q4',
          type: 'choice',
          prompt:
            '对同一矩阵 $A$ 需要求解 10 个不同右端项 $\\mathbf{b}_1,\\dots,\\mathbf{b}_{10}$，最优策略是：',
          options: [
            '10 次重新做高斯消元（共 $10O(n^3)$）',
            '做一次 LU 分解，每次用前代+回代（$O(n^3)+10O(n^2)$）',
            '先求 $A^{-1}$，再逐一乘',
            '直接用 RREF，一次读出 10 组解',
          ],
          answer: 1,
          explain:
            'LU 分解一次（$O(n^3)$），之后每次求解仅 $O(n^2)$，总代价远低于重复消元。',
        },
        {
          id: 'u2-l4-q5',
          type: 'input',
          prompt:
            '对 RREF $\\left(\\begin{array}{cccc|c}1&0&2&0&3\\\\0&1&-1&0&1\\\\0&0&0&1&2\\end{array}\\right)$，方程组有几个自由变量？',
          accept: ['1'],
          placeholder: '输入整数',
          explain:
            '共 4 个未知量，主元列为 1、2、4，主元数 $r=3$，自由变量数 $=4-3=1$（$x_3$ 是自由变量）。',
        },
        {
          id: 'u2-l4-q6',
          type: 'choice',
          prompt:
            '从 RREF $\\left(\\begin{array}{ccc|c}1&0&0&5\\\\0&1&0&-2\\\\0&0&1&3\\end{array}\\right)$ 直接读出 $x_2=$？',
          options: ['5', '-2', '3', '0'],
          answer: 1,
          explain:
            'RREF 的第 2 行直接给出 $x_2=-2$，无需回代。',
        },
        {
          id: 'u2-l4-q7',
          type: 'match',
          prompt: '将消元方法与特点对应。',
          left: [
            '高斯消元（前向消元）',
            'Gauss-Jordan 消元',
            'LU 分解',
          ],
          right: ['直接读解，无需回代', '一次分解，多次 $O(n^2)$ 求解', '化为 REF 后回代'],
          answer: [2, 0, 1],
          explain:
            '高斯消元化 REF 再回代；Gauss-Jordan 化 RREF 直接读解；LU 分解一次复用。',
        },
      ],
    },

    // ─────────────────────────────────────────
    // 关 5  逆矩阵的定义与求法
    // ─────────────────────────────────────────
    {
      id: 'u2-l5',
      title: '逆矩阵的定义与求法',
      subtitle: '定义·唯一性·2×2 公式·行变换法',
      intro: [
        {
          title: '逆矩阵是什么？',
          body:
            '设 $A$ 为 $n\\times n$ 方阵，若存在矩阵 $B$ 使\n\n' +
            '$AB=BA=I_n$\n\n' +
            '则称 $A$ **可逆**，$B$ 是 $A$ 的**逆矩阵**，记作 $A^{-1}$。\n\n' +
            '**几何意义**：若 $A$ 对应一个线性变换（旋转、缩放等），$A^{-1}$ 对应"撤销"该变换。\n\n' +
            '**关键限制**：逆矩阵只对**方阵**定义，且可逆时**唯一**。',
          tip: '奇异矩阵（行列式为零）把空间压缩到更低维度，信息丢失，无法还原，故不可逆。',
        },
        {
          title: '逆矩阵的唯一性',
          body:
            '**定理**：若 $A$ 可逆，其逆矩阵唯一。\n\n' +
            '**证明**：假设 $B$ 和 $C$ 都是 $A$ 的逆矩阵，即 $AB=BA=I$ 且 $AC=CA=I$，则\n\n' +
            '$B=BI=B(AC)=(BA)C=IC=C$\n\n' +
            '故 $B=C$，逆矩阵唯一。\n\n' +
            '**奇异矩阵**：行列式为零的矩阵，例如 $\\begin{pmatrix}1&2\\\\2&4\\end{pmatrix}$（第二行是第一行的 2 倍），将平面压成一条直线，无法还原，不可逆。',
          reveal: {
            q: '若 $AB=I$ 但 $BA\\ne I$，$A$ 算可逆吗？',
            a: '对**方阵**，$AB=I$ 可推出 $BA=I$，所以只需验证一侧即可。非方阵则不然。',
          },
        },
        {
          title: '$2\\times 2$ 逆矩阵公式',
          body:
            '对 $A=\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}$，规律是：**主对角元素交换位置**，**副对角元素变号**，整体除以 $\\det(A)=ad-bc$：\n\n' +
            '**推导**：设 $B=\\begin{pmatrix}d&-b\\\\-c&a\\end{pmatrix}$，则 $AB=(ad-bc)I$，故 $A^{-1}=\\frac{B}{ad-bc}$。\n\n' +
            '**例**：$A=\\begin{pmatrix}3&1\\\\5&2\\end{pmatrix}$，$\\det=6-5=1$，故 $A^{-1}=\\begin{pmatrix}2&-1\\\\-5&3\\end{pmatrix}$。\n\n' +
            '验证：$AA^{-1}=\\begin{pmatrix}6-5&-3+3\\\\10-10&-5+6\\end{pmatrix}=I$。',
          formula:
            '\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}^{-1}=\\frac{1}{ad-bc}\\begin{pmatrix}d&-b\\\\-c&a\\end{pmatrix}',
          tip: '先算 $\\det=ad-bc$；若为 0，矩阵不可逆，停止。',
        },
        {
          title: '行变换法求逆（原理）',
          body:
            '思路：$[A\\mid I]\\xrightarrow{\\text{RREF}}[I\\mid A^{-1}]$\n\n' +
            '**原理**：若初等矩阵序列 $E_k\\cdots E_1\\cdot A=I$，则同样操作施于右侧：$E_k\\cdots E_1\\cdot I=A^{-1}$。\n\n' +
            '**判断不可逆**：若化简过程中左边出现全零行，说明 $A$ 不可逆，立刻停止。\n\n' +
            '**重要**：必须对整行（含右半部）施行同样的变换——只变左半部是常见失误。',
          tip: '行变换法是求逆的通用方法，适用于任意阶方阵，$2\\times 2$ 公式只是它的特例。',
        },
        {
          title: '行变换法完整演示（$3\\times 3$）',
          body:
            '求 $A=\\begin{pmatrix}1&2&1\\\\2&5&3\\\\0&1&2\\end{pmatrix}$ 的逆。\n\n' +
            '**初始**：$\\left(\\begin{array}{ccc|ccc}1&2&1&1&0&0\\\\2&5&3&0&1&0\\\\0&1&2&0&0&1\\end{array}\\right)$\n\n' +
            '**$R_2-2R_1$，$R_3-R_2$**（前向消元）后继续向上消并归一，最终得\n\n' +
            '$\\left(\\begin{array}{ccc|ccc}1&0&0&7&-3&1\\\\0&1&0&-4&2&-1\\\\0&0&1&2&-1&1\\end{array}\\right)$\n\n' +
            '故 $A^{-1}=\\begin{pmatrix}7&-3&1\\\\-4&2&-1\\\\2&-1&1\\end{pmatrix}$。',
        },
        {
          title: '易错点总结',
          body:
            '**易错 1**：$2\\times 2$ 公式副对角号搞反——$b,c$ 要变号，不能只换位置不变号。\n\n' +
            '**易错 2**：行变换法中只变左半部，忘了同步右半部（$I$ 那侧）——两侧必须做同样操作。\n\n' +
            '**易错 3**：把 $\\det(A)=0$ 当作"逆是无穷大"——逆根本不存在，是结构性无法恢复。\n\n' +
            '**条件反射**：\n\n' +
            '- 看到"求逆" → 先算 $\\det$；$\\det=0$ 不可逆，停止\n\n' +
            '- $2\\times 2$ → 用公式；$n\\ge 3$ → 用行变换法\n\n' +
            '- $[A\\mid I]$ 化简后左边出现全零行 → 不可逆，停止',
        },
      ],
      questions: [
        {
          id: 'u2-l5-q1',
          type: 'choice',
          prompt:
            '矩阵 $A=\\begin{pmatrix}2&4\\\\1&2\\end{pmatrix}$ 是否可逆？',
          options: [
            '可逆，$\\det(A)=4$',
            '可逆，$\\det(A)=0$',
            '不可逆，$\\det(A)=0$',
            '需要更多信息',
          ],
          answer: 2,
          explain:
            '$\\det(A)=2\\times 2-4\\times 1=0$，行列式为零，矩阵不可逆（奇异矩阵）。',
        },
        {
          id: 'u2-l5-q2',
          type: 'input',
          prompt:
            '用 $2\\times 2$ 公式求 $A=\\begin{pmatrix}3&1\\\\5&2\\end{pmatrix}$ 的逆矩阵，$A^{-1}$ 的 $(1,1)$ 元素是多少？',
          accept: ['2'],
          placeholder: '输入数字',
          explain:
            '$\\det(A)=6-5=1$，$A^{-1}=\\begin{pmatrix}2&-1\\\\-5&3\\end{pmatrix}$，$(1,1)$ 元素为 $2$。',
        },
        {
          id: 'u2-l5-q3',
          type: 'input',
          prompt:
            '$A=\\begin{pmatrix}4&3\\\\3&2\\end{pmatrix}$，$\\det(A)=$？',
          accept: ['-1'],
          placeholder: '输入数字',
          explain:
            '$\\det(A)=4\\times 2-3\\times 3=8-9=-1$。',
        },
        {
          id: 'u2-l5-q4',
          type: 'choice',
          prompt:
            '用行变换法求逆时，$[A\\mid I]$ 化简后左边出现全零行，说明：',
          options: [
            '$A$ 可逆，继续化简',
            '$A$ 不可逆，停止',
            '出错了，需要重新消元',
            '右边就是 $A^{-1}$',
          ],
          answer: 1,
          explain:
            '左边出现全零行说明 $A$ 的秩 $<n$，$A$ 不可逆，行变换法无法继续。',
        },
        {
          id: 'u2-l5-q5',
          type: 'judge',
          prompt:
            '逆矩阵的逆等于原矩阵：$(A^{-1})^{-1}=A$。',
          answer: true,
          explain:
            '由定义 $A\\cdot A^{-1}=I$，这说明 $A$ 恰好是 $A^{-1}$ 的逆，故 $(A^{-1})^{-1}=A$。',
        },
        {
          id: 'u2-l5-q6',
          type: 'choice',
          prompt:
            '已知 $A=\\begin{pmatrix}1&0\\\\2&1\\end{pmatrix}$，$A^{-1}=\\begin{pmatrix}1&0\\\\-2&1\\end{pmatrix}$，解 $A\\mathbf{x}=\\begin{pmatrix}3\\\\5\\end{pmatrix}$，$x_2=$？',
          options: ['-1', '1', '5', '-5'],
          answer: 0,
          explain:
            '$\\mathbf{x}=A^{-1}\\mathbf{b}=\\begin{pmatrix}1&0\\\\-2&1\\end{pmatrix}\\begin{pmatrix}3\\\\5\\end{pmatrix}=\\begin{pmatrix}3\\\\-1\\end{pmatrix}$，$x_2=-1$。',
        },
        {
          id: 'u2-l5-q7',
          type: 'input',
          prompt:
            '用 $2\\times 2$ 公式求 $A=\\begin{pmatrix}4&7\\\\2&6\\end{pmatrix}$ 的逆，$A^{-1}$ 的 $(2,1)$ 元素是多少？',
          accept: ['-0.2', '-1/5'],
          placeholder: '输入数字（可用小数）',
          explain:
            '$\\det(A)=24-14=10$，$A^{-1}=\\frac{1}{10}\\begin{pmatrix}6&-7\\\\-2&4\\end{pmatrix}$，$(2,1)$ 元素为 $-2/10=-0.2$。',
        },
      ],
    },

    // ─────────────────────────────────────────
    // 关 6  逆的性质与可逆判据
    // ─────────────────────────────────────────
    {
      id: 'u2-l6',
      title: '逆的性质与可逆判据',
      subtitle: '乘积逆·转置逆·可逆矩阵定理·等价条件',
      intro: [
        {
          title: '逆矩阵的代数性质',
          body:
            '设 $A,B$ 均为可逆 $n\\times n$ 矩阵，$c\\ne 0$：\n\n' +
            '- **双重逆**：$(A^{-1})^{-1}=A$\n\n' +
            '- **乘积逆**：$(AB)^{-1}=B^{-1}A^{-1}$（顺序反转！）\n\n' +
            '- **转置逆**：$(A^T)^{-1}=(A^{-1})^T$\n\n' +
            '- **标量逆**：$(cA)^{-1}=\\dfrac{1}{c}A^{-1}$\n\n' +
            '- **幂次逆**：$(A^k)^{-1}=(A^{-1})^k$，记作 $A^{-k}$',
          tip: '记忆口诀：先穿袜子再穿鞋（$AB$），脱的时候先脱鞋再脱袜子（$B^{-1}A^{-1}$）。',
        },
        {
          title: '乘积逆的证明',
          body:
            '**证明 $(AB)^{-1}=B^{-1}A^{-1}$**：\n\n' +
            '左乘验证：$(AB)(B^{-1}A^{-1})=A(BB^{-1})A^{-1}=AIA^{-1}=AA^{-1}=I$\n\n' +
            '右乘验证：$(B^{-1}A^{-1})(AB)=B^{-1}(A^{-1}A)B=B^{-1}IB=B^{-1}B=I$\n\n' +
            '由逆矩阵唯一性，$(AB)^{-1}=B^{-1}A^{-1}$。\n\n' +
            '**推广**：$(A_1A_2\\cdots A_k)^{-1}=A_k^{-1}\\cdots A_2^{-1}A_1^{-1}$，顺序全部反转。',
          reveal: {
            q: '$(ABC)^{-1}=$？',
            a: '$C^{-1}B^{-1}A^{-1}$，顺序全部反转。',
          },
        },
        {
          title: '转置逆的证明',
          body:
            '**证明 $(A^T)^{-1}=(A^{-1})^T$**：\n\n' +
            '$(A^{-1})^T\\cdot A^T=(A\\cdot A^{-1})^T=I^T=I$\n\n' +
            '同理 $A^T\\cdot(A^{-1})^T=I$，故 $(A^T)^{-1}=(A^{-1})^T$。\n\n' +
            '**推论**：若 $A$ 是可逆**对称**矩阵（$A^T=A$），则 $A^{-1}$ 也是对称矩阵：\n\n' +
            '$(A^{-1})^T=(A^T)^{-1}=A^{-1}$',
          tip: '$(A^T)^{-1}=(A^{-1})^T$ 意味着转置与求逆可以互换顺序，用哪个方便用哪个。',
        },
        {
          title: '可逆矩阵定理（9 个等价条件）',
          body:
            '对 $n\\times n$ 方阵 $A$，以下命题互相等价，任一成立则全部成立：\n\n' +
            '1. $A$ 可逆（存在 $A^{-1}$）\n\n' +
            '2. $\\det(A)\\ne 0$\n\n' +
            '3. $\\text{rank}(A)=n$（满秩）\n\n' +
            '4. $A$ 的各列线性无关\n\n' +
            '5. $A$ 的各行线性无关\n\n' +
            '6. $A\\mathbf{x}=\\mathbf{0}$ 只有零解（零空间平凡）\n\n' +
            '7. 对任意 $\\mathbf{b}$，$A\\mathbf{x}=\\mathbf{b}$ 有唯一解\n\n' +
            '8. $A$ 可通过初等行变换化为 $I_n$\n\n' +
            '9. $A$ 是若干初等矩阵的乘积',
          tip: '做题时选最方便的那个等价条件用。常见：判断可逆用 $\\det\\ne 0$；判断零空间用条件 6；解方程用条件 7。',
        },
        {
          title: '消去律与应用',
          body:
            '**消去律**：若 $A$ 可逆且 $AB=AC$，则 $B=C$（左乘 $A^{-1}$）。\n\n' +
            '**注意**：若 $A$ 不可逆，消去律不成立。例如 $A=\\begin{pmatrix}0&0\\\\0&0\\end{pmatrix}$，$B\\ne C$ 但 $AB=AC=O$。\n\n' +
            '**矩阵方程化简**：利用可逆矩阵的性质，常见化简：\n\n' +
            '- $(A^TA)^{-1}A^T$：当 $A$ 方阵且可逆时，$=(A^T)^{-1}(A^T)^{T-1}\\cdot A^T$；更直接：$(A^TA)^{-1}A^T\\cdot A=I$，故等于 $A^{-1}$\n\n' +
            '- 在最小二乘中 $(X^TX)^{-1}X^T$ 是 $X$ 的**左伪逆**',
          reveal: {
            q: '化简 $(B^{-1}A)^{-1}(B^{-1}AB)$',
            a: '$(B^{-1}A)^{-1}=A^{-1}B$，再乘 $B^{-1}AB$：$A^{-1}(BB^{-1})AB=A^{-1}AB=B$。',
          },
        },
        {
          title: '易错点总结',
          body:
            '**易错 1**：$(AB)^{-1}=A^{-1}B^{-1}$（顺序未反转）——矩阵乘法不交换，必须反转。\n\n' +
            '**易错 2**：忘记 $(A^T)^{-1}=(A^{-1})^T$，把转置和逆的顺序搞错。\n\n' +
            '**易错 3**：在不可逆矩阵上使用消去律——$AB=AC$ 不能推出 $B=C$，除非 $A$ 可逆。\n\n' +
            '**条件反射**：\n\n' +
            '- 看到 $(AB)^{-1}$ → 反转顺序 $B^{-1}A^{-1}$\n\n' +
            '- 看到 $(A^T)^{-1}$ → 等于 $(A^{-1})^T$，两者可互换\n\n' +
            '- 看到"可逆"等价条件 → 9 个条件任选最方便的',
        },
      ],
      questions: [
        {
          id: 'u2-l6-q1',
          type: 'choice',
          prompt: '$(AB)^{-1}=$？（$A,B$ 均可逆）',
          options: [
            '$A^{-1}B^{-1}$',
            '$B^{-1}A^{-1}$',
            '$BA$',
            '$A^{-1}B$',
          ],
          answer: 1,
          explain:
            '$(AB)(B^{-1}A^{-1})=A(BB^{-1})A^{-1}=I$，故 $(AB)^{-1}=B^{-1}A^{-1}$，顺序反转。',
        },
        {
          id: 'u2-l6-q2',
          type: 'judge',
          prompt:
            '$(A^T)^{-1}=(A^{-1})^T$，即转置与求逆可以互换顺序。',
          answer: true,
          explain:
            '$(A^{-1})^T\\cdot A^T=(AA^{-1})^T=I^T=I$，两侧验证后得 $(A^T)^{-1}=(A^{-1})^T$。',
        },
        {
          id: 'u2-l6-q3',
          type: 'match',
          prompt: '将可逆矩阵的等价条件与对应描述配对。',
          left: [
            '$\\det(A)\\ne 0$',
            '$\\text{rank}(A)=n$',
            '$A\\mathbf{x}=\\mathbf{0}$ 只有零解',
          ],
          right: ['行列式不为零', '零空间平凡', '满秩'],
          answer: [0, 2, 1],
          explain:
            '三个等价条件：行列式不为零；满秩（秩等于阶数）；零空间只含零向量。',
        },
        {
          id: 'u2-l6-q4',
          type: 'choice',
          prompt:
            '设 $A,B$ 均可逆，化简 $(AB^{-1})^{-1}$：',
          options: [
            '$A^{-1}B$',
            '$BA^{-1}$',
            '$B^{-1}A^{-1}$',
            '$AB$',
          ],
          answer: 1,
          explain:
            '$(AB^{-1})^{-1}=(B^{-1})^{-1}A^{-1}=BA^{-1}$，顺序反转，$B^{-1}$ 的逆是 $B$。',
        },
        {
          id: 'u2-l6-q5',
          type: 'judge',
          prompt:
            '若 $A$ 是可逆对称矩阵（$A^T=A$），则 $A^{-1}$ 也是对称矩阵。',
          answer: true,
          explain:
            '$(A^{-1})^T=(A^T)^{-1}=A^{-1}$，故 $A^{-1}$ 对称。',
        },
        {
          id: 'u2-l6-q6',
          type: 'choice',
          prompt:
            '矩阵 $A$ 满足哪个条件时，对所有 $\\mathbf{b}$，$A\\mathbf{x}=\\mathbf{b}$ 有唯一解？',
          options: [
            '$m>n$（行多于列）',
            '$A$ 是方阵且 $\\text{rank}(A)=n$',
            '$m=n$（方阵）',
            '$\\text{rank}(A)<n$',
          ],
          answer: 1,
          explain:
            '$A$ 可逆（满秩方阵）是保证对所有 $\\mathbf{b}$ 有唯一解的等价条件（可逆矩阵定理条件 7）。',
        },
        {
          id: 'u2-l6-q7',
          type: 'input',
          prompt:
            '$(A_1A_2A_3)^{-1}=A_3^{-1}A_2^{-1}A_1^{-1}$，这利用了哪个性质？请用中文一句话描述。',
          accept: ['乘积逆顺序反转', '乘积的逆等于逆的乘积顺序反转', '顺序反转'],
          placeholder: '简短描述',
          explain:
            '乘积逆：$(A_1A_2\\cdots A_k)^{-1}=A_k^{-1}\\cdots A_1^{-1}$，顺序全部反转。',
        },
      ],
    },

    // ─────────────────────────────────────────
    // 关 7  逆矩阵的应用与工程准则
    // ─────────────────────────────────────────
    {
      id: 'u2-l7',
      title: '逆矩阵的应用与工程准则',
      subtitle: '解方程·工程准则·Sherman-Morrison·AI 关联',
      intro: [
        {
          title: '用逆矩阵解方程组',
          body:
            '若 $A$ 可逆，$A\\mathbf{x}=\\mathbf{b}$ 有唯一解。两边左乘 $A^{-1}$：\n\n' +
            '$\\mathbf{x}=A^{-1}\\mathbf{b}$\n\n' +
            '**示例**：$A=\\begin{pmatrix}1&2&1\\\\2&5&3\\\\0&1&2\\end{pmatrix}$，$A^{-1}=\\begin{pmatrix}7&-3&1\\\\-4&2&-1\\\\2&-1&1\\end{pmatrix}$，$\\mathbf{b}=\\begin{pmatrix}1\\\\3\\\\2\\end{pmatrix}$\n\n' +
            '$\\mathbf{x}=A^{-1}\\mathbf{b}=\\begin{pmatrix}7-9+2\\\\-4+6-2\\\\2-3+2\\end{pmatrix}=\\begin{pmatrix}0\\\\0\\\\1\\end{pmatrix}$',
          tip: '教材写 $A^{-1}\\mathbf{b}$ 是**记号**，表示"存在唯一解"；实际计算应用 LU 分解（`solve`）。',
        },
        {
          title: '为什么工程中不显式求逆？',
          body:
            '尽管 $\\mathbf{x}=A^{-1}\\mathbf{b}$ 形式简洁，**实际数值计算中几乎不这样做**：\n\n' +
            '**1. 计算量更大**：求 $A^{-1}$ 相当于对 $n$ 个右端向量分别消元，常数因子约是直接 `solve` 的 $n$ 倍。\n\n' +
            '**2. 数值精度更差**：显式求逆引入更多浮点误差，对病态矩阵（条件数大）误差急剧放大。\n\n' +
            '**3. 破坏稀疏性**：稀疏矩阵（$O(n)$ 非零元）的逆通常稠密（$O(n^2)$），存储爆炸。\n\n' +
            '**工程准则**：遇到 $A\\mathbf{x}=\\mathbf{b}$，用 `numpy.linalg.solve(A, b)`，不要 `inv(A) @ b`。',
          tip: '即使两种方法数学上等价，数值上 `solve` 永远优于 `inv(A) @ b`——这是数值线性代数的铁律。',
        },
        {
          title: 'Sherman-Morrison 公式：秩 1 更新',
          body:
            '若 $A$ 可逆，$\\mathbf{u},\\mathbf{v}$ 为列向量，且 $1+\\mathbf{v}^TA^{-1}\\mathbf{u}\\ne 0$，则：\n\n' +
            '$(A+\\mathbf{u}\\mathbf{v}^T)^{-1}=A^{-1}-\\dfrac{A^{-1}\\mathbf{u}\\mathbf{v}^TA^{-1}}{1+\\mathbf{v}^TA^{-1}\\mathbf{u}}$\n\n' +
            '**意义**：当矩阵发生**秩 1 更新**（$A\\to A+\\mathbf{u}\\mathbf{v}^T$）时，逆矩阵可以**增量更新**，无需重新计算，复杂度从 $O(n^3)$ 降为 $O(n^2)$。\n\n' +
            '**应用**：在线学习（每收到一个新样本，协方差矩阵发生秩 1 更新）、Kalman 滤波、自然梯度等。',
        },
        {
          title: 'AI 中的逆矩阵：理论核心，工程替代',
          body:
            '**Newton 法**更新步：$\\boldsymbol{\\theta}_{t+1}=\\boldsymbol{\\theta}_t-H^{-1}\\nabla f(\\boldsymbol{\\theta}_t)$，其中 $H$ 是 Hessian 矩阵。\n\n' +
            '**自然梯度**：$\\boldsymbol{\\theta}_{t+1}=\\boldsymbol{\\theta}_t-\\eta F^{-1}\\nabla\\mathcal{L}$，其中 $F$ 是 Fisher 信息矩阵。\n\n' +
            '**实际从不显式计算**：百万参数模型的 Hessian 有 $10^{12}$ 个元素，存储和求逆完全不现实。\n\n' +
            '**替代方案**：\n\n' +
            '- 共轭梯度法：迭代求解 $H\\mathbf{x}=\\mathbf{v}$，只需 Hessian-向量积\n\n' +
            '- K-FAC：Kronecker 因子化近似 Fisher 矩阵，降至 $O(n\\sqrt{n})$ 存储\n\n' +
            '- 批归一化：隐式改善 Hessian 条件数，让一阶方法也有二阶效果',
          tip: '理解"逆矩阵存在但不可显式求"是深入理解深度学习优化的关键认知。',
        },
        {
          title: '综合易错点',
          body:
            '**易错 1**：把"可逆"和"方阵"分开——逆矩阵只对方阵定义，非方阵没有双侧逆。\n\n' +
            '**易错 2**：$(AB)^{-1}=A^{-1}B^{-1}$（顺序未反转）——必须是 $B^{-1}A^{-1}$。\n\n' +
            '**易错 3**：见到 $A\\mathbf{x}=\\mathbf{b}$ 就求 $A^{-1}$——工程上应用 `solve`。\n\n' +
            '**总结条件反射**：\n\n' +
            '- 看到"解 $A\\mathbf{x}=\\mathbf{b}$" → 用 `solve`，不求逆\n\n' +
            '- 看到"矩阵秩 1 更新" → Sherman-Morrison，$O(n^2)$ 更新逆\n\n' +
            '- 看到"稀疏矩阵的逆" → 警觉！逆通常稠密，存储从 $O(n)$ 变 $O(n^2)$',
        },
        {
          title: '单元总结：三章核心脉络',
          body:
            '**Ch4 线性方程组**：矩阵方程 $A\\mathbf{x}=\\mathbf{b}$，增广矩阵判断解的三种情形（秩对比），零空间与解结构（特解 + 齐次通解）。\n\n' +
            '**Ch5 高斯消元**：三种初等行变换保解集，REF/RREF 的区别，自由变量参数化，LU 分解复用优势，部分选主元保数值稳定。\n\n' +
            '**Ch6 逆矩阵**：$[A\\mid I]\\to[I\\mid A^{-1}]$，乘积逆反转顺序，9 个可逆等价条件，工程中用 `solve` 不用 `inv`，Sherman-Morrison 秩 1 更新。\n\n' +
            '三章形成完整闭环：**建模** $A\\mathbf{x}=\\mathbf{b}$ → **算法** 高斯消元/LU → **理论** 可逆性与逆矩阵。',
        },
      ],
      questions: [
        {
          id: 'u2-l7-q1',
          type: 'judge',
          prompt:
            '实际工程中，求解 $A\\mathbf{x}=\\mathbf{b}$ 推荐先计算 $A^{-1}$，再乘以 $\\mathbf{b}$。',
          answer: false,
          explain:
            '工程准则：用 `numpy.linalg.solve(A, b)` 而非 `inv(A) @ b`——前者更快（LU 分解）、数值更稳定、不破坏稀疏性。',
        },
        {
          id: 'u2-l7-q2',
          type: 'choice',
          prompt:
            '稀疏矩阵（大多数元素为零）工程中为何不显式求逆？',
          options: [
            '因为稀疏矩阵一定不可逆',
            '因为稀疏矩阵的逆通常是稠密矩阵，存储从 $O(n)$ 变 $O(n^2)$',
            '因为对稀疏矩阵无法做高斯消元',
            '因为稀疏矩阵的条件数一定很大',
          ],
          answer: 1,
          explain:
            '稀疏矩阵可逆，但其逆通常稠密，存储从 $O(n)$ 变 $O(n^2)$，完全失去稀疏优势。',
        },
        {
          id: 'u2-l7-q3',
          type: 'choice',
          prompt:
            'Sherman-Morrison 公式描述的是哪种操作的逆矩阵增量更新？',
          options: [
            '矩阵转置',
            '矩阵秩 1 更新 $A\\to A+\\mathbf{u}\\mathbf{v}^T$',
            '矩阵乘积 $AB$',
            '矩阵置换行',
          ],
          answer: 1,
          explain:
            'Sherman-Morrison 处理矩阵发生秩 1 更新时的逆矩阵增量计算，复杂度从 $O(n^3)$ 降至 $O(n^2)$。',
        },
        {
          id: 'u2-l7-q4',
          type: 'match',
          prompt: '将下列场景与最优工程策略配对。',
          left: [
            '求解一次 $A\\mathbf{x}=\\mathbf{b}$',
            '同一 $A$，多个右端项 $\\mathbf{b}_1,\\dots,\\mathbf{b}_{10}$',
            '矩阵发生秩 1 更新，已知旧逆矩阵',
          ],
          right: ['Sherman-Morrison，$O(n^2)$', 'LU 分解一次，前代+回代 10 次', '直接用 `solve`，底层 LU'],
          answer: [2, 1, 0],
          explain:
            '单次求解用 `solve`；多右端项用 LU 分解复用；秩 1 更新用 Sherman-Morrison。',
        },
        {
          id: 'u2-l7-q5',
          type: 'judge',
          prompt:
            '对任意可逆方阵 $A$，$(A^TA)^{-1}A^T\\cdot A=I$，即 $(A^TA)^{-1}A^T$ 是 $A$ 的左逆。',
          answer: true,
          explain:
            '$(A^TA)^{-1}(A^TA)=I$，故 $(A^TA)^{-1}A^T\\cdot A=(A^TA)^{-1}(A^TA)=I$。当 $A$ 方阵且可逆时，左逆即 $A^{-1}$。',
        },
        {
          id: 'u2-l7-q6',
          type: 'choice',
          prompt:
            '深度学习中，Newton 法更新步 $\\boldsymbol{\\theta}_{t+1}=\\boldsymbol{\\theta}_t-H^{-1}g$ 实际上为何不显式计算 $H^{-1}$？',
          options: [
            'Hessian 矩阵 $H$ 一定不可逆',
            '参数量大时 $H$ 有 $O(n^2)$ 个元素，存储和求逆不现实',
            '只有线性模型才有 Hessian 矩阵',
            '深度学习中梯度恒为零',
          ],
          answer: 1,
          explain:
            '百万参数模型的 Hessian 有 $10^{12}$ 个元素，存储和求逆（$O(n^3)$）完全不可行，改用共轭梯度、K-FAC 等近似方法。',
        },
      ],
    },
  ],
}
