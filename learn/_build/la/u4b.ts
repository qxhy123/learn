export const LESSONS = [
  // =========================================================================
  // u4-l4  基与坐标
  // =========================================================================
  {
    id: 'u4-l4',
    title: '基与坐标',
    subtitle: '坐标系的选择决定表达效率',
    intro: [
      {
        title: '什么是基？',
        body: '向量空间 $V$ 的一个**基**（Basis）是同时满足两个条件的向量组 $\\mathcal{B}=\\{\\mathbf{b}_1,\\ldots,\\mathbf{b}_n\\}$：\n\n**条件 1——线性无关**：方程 $c_1\\mathbf{b}_1+\\cdots+c_n\\mathbf{b}_n=\\mathbf{0}$ 只有零解。\n\n**条件 2——张成全空间**：$V=\\operatorname{span}\\{\\mathbf{b}_1,\\ldots,\\mathbf{b}_n\\}$，即 $V$ 中每个向量都能写成 $\\mathbf{b}_i$ 的线性组合。\n\n**两条缺一不可**：只有张成但存在冗余（线性相关），意味着某个向量是多余的；只有线性无关但覆盖不了全部空间，意味着描述不完整——两种情况都不叫基。',
        tip: '在 $\\mathbb{R}^n$ 中，$n$ 个向量构成基当且仅当它们线性无关（$n$ 个无关向量自动张成 $\\mathbb{R}^n$），等价于方阵行列式非零，等价于矩阵可逆。',
      },
      {
        title: '标准基',
        body: '$\\mathbb{R}^n$ 的**标准基**（也称自然基）是：\n$e_1,e_2,\\ldots,e_n$，其中 $e_j$ 的第 $j$ 个分量为 $1$，其余为 $0$。\n\n**验证**：线性无关——$c_1e_1+\\cdots+c_ne_n=\\mathbf{0}$ 逐分量得 $c_j=0$；张成——任意 $\\mathbf{x}=(x_1,\\ldots,x_n)^T=x_1e_1+\\cdots+x_ne_n$。\n\n**其他空间的标准基**：\n- $\\mathbb{R}^{2\\times 2}$（$2\\times2$ 矩阵）：$E_{11},E_{12},E_{21},E_{22}$，共 4 个基矩阵。\n- $\\mathcal{P}_n$（次数 $\\leq n$ 多项式）：$\\{1,x,x^2,\\ldots,x^n\\}$，共 $n+1$ 个。',
        formula: 'e_1=\\begin{pmatrix}1\\\\0\\\\\\vdots\\\\0\\end{pmatrix},\\quad e_2=\\begin{pmatrix}0\\\\1\\\\\\vdots\\\\0\\end{pmatrix},\\quad\\ldots,\\quad e_n=\\begin{pmatrix}0\\\\0\\\\\\vdots\\\\1\\end{pmatrix}',
      },
      {
        title: '基的非唯一性与维数',
        body: '同一向量空间可以有无数不同的基。以 $\\mathbb{R}^2$ 为例：\n\n- 标准基：$\\{e_1,e_2\\}=\\{(1,0)^T,(0,1)^T\\}$\n- 另一个合法的基：$\\{(1,1)^T,(1,-1)^T\\}$——线性无关且张成 $\\mathbb{R}^2$\n- 再一个：$\\{(2,0)^T,(0,3)^T\\}$\n\n**关键事实**：虽然基不唯一，但任意两个基包含**相同数量**的向量。这个共同的数量称为**维数** $\\dim(V)$，是良定义的整数。\n\n**子空间的维数不等式**：若 $W$ 是 $V$ 的子空间，则 $\\dim(W)\\leq\\dim(V)$，等号成立当且仅当 $W=V$。',
        formula: '\\dim(\\mathbb{R}^n)=n,\\quad \\dim(\\mathbb{R}^{m\\times n})=mn,\\quad \\dim(\\mathcal{P}_n)=n+1',
      },
      {
        title: '坐标的唯一性',
        body: '设 $\\mathcal{B}=\\{\\mathbf{b}_1,\\ldots,\\mathbf{b}_n\\}$ 是向量空间 $V$ 的一个基，则 $V$ 中每个向量 $\\mathbf{x}$ 都能**唯一地**写成：\n\n$\\mathbf{x}=c_1\\mathbf{b}_1+c_2\\mathbf{b}_2+\\cdots+c_n\\mathbf{b}_n$\n\n系数 $c_1,\\ldots,c_n$ 称为 $\\mathbf{x}$ 在基 $\\mathcal{B}$ 下的**坐标**，列向量 $[\\mathbf{x}]_{\\mathcal{B}}=(c_1,\\ldots,c_n)^T$ 称为**坐标向量**。\n\n**唯一性的来源**：基的线性无关性保证了表示方式唯一——若有两种表示，相减得到一个齐次方程，线性无关性迫使系数全为零。\n\n**哲学直觉**：坐标是描述工具，向量是客观存在——换一个基，坐标改变，但 $\\mathbf{x}$ 本身没有变。',
        tip: '坐标向量 $[\\mathbf{x}]_{\\mathcal{B}}$ 与 $\\mathbf{x}$ 本身不同：$\\mathbf{x}\\in V$（可以是多项式、矩阵），但 $[\\mathbf{x}]_{\\mathcal{B}}\\in\\mathbb{R}^n$（永远是数字列向量）。',
      },
      {
        title: '求坐标：增广矩阵方法',
        body: '**问题**：已知基 $\\mathcal{B}=\\{\\mathbf{b}_1,\\ldots,\\mathbf{b}_n\\}$，求 $\\mathbf{x}$ 在 $\\mathcal{B}$ 下的坐标。\n\n**等价问题**：求解线性方程组 $[\\mathbf{b}_1|\\mathbf{b}_2|\\cdots|\\mathbf{b}_n]\\mathbf{c}=\\mathbf{x}$，解 $\\mathbf{c}$ 即坐标向量。\n\n**操作**：对增广矩阵 $[\\mathbf{b}_1|\\cdots|\\mathbf{b}_n|\\mathbf{x}]$ 行化简到 RREF，右列即为坐标向量。',
        formula: '[\\mathbf{b}_1\\mid\\cdots\\mid\\mathbf{b}_n\\mid\\mathbf{x}] \\xrightarrow{\\text{RREF}} [I\\mid [\\mathbf{x}]_{\\mathcal{B}}]',
        steps: [
          '例：基 $\\mathcal{B}=\\{(1,2)^T,(3,5)^T\\}$，求 $\\mathbf{x}=(7,11)^T$ 的坐标。',
          '增广矩阵：$\\begin{pmatrix}1&3&7\\\\2&5&11\\end{pmatrix}\\xrightarrow{R_2-2R_1}\\begin{pmatrix}1&3&7\\\\0&-1&-3\\end{pmatrix}\\xrightarrow{-R_2}\\begin{pmatrix}1&3&7\\\\0&1&3\\end{pmatrix}\\xrightarrow{R_1-3R_2}\\begin{pmatrix}1&0&-2\\\\0&1&3\\end{pmatrix}$',
          '故 $c_1=-2,c_2=3$，即 $[\\mathbf{x}]_{\\mathcal{B}}=(-2,3)^T$。',
          '验证：$-2(1,2)^T+3(3,5)^T=(-2,-4)^T+(9,15)^T=(7,11)^T$ ✓',
        ],
      },
      {
        title: '扩充为基与 PCA 联系',
        body: '**扩充为基**：给定 $k$ 个线性无关向量（$k<n$），可将其扩充为 $\\mathbb{R}^n$ 的基——把这 $k$ 个向量与标准基 $e_1,\\ldots,e_n$ 合并成矩阵，行化简，选出 $n$ 个主元列即为扩充后的基。\n\n**AI 联系——PCA**：主成分分析的本质是寻找协方差矩阵的前 $k$ 个特征向量，构成一组正交基，然后把数据坐标变换到这组新基下。信息浓缩在前几个坐标里，后面的坐标接近 $0$ 可以安全丢弃——这正是"选更好的基，重新坐标化"的过程。\n\n**易错点**：求列空间的基，必须取**原矩阵**的主元列，不能取行阶梯形的列（行变换改变了列向量的值）。',
        reveal: {
          q: '若 $A$ 是 $4\\times6$ 矩阵，$\\operatorname{rank}(A)=3$，其零空间的基有几个向量？',
          a: '由秩-零化度定理（$n=6$ 列）：$\\operatorname{nullity}(A)=6-3=3$。零空间的基含 **3 个**向量。直觉：输入空间 $\\mathbb{R}^6$ 的 6 个维度中，有 3 个被 $A$ 有效映射（行空间），另外 3 个被压缩为零（零空间）。',
        },
      },
    ],
    questions: [
      {
        id: 'u4-l4-q1',
        type: 'choice',
        prompt: '设 $\\mathbf{b}_1=(1,1,0)^T$，$\\mathbf{b}_2=(1,0,1)^T$，$\\mathbf{b}_3=(0,1,1)^T$。计算行列式 $\\det\\begin{pmatrix}1&1&0\\\\1&0&1\\\\0&1&1\\end{pmatrix}$ 以判断 $\\{\\mathbf{b}_1,\\mathbf{b}_2,\\mathbf{b}_3\\}$ 是否构成 $\\mathbb{R}^3$ 的基。',
        options: [
          '行列式 $= -2\\neq 0$，构成基',
          '行列式 $= 0$，不构成基',
          '行列式 $= 2\\neq 0$，构成基',
          '行列式 $= -1\\neq 0$，构成基',
        ],
        answer: 0,
        explain: '按第一行展开：$1(0-1)-1(1-0)+0=-1-1=-2\\neq 0$。行列式非零，三向量线性无关，在 $\\mathbb{R}^3$ 中构成基。',
      },
      {
        id: 'u4-l4-q2',
        type: 'input',
        prompt: '设基 $\\mathcal{B}=\\{\\mathbf{b}_1=(1,2)^T,\\mathbf{b}_2=(3,5)^T\\}$，向量 $\\mathbf{x}=(7,11)^T$ 在 $\\mathcal{B}$ 下的坐标 $c_2$ 是多少？（填整数）',
        accept: ['3'],
        explain: '增广矩阵行化简得 $c_1=-2,c_2=3$。验证：$-2(1,2)^T+3(3,5)^T=(7,11)^T$ ✓。坐标向量为 $(-2,3)^T$，故 $c_2=3$。',
      },
      {
        id: 'u4-l4-q3',
        type: 'judge',
        prompt: '同一个向量空间可以有无数个不同的基，但任意两个基中向量的个数一定相同。',
        answer: true,
        explain: '基的非唯一性与维数的唯一性是两回事。由 Steinitz 替换引理可证：任意两个基的元素个数相同，这个共同的数量定义为维数 $\\dim(V)$，是良定义的。',
      },
      {
        id: 'u4-l4-q4',
        type: 'choice',
        prompt: '$\\mathcal{P}_3$（次数不超过 3 的实多项式）的维数是多少？',
        options: ['$3$', '$4$', '$2$', '无穷'],
        answer: 1,
        explain: '$\\mathcal{P}_3$ 的标准基是 $\\{1,x,x^2,x^3\\}$，共 4 个元素，故 $\\dim(\\mathcal{P}_3)=4=3+1$。一般地，$\\dim(\\mathcal{P}_n)=n+1$。',
      },
      {
        id: 'u4-l4-q5',
        type: 'match',
        prompt: '将各向量空间与其维数配对：',
        left: [
          '$\\mathbb{R}^7$',
          '$\\mathbb{R}^{3\\times 4}$（$3\\times4$ 矩阵）',
          '$\\mathcal{P}_4$（次数 $\\leq4$ 多项式）',
          '$\\{\\mathbf{0}\\}$（零空间）',
        ],
        right: ['$7$', '$12$', '$5$', '$0$'],
        explain: '$\\dim(\\mathbb{R}^n)=n=7$；$\\dim(\\mathbb{R}^{m\\times n})=mn=12$；$\\dim(\\mathcal{P}_4)=4+1=5$；$\\{\\mathbf{0}\\}$ 的基是空集，维数为 $0$。',
      },
      {
        id: 'u4-l4-q6',
        type: 'judge',
        prompt: '求矩阵列空间的基时，可以直接取行化简后行阶梯形的主元列作为列空间的基向量。',
        answer: false,
        explain: '列空间的基必须取**原矩阵**的主元列，不能取行阶梯形的列。行化简只用于定位主元位置（哪几列有主元），但行变换改变了列向量本身的值。',
      },
      {
        id: 'u4-l4-q7',
        type: 'choice',
        prompt: '设基 $\\mathcal{B}=\\{\\mathbf{b}_1=(2,-1)^T,\\mathbf{b}_2=(-3,4)^T\\}$，向量 $\\mathbf{x}=(4,-5)^T$ 在基 $\\mathcal{B}$ 下的坐标向量 $[\\mathbf{x}]_{\\mathcal{B}}$ 是？',
        options: [
          '$(1/5,\\,-6/5)^T$',
          '$(-2,\\,3)^T$',
          '$(1,\\,-1)^T$',
          '$(4,\\,-5)^T$',
        ],
        answer: 0,
        explain: '解方程组：系数矩阵行列式 $=8-3=5$，逆矩阵为 $\\frac{1}{5}\\begin{pmatrix}4&3\\\\1&2\\end{pmatrix}$。$[\\mathbf{x}]_{\\mathcal{B}}=\\frac{1}{5}\\begin{pmatrix}4&3\\\\1&2\\end{pmatrix}\\begin{pmatrix}4\\\\-5\\end{pmatrix}=\\frac{1}{5}\\begin{pmatrix}1\\\\-6\\end{pmatrix}=(1/5,-6/5)^T$。验证：$\\frac{1}{5}(2,-1)^T-\\frac{6}{5}(-3,4)^T=(2/5+18/5,\\,-1/5-24/5)^T=(4,-5)^T$ ✓',
      },
    ],
  },

  // =========================================================================
  // u4-l5  维数与秩-零化度定理
  // =========================================================================
  {
    id: 'u4-l5',
    title: '维数与秩-零化度',
    subtitle: 'rank + nullity = n——输入空间的完整分解',
    intro: [
      {
        title: '维数的严格定义',
        body: '向量空间 $V$ 的**维数** $\\dim(V)$ 定义为 $V$ 的任意一个基中向量的个数。\n\n**为什么良定义？** 由 Steinitz 替换引理（交换引理）：若 $\\{\\mathbf{w}_1,\\ldots,\\mathbf{w}_m\\}$ 张成 $V$ 而 $\\{\\mathbf{u}_1,\\ldots,\\mathbf{u}_p\\}$ 线性无关，则 $p\\leq m$。用这个引理分别在两个方向上估计，可证任意两个基的元素个数相同。\n\n**特殊情形**：\n- $\\dim(\\{\\mathbf{0}\\})=0$（空集是零空间的基）\n- 无穷维空间（如 $C[a,b]$）的任何有限集都不能张成整个空间\n\n**推论**：若 $\\dim(V)=n$，则：① $V$ 中任意 $n+1$ 个向量必线性相关；② $n$ 个线性无关向量自动构成 $V$ 的基；③ 子空间 $W$ 满足 $\\dim(W)\\leq n$，等号当且仅当 $W=V$。',
        tip: '推论②最实用：在 $\\mathbb{R}^n$ 中验证一组 $n$ 个向量是否构成基，只需验证线性无关（行列式非零或秩等于 $n$），无需另外验证张成性。',
      },
      {
        title: '秩与零化度',
        body: '设 $A$ 是 $m\\times n$ 矩阵。\n\n**秩**（rank）：$\\operatorname{rank}(A)=\\dim(\\operatorname{Col}(A))=\\dim(\\operatorname{Row}(A))$。行化简后主元的个数。\n\n**零化度**（nullity）：$\\operatorname{nullity}(A)=\\dim(\\operatorname{Null}(A))$，即 $A\\mathbf{x}=\\mathbf{0}$ 的解集的维数。\n\n**重要性质**：\n- $\\operatorname{rank}(A)=\\operatorname{rank}(A^T)$（行秩等于列秩）\n- $\\operatorname{rank}(A)\\leq\\min(m,n)$（不超过行数和列数中较小的那个）\n- 行变换不改变秩，也不改变零空间',
        formula: '\\operatorname{rank}(A) = \\dim(\\operatorname{Col}(A)) = \\dim(\\operatorname{Row}(A))',
      },
      {
        title: '秩-零化度定理',
        body: '**定理**（Rank-Nullity Theorem）：对任意 $m\\times n$ 矩阵 $A$，有：',
        formula: '\\operatorname{rank}(A) + \\operatorname{nullity}(A) = n',
        tip: '这里 $n$ 是**列数**（输入空间 $\\mathbb{R}^n$ 的维数），不是行数。秩是"有效方向数"，零化度是"被压缩为零的方向数"，两者合计恰好填满输入空间 $\\mathbb{R}^n$。',
      },
      {
        title: '定理的几何含义',
        body: '秩-零化度定理揭示了矩阵对输入空间的"分割"：\n\n- **行空间**（$r$ 维）：这个方向上的输入被 $A$ 有效映射到输出，携带信息。\n- **零空间**（$n-r$ 维）：这个方向上的输入被 $A$ 压缩为零，信息丢失。\n\n两部分维数之和恰为 $n$——输入空间被完整分解，没有遗漏，没有重叠（它们互相正交）：\n\n$\\mathbb{R}^n = \\operatorname{Row}(A) \\oplus \\operatorname{Null}(A)$（正交直和）\n\n| 矩阵类型 | 秩 | 零化度 | 解的情形 |\n|----------|----|---------|---------|\n| 列满秩 $r=n$ | $n$ | $0$ | $A\\mathbf{x}=\\mathbf{0}$ 仅有零解 |\n| 行满秩 $r=m$ | $m$ | $n-m$ | $A\\mathbf{x}=\\mathbf{b}$ 对所有 $\\mathbf{b}$ 有解 |\n| 满秩方阵 $r=m=n$ | $n$ | $0$ | $A\\mathbf{x}=\\mathbf{b}$ 有唯一解 |',
      },
      {
        title: '应用：快速求子空间维数',
        body: '**不需要真正求基**，用秩-零化度定理直接得维数：\n\n**例**：$A=\\begin{pmatrix}1&2&1\\\\2&4&2\\\\3&6&3\\end{pmatrix}$，三行均成比例，$\\operatorname{rank}(A)=1$（不做完整行化简也能看出），故：\n- $\\dim(\\operatorname{Null}(A))=3-1=2$（秩-零化度，$n=3$ 列）\n- $\\dim(\\operatorname{Null}(A^T))=3-1=2$（对 $A^T$ 用，$m=3$ 行）\n\n**LoRA 应用**：LoRA 的更新量 $\\Delta W=BA$（$B\\in\\mathbb{R}^{d\\times r}$，$A\\in\\mathbb{R}^{r\\times d}$），由秩的乘积不等式 $\\operatorname{rank}(BA)\\leq\\min(r,r)=r$，因此 $\\Delta W$ 的列空间维数最多 $r$——LoRA 只在 $r$ 维子空间内微调，这正是参数高效的原因。',
        reveal: {
          q: '$A$ 是 $5\\times7$ 矩阵，$\\operatorname{rank}(A)=3$。$\\operatorname{Null}(A)$ 的维数是多少？$\\operatorname{Null}(A^T)$ 的维数是多少？',
          a: '$\\operatorname{nullity}(A)=7-3=4$（$n=7$ 列）。$\\dim(\\operatorname{Null}(A^T))=m-r=5-3=2$（$A^T$ 是 $7\\times5$ 矩阵，列数为 $m=5$）。',
        },
      },
      {
        title: '秩的常用不等式',
        body: '设 $A\\in\\mathbb{R}^{m\\times n}$，$B\\in\\mathbb{R}^{n\\times p}$：\n\n- **乘积秩上界**：$\\operatorname{rank}(AB)\\leq\\min(\\operatorname{rank}(A),\\operatorname{rank}(B))$\n- **Sylvester 下界**：$\\operatorname{rank}(AB)\\geq\\operatorname{rank}(A)+\\operatorname{rank}(B)-n$\n- **次可加性**：$\\operatorname{rank}(A+B)\\leq\\operatorname{rank}(A)+\\operatorname{rank}(B)$\n- **转置保秩**：$\\operatorname{rank}(A^T A)=\\operatorname{rank}(A)$\n\n**例**：$A$ 是 $3\\times5$ 矩阵，$\\operatorname{rank}(A)=3$；$B$ 是 $5\\times4$ 矩阵，$\\operatorname{rank}(B)=4$。则 $\\operatorname{rank}(AB)$ 的范围：上界 $\\min(3,4)=3$，Sylvester 下界 $3+4-5=2$，故 $2\\leq\\operatorname{rank}(AB)\\leq3$。',
        formula: '\\operatorname{rank}(AB) \\leq \\min(\\operatorname{rank}(A),\\, \\operatorname{rank}(B))',
      },
    ],
    questions: [
      {
        id: 'u4-l5-q1',
        type: 'judge',
        prompt: '$A$ 是 $3\\times5$ 矩阵，$\\operatorname{rank}(A)=3$，则 $\\operatorname{nullity}(A)=2$。',
        answer: true,
        explain: '由秩-零化度定理（$n=5$ 列）：$\\operatorname{nullity}(A)=5-3=2$。零空间是 $\\mathbb{R}^5$ 中的 2 维子空间，方程 $A\\mathbf{x}=\\mathbf{0}$ 的通解含 2 个自由变量。',
      },
      {
        id: 'u4-l5-q2',
        type: 'choice',
        prompt: '矩阵 $A=\\begin{pmatrix}1&2&3\\\\2&4&6\\\\1&2&3\\end{pmatrix}$ 的 $\\operatorname{nullity}(A)$ 是多少？',
        options: ['$0$', '$1$', '$2$', '$3$'],
        answer: 2,
        explain: '三行均成比例（行2=2行1，行3=行1），$\\operatorname{rank}(A)=1$。由秩-零化度定理（$n=3$ 列）：$\\operatorname{nullity}(A)=3-1=2$。',
      },
      {
        id: 'u4-l5-q3',
        type: 'input',
        prompt: '$A$ 是 $4\\times6$ 矩阵，$\\operatorname{rank}(A)=4$，则 $\\operatorname{nullity}(A)$ 等于多少？（填整数）',
        accept: ['2'],
        explain: '由秩-零化度定理（$n=6$ 列）：$\\operatorname{nullity}(A)=6-4=2$。',
      },
      {
        id: 'u4-l5-q4',
        type: 'choice',
        prompt: '$\\mathbb{R}^n$ 中 $n$ 个向量构成 $\\mathbb{R}^n$ 的基，以下哪个条件**单独**就能判断？',
        options: [
          '这 $n$ 个向量张成 $\\mathbb{R}^n$（无需验证线性无关）',
          '这 $n$ 个向量线性无关（无需验证张成）',
          '这 $n$ 个向量两两不同（不成比例）',
          '两个条件都必须分别验证，缺一不可',
        ],
        answer: 1,
        explain: '在 $\\mathbb{R}^n$ 中，$n$ 个向量线性无关自动蕴含张成（由维数推论：$n$ 维空间中 $n$ 个无关向量构成基）。因此只需验证线性无关（等价于行列式非零或秩等于 $n$）。',
      },
      {
        id: 'u4-l5-q5',
        type: 'match',
        prompt: '将各矩阵情形与 $\\operatorname{nullity}$ 配对（$n$ 为列数）：',
        left: [
          '$3\\times4$ 矩阵，$\\operatorname{rank}=3$',
          '$2\\times5$ 矩阵，$\\operatorname{rank}=2$',
          '$4\\times4$ 满秩方阵',
          '$3\\times3$ 矩阵，$\\operatorname{rank}=1$',
        ],
        right: ['$1$', '$3$', '$0$', '$2$'],
        explain: '秩-零化度定理 $\\operatorname{nullity}=n-\\operatorname{rank}$（$n$ 为列数）：$4-3=1$；$5-2=3$；$4-4=0$；$3-1=2$。',
      },
      {
        id: 'u4-l5-q6',
        type: 'judge',
        prompt: '若 $A$ 是列满秩矩阵（$\\operatorname{rank}(A)=n$，列数），则 $A\\mathbf{x}=\\mathbf{0}$ 只有零解。',
        answer: true,
        explain: '列满秩时 $\\operatorname{nullity}(A)=n-n=0$，零空间只含零向量，即 $\\operatorname{Null}(A)=\\{\\mathbf{0}\\}$，故 $A\\mathbf{x}=\\mathbf{0}$ 的唯一解是 $\\mathbf{x}=\\mathbf{0}$。',
      },
      {
        id: 'u4-l5-q7',
        type: 'choice',
        prompt: 'LoRA 中更新量 $\\Delta W=BA$，其中 $B\\in\\mathbb{R}^{512\\times8}$，$A\\in\\mathbb{R}^{8\\times512}$。$\\operatorname{rank}(\\Delta W)$ 最大是多少？',
        options: ['$512$', '$256$', '$8$', '$4$'],
        answer: 2,
        explain: '由秩的乘积不等式：$\\operatorname{rank}(\\Delta W)=\\operatorname{rank}(BA)\\leq\\min(\\operatorname{rank}(B),\\operatorname{rank}(A))\\leq\\min(8,8)=8$。故 $\\Delta W$ 的秩最大为 $8$（LoRA 的 $r$），列空间是 $\\mathbb{R}^{512}$ 中最多 8 维的子空间。',
      },
    ],
  },

  // =========================================================================
  // u4-l6  四大基本子空间
  // =========================================================================
  {
    id: 'u4-l6',
    title: '四大基本子空间',
    subtitle: '列空间、行空间、零空间、左零空间——完整刻画矩阵结构',
    intro: [
      {
        title: '四大基本子空间概览',
        body: '设 $A$ 是 $m\\times n$ 矩阵，秩为 $r$。矩阵 $A$ 自然关联四个子空间，分布在两个空间中：\n\n| 子空间 | 符号 | 定义 | 所在空间 | 维数 |\n|--------|------|------|----------|------|\n| 列空间 | $\\operatorname{Col}(A)$ | $A$ 的列的线性组合 | $\\mathbb{R}^m$ | $r$ |\n| 行空间 | $\\operatorname{Row}(A)$ | $A$ 的行的线性组合 | $\\mathbb{R}^n$ | $r$ |\n| 零空间 | $\\operatorname{Null}(A)$ | $A\\mathbf{x}=\\mathbf{0}$ 的解集 | $\\mathbb{R}^n$ | $n-r$ |\n| 左零空间 | $\\operatorname{Null}(A^T)$ | $A^T\\mathbf{y}=\\mathbf{0}$ 的解集 | $\\mathbb{R}^m$ | $m-r$ |\n\n**记忆口诀**：输入空间 $\\mathbb{R}^n$ 中有行空间（$r$ 维）和零空间（$n-r$ 维）；输出空间 $\\mathbb{R}^m$ 中有列空间（$r$ 维）和左零空间（$m-r$ 维）。',
      },
      {
        title: '列空间与行空间',
        body: '**列空间** $\\operatorname{Col}(A)=\\{A\\mathbf{x}\\mid\\mathbf{x}\\in\\mathbb{R}^n\\}$：$A$ 作为线性映射的**像**——所有可能的输出。\n\n**几何意义**：$A\\mathbf{x}=\\mathbf{b}$ 有解 $\\Leftrightarrow$ $\\mathbf{b}\\in\\operatorname{Col}(A)$。\n\n**求列空间的基**：行化简 $A$，主元在哪几列，就取**原矩阵** $A$ 的那几列作为基。\n\n**行空间** $\\operatorname{Row}(A)=\\operatorname{Col}(A^T)$：$A$ 的行向量的线性组合，属于输入空间 $\\mathbb{R}^n$。\n\n**求行空间的基**：行化简 $A$，取**行阶梯形**（或 RREF）的非零行（视为行向量），它们构成行空间的基。\n\n**重要性质**：行变换不改变行空间（把行替换为行的线性组合，张成不变）；行变换也不改变列之间的线性相关关系（因此零空间不变，从而列空间的"结构"信息保留）。',
        tip: '列秩 $=$ 行秩 $= r$：这不是巧合，而是深刻的定理。行化简让两侧的主元数量一致。',
      },
      {
        title: '零空间与解的结构',
        body: '**零空间** $\\operatorname{Null}(A)=\\{\\mathbf{x}\\in\\mathbb{R}^n\\mid A\\mathbf{x}=\\mathbf{0}\\}$：被 $A$ "压缩"成零的所有输入方向。\n\n**它是子空间**（验证三步）：$A\\mathbf{0}=\\mathbf{0}$ ✓；若 $A\\mathbf{u}=A\\mathbf{v}=\\mathbf{0}$ 则 $A(\\mathbf{u}+\\mathbf{v})=\\mathbf{0}$ ✓；若 $A\\mathbf{v}=\\mathbf{0}$ 则 $A(c\\mathbf{v})=c\\mathbf{0}=\\mathbf{0}$ ✓。\n\n**求零空间的基**：行化简 $A$ 到 RREF，对每个自由变量令其等于 $1$、其余自由变量等于 $0$，回代求出其余分量，得到一个基向量。自由变量个数 $=$ 零空间维数 $=n-r$。\n\n**解的结构定理**：若 $A\\mathbf{x}=\\mathbf{b}$ 有特解 $\\mathbf{x}_p$，则通解为\n$\\mathbf{x}=\\mathbf{x}_p+\\mathbf{x}_h,\\quad \\mathbf{x}_h\\in\\operatorname{Null}(A)$\n零空间决定通解的"形状"，特解是"偏移"。',
        formula: '\\mathbf{x} = \\mathbf{x}_p + \\mathbf{x}_h, \\quad \\mathbf{x}_h \\in \\operatorname{Null}(A)',
      },
      {
        title: '左零空间',
        body: '**左零空间** $\\operatorname{Null}(A^T)=\\{\\mathbf{y}\\in\\mathbb{R}^m\\mid A^T\\mathbf{y}=\\mathbf{0}\\}$，等价地，$\\mathbf{y}^T A=\\mathbf{0}^T$（向量从左边乘 $A$，故名"左"零空间）。\n\n**所在空间**：$\\operatorname{Null}(A^T)\\subseteq\\mathbb{R}^m$（输出空间）。\n\n**维数**：$m-r$（对 $A^T$ 用秩-零化度定理，$A^T$ 的列数为 $m$）。\n\n**几何意义**：左零空间是列空间的正交补——$\\mathbf{y}\\in\\operatorname{Null}(A^T)$ 当且仅当 $\\mathbf{y}$ 与 $\\operatorname{Col}(A)$ 中的每个向量垂直。\n\n**求法**：对 $A^T$ 行化简求零空间；或在 $[A\\mid I_m]$ 行化简到 $[R\\mid E]$ 后，$R$ 的零行对应的 $E$ 的行就是左零空间的基（Strang 方法）。',
        tip: '维数验证：$\\dim(\\operatorname{Col})+\\dim(\\operatorname{Null}(A^T))=r+(m-r)=m$；$\\dim(\\operatorname{Row})+\\dim(\\operatorname{Null})=r+(n-r)=n$。两对互为正交补。',
      },
      {
        title: '正交关系与直和分解',
        body: '四个子空间以两对正交关系配对：\n\n$\\operatorname{Row}(A)\\perp\\operatorname{Null}(A)$（均在 $\\mathbb{R}^n$ 中）\n$\\operatorname{Col}(A)\\perp\\operatorname{Null}(A^T)$（均在 $\\mathbb{R}^m$ 中）\n\n进一步，两对构成**正交直和**分解：\n\n$\\mathbb{R}^n = \\operatorname{Row}(A)\\oplus\\operatorname{Null}(A)$（行空间与零空间正交直和）\n$\\mathbb{R}^m = \\operatorname{Col}(A)\\oplus\\operatorname{Null}(A^T)$（列空间与左零空间正交直和）\n\n**直觉**：任何输入向量 $\\mathbf{x}\\in\\mathbb{R}^n$ 可唯一分解为行空间分量（被 $A$ 有效映射）和零空间分量（被 $A$ 压缩为零）。这正是解的结构定理的几何核心。',
        formula: '\\mathbb{R}^n = \\operatorname{Row}(A) \\oplus \\operatorname{Null}(A),\\quad \\mathbb{R}^m = \\operatorname{Col}(A) \\oplus \\operatorname{Null}(A^T)',
      },
      {
        title: '例题：完整计算四大子空间',
        body: '**题目**：$A=\\begin{pmatrix}1&2&1\\\\0&1&2\\\\1&3&3\\end{pmatrix}$，求四大基本子空间的基和维数。',
        steps: [
          '行化简：$\\begin{pmatrix}1&2&1\\\\0&1&2\\\\1&3&3\\end{pmatrix}\\xrightarrow{R_3-R_1}\\begin{pmatrix}1&2&1\\\\0&1&2\\\\0&1&2\\end{pmatrix}\\xrightarrow{R_3-R_2}\\begin{pmatrix}1&2&1\\\\0&1&2\\\\0&0&0\\end{pmatrix}$，$r=2$。',
          '列空间：主元在第 1、2 列，取原矩阵第 1、2 列，基为 $\\{(1,0,1)^T,(2,1,3)^T\\}$，$\\dim=2$，在 $\\mathbb{R}^3$（行数 $m=3$）中。',
          '行空间：取行阶梯形非零行，基为 $\\{(1,2,1)^T,(0,1,2)^T\\}$，$\\dim=2$，在 $\\mathbb{R}^3$（列数 $n=3$）中。',
          '零空间：$n-r=1$，自由变量 $x_3=t$，回代得 $x_2=-2t,x_1=3t$，基为 $\\{(3,-2,1)^T\\}$，$\\dim=1$。',
          '左零空间：$m-r=1$，解 $A^T\\mathbf{y}=\\mathbf{0}$，得基为 $\\{(-1,-1,1)^T\\}$，$\\dim=1$。',
          '验证：$\\dim(\\operatorname{Col})+\\dim(\\operatorname{Null}(A^T))=2+1=3=m$ ✓；$\\dim(\\operatorname{Row})+\\dim(\\operatorname{Null})=2+1=3=n$ ✓。',
        ],
      },
    ],
    questions: [
      {
        id: 'u4-l6-q1',
        type: 'choice',
        prompt: '设 $A$ 是 $5\\times7$ 矩阵，$\\operatorname{rank}(A)=3$。下列维数中哪个是**正确**的？',
        options: [
          '$\\dim(\\operatorname{Col}(A))=5$，$\\dim(\\operatorname{Null}(A))=4$',
          '$\\dim(\\operatorname{Col}(A))=3$，$\\dim(\\operatorname{Null}(A))=4$',
          '$\\dim(\\operatorname{Col}(A))=3$，$\\dim(\\operatorname{Null}(A))=2$',
          '$\\dim(\\operatorname{Row}(A))=5$，$\\dim(\\operatorname{Null}(A^T))=2$',
        ],
        answer: 1,
        explain: '$\\dim(\\operatorname{Col})=r=3$，在 $\\mathbb{R}^5$ 中；$\\operatorname{nullity}(A)=n-r=7-3=4$，在 $\\mathbb{R}^7$ 中；$\\dim(\\operatorname{Null}(A^T))=m-r=5-3=2$，在 $\\mathbb{R}^5$ 中。',
      },
      {
        id: 'u4-l6-q2',
        type: 'judge',
        prompt: '$A\\mathbf{x}=\\mathbf{b}$ 有解的充要条件是 $\\mathbf{b}\\in\\operatorname{Col}(A)$。',
        answer: true,
        explain: '列空间的定义：$\\operatorname{Col}(A)=\\{A\\mathbf{x}\\mid\\mathbf{x}\\in\\mathbb{R}^n\\}$，即所有可能的输出。故 $A\\mathbf{x}=\\mathbf{b}$ 有解当且仅当 $\\mathbf{b}$ 属于列空间。',
      },
      {
        id: 'u4-l6-q3',
        type: 'input',
        prompt: '设 $A$ 是 $4\\times5$ 矩阵，$\\operatorname{rank}(A)=3$，则 $\\dim(\\operatorname{Null}(A^T))$ 等于多少？（填整数）',
        accept: ['1'],
        explain: '$\\operatorname{Null}(A^T)\\subseteq\\mathbb{R}^m=\\mathbb{R}^4$，维数为 $m-r=4-3=1$。',
      },
      {
        id: 'u4-l6-q4',
        type: 'choice',
        prompt: '矩阵 $A=\\begin{pmatrix}1&0&2\\\\0&1&-1\\\\2&1&3\\end{pmatrix}$ 行化简后主元在第 1、2 列，则列空间的基为？',
        options: [
          '行阶梯形的第 1、2 列',
          '原矩阵 $A$ 的第 1、2 列：$(1,0,2)^T$ 和 $(0,1,1)^T$',
          '原矩阵 $A$ 的第 1、3 列',
          'RREF 的非零行',
        ],
        answer: 1,
        explain: '列空间的基取**原矩阵**的主元列（第 1、2 列），即 $(1,0,2)^T$ 和 $(0,1,1)^T$，维数为 $2$。行化简仅用于定位主元位置。',
      },
      {
        id: 'u4-l6-q5',
        type: 'match',
        prompt: '将各子空间与其所在空间配对（$A$ 是 $m\\times n$ 矩阵）：',
        left: [
          '$\\operatorname{Col}(A)$',
          '$\\operatorname{Row}(A)$',
          '$\\operatorname{Null}(A)$',
          '$\\operatorname{Null}(A^T)$',
        ],
        right: [
          '$\\mathbb{R}^m$（输出空间）',
          '$\\mathbb{R}^n$（输入空间）',
          '$\\mathbb{R}^n$（输入空间）',
          '$\\mathbb{R}^m$（输出空间）',
        ],
        explain: '列空间和左零空间都在输出空间 $\\mathbb{R}^m$ 中（一对正交补）；行空间和零空间都在输入空间 $\\mathbb{R}^n$ 中（另一对正交补）。',
      },
      {
        id: 'u4-l6-q6',
        type: 'judge',
        prompt: '行变换会改变矩阵的零空间。',
        answer: false,
        explain: '行变换（初等行操作）不改变零空间：$A\\mathbf{x}=\\mathbf{0}$ 与经行变换得到的 $R\\mathbf{x}=\\mathbf{0}$ 具有完全相同的解集。正因如此，才能通过行化简来求零空间的基。',
      },
      {
        id: 'u4-l6-q7',
        type: 'choice',
        prompt: 'LoRA 中 $\\Delta W=BA$（$B\\in\\mathbb{R}^{d\\times r}$，$A\\in\\mathbb{R}^{r\\times d}$，$r\\ll d$），$\\operatorname{Col}(\\Delta W)$ 的维数最大为多少？它是 $\\mathbb{R}^d$ 中什么维度的子空间？',
        options: [
          '维数最大为 $d$，整个 $\\mathbb{R}^d$',
          '维数最大为 $r$，$\\mathbb{R}^d$ 中 $r$ 维子空间',
          '维数最大为 $2r$，$\\mathbb{R}^d$ 中 $2r$ 维子空间',
          '维数最大为 $d/2$',
        ],
        answer: 1,
        explain: '$\\operatorname{Col}(\\Delta W)=\\operatorname{Col}(BA)\\subseteq\\operatorname{Col}(B)$，而 $B\\in\\mathbb{R}^{d\\times r}$ 故 $\\dim(\\operatorname{Col}(B))\\leq r$。因此 $\\Delta W$ 的列空间维数最大为 $r$——LoRA 微调时只在 $\\mathbb{R}^d$ 中 $r$ 维的低维子空间内更新权重，极大压缩了参数量。',
      },
    ],
  },
]
