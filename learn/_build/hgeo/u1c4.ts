export const LESSONS = [
  {
    id: 'u1c4',
    title: '基底与坐标表示',
    subtitle: '平面向量基本定理——给每个向量一个"身份证"',
    intro: [
      {
        title: '引入：用基底做运算',
        body: '已知 $\\vec{e}_1, \\vec{e}_2$ 不共线，$\\vec{a} = 2\\vec{e}_1 - 3\\vec{e}_2$，$\\vec{b} = \\vec{e}_1 + 4\\vec{e}_2$，求 $\\vec{a}+\\vec{b}$ 和 $2\\vec{a}-\\vec{b}$。\n\n**关键感悟**：$\\vec{e}_1, \\vec{e}_2$ 就像"$x$"和"$y$"——它们只是承载系数的载体，本身不参与数运算，只等着被收集。\n\n$$\\vec{a}+\\vec{b} = (2+1)\\vec{e}_1 + (-3+4)\\vec{e}_2 = 3\\vec{e}_1+\\vec{e}_2$$\n$$2\\vec{a}-\\vec{b} = (4-1)\\vec{e}_1+(-6-4)\\vec{e}_2 = 3\\vec{e}_1-10\\vec{e}_2$$\n\n这正是**坐标**的本质：把向量运算转化为系数的数值运算。',
        tip: '基底运算节奏：识别基底 → 按运算律展开 → 分别合并各基底前的系数 → 写结果。',
      },
      {
        title: '平面向量基本定理',
        body: '**定理**：如果 $\\vec{e}_1, \\vec{e}_2$ 是两个**不共线**的向量，那么对平面内**任意**向量 $\\vec{a}$，存在**唯一**一对实数 $\\lambda_1, \\lambda_2$，使得：\n\n$$\\vec{a} = \\lambda_1\\vec{e}_1 + \\lambda_2\\vec{e}_2$$\n\n这样的 $\\vec{e}_1, \\vec{e}_2$ 称为该平面的一组**基底**（basis）。\n\n**三个关键词**：\n- **不共线**：基底要"张成"整个平面，共线的两向量只能描述一条线\n- **任意**：平面内每个向量都能分解，没有遗漏\n- **唯一**：分解方式只有一种，$\\lambda_1, \\lambda_2$ 确定',
        formula: '\\vec{a} = \\lambda_1\\vec{e}_1 + \\lambda_2\\vec{e}_2',
        image: 'geo-p1-04-1.svg',
        imageCaption: '平行四边形分解示意：任意向量 $\\vec{a}$ 在基底 $\\vec{e}_1, \\vec{e}_2$ 下的唯一分解',
        tip: '若 $\\vec{e}_1 \\parallel \\vec{e}_2$（共线），则它们无法"张开"平面，只能描述一条直线上的向量，失去了基底的作用。',
      },
      {
        title: '为什么要不共线——严格推导',
        body: '若 $\\vec{e}_1$ 与 $\\vec{e}_2$ 共线（比如 $\\vec{e}_2 = 2\\vec{e}_1$），则：\n\n$$\\lambda_1\\vec{e}_1 + \\lambda_2\\vec{e}_2 = \\lambda_1\\vec{e}_1 + 2\\lambda_2\\vec{e}_1 = (\\lambda_1 + 2\\lambda_2)\\vec{e}_1$$\n\n结果永远在 $\\vec{e}_1$ 所在直线上，无法表示"平面内所有方向"的向量。\n\n更严重的问题：给定和向量 $\\vec{a}$（若恰好在该直线上），满足 $(\\lambda_1+2\\lambda_2)\\vec{e}_1 = \\vec{a}$ 的 $\\lambda_1, \\lambda_2$ 有**无穷多组**（只要 $\\lambda_1+2\\lambda_2$ 固定即可），失去唯一性。',
        steps: [
          '第一步：确认两个基底向量不共线（方向不平行）',
          '第二步：任意平面向量都能用这两个基底唯一线性表示',
          '第三步：通过合并同类项（按基底）求出系数 $\\lambda_1, \\lambda_2$',
        ],
      },
      {
        title: '正交基底与坐标',
        body: '当基底满足以下两个条件时，称为**正交（单位）基底**：\n① $\\vec{e}_1 \\perp \\vec{e}_2$（互相垂直）\n② $|\\vec{e}_1| = |\\vec{e}_2| = 1$（均为单位向量）\n\n通常记 $\\vec{e}_1 = \\vec{i} = (1,0)$，$\\vec{e}_2 = \\vec{j} = (0,1)$（直角坐标系的两个方向）。\n\n此时 $\\vec{a} = x\\vec{i}+y\\vec{j}$ 简记为 $\\vec{a} = (x, y)$。\n$x$ 是 $\\vec{a}$ 在 $x$ 轴方向的分量，$y$ 是 $y$ 轴方向的分量。',
        formula: '\\vec{a} = x\\vec{i}+y\\vec{j} \\longleftrightarrow \\vec{a}=(x,y)',
        image: 'geo-p1-04-2.svg',
        imageCaption: '正交基底坐标表示：向量 $(x, y)$ 在直角坐标系中的分解',
      },
      {
        title: '坐标的四则运算',
        body: '设 $\\vec{a} = (x_1, y_1)$，$\\vec{b} = (x_2, y_2)$，$\\lambda \\in \\mathbb{R}$：\n\n- **加法**：$\\vec{a}+\\vec{b} = (x_1+x_2,\\; y_1+y_2)$\n- **减法**：$\\vec{a}-\\vec{b} = (x_1-x_2,\\; y_1-y_2)$\n- **数乘**：$\\lambda\\vec{a} = (\\lambda x_1,\\; \\lambda y_1)$\n- **模长**：$|\\vec{a}| = \\sqrt{x_1^2+y_1^2}$\n\n**本质**：坐标运算就是对两个分量分别做对应的实数运算，$x$ 坐标只和 $x$ 坐标运算，$y$ 坐标只和 $y$ 坐标运算，不交叉。\n\n**起点终点关系**：设 $A(x_A, y_A)$，$B(x_B, y_B)$，则 $\\overrightarrow{AB} = (x_B-x_A,\\; y_B-y_A)$（终点减起点）。',
        formula: '\\overrightarrow{AB} = (x_B-x_A,\\; y_B-y_A),\\quad |\\vec{a}| = \\sqrt{x_1^2+y_1^2}',
        tip: '$\\overrightarrow{AB}$ 是终点 $B$ 减起点 $A$，不要反过来！$\\overrightarrow{BA} = -\\overrightarrow{AB}$。',
      },
      {
        title: '坐标判定共线、中点与重心',
        body: '**共线判定**：$\\vec{a} = (x_1, y_1)$ 与 $\\vec{b} = (x_2, y_2)$ 共线 $\\Leftrightarrow$\n\n$$x_1 y_2 - x_2 y_1 = 0$$\n\n（交叉相乘之差为零，即 $2\\times2$ 行列式为零）\n\n**中点公式**：线段 $AB$ 的中点坐标为 $M = \\left(\\dfrac{x_A+x_B}{2},\\; \\dfrac{y_A+y_B}{2}\\right)$\n\n**重心公式**：三角形 $A(x_1,y_1), B(x_2,y_2), C(x_3,y_3)$ 的重心为\n$$G = \\left(\\dfrac{x_1+x_2+x_3}{3},\\; \\dfrac{y_1+y_2+y_3}{3}\\right)$$\n\n重心满足 $\\overrightarrow{OG} = \\dfrac{1}{3}(\\overrightarrow{OA}+\\overrightarrow{OB}+\\overrightarrow{OC})$，将每条中线按 $2:1$ 分割（从顶点到对边中点）。',
        formula: 'x_1 y_2 - x_2 y_1 = 0',
        tip: '中点除以 $2$，重心除以 $3$。形式相似，场景不同，不要混淆！',
      },
      {
        title: '典型例题',
        body: '**例题**：已知 $A(1,2)$，$B(3,6)$，$C(k,10)$，求使 $A, B, C$ 三点共线的 $k$ 值。\n\n**解**：$\\overrightarrow{AB} = (2, 4)$，$\\overrightarrow{AC} = (k-1, 8)$。\n\n三点共线 $\\Leftrightarrow$ $2 \\times 8 - (k-1) \\times 4 = 0$，即 $16 - 4(k-1) = 0$，$k-1 = 4$，$k = 5$。\n\n**验证**：$\\overrightarrow{AC} = (4, 8) = 2\\overrightarrow{AB}$，确实共线。\n\n**例题2**：三角形顶点 $A(2,1), B(4,7), C(-2,3)$，求重心 $G$。\n\n**解**：$G = \\left(\\dfrac{2+4+(-2)}{3}, \\dfrac{1+7+3}{3}\\right) = \\left(\\dfrac{4}{3}, \\dfrac{11}{3}\\right)$。',
        reveal: {
          q: '已知 $A(-1, 3)$，$B(5, 1)$，求 $\\overrightarrow{AB}$ 的坐标和 $|\\overrightarrow{AB}|$。',
          a: '$\\overrightarrow{AB} = (5-(-1),\\; 1-3) = (6, -2)$。$|\\overrightarrow{AB}| = \\sqrt{6^2+(-2)^2} = \\sqrt{36+4} = \\sqrt{40} = 2\\sqrt{10}$。',
        },
      },
      {
        title: '易错点汇总',
        body: '**易错1**：基底共线时强行分解——必须先确认 $\\vec{e}_1, \\vec{e}_2$ 不共线，否则分解不唯一甚至不可能。\n\n**易错2**：$\\overrightarrow{AB}$ 坐标写反——是终点 $B$ 减起点 $A$，不是 $A$ 减 $B$。\n\n**易错3**：模长公式漏根号——$|\\vec{a}| = \\sqrt{x^2+y^2}$，不是 $x^2+y^2$，更不是 $|x|+|y|$（后者是出租车距离）。\n\n**易错4**：共线公式符号出错——$x_1 y_2 - x_2 y_1 = 0$ 是交叉相乘之差，不是 $x_1 y_1 - x_2 y_2$。\n\n**易错5**：重心公式误用为中点公式——中点除以 $2$（两个顶点），重心除以 $3$（三个顶点）。',
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '平面向量基本定理要求基底 $\\vec{e}_1, \\vec{e}_2$ 满足的条件是（　　）',
        options: [
          '互相垂直',
          '模均为 $1$',
          '不共线（不平行）',
          '互相垂直且模均为 $1$',
        ],
        answer: 2,
        explain: '选 C。平面向量基本定理只要求 $\\vec{e}_1$ 与 $\\vec{e}_2$ **不共线**（不平行），不要求垂直或模为 $1$。正交单位基底是满足额外条件的特殊情形，此时退化为直角坐标表示。',
      },
      {
        type: 'input',
        prompt: '已知 $\\vec{e}_1, \\vec{e}_2$ 不共线，$\\vec{m} = 5\\vec{e}_1-2\\vec{e}_2$，$\\vec{n} = -\\vec{e}_1+3\\vec{e}_2$，求 $3\\vec{m}+2\\vec{n}$ 中 $\\vec{e}_1$ 的系数。',
        accept: ['13'],
        explain: '$3\\vec{m}+2\\vec{n} = 3(5\\vec{e}_1-2\\vec{e}_2)+2(-\\vec{e}_1+3\\vec{e}_2) = (15-2)\\vec{e}_1+(-6+6)\\vec{e}_2 = 13\\vec{e}_1$。$\\vec{e}_1$ 的系数为 $13$，$\\vec{e}_2$ 的系数为 $0$（消掉了）。',
      },
      {
        type: 'choice',
        prompt: '已知 $A(-1, 3)$，$B(5, 1)$，则 $|\\overrightarrow{AB}|$ 等于（　　）',
        options: [
          '$\\sqrt{8}$',
          '$2\\sqrt{10}$',
          '$\\sqrt{52}$',
          '$8$',
        ],
        answer: 1,
        explain: '选 B。$\\overrightarrow{AB} = (5-(-1),\\; 1-3) = (6,-2)$。$|\\overrightarrow{AB}| = \\sqrt{6^2+(-2)^2} = \\sqrt{36+4} = \\sqrt{40} = 2\\sqrt{10}$。',
      },
      {
        type: 'judge',
        prompt: '若 $\\vec{a} = (3,-6)$，$\\vec{b} = (-2, 4)$，则 $\\vec{a}$ 与 $\\vec{b}$ 共线。',
        answer: true,
        explain: '正确。$x_1 y_2 - x_2 y_1 = 3 \\times 4 - (-2) \\times (-6) = 12 - 12 = 0$，交叉积为零，故共线。也可直接验证：$\\vec{b} = -\\dfrac{2}{3}\\vec{a}$，是数乘关系。',
      },
      {
        type: 'choice',
        prompt: '三角形顶点 $A(2,1)$，$B(4,7)$，$C(-2,3)$，重心 $G$ 的坐标是（　　）',
        options: [
          '$(1, 4)$',
          '$\\left(\\dfrac{4}{3}, \\dfrac{11}{3}\\right)$',
          '$(2, \\dfrac{11}{3})$',
          '$(3, 4)$',
        ],
        answer: 1,
        explain: '选 B。重心公式：$G = \\left(\\dfrac{x_1+x_2+x_3}{3}, \\dfrac{y_1+y_2+y_3}{3}\\right) = \\left(\\dfrac{2+4+(-2)}{3}, \\dfrac{1+7+3}{3}\\right) = \\left(\\dfrac{4}{3}, \\dfrac{11}{3}\\right)$。',
      },
      {
        type: 'input',
        prompt: '已知 $M(1,4)$ 是 $AB$ 的中点，$A(a, 2)$，$B(3, b)$，求 $a+b$ 的值。',
        accept: ['5'],
        explain: '由中点公式：$\\dfrac{a+3}{2} = 1 \\Rightarrow a = -1$；$\\dfrac{2+b}{2} = 4 \\Rightarrow b = 6$。故 $a+b = -1+6 = 5$。',
      },
      {
        type: 'match',
        prompt: '将左侧工具与右侧对应用途配对',
        left: [
          '$\\overrightarrow{AB} = (x_B-x_A,\\; y_B-y_A)$',
          '$x_1 y_2 - x_2 y_1 = 0$',
          '$G = \\left(\\dfrac{x_1+x_2+x_3}{3},\\; \\dfrac{y_1+y_2+y_3}{3}\\right)$',
        ],
        right: [
          '求三角形三个顶点确定的重心坐标',
          '由两点坐标求有向线段的坐标表示',
          '判断两向量（或三点）是否共线',
        ],
        explain: '$\\overrightarrow{AB} = (x_B-x_A, y_B-y_A)$ 是终点减起点，用于求有向线段坐标。$x_1 y_2-x_2 y_1=0$ 是共线的坐标判定（叉积为零）。重心公式是三顶点坐标各取平均。',
      },
      {
        type: 'choice',
        prompt: '已知 $A(1,2)$，$B(3,6)$，$C(k,10)$ 三点共线，则 $k$ 等于（　　）',
        options: ['$4$', '$5$', '$6$', '$7$'],
        answer: 1,
        explain: '选 B。$\\overrightarrow{AB} = (2,4)$，$\\overrightarrow{AC} = (k-1, 8)$。共线条件：$2 \\times 8 - (k-1) \\times 4 = 0$，即 $16-4(k-1) = 0$，$k-1 = 4$，$k = 5$。',
      },
    ],
  },
]
