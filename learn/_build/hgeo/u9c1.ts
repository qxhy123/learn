export const LESSONS = [
  {
    id: 'u9c1',
    title: '空间向量的基本概念与运算',
    subtitle: '从平面走向三维，掌握共线与共面定理',
    intro: [
      {
        title: '为什么要学空间向量？',
        body: '平面向量只能描述二维运动——飞机位移、平面上的力。但现实世界是三维的：火箭轨迹、卫星姿态、建筑应力都需要三维工具。空间向量将平面向量的概念和运算全面推广到三维空间，同时保留"自由向量"的核心性质：向量不固定在某个起点，只由大小和方向决定，可以自由平移。',
      },
      {
        title: '空间向量的定义',
        body: '空间向量是空间中既有大小又有方向的量。几何上用有向线段 $\\vec{AB}$ 表示，起点 $A$，终点 $B$，箭头指向 $B$。向量的模（大小）记作 $|\\vec{AB}|$ 或 $|\\vec{a}|$。相同大小、相同方向的向量，无论起点在何处，都视为相等向量（自由性）。零向量 $\\vec{0}$ 的模为 $0$，方向任意；单位向量满足 $|\\vec{e}|=1$。',
      },
      {
        title: '平面向量 vs 空间向量',
        body: '平面向量在 2 维空间中，坐标为 $(x,y)$，用 2 个不共线向量就能构成基底。空间向量在 3 维空间中，坐标为 $(x,y,z)$，需要 3 个不共面的向量才能构成基底，张成整个空间。关键差异：在平面中任意两向量都共面；在空间中必须额外验证三个向量是否共面。',
      },
      {
        title: '加法：三角形法则与平行六面体法则',
        body: '三角形法则（首尾相接）与平面情形完全相同：$\\vec{AB}+\\vec{BC}=\\vec{AC}$，封闭折线向量和为零。\n\n三个共起点向量在空间中用平行六面体法则：以 $\\vec{a},\\vec{b},\\vec{c}$ 为三条棱，先用平行四边形法则得 $\\vec{a}+\\vec{b}$，再与 $\\vec{c}$ 用三角形法则合并，结果是从公共起点穿过六面体中心到对顶点的体对角线向量。',
        formula: '\\vec{a}+\\vec{b}+\\vec{c} = \\overrightarrow{AG}\\;(\\text{body diagonal})',
        image: 'geo-p9-01-1.svg',
        imageCaption: '平行六面体法则示意：三个共起点向量之和为体对角线',
      },
      {
        title: '减法与数乘',
        body: '减法：$\\vec{a}-\\vec{b}=\\vec{a}+(-\\vec{b})$。共起点几何意义：$\\vec{OA}-\\vec{OB}=\\vec{BA}$，即从减数的终点指向被减数的终点。\n\n数乘 $\\lambda\\vec{a}$：模为 $|\\lambda|\\cdot|\\vec{a}|$；$\\lambda>0$ 时同向，$\\lambda<0$ 时反向，$\\lambda=0$ 时得零向量。运算律：$\\lambda(\\mu\\vec{a})=(\\lambda\\mu)\\vec{a}$，$(\\lambda+\\mu)\\vec{a}=\\lambda\\vec{a}+\\mu\\vec{a}$，$\\lambda(\\vec{a}+\\vec{b})=\\lambda\\vec{a}+\\lambda\\vec{b}$。',
        formula: '|\\lambda\\vec{a}| = |\\lambda|\\cdot|\\vec{a}|',
      },
      {
        title: '共线向量定理',
        body: '定理：设 $\\vec{b}\\neq\\vec{0}$，则 $\\vec{a}\\parallel\\vec{b}$ 当且仅当存在唯一实数 $\\lambda$，使得 $\\vec{a}=\\lambda\\vec{b}$。\n\n证明思路：若 $\\vec{a}\\parallel\\vec{b}$，设 $|\\vec{a}|/|\\vec{b}|=r$，同向取 $\\lambda=r$，反向取 $\\lambda=-r$；反之若 $\\vec{a}=\\lambda\\vec{b}$，方向由 $\\lambda$ 的正负决定，故共线。\n\n注意：零向量 $\\vec{0}$ 与任意向量共线，需单独说明。',
        formula: '\\vec{a}\\parallel\\vec{b} \\Leftrightarrow \\vec{a}=\\lambda\\vec{b} \\quad (\\vec{b}\\neq\\vec{0})',
      },
      {
        title: '共面向量定理与空间基本定理',
        body: '共面向量定理：$\\vec{a},\\vec{b}$ 不共线，则向量 $\\vec{p}$ 与 $\\vec{a},\\vec{b}$ 共面，当且仅当存在唯一实数 $x,y$，使 $\\vec{p}=x\\vec{a}+y\\vec{b}$。\n\n空间向量基本定理：若 $\\vec{e}_1,\\vec{e}_2,\\vec{e}_3$ 不共面，则任意向量 $\\vec{p}$ 都能唯一表示为 $\\vec{p}=x\\vec{e}_1+y\\vec{e}_2+z\\vec{e}_3$。取标准基 $\\vec{i}=(1,0,0)$，$\\vec{j}=(0,1,0)$，$\\vec{k}=(0,0,1)$，即得坐标表示 $\\vec{a}=(a_1,a_2,a_3)$。',
        formula: '\\vec{p} = x\\vec{e}_1 + y\\vec{e}_2 + z\\vec{e}_3 \\quad (x,y,z\\in\\mathbb{R})',
        image: 'geo-p9-01-2.svg',
        imageCaption: '共面向量定理：$\\vec{p}$ 在 $\\vec{a},\\vec{b}$ 张成的平面内当且仅当 $\\vec{p}=x\\vec{a}+y\\vec{b}$',
      },
      {
        title: '例题与易错',
        body: '例：已知 $\\vec{a}=(1,2,-1)$，$\\vec{b}=(2,t,-2)$，$\\vec{a}\\parallel\\vec{b}$，求 $t$。\n\n解：$\\vec{b}=\\lambda\\vec{a}$，由第一分量 $2=\\lambda\\cdot1$ 得 $\\lambda=2$；由第三分量 $-2=\\lambda\\cdot(-1)$ 得 $\\lambda=2$（一致）；由第二分量 $t=2\\times2=4$。\n\n易错提示：①不要把平面"任意两向量共面"误用于空间；②共线定理中必须排除 $\\vec{b}=\\vec{0}$；③平行六面体法则中，$\\vec{a}+\\vec{b}+\\vec{c}$ 对应体对角线，不是面对角线；④三个向量不共面的关键是任一个都不能用其余两个的线性组合表示。',
        reveal: {
          q: '正三棱柱 $ABC$-$A_1B_1C_1$ 中，$\\vec{AB}=\\vec{a}$，$\\vec{AC}=\\vec{b}$，$\\vec{AA_1}=\\vec{c}$，用 $\\vec{a},\\vec{b},\\vec{c}$ 表示 $\\vec{BC_1}$。',
          a: '$\\vec{BC_1}=\\vec{BA}+\\vec{AC}+\\vec{CC_1}=(-\\vec{a})+\\vec{b}+\\vec{c}=\\vec{b}-\\vec{a}+\\vec{c}$',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '下列关于空间向量的说法，正确的是',
        options: [
          '空间中任意两个非零向量都共面',
          '空间向量可以自由平移，起点不同但大小和方向相同的向量是相等向量',
          '三个向量不共线就能构成空间基底',
          '零向量没有方向，因此与任何向量都不共线',
        ],
        answer: 1,
        explain: '空间向量的自由性：只要大小和方向相同，不论起点在何处，均视为相等向量。选项 A 错误，空间中三个向量不一定共面；选项 C 错误，需要三个不共面（而非不共线）的向量才能构成基底；选项 D 错误，零向量规定与任意向量共线。',
      },
      {
        type: 'choice',
        prompt: '已知向量 $\\vec{a}=(1,-2,3)$，$\\vec{b}=(2,-4,k)$，若 $\\vec{a}\\parallel\\vec{b}$，则 $k=$',
        options: ['3', '6', '5', '9'],
        answer: 1,
        explain: '$\\vec{a}\\parallel\\vec{b}$ 且 $\\vec{a}\\neq\\vec{0}$，故 $\\vec{b}=\\lambda\\vec{a}$。由第一分量：$\\lambda=2$。由第二分量：$-4=2\\times(-2)=-4$，一致。由第三分量：$k=2\\times3=6$。',
      },
      {
        type: 'choice',
        prompt: '在长方体 $ABCD$-$A_1B_1C_1D_1$ 中，设 $\\vec{AB}=\\vec{a}$，$\\vec{AD}=\\vec{b}$，$\\vec{AA_1}=\\vec{c}$，则体对角线向量 $\\vec{AC_1}$ 等于',
        options: [
          '$\\vec{a}+\\vec{b}$',
          '$\\vec{a}+\\vec{c}$',
          '$\\vec{a}+\\vec{b}+\\vec{c}$',
          '$\\vec{b}+\\vec{c}$',
        ],
        answer: 2,
        explain: '首尾相接：$\\vec{AC_1}=\\vec{AB}+\\vec{BC}+\\vec{CC_1}=\\vec{a}+\\vec{b}+\\vec{c}$。其中 $\\vec{BC}=\\vec{AD}=\\vec{b}$，$\\vec{CC_1}=\\vec{AA_1}=\\vec{c}$（棱柱侧棱平行且相等）。',
      },
      {
        type: 'judge',
        prompt: '若三个向量 $\\vec{e}_1=(1,1,0)$，$\\vec{e}_2=(0,1,1)$，$\\vec{e}_3=(1,0,-1)$ 满足 $\\vec{e}_3=\\vec{e}_1-\\vec{e}_2$，所以它们共面，不能构成空间基底。',
        answer: true,
        explain: '$\\vec{e}_3=(1,0,-1)=(1,1,0)-(0,1,1)=\\vec{e}_1-\\vec{e}_2$，即 $\\vec{e}_3$ 可以用 $\\vec{e}_1,\\vec{e}_2$ 的线性组合表示，故三个向量共面，不能构成空间基底。（注意本题判断的是"不能构成基底"这一结论为真。）',
      },
      {
        type: 'input',
        prompt: '已知 $\\vec{a}=(2,1,-1)$，$\\vec{b}=(-4,-2,2)$，则 $\\vec{b}=\\lambda\\vec{a}$ 中 $\\lambda=$',
        accept: ['-2'],
        explain: '$\\vec{b}$ 各分量是 $\\vec{a}$ 对应分量的 $-2$ 倍：$-4=(-2)\\times2$，$-2=(-2)\\times1$，$2=(-2)\\times(-1)$，故 $\\lambda=-2$，两向量共线（反向）。',
      },
      {
        type: 'match',
        prompt: '将下列加法结果与正确表达式配对',
        left: [
          '$\\vec{AB}+\\vec{BC}$',
          '$\\vec{OA}-\\vec{OB}$',
          '$\\vec{AB}+\\vec{BC}+\\vec{CA}$',
          '$\\vec{a}+\\vec{b}+\\vec{c}$（三向量共起点）',
        ],
        right: [
          '$\\vec{0}$',
          '$\\vec{BA}$',
          '平行六面体体对角线向量',
          '$\\vec{AC}$',
        ],
        explain: '$\\vec{AB}+\\vec{BC}=\\vec{AC}$（首尾相接）；$\\vec{OA}-\\vec{OB}=\\vec{BA}$（从减数终点到被减数终点）；封闭折线 $\\vec{AB}+\\vec{BC}+\\vec{CA}=\\vec{0}$；三共起点向量之和为平行六面体体对角线向量。',
      },
      {
        type: 'choice',
        prompt: '空间四边形 $OABC$ 中，$\\vec{OA}=\\vec{a}$，$\\vec{OB}=\\vec{b}$，$\\vec{OC}=\\vec{c}$，点 $M$ 满足 $\\vec{OM}=\\dfrac{1}{3}\\vec{a}+\\dfrac{1}{3}\\vec{b}+\\dfrac{1}{3}\\vec{c}$，则 $M$ 是',
        options: [
          '线段 $OA$ 的三等分点',
          '体对角线上的点，不在平面 $ABC$ 内',
          '三角形 $ABC$ 的重心',
          '三角形 $OAB$ 的重心',
        ],
        answer: 2,
        explain: '平面 $ABC$ 上的点可写为 $\\vec{OP}=(1-s-t)\\vec{a}+s\\vec{b}+t\\vec{c}$，三系数之和为 $1$。$M$ 的系数 $\\frac{1}{3}+\\frac{1}{3}+\\frac{1}{3}=1$，故 $M$ 在平面 $ABC$ 内，且三系数相等均为 $\\frac{1}{3}$，$M$ 是 $\\triangle ABC$ 的重心。',
      },
    ],
  },
]
