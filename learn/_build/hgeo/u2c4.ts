export const LESSONS = [
  {
    id: 'u2c4',
    title: '向量的物理与综合应用',
    subtitle: '力的合成、速度叠加与几何综合证明',
    intro: [
      {
        title: '向量是物理与数学的桥梁',
        body: '力、速度、位移都是有大小又有方向的量——向量。物理中的平行四边形定则、速度合成、位移叠加，在数学里都归结为向量加法。向量语言把物理直觉变成精确的代数运算，让我们能够计算合力大小、确定船的偏角、证明几何定理。',
      },
      {
        title: '力的合成：平行四边形定则',
        body: '两个力 $\\vec{F_1}$ 与 $\\vec{F_2}$ 作用于同一点时，合力 $\\vec{F} = \\vec{F_1} + \\vec{F_2}$，大小由模长展开公式给出：',
        formula: '|\\vec{F}|^2 = |\\vec{F_1}|^2 + 2|\\vec{F_1}||\\vec{F_2}|\\cos\\theta + |\\vec{F_2}|^2',
        steps: [
          '同向（$\\theta = 0°$）：$|\\vec{F}| = |\\vec{F_1}| + |\\vec{F_2}|$，合力最大',
          '垂直（$\\theta = 90°$）：$|\\vec{F}| = \\sqrt{|\\vec{F_1}|^2 + |\\vec{F_2}|^2}$，勾股定理',
          '反向（$\\theta = 180°$）：$|\\vec{F}| = \\big||\\vec{F_1}| - |\\vec{F_2}|\\big|$，合力最小',
          '一般情形：先用坐标分量分别相加，再求合力向量的模长',
        ],
        tip: '合力范围：$\\big||\\vec{F_1}| - |\\vec{F_2}|\\big| \\leq |\\vec{F}| \\leq |\\vec{F_1}| + |\\vec{F_2}|$，当两力夹角从 $0°$ 增大到 $180°$ 时合力从最大减到最小。',
      },
      {
        title: '速度合成：船渡河问题',
        body: '船在有水流的河中行驶，实际速度 $\\vec{v}$ 由船相对水的速度 $\\vec{v}_{\\text{船}}$ 与水流速度 $\\vec{v}_{\\text{水}}$ 叠加：',
        formula: '\\vec{v} = \\vec{v}_{\\text{船}} + \\vec{v}_{\\text{水}}',
        steps: [
          '两速度方向一般垂直：船头垂直河岸，水流沿河岸方向，合速度斜指对岸下游',
          '合速度大小：$|\\vec{v}| = \\sqrt{v_{\\text{船}}^2 + v_{\\text{水}}^2}$（勾股定理）',
          '最短时间策略：船头垂直河岸，$t_{\\min} = \\dfrac{d}{v_{\\text{船}}}$，与水速无关；但会漂移',
          '最短路程策略：船头偏向上游，使合速度垂直河岸，漂移为零但过河时间更长',
        ],
        tip: '陷阱："最短时间"和"最短路程"是两种不同的策略，审题要看题目问的是哪个量。',
      },
      {
        title: '引入例：两力垂直求合力',
        body: '两力 $|\\vec{F_1}| = 3\\,\\text{N}$，$|\\vec{F_2}| = 4\\,\\text{N}$，夹角 $90°$，求合力大小及合力与 $\\vec{F_1}$ 的夹角：',
        steps: [
          '合力大小：$|\\vec{F}|^2 = 3^2 + 4^2 = 25$，$|\\vec{F}| = 5\\,\\text{N}$',
          '夹角 $\\alpha$：$\\tan\\alpha = \\dfrac{|\\vec{F_2}|}{|\\vec{F_1}|} = \\dfrac{4}{3}$，$\\alpha \\approx 53.1°$',
          '验证：满足 $3$-$4$-$5$ 勾股比，结果合理',
        ],
        tip: '两力垂直时，合力、$\\vec{F_1}$、$\\vec{F_2}$ 构成直角三角形，直接用反三角函数求角。',
      },
      {
        title: '三角不等式的向量证明',
        body: '向量三角不等式：$|\\vec{a}+\\vec{b}| \\leq |\\vec{a}| + |\\vec{b}|$，证明如下：',
        steps: [
          '展开：$|\\vec{a}+\\vec{b}|^2 = |\\vec{a}|^2 + 2\\vec{a}\\cdot\\vec{b} + |\\vec{b}|^2$',
          '由 $\\vec{a}\\cdot\\vec{b} \\leq |\\vec{a}||\\vec{b}|$（数量积不超过模长之积），代入得 $|\\vec{a}+\\vec{b}|^2 \\leq (|\\vec{a}|+|\\vec{b}|)^2$',
          '两边取非负平方根：$|\\vec{a}+\\vec{b}| \\leq |\\vec{a}| + |\\vec{b}|$',
          '等号成立条件：$\\vec{a}$ 与 $\\vec{b}$ 同向（$\\theta = 0°$），几何意义是三角形退化为线段',
        ],
        tip: '配合不等式 $|\\vec{a}+\\vec{b}| \\geq \\big||\\vec{a}| - |\\vec{b}|\\big|$（下界，反向时取等），完整描述合向量模长的范围。',
      },
      {
        title: '向量与三角的交叉：单位向量点积',
        body: '设 $\\vec{a} = (\\cos\\alpha, \\sin\\alpha)$，$\\vec{b} = (\\cos\\beta, \\sin\\beta)$ 均为单位向量，则：',
        formula: '\\vec{a} \\cdot \\vec{b} = \\cos\\alpha\\cos\\beta + \\sin\\alpha\\sin\\beta = \\cos(\\alpha - \\beta)',
        steps: [
          '这是和差化积公式 $\\cos(\\alpha-\\beta) = \\cos\\alpha\\cos\\beta + \\sin\\alpha\\sin\\beta$ 的向量解读',
          '单位向量的点积恰好等于夹角的余弦，与数量积定义一致',
          '推论：若两单位向量点积为 $\\dfrac{1}{2}$，则夹角为 $60°$',
          '应用：含 $\\sin\\theta$、$\\cos\\theta$ 的向量模长问题，先展开再用 $\\sin^2+\\cos^2=1$ 化简',
        ],
      },
      {
        title: '向量证明平行四边形对角线互相平分',
        body: '用向量方法证明平行四边形 $ABCD$ 的对角线 $AC$ 与 $BD$ 互相平分：',
        steps: [
          '平行四边形条件：$\\vec{AB} = \\vec{DC}$，即 $\\vec{OB}-\\vec{OA} = \\vec{OC}-\\vec{OD}$，整理得 $\\vec{OA}+\\vec{OC} = \\vec{OB}+\\vec{OD}$',
          '$AC$ 的中点 $M_1$：$\\vec{OM_1} = \\dfrac{\\vec{OA}+\\vec{OC}}{2}$',
          '$BD$ 的中点 $M_2$：$\\vec{OM_2} = \\dfrac{\\vec{OB}+\\vec{OD}}{2}$',
          '由第一步结论：$\\vec{OM_1} = \\vec{OM_2}$，即 $M_1 = M_2$，两对角线中点重合，互相平分',
        ],
        tip: '向量证明几何定理的模板：① 建立条件的向量等式；② 写出目标点的位置向量；③ 推导两位置向量相等。',
      },
      {
        title: '易错辨析与综合提醒',
        body: '本章综合性强，常见的四类错误：',
        steps: [
          '错误1：物理题漏写单位——力的单位 N、速度 m/s，合力/合速度的单位与分量相同，不要漏',
          '错误2：船渡河混淆策略——"最短时间"让船头垂直河岸，"不偏移/最短路程"让船头斜向上游',
          '错误3：三角不等式等号条件——$|\\vec{a}+\\vec{b}| = |\\vec{a}|+|\\vec{b}|$ 当且仅当 $\\vec{a}$ 与 $\\vec{b}$ **同向**，不是任意情形',
          '错误4：向量综合题忘选基底——证明几何命题必须先声明基底，否则"所有向量用 $\\vec{AB}$、$\\vec{AC}$ 表示"这一步缺失会失分',
        ],
        reveal: {
          q: '已知 $|\\vec{a}| = 3$，$|\\vec{b}| = 4$，$|\\vec{a}+\\vec{b}| = 5$，求 $\\vec{a}\\cdot\\vec{b}$。',
          a: '$25 = 9 + 2\\vec{a}\\cdot\\vec{b} + 16$，故 $2\\vec{a}\\cdot\\vec{b} = 0$，$\\vec{a}\\cdot\\vec{b} = 0$，$\\vec{a} \\perp \\vec{b}$（满足勾股定理 $3^2+4^2=5^2$）。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '两力 $|\\vec{F_1}| = 5\\,\\text{N}$，$|\\vec{F_2}| = 5\\,\\text{N}$，夹角 $\\theta = 120°$，则合力大小为：',
        options: ['$5\\,\\text{N}$', '$10\\,\\text{N}$', '$5\\sqrt{3}\\,\\text{N}$', '$0\\,\\text{N}$'],
        answer: 0,
        explain: '$|\\vec{F}|^2 = 25 + 2\\times5\\times5\\times\\cos120° + 25 = 50 + 50\\times(-\\dfrac{1}{2}) = 25$，$|\\vec{F}| = 5\\,\\text{N}$。（等边三角形情形，合力与分力等大。）',
      },
      {
        type: 'judge',
        prompt: '向量三角不等式 $|\\vec{a}+\\vec{b}| \\leq |\\vec{a}| + |\\vec{b}|$ 的等号，当 $\\vec{a}$ 与 $\\vec{b}$ 方向相反时成立。',
        answer: false,
        explain: '等号成立条件是 $\\vec{a}$ 与 $\\vec{b}$ **同向**（$\\theta = 0°$）。反向时 $|\\vec{a}+\\vec{b}| = \\big||\\vec{a}| - |\\vec{b}|\\big|$，取的是下界不等式的等号，不是上界。',
      },
      {
        type: 'choice',
        prompt: '设 $\\vec{a} = (\\cos\\alpha, \\sin\\alpha)$，$\\vec{b} = (\\cos\\beta, \\sin\\beta)$，则 $\\vec{a}\\cdot\\vec{b}$ 等于：',
        options: ['$\\cos(\\alpha-\\beta)$', '$\\cos(\\alpha+\\beta)$', '$\\sin(\\alpha-\\beta)$', '$1$'],
        answer: 0,
        explain: '$\\vec{a}\\cdot\\vec{b} = \\cos\\alpha\\cos\\beta + \\sin\\alpha\\sin\\beta = \\cos(\\alpha-\\beta)$（和差化积公式）。两单位向量点积等于夹角的余弦。',
      },
      {
        type: 'input',
        prompt: '已知 $|\\vec{a}| = 3$，$|\\vec{b}| = 4$，$|\\vec{a}+\\vec{b}| = 5$，则 $\\vec{a}\\cdot\\vec{b}$ 等于多少（填整数）？',
        accept: ['0'],
        explain: '$|\\vec{a}+\\vec{b}|^2 = |\\vec{a}|^2 + 2\\vec{a}\\cdot\\vec{b} + |\\vec{b}|^2$，即 $25 = 9 + 2\\vec{a}\\cdot\\vec{b} + 16$，解得 $\\vec{a}\\cdot\\vec{b} = 0$。',
      },
      {
        type: 'judge',
        prompt: '船渡河时，为了用最短时间到达对岸，船头应朝向斜向上游的方向。',
        answer: false,
        explain: '最短时间策略是船头垂直河岸，过河时间 $t = \\dfrac{d}{v_{\\text{船}}}$ 与水速无关，但会漂向下游。斜向上游是为了使合速度垂直河岸（最短路程/不偏移），过河时间反而更长。',
      },
      {
        type: 'match',
        prompt: '将两力夹角与合力特征配对（$|\\vec{F_1}| = |\\vec{F_2}| = F$）：',
        left: ['$\\theta = 0°$', '$\\theta = 90°$', '$\\theta = 120°$', '$\\theta = 180°$'],
        right: [
          '$|\\vec{F}| = 2F$（最大）',
          '$|\\vec{F}| = F\\sqrt{2}$',
          '$|\\vec{F}| = F$',
          '$|\\vec{F}| = 0$（最小）',
        ],
        explain: '$\\theta=0°$：同向合力最大 $2F$；$\\theta=90°$：勾股定理 $F\\sqrt{2}$；$\\theta=120°$：$|\\vec{F}|^2=F^2+2F^2\\cos120°+F^2=F^2$，故 $|\\vec{F}|=F$；$\\theta=180°$：反向抵消，合力为 $0$。',
      },
      {
        type: 'choice',
        prompt: '平行四边形 $ABCD$ 中，$\\vec{AB}=\\vec{b}$，$\\vec{AD}=\\vec{d}$。对角线 $AC$ 与 $BD$ 的交点 $O$ 满足 $\\vec{AO}$ 等于：',
        options: [
          '$\\dfrac{1}{2}(\\vec{b}+\\vec{d})$',
          '$\\vec{b}+\\vec{d}$',
          '$\\dfrac{1}{2}\\vec{b}$',
          '$\\dfrac{1}{2}\\vec{d}$',
        ],
        answer: 0,
        explain: '平行四边形对角线互相平分，$O$ 是 $AC$ 的中点。$\\vec{AC} = \\vec{AB}+\\vec{BC} = \\vec{b}+\\vec{d}$，故 $\\vec{AO} = \\dfrac{1}{2}\\vec{AC} = \\dfrac{1}{2}(\\vec{b}+\\vec{d})$。',
      },
      {
        type: 'choice',
        prompt: '已知两向量 $|\\vec{a}| = 2$，$|\\vec{b}| = 3$，夹角 $\\theta = 60°$，则 $|\\vec{a}-\\vec{b}|$ 等于：',
        options: ['$\\sqrt{7}$', '$\\sqrt{19}$', '$1$', '$5$'],
        answer: 0,
        explain: '$\\vec{a}\\cdot\\vec{b} = 2\\times3\\times\\cos60° = 3$。$|\\vec{a}-\\vec{b}|^2 = |\\vec{a}|^2 - 2\\vec{a}\\cdot\\vec{b} + |\\vec{b}|^2 = 4 - 6 + 9 = 7$，故 $|\\vec{a}-\\vec{b}| = \\sqrt{7}$。',
      },
    ],
  },
]
