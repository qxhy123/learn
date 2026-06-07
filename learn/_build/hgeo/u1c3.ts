export const LESSONS = [
  {
    id: 'u1c3',
    title: '向量的数乘',
    subtitle: '拉伸、压缩、翻转——实数与向量的对话',
    intro: [
      {
        title: '引入：把向量"放大"或"翻转"',
        body: '已知位移向量 $\\vec{a}$（向右 $1\\,\\text{m}$），如何描述"向右 $2\\,\\text{m}$"和"向左 $3\\,\\text{m}$"？\n\n直觉告诉我们：向右 $2\\,\\text{m}$ 就是"两个 $\\vec{a}$ 叠加"；向左 $3\\,\\text{m}$ 就是"三个 $\\vec{a}$ 反向叠加"。这就是**数乘**的动机——用实数拉伸、压缩或翻转一个向量。',
        tip: '数乘 $\\lambda\\vec{a}$：$\\lambda$ 控制"拉多长"（模），$\\lambda$ 的符号控制"翻不翻转"（方向）。',
      },
      {
        title: '数乘的定义——三种情形',
        body: '实数 $\\lambda$ 与向量 $\\vec{a}$ 的**数乘** $\\lambda\\vec{a}$ 是一个新向量，按如下规则定义：\n\n- $\\lambda > 0$：$|\\lambda\\vec{a}| = \\lambda|\\vec{a}|$，方向与 $\\vec{a}$ **相同**\n- $\\lambda < 0$：$|\\lambda\\vec{a}| = |\\lambda||\\vec{a}|$，方向与 $\\vec{a}$ **相反**\n- $\\lambda = 0$：$0 \\cdot \\vec{a} = \\vec{0}$（零向量，方向不定）\n\n特别地，$\\vec{a} = \\vec{0}$ 时，$\\lambda\\vec{0} = \\vec{0}$（对任意实数 $\\lambda$）。',
        formula: '|\\lambda\\vec{a}| = |\\lambda|\\,|\\vec{a}|',
        image: 'geo-p1-03-1.svg',
        imageCaption: '以同一起点画出 $\\vec{a},\\,2\\vec{a},\\,-\\vec{a},\\,\\tfrac{1}{2}\\vec{a}$：同向拉伸、翻转、缩短',
        tip: '关键：模长公式取 $\\lambda$ 的**绝对值**，与方向无关。$\\lambda < 0$ 时也不会出现"负模长"。',
      },
      {
        title: '数乘的运算律',
        body: '设 $\\lambda, \\mu \\in \\mathbb{R}$，$\\vec{a}, \\vec{b}$ 为向量，以下三条运算律成立：\n\n**结合律**：$\\lambda(\\mu\\vec{a}) = (\\lambda\\mu)\\vec{a}$（例：$2(3\\vec{a}) = 6\\vec{a}$）\n\n**对标量加法的分配律**：$(\\lambda + \\mu)\\vec{a} = \\lambda\\vec{a} + \\mu\\vec{a}$（例：$(2+3)\\vec{a} = 2\\vec{a}+3\\vec{a} = 5\\vec{a}$）\n\n**对向量加法的分配律**：$\\lambda(\\vec{a} + \\vec{b}) = \\lambda\\vec{a} + \\lambda\\vec{b}$（例：$2(\\vec{a}+\\vec{b}) = 2\\vec{a}+2\\vec{b}$）\n\n**常用推论**：$1\\cdot\\vec{a} = \\vec{a}$，$(-1)\\vec{a} = -\\vec{a}$，$\\lambda(\\vec{a}-\\vec{b}) = \\lambda\\vec{a}-\\lambda\\vec{b}$。',
        formula: '\\lambda(\\vec{a}+\\vec{b}) = \\lambda\\vec{a}+\\lambda\\vec{b},\\quad (\\lambda+\\mu)\\vec{a} = \\lambda\\vec{a}+\\mu\\vec{a}',
      },
      {
        title: '单位化——提取向量的"方向名片"',
        body: '任意非零向量 $\\vec{a}$ 可以单位化，得到与 $\\vec{a}$ 同向的单位向量：\n\n$$\\hat{a} = \\dfrac{\\vec{a}}{|\\vec{a}|}$$\n\n这是数乘的一个重要应用：$\\hat{a} = \\dfrac{1}{|\\vec{a}|}\\vec{a}$，即用 $\\lambda = \\dfrac{1}{|\\vec{a}|} > 0$ 对 $\\vec{a}$ 进行缩放。\n\n**验证**：$|\\hat{a}| = \\left|\\dfrac{1}{|\\vec{a}|}\\right||\\vec{a}| = \\dfrac{1}{|\\vec{a}|} \\cdot |\\vec{a}| = 1$。模确实为 $1$。',
        formula: '\\hat{a} = \\dfrac{\\vec{a}}{|\\vec{a}|}',
        tip: '单位化公式：原向量除以自身的模。结果与原向量同向，模为 $1$。',
      },
      {
        title: '共线向量定理',
        body: '**定理**：设 $\\vec{a} \\neq \\vec{0}$，则向量 $\\vec{b}$ 与 $\\vec{a}$ 共线（平行）的**充要条件**是存在**唯一**实数 $\\lambda$，使得\n\n$$\\vec{b} = \\lambda\\vec{a}$$\n\n**为什么唯一？** 因为 $\\vec{a} \\neq \\vec{0}$，等式 $\\vec{b} = \\lambda\\vec{a}$ 中，$\\lambda = \\dfrac{|\\vec{b}|}{|\\vec{a}|}$（同向）或 $\\lambda = -\\dfrac{|\\vec{b}|}{|\\vec{a}|}$（反向），由 $\\vec{b}$ 的方向唯一确定。\n\n**为什么要求 $\\vec{a} \\neq \\vec{0}$？** 若 $\\vec{a} = \\vec{0}$，则 $\\lambda\\vec{a} = \\vec{0}$ 对任意 $\\lambda$ 成立，$\\lambda$ 不唯一，定理不适用。\n\n**三点共线**：$A, B, C$ 三点共线 $\\Leftrightarrow$ $\\overrightarrow{AB} = \\lambda\\overrightarrow{AC}$（$\\overrightarrow{AC} \\neq \\vec{0}$）。',
        formula: '\\vec{b} \\parallel \\vec{a}\\;(\\vec{a}\\neq\\vec{0}) \\iff \\exists\\,\\lambda\\in\\mathbb{R},\\;\\vec{b}=\\lambda\\vec{a}',
        tip: '判定共线：从同一公共点出发写两个向量，看是否是数乘关系。',
      },
      {
        title: '中点公式',
        body: '设 $M$ 是线段 $AB$ 的中点，$O$ 是任意一点，则：\n\n$$\\overrightarrow{OM} = \\dfrac{1}{2}(\\overrightarrow{OA} + \\overrightarrow{OB})$$\n\n**推导**（一步步来）：\n\n$\\overrightarrow{OM} = \\overrightarrow{OA} + \\overrightarrow{AM}$（三角形法则）\n\n$= \\overrightarrow{OA} + \\dfrac{1}{2}\\overrightarrow{AB}$（$M$ 是中点，故 $\\overrightarrow{AM} = \\dfrac{1}{2}\\overrightarrow{AB}$）\n\n$= \\overrightarrow{OA} + \\dfrac{1}{2}(\\overrightarrow{OB} - \\overrightarrow{OA})$（减法表示 $\\overrightarrow{AB}$）\n\n$= \\dfrac{1}{2}\\overrightarrow{OA} + \\dfrac{1}{2}\\overrightarrow{OB} = \\dfrac{1}{2}(\\overrightarrow{OA}+\\overrightarrow{OB})$',
        formula: '\\overrightarrow{OM} = \\dfrac{1}{2}(\\overrightarrow{OA}+\\overrightarrow{OB})',
      },
      {
        title: '典型例题与易错点',
        body: '**例题**：$\\overrightarrow{OA} = \\vec{a}$，$\\overrightarrow{OB} = \\vec{b}$，$\\overrightarrow{OP} = \\vec{a}+2\\vec{b}$，$\\overrightarrow{OQ} = 4\\vec{a}+2\\vec{b}$，$\\overrightarrow{OR} = -2\\vec{a}+2\\vec{b}$，判断 $P, Q, R$ 是否共线。\n\n**解**：$\\overrightarrow{PQ} = (4\\vec{a}+2\\vec{b})-(\\vec{a}+2\\vec{b}) = 3\\vec{a}$；$\\overrightarrow{PR} = (-2\\vec{a}+2\\vec{b})-(\\vec{a}+2\\vec{b}) = -3\\vec{a}$。\n故 $\\overrightarrow{PR} = -1 \\cdot \\overrightarrow{PQ}$，共线且共享点 $P$，所以 $P, Q, R$ 三点共线。\n\n**易错1**：$|\\lambda\\vec{a}| = |\\lambda||\\vec{a}|$，不是 $\\lambda|\\vec{a}|$（当 $\\lambda < 0$ 时会出现负的"模长"）。\n**易错2**：中点公式中 $O$ 必须是同一起点，不要混用 $\\overrightarrow{AO}$ 和 $\\overrightarrow{OB}$。',
        reveal: {
          q: '已知 $|\\vec{a}| = 5$，$\\lambda = -\\dfrac{3}{5}$，求 $|\\lambda\\vec{a}|$，并判断 $\\lambda\\vec{a}$ 与 $\\vec{a}$ 的方向关系。',
          a: '$|\\lambda\\vec{a}| = \\left|-\\dfrac{3}{5}\\right| \\times 5 = \\dfrac{3}{5} \\times 5 = 3$。因 $\\lambda = -\\dfrac{3}{5} < 0$，故 $\\lambda\\vec{a}$ 与 $\\vec{a}$ 方向**相反**，且模为 $3$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '已知 $|\\vec{a}| = 3$，则 $|-3\\vec{a}|$ 等于（　　）',
        options: ['$-9$', '$-3$', '$3$', '$9$'],
        answer: 3,
        explain: '选 D。$|-3\\vec{a}| = |-3| \\times |\\vec{a}| = 3 \\times 3 = 9$。模长公式取绝对值，结果永远非负，所以不会是 $-9$ 或 $-3$。',
      },
      {
        type: 'judge',
        prompt: '若 $\\vec{b} = \\lambda\\vec{a}$（$\\lambda$ 为实数），则 $\\vec{a}$ 与 $\\vec{b}$ 共线。',
        answer: true,
        explain: '正确。这正是共线向量定理的充分方向：$\\vec{b} = \\lambda\\vec{a}$ 直接说明 $\\vec{b}$ 可以表示为 $\\vec{a}$ 的数乘，故 $\\vec{b}$ 与 $\\vec{a}$ 共线（或 $\\vec{b} = \\vec{0}$，零向量与任意向量共线）。',
      },
      {
        type: 'choice',
        prompt: '向量 $\\vec{p} = 3\\vec{a} - \\vec{b}$ 与 $\\vec{q} = -6\\vec{a} + 2\\vec{b}$ 的关系是（　　）',
        options: [
          '共线，且方向相同',
          '共线，且方向相反',
          '不共线',
          '无法判断',
        ],
        answer: 1,
        explain: '选 B。$\\vec{q} = -6\\vec{a}+2\\vec{b} = -2(3\\vec{a}-\\vec{b}) = -2\\vec{p}$，故 $\\vec{q} = (-2)\\vec{p}$。系数 $-2 < 0$，方向与 $\\vec{p}$ 相反，故共线且方向相反。',
      },
      {
        type: 'input',
        prompt: '已知 $\\overrightarrow{OA} = 2\\vec{i}+3\\vec{j}$，$\\overrightarrow{OB} = 4\\vec{i}-\\vec{j}$（$\\vec{i}, \\vec{j}$ 为正交单位向量），$AB$ 中点 $M$ 的 $\\overrightarrow{OM}$ 中 $\\vec{i}$ 的系数是多少？',
        accept: ['3'],
        explain: '$\\overrightarrow{OM} = \\dfrac{1}{2}(\\overrightarrow{OA}+\\overrightarrow{OB}) = \\dfrac{1}{2}((2+4)\\vec{i}+(3-1)\\vec{j}) = \\dfrac{1}{2}(6\\vec{i}+2\\vec{j}) = 3\\vec{i}+\\vec{j}$。$\\vec{i}$ 的系数为 $3$。',
      },
      {
        type: 'choice',
        prompt: '已知 $A(1,2)$，$B(3,6)$，$C(5,10)$，判断三点是否共线的依据是（　　）',
        options: [
          '$\\overrightarrow{AB} = (2,4)$，$\\overrightarrow{AC} = (4,8)$，$\\overrightarrow{AC} = 2\\overrightarrow{AB}$，故共线',
          '$\\overrightarrow{AB} = (2,4)$，$\\overrightarrow{BC} = (2,4)$，方向相同，故共线',
          '以上两种方法都正确',
          '无法用向量法判断',
        ],
        answer: 2,
        explain: '选 C。两种方法都正确。方法一：从公共点 $A$ 出发，$\\overrightarrow{AC} = 2\\overrightarrow{AB}$，故 $A, B, C$ 共线（共线向量定理）。方法二：$\\overrightarrow{AB} = \\overrightarrow{BC}$，故 $B$ 是 $\\overrightarrow{AC}$ 上的点，三点共线。',
      },
      {
        type: 'judge',
        prompt: '若 $\\vec{a} \\neq \\vec{0}$，则 $\\dfrac{\\vec{a}}{|\\vec{a}|}$ 是与 $\\vec{a}$ 同向的单位向量。',
        answer: true,
        explain: '正确。$\\lambda = \\dfrac{1}{|\\vec{a}|} > 0$（因为 $|\\vec{a}| > 0$），所以 $\\dfrac{\\vec{a}}{|\\vec{a}|}$ 与 $\\vec{a}$ 同向；其模为 $\\dfrac{|\\vec{a}|}{|\\vec{a}|} = 1$，是单位向量。',
      },
      {
        type: 'match',
        prompt: '将数乘 $\\lambda\\vec{a}$（$\\vec{a} \\neq \\vec{0}$）的情形与对应描述配对',
        left: [
          '$\\lambda > 0$',
          '$\\lambda < 0$',
          '$\\lambda = 0$',
        ],
        right: [
          '结果为零向量 $\\vec{0}$，方向不定',
          '与 $\\vec{a}$ 同向，模为 $\\lambda|\\vec{a}|$',
          '与 $\\vec{a}$ 反向，模为 $|\\lambda||\\vec{a}|$',
        ],
        explain: '$\\lambda > 0$：同向，模 $= \\lambda|\\vec{a}|$；$\\lambda < 0$：反向，模 $= |\\lambda||\\vec{a}|$（取绝对值）；$\\lambda = 0$：结果是零向量，方向不定。',
      },
      {
        type: 'input',
        prompt: '平行四边形 $ABCD$ 中，$O$ 是对角线交点（也是中点），$\\overrightarrow{OA} = \\vec{a}$，$\\overrightarrow{OB} = \\vec{b}$，则 $\\overrightarrow{OC}$ 等于（用 $\\vec{a}$ 的倍数表示，如 $2\\vec{a}$ 或 $-\\vec{a}$）',
        accept: ['-\\vec{a}', '-a', '负向量a'],
        explain: '$O$ 是 $AC$ 的中点，由中点公式 $\\overrightarrow{0} = \\dfrac{1}{2}(\\overrightarrow{OA}+\\overrightarrow{OC})$，故 $\\overrightarrow{OC} = -\\overrightarrow{OA} = -\\vec{a}$。（$\\overrightarrow{OA}+\\overrightarrow{OC} = \\vec{0}$ 因为 $O$ 是中点，两者方向相反、模相等。）',
      },
    ],
  },
]
