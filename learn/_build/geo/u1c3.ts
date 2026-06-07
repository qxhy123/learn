export const LESSONS = [
  {
    id: 'u1c3',
    title: '角平分线',
    subtitle: '把角"对折"的射线与等距性质',
    intro: [
      {
        title: '什么是角平分线？',
        body: '角平分线是从角的顶点引出的、把角分成两个**相等部分**的射线。\n\n三个要点：\n- **起点**是角的顶点，不能从别处出发。\n- 它是一条**射线**，不是直线，也不是线段。\n- 它必须落在角的**内部**，否则无法把角分成两个相等的部分。\n\n形象地说，角平分线像一把"对折"角的折痕——沿平分线把角翻折，两条边会完全重合。',
        tip: '"角平分线的长度"这种说法是错误的——射线没有有限长度。',
      },
      {
        title: '定义与等价公式',
        body: '**定义：** 在 $\\angle AOB$ 的内部有一条射线 $OC$，若 $\\angle AOC = \\angle COB$，则称射线 $OC$ 是 $\\angle AOB$ 的角平分线。\n\n**等价表达：**\n$\\angle AOC = \\angle COB = \\frac{1}{2}\\angle AOB$\n\n或者反过来：\n$\\angle AOB = 2\\angle AOC = 2\\angle COB$\n\n做题时，两种方向都要熟练：知道角平分线 $\\Rightarrow$ 半角关系；知道半角关系 $\\Rightarrow$ 是角平分线。',
        image: 'def-angle-bisector.svg',
        imageCaption: '$OC$ 平分 $\\angle AOB$，把角分为相等的两半',
        formula: '\\angle AOC = \\angle COB = \\dfrac{1}{2}\\angle AOB',
      },
      {
        title: '核心性质：到两边的距离相等',
        body: '**性质：** 角平分线上任意一点到角两边的距离相等。\n\n用符号表述：若 $OC$ 平分 $\\angle AOB$，$P$ 是 $OC$ 上任一点，$PM \\perp OA$ 于 $M$，$PN \\perp OB$ 于 $N$，则 $PM = PN$。\n\n**逆定理：** 在角的内部，到角两边距离相等的点一定在这个角的平分线上。\n\n正定理 + 逆定理合起来，给出了一个"位置 $\\Leftrightarrow$ 距离"的完全等价刻画。',
        image: 'thm-angle-bisector-property.svg',
        imageCaption: '核心性质：$P$ 在角平分线上，$PM\\perp OA$、$PN\\perp OB$，则 $PM=PN$',
        formula: 'P \\in \\text{角平分线} \\iff PM = PN',
      },
      {
        title: '性质的完整证明（AAS 全等）',
        body: '**已知：** $OC$ 平分 $\\angle AOB$，$P$ 在 $OC$ 上，$PM \\perp OA$，$PN \\perp OB$。\n**求证：** $PM = PN$。\n\n考察 $\\triangle OPM$ 与 $\\triangle OPN$：',
        steps: [
          '$OC$ 平分 $\\angle AOB$，故 $\\angle MOP = \\angle NOP$（角平分线定义）',
          '$PM \\perp OA$，$PN \\perp OB$，故 $\\angle PMO = \\angle PNO = 90^\\circ$',
          '$OP$ 是两个三角形的公共边，$OP = OP$',
          '由 AAS（两角及对应边），$\\triangle OPM \\cong \\triangle OPN$',
          '全等三角形对应边相等：$PM = PN$。$\\blacksquare$',
        ],
        tip: '此处用 AAS：两个角（$\\angle MOP = \\angle NOP$，$\\angle PMO = \\angle PNO$）和它们夹的对边（$OP$）。',
      },
      {
        title: '例题 1：基本计算',
        body: '**问题 1：** 已知 $OC$ 平分 $\\angle AOB$，$\\angle AOB = 70^\\circ$，求 $\\angle AOC$。\n\n**解：** $\\angle AOC = \\dfrac{1}{2}\\angle AOB = \\dfrac{1}{2} \\times 70^\\circ = 35^\\circ$。\n\n**问题 2（自测 2 改编）：** $\\angle AOB = 120^\\circ$，$OC$ 平分 $\\angle AOB$，$OD$ 平分 $\\angle AOC$，求 $\\angle BOD$。\n\n**解：** $\\angle AOC = 60^\\circ$，$\\angle AOD = 30^\\circ$，$\\angle BOD = \\angle AOB - \\angle AOD = 120^\\circ - 30^\\circ = 90^\\circ$。',
        formula: '\\angle BOD = \\angle AOB - \\angle AOD',
      },
      {
        title: '例题 2：三角形内心',
        body: '**结论：** 三角形的三条内角平分线交于一点，这一点到三边的距离相等，称为三角形的**内心**（内切圆的圆心）。\n\n**思路：** 设 $\\triangle ABC$ 中，$\\angle A$ 的平分线与 $\\angle B$ 的平分线交于点 $I$。\n- $I$ 在 $\\angle A$ 的平分线上 $\\Rightarrow$ $I$ 到 $AB$、$AC$ 的距离相等：$d_{AB} = d_{AC}$\n- $I$ 在 $\\angle B$ 的平分线上 $\\Rightarrow$ $I$ 到 $AB$、$BC$ 的距离相等：$d_{AB} = d_{BC}$\n- 连起来：$d_{AC} = d_{BC}$，由逆定理，$I$ 也在 $\\angle C$ 的平分线上。\n\n故三条角平分线共点，且 $I$ 到三边距离相等。',
        image: 'ex-p1-03-3.svg',
        imageCaption: '三角形内心 $I$：三条内角平分线交于一点，$I$ 即内切圆圆心',
      },
      {
        title: '易错点与辨析',
        body: '1. **角平分线是射线，不是线段：** 不能说角平分线"有多长"。\n2. **"距离"指垂线段长度：** 点到角的一边的距离，是过该点向边所作垂线的垂线段长度，绝不是该点到顶点的距离 $OP$。\n3. **逆定理需要"在角内部"：** 到两条射线距离相等的点，也可能在外角平分线上；要先确认点在角的内部，才能用逆定理。\n4. **与线段垂直平分线区分：** 角平分线到两"边"距离相等；线段垂直平分线到两"端点"距离相等——两个性质完全不同。',
        reveal: {
          q: '点 $P$ 在 $\\angle AOB$ 内部，且到 $OA$、$OB$ 的距离都等于 $3$。$P$ 一定在角平分线上吗？',
          a: '一定在。这正是角平分线**逆定理**的内容：在角的内部，到角两边距离相等的点必在该角的平分线上。（若 $P$ 不在角内部，则不能用此逆定理。）',
        },
      },
    ],
    questions: [
      {
        type: 'input',
        prompt: '$OC$ 平分 $\\angle AOB$，$\\angle AOC = 28^\\circ$，求 $\\angle AOB$。',
        accept: ['56°', '56'],
        explain: '由角平分线定义，$\\angle AOB = 2\\angle AOC = 2 \\times 28^\\circ = 56^\\circ$。',
      },
      {
        type: 'choice',
        prompt: '下列关于角平分线的说法，正确的是',
        options: [
          '角平分线是一条线段',
          '角平分线是从角的顶点引出的一条直线',
          '角平分线是一条射线，从顶点出发落在角的内部',
          '角平分线可以在角的外部',
        ],
        answer: 2,
        explain: '角平分线是从顶点出发、落在角内部的射线。它不是线段（线段有有限长度），也不是直线（直线无端点），更不在角的外部。',
      },
      {
        type: 'judge',
        prompt: '$OC$ 平分 $\\angle AOB$，$P$ 是 $OC$ 上任一点，则 $P$ 到 $OA$ 的距离等于 $P$ 到顶点 $O$ 的距离。',
        answer: false,
        explain: '角平分线性质说的是：$P$ 到 $OA$ 的距离（垂线段长）等于 $P$ 到 $OB$ 的距离（垂线段长）。"距离"指垂线段，不是 $OP$。混淆这两种距离是常见错误。',
      },
      {
        type: 'input',
        prompt: '$\\angle AOB = 120^\\circ$，$OC$ 平分 $\\angle AOB$，$OD$ 平分 $\\angle AOC$，求 $\\angle BOD$ 的度数。',
        accept: ['90°', '90'],
        explain: '$\\angle AOC = 60^\\circ$（$OC$ 平分 $\\angle AOB$），$\\angle AOD = 30^\\circ$（$OD$ 平分 $\\angle AOC$），$\\angle BOD = \\angle AOB - \\angle AOD = 120^\\circ - 30^\\circ = 90^\\circ$。',
      },
      {
        type: 'match',
        prompt: '把左边的条件与右边的结论正确连线',
        left: [
          '$OC$ 平分 $\\angle AOB$，$P \\in OC$',
          '角平分线正定理',
          '角平分线逆定理',
        ],
        right: [
          '$P$ 在角内部且到两边距离相等 $\\Rightarrow$ $P$ 在平分线上',
          '$P$ 到 $OA$、$OB$ 的距离相等',
          '$P$ 在角平分线上 $\\Rightarrow$ 到两边距离相等',
        ],
        explain: '正定理：在平分线上 $\\Rightarrow$ 等距；逆定理：在角内部且等距 $\\Rightarrow$ 在平分线上。$P \\in OC$ 且 $OC$ 是角平分线，结论是 $P$ 到两边距离相等。',
      },
      {
        type: 'judge',
        prompt: '三角形三条内角平分线的交点（内心）到三条边的距离相等。',
        answer: true,
        explain: '内心是三条内角平分线的交点。由角平分线性质，内心到每两条边的距离相等，连锁推出内心到三条边的距离全部相等，是内切圆的圆心。',
      },
      {
        type: 'choice',
        prompt: '在 $\\triangle ABC$ 中，$AD$ 是 $\\angle BAC$ 的平分线，$D$ 在 $BC$ 上，$DE \\perp AB$ 于 $E$，$DF \\perp AC$ 于 $F$。下列结论正确的是',
        options: [
          '$DE = DA$',
          '$DE = DF$',
          '$DE = DB$',
          '$DF = DA$',
        ],
        answer: 1,
        explain: '$D$ 在 $\\angle BAC$ 的平分线 $AD$ 上，$DE \\perp AB$，$DF \\perp AC$，由角平分线的核心性质，$D$ 到角两边的距离相等，即 $DE = DF$。',
      },
      {
        type: 'input',
        prompt: '$OC$ 平分 $\\angle AOB$，$\\angle AOB = 84^\\circ$，求 $\\angle BOC$ 的度数。',
        accept: ['42°', '42'],
        explain: '$\\angle BOC = \\dfrac{1}{2}\\angle AOB = \\dfrac{1}{2} \\times 84^\\circ = 42^\\circ$。',
      },
    ],
  },
]
