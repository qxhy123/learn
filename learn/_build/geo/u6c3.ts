export const LESSONS = [
  {
    id: 'u6c3',
    title: '母子相似与射影定理',
    subtitle: '直角三角形斜边上的高 → 一图三相似',
    intro: [
      {
        title: '一图速记',
        body:
          '**直角三角形 + 斜边上的高 = 母子相似**。设 $\\angle C = 90°$，$CD \\perp AB$ 于 $D$，则斜边上的高把大直角三角形（母）分成两个小直角三角形（子），三者彼此相似。这一结构推出**射影定理**三公式，是初中相似三角形最经典的"一图三相似"。',
        image: 'model-mother-child.svg',
        imageCaption: '母子相似：直角三角形斜边上的高',
      },
      {
        title: '符号约定与图形',
        body:
          '直角 $\\triangle ABC$，$\\angle C = 90°$，$CD \\perp AB$ 于 $D$。记：\n- $AD = p$，$BD = q$，$CD = h$（斜边上的高）；\n- $BC = a$，$CA = b$，$AB = c$（斜边）。\n\n三相似关系：$\\triangle ACD \\sim \\triangle CBD \\sim \\triangle ABC$。',
        image: 'ex-p3s03-1.svg',
        imageCaption: '母子相似例题图（$p=4, q=9$）',
      },
      {
        title: '思维路径：为什么三者相似？',
        body:
          '**$\\triangle ACD \\sim \\triangle ABC$**：$\\angle ADC = 90° = \\angle ACB$，共锐角 $\\angle A$ $\\Rightarrow$ AA 相似，对应 $A\\leftrightarrow A, C\\leftrightarrow B, D\\leftrightarrow C$。\n\n**$\\triangle CBD \\sim \\triangle ABC$**：$\\angle BDC = 90° = \\angle ACB$，共锐角 $\\angle B$ $\\Rightarrow$ AA 相似，对应 $C\\leftrightarrow A, B\\leftrightarrow B, D\\leftrightarrow C$。\n\n两个小三角形都与大三角形相似，由相似的传递性，两小三角形彼此也相似。',
        tip: '识别关键：每对三角形都共一个直角 + 共一个锐角，AA 唾手可得。',
      },
      {
        title: '射影定理三公式',
        body:
          '由三相似的对应边比例，交叉相乘直接得到：',
        formula:
          'h^2 = pq,\\qquad b^2 = pc,\\qquad a^2 = qc',
        tip: '记忆口诀："高平方等于两射影之积；直角边平方等于它在斜边上的射影乘斜边。"',
      },
      {
        title: '射影定理推出勾股定理',
        body:
          '这是最优雅的勾股定理证明之一，只用相似，无需面积拼接：',
        formula:
          'a^2 + b^2 = qc + pc = (p+q)c = c \\cdot c = c^2',
        steps: [
          '由 $\\triangle CBD \\sim \\triangle ABC$ 得 $a^2 = qc$',
          '由 $\\triangle ACD \\sim \\triangle ABC$ 得 $b^2 = pc$',
          '两式相加：$a^2 + b^2 = (p+q)c = c^2$',
        ],
      },
      {
        title: '思路路标',
        body:
          '- 看到**直角三角形 + 斜边上的高** → 立刻调出射影定理三公式；\n- 求 $h$ → 用 $h^2 = pq$，即 $h = \\sqrt{pq}$（$h$ 是 $p, q$ 的几何平均）；\n- 求直角边 $a$ 或 $b$ → 用 $a^2 = qc$ 或 $b^2 = pc$；\n- 看到**圆中直径 + 圆周角 + 垂线** → 直径所对圆周角为直角，构成母子相似；\n- 找不到突破口时，作高是制造母子相似的标准辅助线。',
        reveal: {
          q: '直角 $\\triangle ABC$ 中 $\\angle C = 90°$，$CD \\perp AB$，$AD = 4$，$BD = 9$，求 $CD$。',
          a: '由 $h^2 = pq = 4 \\times 9 = 36$，得 $CD = h = 6$。',
        },
      },
      {
        title: '圆中应用',
        body:
          '圆 $O$ 的直径 $AB$，$C$ 是圆上一点，$CD \\perp AB$ 于 $D$。由"直径所对圆周角为直角"，$\\angle ACB = 90°$，整张图就是标准母子相似图。\n\n由射影定理 $CD^2 = AD \\cdot DB$，可直接求 $CD$，无需其他辅助。',
        image: 'ex-p3s03-3.svg',
        imageCaption: '圆中母子相似：直径 + 圆周角 + 垂线',
        formula: 'CD^2 = AD \\cdot DB',
      },
    ],
    questions: [
      {
        type: 'input',
        prompt:
          '直角 $\\triangle ABC$ 中，$\\angle C = 90°$，$CD \\perp AB$ 于 $D$，$AD = 4$，$BD = 9$，求 $CD$。',
        accept: ['6'],
        explain:
          '由射影定理 $CD^2 = AD \\cdot BD = 4 \\times 9 = 36$，故 $CD = 6$。',
      },
      {
        type: 'choice',
        prompt:
          '直角 $\\triangle ABC$ 中，$\\angle C = 90°$，$CD \\perp AB$ 于 $D$，$AC = b$，$AB = c$，$AD = p$，则以下哪个等式正确？',
        options: [
          '$b^2 = qc$',
          '$b^2 = pc$',
          '$h^2 = pc$',
          '$b = p + q$',
        ],
        answer: 1,
        explain:
          '射影定理：$b^2 = pc$（直角边 $b = CA$ 在斜边 $c = AB$ 上的射影是 $p = AD$，因为 $\\angle A$ 是 $\\triangle ACD$ 和 $\\triangle ABC$ 的公共角，$C \\leftrightarrow B$）。',
      },
      {
        type: 'judge',
        prompt:
          '在母子相似中，$\\triangle ACD \\sim \\triangle CBD$ 是由相似的传递性得到的，不需要单独证明。',
        answer: true,
        explain:
          '$\\triangle ACD \\sim \\triangle ABC$ 且 $\\triangle CBD \\sim \\triangle ABC$，由相似的传递性（与同一三角形相似的两个三角形彼此相似），得 $\\triangle ACD \\sim \\triangle CBD$，无需重新验证角度。',
      },
      {
        type: 'input',
        prompt:
          '圆的直径 $AB = 10$，$C$ 是圆上一点，$CD \\perp AB$ 于 $D$，$AD = 2$，求 $CD$。',
        accept: ['4'],
        explain:
          '$\\angle ACB = 90°$（直径所对圆周角），构成母子相似，$DB = AB - AD = 10 - 2 = 8$，由 $CD^2 = AD \\cdot DB = 2 \\times 8 = 16$，得 $CD = 4$。',
      },
      {
        type: 'match',
        prompt: '将射影定理各公式与其描述配对。',
        left: [
          '$h^2 = pq$',
          '$b^2 = pc$',
          '$a^2 = qc$',
        ],
        right: [
          '直角边 $a = BC$ 的平方等于其射影 $q = BD$ 乘斜边',
          '斜边上高的平方等于高两侧射影之积',
          '直角边 $b = CA$ 的平方等于其射影 $p = AD$ 乘斜边',
        ],
        explain:
          '$h^2 = pq$ 是高与两射影的关系；$b^2 = pc$ 是 $CA$ 边与射影 $AD$ 的关系；$a^2 = qc$ 是 $BC$ 边与射影 $BD$ 的关系。',
      },
      {
        type: 'choice',
        prompt:
          '直角 $\\triangle ABC$ 中，$\\angle C = 90°$，$CD \\perp AB$ 于 $D$，$AC = 6$，$AB = 9$，则 $AD = $',
        options: ['$3$', '$4$', '$5$', '$6$'],
        answer: 1,
        explain:
          '由 $b^2 = pc$，即 $AC^2 = AD \\cdot AB$，$36 = AD \\times 9$，故 $AD = 4$。',
      },
      {
        type: 'judge',
        prompt:
          '射影定理 $h^2 = pq$ 可以直接推导出勾股定理 $a^2 + b^2 = c^2$。',
        answer: true,
        explain:
          '由射影定理 $a^2 = qc$，$b^2 = pc$，相加得 $a^2 + b^2 = (p+q)c = c \\cdot c = c^2$，这正是勾股定理。只用相似，无需面积拼接，是最简洁的证明之一。',
      },
    ],
  },
]
