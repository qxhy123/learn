export const LESSONS = [
  {
    id: 'u6c2',
    title: 'A 字与 X 字相似模型',
    subtitle: '两平行线 + 共顶点截线 → 两相似三角形',
    intro: [
      {
        title: '一图速记',
        body:
          '**两条平行线被两条共顶点的直线所截**，必然形成两个相似三角形。共顶点在两平行线**同侧** → **A 字模型**；共顶点在两平行线**之间** → **X 字（8 字）模型**。这是本章所有比例线段题的根基。',
        image: 'model-a-shape.svg',
        imageCaption: 'A 字型相似：共顶点在两平行线同侧',
      },
      {
        title: '引入：三角形中的平行截线',
        body:
          '$\\triangle ABC$ 中，$DE \\parallel BC$，$D \\in AB$，$E \\in AC$。\n\n平行线能给什么？**唯一能用的就是同位角或内错角相等**。\n- $DE \\parallel BC$ 被 $AB$ 截 $\\Rightarrow$ 同位角 $\\angle ADE = \\angle ABC$；\n- 公共角 $\\angle A = \\angle A$。\n\n两组角相等，AA 直接得 $\\triangle ADE \\sim \\triangle ABC$。',
        image: 'ex-p3s02-0.svg',
        imageCaption: 'A 字引入：三角形内的平行截线',
      },
      {
        title: 'A 字模型',
        body:
          '**图形特征**：平行线 $l_1 \\parallel l_2$，两条直线共顶点 $P$，$P$ 在两平行线**同侧**（$l_1, l_2$ 都在 $P$ 的一边）。\n\n**凑 AA 方式**：平行 $\\Rightarrow$ 同位角相等；顶点公共 $\\Rightarrow$ 公共角相等。\n\n**核心比例**：$\\dfrac{AD}{AB} = \\dfrac{AE}{AC} = \\dfrac{DE}{BC}$（对应边之比 $=$ 相似比）。',
        formula: '\\frac{AD}{AB} = \\frac{AE}{AC} = \\frac{DE}{BC}',
        tip: 'A 字里"直接比例"：分子是短段，分母是长段（整体），$D$ 对 $B$，$E$ 对 $C$。',
      },
      {
        title: 'X 字 / 8 字模型',
        body:
          '**图形特征**：平行线 $l_1 \\parallel l_2$，两条直线共顶点 $P$，$P$ 在两平行线**之间**。两个三角形"头对头、尖儿冲尖儿"，形如字母 X 或数字 8。\n\n**凑 AA 方式**：平行 $\\Rightarrow$ 内错角相等；共顶点 $\\Rightarrow$ 对顶角相等。\n\n结论仍是两相似三角形，对应边成比例。',
        image: 'ex-p3s02-2.svg',
        imageCaption: 'X 字例题：两平行线 + 共顶点，共顶点在两线之间',
      },
      {
        title: 'A 字 vs X 字 核心区别',
        body:
          '| | A 字 | X 字 |\n|---|---|---|\n| 共顶点位置 | 两平行线同侧（外侧） | 两平行线之间 |\n| 凑 AA 的角 | 同位角 + 公共角 | 内错角 + 对顶角 |\n\n**本质相同**：都是"两平行线 + 共顶点两截线 $\\Rightarrow$ AA 相似"。区别只是顶点站哪一侧。',
        tip: '看到比例线段题，第一步永远是：找（或作）平行线，然后识别 A 字 / X 字。',
      },
      {
        title: '重要变形：梯形里找 A 字',
        body:
          '梯形 $ABCD$（$AD \\parallel BC$），延长两腰 $BA$、$CD$ 交于 $P$，则 $\\triangle PAD \\sim \\triangle PBC$（A 字相似）。\n\n相似比 $= AD:BC$（上底比下底）。由此可求腰的延长段长度，是中考高频套路。',
        image: 'ex-p3s02-3.svg',
        imageCaption: '梯形延长两腰交于 P，构成 A 字相似',
        formula: '\\frac{PA}{PB} = \\frac{PD}{PC} = \\frac{AD}{BC}',
      },
      {
        title: '逆向 A 字与思路路标',
        body:
          'A 字不仅可以"平行 $\\Rightarrow$ 相似 $\\Rightarrow$ 比例"，也可以逆用：\n- 已知 $\\dfrac{AD}{AB} = \\dfrac{AE}{AC}$ 且有公共角 $\\angle A$ $\\Rightarrow$ SAS 相似 $\\Rightarrow \\angle ADE = \\angle ABC$ $\\Rightarrow DE \\parallel BC$。\n\n**路标速查**：\n- 三角形 + 内部平行线 → A 字；\n- 两直线交一点 + 两平行线 → X 字；\n- 梯形 + 延长两腰 → A 字（相似比 = 上底/下底）；\n- 题目要求比例式 → 先找（或作）平行线。',
        reveal: {
          q: '$\\triangle ABC$ 中 $DE \\parallel BC$，$\\dfrac{AD}{DB} = \\dfrac{2}{3}$，$DE = 4$，则 $BC = ?$',
          a: '$\\dfrac{AD}{AB} = \\dfrac{2}{5}$（因为 $AB = AD + DB$），由 A 字相似 $\\dfrac{DE}{BC} = \\dfrac{2}{5}$，故 $BC = 10$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt:
          '$\\triangle ABC$ 中，$DE \\parallel BC$，$D \\in AB$，$E \\in AC$，$AD = 3$，$DB = 2$，$BC = 10$，则 $DE = $',
        options: ['$5$', '$6$', '$4$', '$8$'],
        answer: 1,
        explain:
          'A 字相似，$\\triangle ADE \\sim \\triangle ABC$，相似比 $k = \\dfrac{AD}{AB} = \\dfrac{3}{5}$，故 $DE = k \\cdot BC = \\dfrac{3}{5} \\times 10 = 6$。',
      },
      {
        type: 'judge',
        prompt:
          'X 字模型中，两个相似三角形的对应角是通过"对顶角相等"和"内错角相等"凑出的 AA 条件。',
        answer: true,
        explain:
          'X 字（共顶点在两平行线之间）：共顶点处形成对顶角，平行线给出内错角，两者合力凑出 AA，从而判定相似。',
      },
      {
        type: 'input',
        prompt:
          '两直线交于 $O$，分别交两平行线 $l_1, l_2$ 于 $A, C$（同一截线）和 $B, D$（同一截线），$OA = 4$，$OC = 6$，$OB = 5$，求 $OD$。',
        accept: ['7.5', '15/2'],
        explain:
          'X 字相似，$\\triangle OAB \\sim \\triangle OCD$（对顶角 + 内错角，AA），故 $\\dfrac{OA}{OC} = \\dfrac{OB}{OD}$，即 $\\dfrac{4}{6} = \\dfrac{5}{OD}$，解得 $OD = \\dfrac{15}{2} = 7.5$。',
      },
      {
        type: 'choice',
        prompt:
          '梯形 $ABCD$ 中 $AD \\parallel BC$，$AD = 4$，$BC = 8$，延长 $BA, CD$ 交于 $P$，$PA = 5$，则 $AB = $',
        options: ['$3$', '$4$', '$5$', '$6$'],
        answer: 2,
        explain:
          'A 字相似 $\\triangle PAD \\sim \\triangle PBC$，相似比 $= \\dfrac{AD}{BC} = \\dfrac{4}{8} = \\dfrac{1}{2}$，故 $\\dfrac{PA}{PB} = \\dfrac{1}{2}$，$PB = 2PA = 10$，$AB = PB - PA = 10 - 5 = 5$。',
      },
      {
        type: 'match',
        prompt: '将图形特征与相似模型对应。',
        left: [
          '共顶点在两平行线同侧，同位角 + 公共角',
          '共顶点在两平行线之间，对顶角 + 内错角',
          '梯形延长两腰交于顶点',
        ],
        right: ['X 字 / 8 字模型', 'A 字模型（梯形变形）', 'A 字模型'],
        explain:
          '共顶点同侧是 A 字；共顶点居中是 X 字；梯形延腰是 A 字的梯形变形版本。',
      },
      {
        type: 'judge',
        prompt:
          '$\\triangle ABC$ 中，$\\dfrac{AD}{AB} = \\dfrac{AE}{AC}$ 且 $D \\in AB$、$E \\in AC$，可以推出 $DE \\parallel BC$。',
        answer: true,
        explain:
          '已知 $\\dfrac{AD}{AB} = \\dfrac{AE}{AC}$ 加公共角 $\\angle A$，由 SAS 相似得 $\\triangle ADE \\sim \\triangle ABC$，从而 $\\angle ADE = \\angle ABC$（同位角相等），故 $DE \\parallel BC$。这是 A 字模型的逆向使用。',
      },
      {
        type: 'input',
        prompt:
          '$\\triangle ABC$ 中 $DE \\parallel BC$，$\\dfrac{AD}{DB} = \\dfrac{1}{3}$，$BC = 20$，求 $DE$。',
        accept: ['5'],
        explain:
          '$\\dfrac{AD}{DB} = \\dfrac{1}{3}$，故 $AD = \\dfrac{1}{4} AB$，相似比 $k = \\dfrac{AD}{AB} = \\dfrac{1}{4}$，$DE = k \\cdot BC = \\dfrac{1}{4} \\times 20 = 5$。',
      },
    ],
  },
]
