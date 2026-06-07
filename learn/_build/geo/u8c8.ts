export const LESSONS = [
  {
    id: 'u8c8',
    title: '中点四边形模型',
    subtitle: '第8单元第8关：四边中点连成的平行四边形',
    intro: [
      {
        title: '引入：一道看似简单的题',
        body: '任意四边形 $ABCD$，$E$、$F$、$G$、$H$ 分别是 $AB$、$BC$、$CD$、$DA$ 的中点，顺次连接得到四边形 $EFGH$。$EFGH$ 是什么形状？结论让人意外：无论原四边形形状如何，$EFGH$ 必然是平行四边形。',
        image: 'model-midpoint-quad-general.svg',
        imageCaption: '任意四边形的中点四边形必是平行四边形',
      },
      {
        title: '思维路径：用对角线作"桥梁"',
        body: '要证 $EFGH$ 是平行四边形，选"一组对边平行且相等"这条判定。看 $EF$：$E$ 是 $AB$ 中点，$F$ 是 $BC$ 中点——这是 $\\triangle ABC$ 的中位线！得 $EF \\parallel AC$ 且 $EF = \\frac{1}{2}AC$。看 $HG$：$H$ 是 $DA$ 中点，$G$ 是 $DC$ 中点——这是 $\\triangle ACD$ 的中位线！得 $HG \\parallel AC$ 且 $HG = \\frac{1}{2}AC$。两组合并：$EF \\parallel HG$ 且 $EF = HG$ → 平行四边形。',
        image: 'ex-p4-08-0.svg',
        imageCaption: '对角线 $AC$ 作为桥梁：$EF$ 和 $HG$ 各是三角形中位线，都平行于 $AC$ 且等于 $\\frac{1}{2}AC$',
        steps: [
          '连接对角线 $AC$',
          '$E$、$F$ 是 $\\triangle ABC$ 两边中点 → $EF \\parallel AC$，$EF = \\frac{1}{2}AC$',
          '$H$、$G$ 是 $\\triangle ACD$ 两边中点 → $HG \\parallel AC$，$HG = \\frac{1}{2}AC$',
          '故 $EF \\parallel HG$ 且 $EF = HG$ → $EFGH$ 是平行四边形',
        ],
      },
      {
        title: '形态映射表',
        body: '中点四边形的具体形态完全由原四边形对角线的两个属性决定：是否相等（给"菱形性"），是否垂直（给"矩形性"）。',
        tip: '核心口诀：原对角线相等 → 中点四边形邻边相等 → 菱形；原对角线垂直 → 中点四边形有直角 → 矩形；两者都满足 → 正方形。',
        image: 'model-midpoint-quad-rectangle.svg',
        imageCaption: '矩形的中点四边形是菱形（因为矩形对角线相等，导致中点四边形邻边相等）',
      },
      {
        title: '为什么原矩形 → 中点菱形？',
        body: '矩形对角线相等：$AC = BD$。由中位线，$EF = \\frac{1}{2}AC$，$EH = \\frac{1}{2}BD$，故 $EF = EH$。邻边相等的平行四边形 → 菱形。同理，菱形对角线垂直 $AC \\perp BD$ → $EF \\parallel AC$，$EH \\parallel BD$，故 $EF \\perp EH$ → 矩形。',
        formula: 'EF = \\tfrac{1}{2}AC,\\quad EH = \\tfrac{1}{2}BD',
      },
      {
        title: '完整的形态映射',
        body: '任意四边形 → 平行四边形；矩形（对角线相等）→ 菱形；菱形（对角线垂直）→ 矩形；正方形（相等且垂直）→ 正方形；等腰梯形（对角线相等）→ 菱形；筝形（对角线垂直）→ 矩形。性质通过对角线这一层"中转"传递。',
        reveal: {
          q: '菱形 $ABCD$ 中，$E$、$F$、$G$、$H$ 是四边中点，判断 $EFGH$ 的形状。',
          a: '菱形对角线互相垂直，传递到中点四边形：$EF \\perp EH$ → $EFGH$ 有一个直角，加上它是平行四边形 → $EFGH$ 是矩形（但不一定是正方形，因为菱形对角线一般不相等）。',
        },
      },
      {
        title: '反向推导：已知中点四边形推原对角线',
        body: '若已知中点四边形 $EFGH$ 是正方形，反推原四边形 $ABCD$ 的对角线条件：正方形 = 菱形 + 矩形 → $EF = EH$（菱形）→ $AC = BD$（相等）；$EF \\perp EH$（矩形）→ $AC \\perp BD$（垂直）。结论：原四边形对角线相等且垂直（原四边形不必是正方形！）。',
        image: 'ex-p4-08-3.svg',
        imageCaption: '中点四边形是正方形 → 原四边形对角线相等且垂直（原四边形可以是特殊筝形等）',
      },
      {
        title: '模型的本质与识别要点',
        body: '中点四边形模型本质是"对角线 + 三角形中位线定理 × 2次"。识别条件：四边形四边中点依次连接。解题路标：看到四中点 → 连对角线 → 两次中位线定理 → 打通两条对边。用原四边形的对角线性质（相等/垂直）决定中点四边形形态。',
        tip: '想证中点四边形是平行四边形而不知从何下手 → 默念"两次中位线 + 同一条对角线作桥梁"。',
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '任意四边形四边的中点依次连接，所得四边形一定是什么？',
        options: ['矩形', '菱形', '正方形', '平行四边形'],
        answer: 3,
        explain: '中点四边形模型的基本结论：无论原四边形形状如何，中点四边形必然是平行四边形。',
      },
      {
        type: 'judge',
        prompt: '矩形的中点四边形是正方形。',
        answer: false,
        explain: '矩形对角线相等，传递到中点四边形使邻边相等，故中点四边形是菱形而非正方形。要是正方形还需要对角线垂直，矩形不保证这一点。',
      },
      {
        type: 'choice',
        prompt: '菱形的中点四边形是什么形状？',
        options: ['菱形', '矩形', '正方形', '梯形'],
        answer: 1,
        explain: '菱形对角线互相垂直，传递到中点四边形使邻边垂直，故中点四边形有直角，加上已是平行四边形 → 矩形。',
      },
      {
        type: 'judge',
        prompt: '若四边形 $ABCD$ 的中点四边形 $EFGH$ 是菱形，则 $ABCD$ 的两条对角线必然相等。',
        answer: true,
        explain: '$EFGH$ 是菱形说明邻边 $EF = EH$，由 $EF = \\frac{1}{2}AC$，$EH = \\frac{1}{2}BD$，得 $AC = BD$，即两对角线相等。',
      },
      {
        type: 'match',
        prompt: '将原四边形类型与其中点四边形形态配对。',
        left: ['任意四边形', '矩形', '菱形', '正方形'],
        right: ['正方形', '矩形', '菱形', '平行四边形'],
        explain: '任意→平行四边形；矩形（对角线相等）→菱形；菱形（对角线垂直）→矩形；正方形（相等且垂直）→正方形。',
      },
      {
        type: 'input',
        prompt: '四边形 $ABCD$ 中，$AC = 12$，$BD = 12$，且 $AC \\perp BD$，$E$、$F$、$G$、$H$ 是四边中点。问 $EF$ 的长是多少（填数字）？',
        accept: ['6', '6cm'],
        explain: '$EF$ 是 $\\triangle ABC$ 的中位线，$EF = \\frac{1}{2}AC = \\frac{1}{2} \\times 12 = 6$。',
      },
      {
        type: 'choice',
        prompt: '若四边形 $ABCD$ 的中点四边形是正方形，则 $ABCD$ 的对角线满足什么条件？',
        options: [
          '相等',
          '垂直',
          '相等且垂直',
          '互相平分',
        ],
        answer: 2,
        explain: '中点四边形是正方形 = 菱形 + 矩形。菱形要求对角线相等，矩形要求对角线垂直，所以原四边形对角线必须相等且垂直。',
      },
      {
        type: 'judge',
        prompt: '中点四边形是矩形，则原四边形的对角线互相垂直。',
        answer: true,
        explain: '$EFGH$ 是矩形 → 邻边 $EF \\perp EH$。由 $EF \\parallel AC$，$EH \\parallel BD$，平行线传递夹角 → $AC \\perp BD$，即原对角线垂直。',
      },
    ],
  },
]
