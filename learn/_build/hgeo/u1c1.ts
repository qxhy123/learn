export const LESSONS = [
  {
    id: 'u1c1',
    title: '向量的概念',
    subtitle: '大小与方向——向量的两张名片',
    intro: [
      {
        title: '为什么需要向量？',
        body: '数量只描述大小（如质量 $5\\,\\text{kg}$、温度 $25\\,^{\\circ}\\text{C}$），但生活中很多量还有方向：力 $\\vec{F}$（$10\\,\\text{N}$，向东）、位移 $\\vec{s}$（$5\\,\\text{m}$，向北）。只知道力的大小而不知道方向，无法判断物体如何运动。因此，数学引入了「向量」这一工具来同时描述大小和方向。',
        tip: '数量（scalar）只有大小；向量（vector）既有大小又有方向。',
      },
      {
        title: '向量的定义',
        body: '既有大小又有方向的量叫做**向量**。向量的大小称为**模**，记作 $|\\vec{a}|$ 或 $|\\overrightarrow{AB}|$，是一个非负实数。几何上用**有向线段**表示向量：起点 $A$ 是出发点，终点 $B$ 是箭头所指，线段长度等于模。',
        formula: '\\overrightarrow{AB}',
        image: 'geo-p1-01-1.svg',
        imageCaption: '有向线段 $\\overrightarrow{AB}$ 示意：起点 $A$，终点 $B$，箭头指向 $B$',
      },
      {
        title: '向量的自由性——与起点无关',
        body: '向量由大小和方向完全决定，与起点在哪里**无关**。把 $\\vec{a}$ 平移到任何位置，只要模和方向不变，就仍然是 $\\vec{a}$。这叫做向量的**自由性**。\n\n例如：在平行四边形 $ABCD$ 中，$\\overrightarrow{AB}$ 与 $\\overrightarrow{DC}$ 长度相等、方向相同（都从左指右），故 $\\overrightarrow{AB} = \\overrightarrow{DC}$，尽管起点不同。',
        tip: '向量是"自由向量"，不固定在某一起点。不同位置的有向线段，只要模和方向相同，就代表同一个向量。',
      },
      {
        title: '特殊向量：零向量与单位向量',
        body: '**零向量** $\\vec{0}$：模为 $0$，方向任意（不确定）。它是加法的单位元：$\\vec{a} + \\vec{0} = \\vec{a}$。\n\n**单位向量** $\\hat{e}$：模等于 $1$，即 $|\\hat{e}| = 1$。单位向量只携带方向信息。任意非零向量 $\\vec{a}$ 都可以除以其模，得到同方向的单位向量 $\\dfrac{\\vec{a}}{|\\vec{a}|}$。',
        formula: '|\\vec{0}| = 0,\\quad |\\hat{e}| = 1',
        tip: '零向量的模是 $0$；单位向量的模是 $1$。两者都特殊，但含义截然不同。',
      },
      {
        title: '相等向量——两个条件缺一不可',
        body: '向量 $\\vec{a}$ 与 $\\vec{b}$ **相等**（记 $\\vec{a} = \\vec{b}$）需同时满足：\n① 模相同：$|\\vec{a}| = |\\vec{b}|$\n② 方向相同\n\n**反例**：$\\overrightarrow{AB}$ 与 $\\overrightarrow{BA}$ 模相同但方向相反，所以 $\\overrightarrow{AB} \\neq \\overrightarrow{BA}$。只满足一个条件不够！',
        steps: [
          '第一步：比较两向量的模（长度）是否相等',
          '第二步：比较两向量的方向是否完全一致（不能相反）',
          '两步都满足 → 相等；任一步不满足 → 不相等',
        ],
      },
      {
        title: '共线向量与相反向量',
        body: '**共线（平行）向量**：两个向量方向相同或相反，记 $\\vec{a} \\parallel \\vec{b}$。\n零向量与任意向量共线（特别规定，为了后续数乘 $\\lambda = 0$ 时不出现例外）。\n\n**相反向量**：模相同、方向相反，记 $\\vec{b} = -\\vec{a}$。性质：$\\vec{a} + (-\\vec{a}) = \\vec{0}$，且 $-\\overrightarrow{AB} = \\overrightarrow{BA}$（起终点互换）。',
        formula: '-\\overrightarrow{AB} = \\overrightarrow{BA}',
        image: 'geo-p1-01-2.svg',
        imageCaption: '相等向量（模同向同）与共线向量（方向同或反）示例',
      },
      {
        title: '典型例题：平行四边形中的向量关系',
        body: '**例题**：平行四边形 $ABCD$ 中（$AB \\parallel DC$，$AD \\parallel BC$），判断 $\\overrightarrow{AB}$ 与 $\\overrightarrow{DC}$ 是否相等；$\\overrightarrow{AB}$ 与 $\\overrightarrow{CD}$ 有何关系；$\\overrightarrow{AB}$ 与 $\\overrightarrow{BA}$ 有何关系？\n\n**解**：\n① $\\overrightarrow{AB}$ 与 $\\overrightarrow{DC}$：$A \\to B$ 与 $D \\to C$ 方向相同（均从左到右），且 $|AB| = |DC|$，故 $\\overrightarrow{AB} = \\overrightarrow{DC}$。\n② $\\overrightarrow{AB}$ 与 $\\overrightarrow{CD}$：$AB \\parallel CD$ 但 $C \\to D$ 与 $A \\to B$ 方向相反，故共线但不相等。\n③ $\\overrightarrow{AB}$ 与 $\\overrightarrow{BA}$：模相同、方向相反，故 $\\overrightarrow{BA} = -\\overrightarrow{AB}$，互为相反向量。',
        tip: '共线只要方向同或反；相等要求方向完全一致（不能相反）。相等比共线条件更强。',
      },
      {
        title: '易错点汇总',
        body: '**易错1**：以为向量绑定在起点——向量是自由的，起点不同但模方向相同就相等。\n\n**易错2**：忘记零向量与任意向量共线——$\\vec{0}$ 与任意 $\\vec{a}$ 共线，这是定义规定。\n\n**易错3**：相等向量只满足一个条件——模相同 $\\neq$ 相等；方向相同 $\\neq$ 相等；必须两者同时满足。\n\n**易错4**：$\\overrightarrow{AB}$ 与 $\\overrightarrow{BA}$ 模相同但不相等，是相反向量。\n\n**易错5**：单位向量有无穷多个（每个方向各有一个），零向量只有一个但方向不定。',
        reveal: {
          q: '正三角形 $ABC$ 中，$D$ 是 $BC$ 的中点，$\\overrightarrow{AB}$ 与 $\\overrightarrow{DC}$ 是否相等？',
          a: '不相等。$AB \\parallel DC$ 且方向相同（均沿同一方向），但 $|DC| = \\dfrac{1}{2}|BC| = \\dfrac{1}{2}|AB|$（正三角形三边相等，$D$ 是中点所以 $DC = \\dfrac{1}{2}BC$），模不相同，故不相等，只是共线。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '下列关于向量的说法中，正确的是（　　）',
        options: [
          '向量 $\\vec{a}$ 与 $\\vec{b}$ 模相同，则 $\\vec{a} = \\vec{b}$',
          '向量 $\\vec{a}$ 与 $\\vec{b}$ 相等，则它们的起点相同',
          '零向量 $\\vec{0}$ 与任意向量共线',
          '平面内的单位向量只有两个（向左和向右各一个）',
        ],
        answer: 2,
        explain: '选 C。A 错：相等需模同且方向同，仅模同不够（如 $\\overrightarrow{AB}$ 与 $\\overrightarrow{BA}$ 模同但方向反）。B 错：向量是自由向量，相等与起点无关。C 正确：零向量与任意向量共线（定义规定）。D 错：每个方向都有单位向量，平面内单位向量有无穷多个。',
      },
      {
        type: 'judge',
        prompt: '若 $\\overrightarrow{AB} = \\overrightarrow{CD}$，则 $A, B, C, D$ 四点构成平行四边形。',
        answer: false,
        explain: '错误。$\\overrightarrow{AB} = \\overrightarrow{CD}$ 说明 $AB \\parallel CD$ 且 $|AB| = |CD|$，方向 $A \\to B$ 与 $C \\to D$ 相同。这意味着 $ABDC$（注意顺序）构成平行四边形，或者 $A, B, C, D$ 四点中 $A = C$ 的情形，不能直接断定 $ABCD$ 构成平行四边形。',
      },
      {
        type: 'choice',
        prompt: '设 $\\vec{a}$ 方向向右、$|\\vec{a}| = 2$，下列向量中与 $\\vec{a}$ 相等的是（　　）',
        options: [
          '方向向右、模为 $3$ 的向量 $\\vec{p}$',
          '方向向左、模为 $2$ 的向量 $\\vec{q}$',
          '方向向右、模为 $2$、起点不同的向量 $\\vec{s}$',
          '方向向右、模为 $2$ 的零向量',
        ],
        answer: 2,
        explain: '选 C。向量与起点无关（自由性），只要方向向右且模为 $2$，无论起点在哪里都与 $\\vec{a}$ 相等。A 错：模不同（$3 \\neq 2$）。B 错：方向相反（$\\vec{q} = -\\vec{a}$）。D 错：模为 $2$ 的向量不是零向量，说法自相矛盾。',
      },
      {
        type: 'input',
        prompt: '在平行四边形 $ABCD$ 中（$AB \\parallel DC$，$AD \\parallel BC$），$\\overrightarrow{AB}$ 的相反向量是哪个标准记法（用字母顺序表示的有向线段）？',
        accept: ['\\overrightarrow{BA}', 'BA', '向量BA'],
        explain: '相反向量模相同、方向相反。$\\overrightarrow{AB}$ 从 $A$ 到 $B$，反方向即从 $B$ 到 $A$，故相反向量为 $\\overrightarrow{BA}$。注意 $\\overrightarrow{CD}$ 也与 $\\overrightarrow{AB}$ 反向（$C \\to D$ 与 $A \\to B$ 方向相反），且 $|CD| = |AB|$，故 $\\overrightarrow{CD}$ 也是 $\\overrightarrow{AB}$ 的相反向量。',
      },
      {
        type: 'match',
        prompt: '将左侧向量关系术语与右侧描述正确配对',
        left: ['相等向量', '共线向量', '相反向量'],
        right: ['模相同且方向相同，起点无关', '模相同且方向相反', '方向相同或相反（模可以不同）'],
        explain: '相等向量：模同 + 方向同（最强条件）。相反向量：模同 + 方向反（$\\vec{b} = -\\vec{a}$）。共线向量：方向同或反，模可以不同（最弱条件，相等和相反都是共线的特例）。',
      },
      {
        type: 'judge',
        prompt: '零向量 $\\vec{0}$ 的方向是不确定的，因此零向量与任意向量都不共线。',
        answer: false,
        explain: '错误。恰恰相反——零向量方向任意，规定它与**任意**向量共线。这样才能保证数乘章节中"$\\lambda = 0$ 时 $\\lambda\\vec{a} = \\vec{0}$ 与 $\\vec{a}$ 共线"不出现例外。',
      },
      {
        type: 'choice',
        prompt: '$M$ 是线段 $AB$ 的中点，$O$ 是平面上任意一点，以下正确的是（　　）',
        options: [
          '$\\overrightarrow{AM} = \\overrightarrow{BM}$',
          '$\\overrightarrow{AM} = \\overrightarrow{MB}$',
          '$\\overrightarrow{OM} = \\overrightarrow{OA}$',
          '$\\overrightarrow{OA}$ 与 $\\overrightarrow{OB}$ 一定方向相同',
        ],
        answer: 1,
        explain: '选 B。$M$ 是 $AB$ 中点，所以 $|AM| = |MB|$，且 $A \\to M$ 与 $M \\to B$ 方向相同（都沿 $AB$ 方向），故 $\\overrightarrow{AM} = \\overrightarrow{MB}$。A 错：$\\overrightarrow{AM}$ 从 $A$ 到 $M$，$\\overrightarrow{BM}$ 从 $B$ 到 $M$，方向相反。C 错：$M \\neq A$，终点不同。D 错：$O$ 是任意点，$\\overrightarrow{OA}$ 和 $\\overrightarrow{OB}$ 方向取决于 $O, A, B$ 的位置，不一定相同。',
      },
      {
        type: 'input',
        prompt: '已知 $|\\vec{a}| = 5$，$\\hat{a}$ 是与 $\\vec{a}$ 同向的单位向量，则 $|\\hat{a}|$ 等于多少？',
        accept: ['1'],
        explain: '单位向量的模定义为 $1$，记作 $|\\hat{a}| = 1$。无论原向量模多大，将其除以自身的模所得单位向量的模均为 $1$：$\\hat{a} = \\dfrac{\\vec{a}}{|\\vec{a}|} = \\dfrac{\\vec{a}}{5}$，故 $|\\hat{a}| = \\dfrac{|\\vec{a}|}{5} = 1$。',
      },
    ],
  },
]
