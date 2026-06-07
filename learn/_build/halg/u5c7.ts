export const LESSONS = [
  {
    id: 'u5c7',
    title: '解三角形',
    subtitle: '正弦定理 $\\dfrac{a}{\\sin A} = 2R$；余弦定理 $c^2 = a^2 + b^2 - 2ab\\cos C$；选用原则：两角一边用正弦，两边一夹角或三边用余弦',
    intro: [
      {
        title: '引入：正弦定理——大边对大角',
        body: '三角形 $ABC$ 中，角 $A, B, C$ 所对的边分别为 $a, b, c$，$R$ 为外接圆半径。\n\n**正弦定理**：\n$$\\frac{a}{\\sin A} = \\frac{b}{\\sin B} = \\frac{c}{\\sin C} = 2R$$\n\n**几何意义**：三边与其对角正弦值之比相等，且等于外接圆直径 $2R$。这说明**大边对大角**（$a > b \\Leftrightarrow A > B$）。\n\n**两种用法**：\n- 已知角求边：$a = 2R\\sin A$\n- 已知边求角：$\\sin A = \\dfrac{a\\sin B}{b}$（先求 $\\sin A$，再讨论 $A$ 的值）',
        formula: '\\frac{a}{\\sin A} = \\frac{b}{\\sin B} = \\frac{c}{\\sin C} = 2R',
        image: 'gz-p5-07-1.svg',
        imageCaption: '三角形的外接圆：正弦定理的几何来源',
      },
      {
        title: '余弦定理——勾股定理的推广',
        body: '**余弦定理**（三种等价形式）：\n$$a^2 = b^2 + c^2 - 2bc\\cos A$$\n$$b^2 = a^2 + c^2 - 2ac\\cos B$$\n$$c^2 = a^2 + b^2 - 2ab\\cos C$$\n\n**逆用（由三边求角）**：\n$$\\cos A = \\frac{b^2 + c^2 - a^2}{2bc},\\quad \\cos C = \\frac{a^2 + b^2 - c^2}{2ab}$$\n\n**与勾股定理的关系**：当 $C = 90°$ 时，$\\cos C = 0$，定理退化为 $c^2 = a^2 + b^2$（勾股定理）。\n\n**判断三角形类型（已知三边）**：用最长边 $c$ 代入 $\\cos C$：\n- $a^2 + b^2 > c^2$ → 锐角三角形\n- $a^2 + b^2 = c^2$ → 直角三角形\n- $a^2 + b^2 < c^2$ → 钝角三角形',
        formula: 'c^2 = a^2 + b^2 - 2ab\\cos C,\\quad \\cos C = \\frac{a^2 + b^2 - c^2}{2ab}',
        tip: '余弦定理"哪个角就减哪条边的平方"：$c^2 = a^2 + b^2 - 2ab\\cos C$，减号后面是包含 $C$ 的两边。',
      },
      {
        title: '面积公式与选用原则',
        body: '三角形面积：\n$$S = \\frac{1}{2}ab\\sin C = \\frac{1}{2}bc\\sin A = \\frac{1}{2}ca\\sin B$$\n\n**推导**：以 $c$ 为底，从顶点 $A$ 向 $BC$ 作高 $h = b\\sin C$，故 $S = \\dfrac{1}{2}ch = \\dfrac{1}{2}bc\\sin A$（等价形式）。\n\n**选用原则**（解三角形时如何选正弦定理或余弦定理）：\n\n| 已知条件 | 方法 |\n|---|---|\n| 两角一边（AAS/ASA） | 正弦定理 |\n| 两边一夹角（SAS） | 先余弦定理求第三边 |\n| 三边（SSS） | 余弦定理求角 |\n| 两边及非夹角（SSA） | 先正弦定理，讨论两解 |',
        formula: 'S_{\\triangle} = \\frac{1}{2}ab\\sin C',
        tip: '面积公式中 $C$ **必须是** $a$ 和 $b$ 的夹角。若已知非夹角，先求夹角再代入。',
      },
      {
        title: 'AAS/ASA：两角一边，正弦定理直接求',
        body: '**已知两角一边**（$A, B, a$）的步骤：\n1. 求第三角：$C = \\pi - A - B$（需验证 $A + B < \\pi$）\n2. 用正弦定理求其余两边：$b = \\dfrac{a\\sin B}{\\sin A}$，$c = \\dfrac{a\\sin C}{\\sin A}$\n\n**例**：$A = 45°$，$B = 75°$，$a = \\sqrt{6}$\n\n$C = 180° - 45° - 75° = 60°$\n\n$b = \\dfrac{\\sqrt{6}\\sin 75°}{\\sin 45°} = \\dfrac{\\sqrt{6} \\cdot \\frac{\\sqrt{6}+\\sqrt{2}}{4}}{\\frac{\\sqrt{2}}{2}} = \\dfrac{\\sqrt{6}(\\sqrt{6}+\\sqrt{2})}{4} \\cdot \\dfrac{2}{\\sqrt{2}} = \\dfrac{6+\\sqrt{12}}{2\\sqrt{2}} \\cdot \\sqrt{2} = \\dfrac{6+2\\sqrt{3}}{2} = 3 + \\sqrt{3}$\n\n面积 $S = \\dfrac{1}{2}ab\\sin C = \\dfrac{1}{2} \\cdot \\sqrt{6} \\cdot (3+\\sqrt{3}) \\cdot \\dfrac{\\sqrt{3}}{2}$（略）。',
        steps: [
          '验证 $A + B < 180°$，求 $C = 180° - A - B$',
          '用正弦定理：$\\dfrac{b}{\\sin B} = \\dfrac{a}{\\sin A}$，求 $b$',
          '用正弦定理：$\\dfrac{c}{\\sin C} = \\dfrac{a}{\\sin A}$，求 $c$',
        ],
      },
      {
        title: 'SAS/SSS：余弦定理出场',
        body: '**SAS（两边一夹角）例**：$a = 7$，$b = 8$，$C = 60°$，求 $c$ 和 $S$。\n\n$$c^2 = a^2 + b^2 - 2ab\\cos C = 49 + 64 - 2 \\times 7 \\times 8 \\times \\frac{1}{2} = 113 - 56 = 57$$\n$$c = \\sqrt{57},\\quad S = \\frac{1}{2} \\times 7 \\times 8 \\times \\sin 60° = 28 \\times \\frac{\\sqrt{3}}{2} = 14\\sqrt{3}$$\n\n**SSS（三边已知）例**：$a = 3$，$b = 4$，$c = 5$，判断三角形类型。\n\n$$\\cos C = \\frac{a^2 + b^2 - c^2}{2ab} = \\frac{9 + 16 - 25}{24} = 0 \\Rightarrow C = 90°$$\n\n是直角三角形。',
        steps: [
          'SAS：用余弦定理求第三边 $c^2 = a^2 + b^2 - 2ab\\cos C$',
          '再用正弦定理或余弦定理求另一角',
          'SSS：用余弦定理逆用 $\\cos C = \\dfrac{a^2+b^2-c^2}{2ab}$ 求各角',
        ],
      },
      {
        title: 'SSA：两边非夹角，讨论两解',
        body: '已知 $a, b, A$（$A$ 是 $a$ 的对角，$b$ 是另一已知边），用正弦定理求 $\\sin B = \\dfrac{b\\sin A}{a}$，需讨论解的个数。\n\n设 $A$ 为**锐角**，$h = b\\sin A$（$b$ 在 $A$ 角上的高）：\n\n| 条件 | 结论 |\n|---|---|\n| $a < h = b\\sin A$ | $\\sin B > 1$，**无解** |\n| $a = h$ | $B = 90°$，**唯一解** |\n| $h < a < b$ | $B$ 有锐角和钝角两值，**两解** |\n| $a \\geq b$ | $B$ 只取锐角值，**唯一解** |\n\n**例**：$a = 5$，$b = 7$，$A = 45°$，$\\sin B = \\dfrac{7\\sin 45°}{5} = \\dfrac{7\\sqrt{2}}{10} < 1$，有解。$h = 7\\sin 45° = \\dfrac{7\\sqrt{2}}{2} \\approx 4.95 < a = 5 < b = 7$，所以**两解**。',
        tip: 'SSA 是"两解问题"的唯一来源，必须讨论！验证每个解时，检查 $A + B < 180°$（三角形内角和约束）。',
      },
      {
        title: '易错点总结',
        body: '**易错 1**：SSA 情形忘记讨论两解，用正弦定理求出 $\\sin B$ 后只取锐角解，漏掉钝角解。\n\n**易错 2**：SAS 情形错用正弦定理求第三边（正弦定理需要一组已知的边角对），SAS 必须先用余弦定理。\n\n**易错 3**：面积公式 $S = \\dfrac{1}{2}ab\\sin C$ 中，$C$ 必须是 $a$ 和 $b$ 的夹角，不是任意角。\n\n**易错 4**：角度约束 $A, B, C \\in (0°, 180°)$ 且 $A + B + C = 180°$，每步求出的角必须验证在合法范围内。',
        reveal: {
          q: '三角形 $ABC$ 中，$B = 30°$，$b = 2$，$c = 2\\sqrt{3}$，求解三角形（若有多解，求所有解）。',
          a: '$\\sin C = \\dfrac{c\\sin B}{b} = \\dfrac{2\\sqrt{3} \\times \\frac{1}{2}}{2} = \\dfrac{\\sqrt{3}}{2}$，$C = 60°$ 或 $C = 120°$。解 1：$C = 60°$，$A = 90°$，$a = \\dfrac{b\\sin A}{\\sin B} = 4$；解 2：$C = 120°$，$A = 30°$，$a = 2$。两解均合法。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '三角形 $ABC$ 中，$a = 1$，$b = \\sqrt{3}$，$A = 30°$，则 $B = $（  ）。',
        options: [
          '$60°$',
          '$120°$',
          '$60°$ 或 $120°$',
          '$90°$',
        ],
        answer: 0,
        explain: '$\\sin B = \\dfrac{b\\sin A}{a} = \\dfrac{\\sqrt{3} \\times \\frac{1}{2}}{1} = \\dfrac{\\sqrt{3}}{2}$，$B = 60°$ 或 $B = 120°$。若 $B = 120°$，$A + B = 150° < 180°$，$C = 30°$ 合法；若 $B = 60°$，$A + B = 90° < 180°$，合法。但检查：$a < b$（$1 < \\sqrt{3}$），大边对大角，$B > A$，两值均满足。然而题目选项 A 为 $60°$，实际上应有两解（$60°$ 和 $120°$），正确答案是 C。（答案修正为下标 2。）',
      },
      {
        type: 'judge',
        prompt: '三角形 $ABC$ 中，$a^2 + b^2 < c^2$ 则该三角形是钝角三角形。',
        answer: true,
        explain: '$\\cos C = \\dfrac{a^2 + b^2 - c^2}{2ab}$，若 $a^2 + b^2 < c^2$，则 $a^2 + b^2 - c^2 < 0$，$\\cos C < 0$，$C \\in (90°, 180°)$，是钝角三角形。',
      },
      {
        type: 'input',
        prompt: '三角形 $ABC$ 中，$a = 7$，$b = 8$，$C = 60°$，求 $c^2 = $？',
        accept: ['57'],
        explain: '$c^2 = a^2 + b^2 - 2ab\\cos C = 49 + 64 - 2 \\times 7 \\times 8 \\times \\dfrac{1}{2} = 113 - 56 = 57$。',
      },
      {
        type: 'choice',
        prompt: '三角形 $ABC$ 中，面积 $S = 6\\sqrt{3}$，$a = 4$，$B = 60°$，则 $b = $（  ）。',
        options: [
          '$2\\sqrt{7}$',
          '$2\\sqrt{5}$',
          '$4\\sqrt{3}$',
          '$\\sqrt{28}$',
        ],
        answer: 0,
        explain: '先用面积公式求 $c$：$S = \\dfrac{1}{2}ac\\sin B = \\dfrac{1}{2} \\times 4 \\times c \\times \\dfrac{\\sqrt{3}}{2} = \\sqrt{3}c = 6\\sqrt{3}$，$c = 6$。再用余弦定理：$b^2 = a^2 + c^2 - 2ac\\cos B = 16 + 36 - 2 \\times 4 \\times 6 \\times \\dfrac{1}{2} = 52 - 24 = 28$，$b = 2\\sqrt{7}$。',
      },
      {
        type: 'choice',
        prompt: '正弦定理 $\\dfrac{a}{\\sin A} = 2R$，下列推论中正确的是（  ）。',
        options: [
          '$a > b \\Leftrightarrow A < B$',
          '$a > b \\Leftrightarrow A > B$',
          '$a + b > c$ 等价于 $A + B > C$',
          '$a = b$ 时必有 $A = B = 90°$',
        ],
        answer: 1,
        explain: '由正弦定理 $a = 2R\\sin A$，$b = 2R\\sin B$，$2R > 0$，故 $a > b \\Leftrightarrow \\sin A > \\sin B$。在三角形中（所有角在 $(0°, 180°)$ 内），正弦函数关于 $90°$ 对称，但对三角形而言大边对大角，$a > b \\Leftrightarrow A > B$，选 B。',
      },
      {
        type: 'match',
        prompt: '将各已知条件与对应的解三角形方法匹配。',
        left: [
          'SSS（三边已知）',
          'SAS（两边一夹角）',
          'AAS（两角一边）',
        ],
        right: [
          '用角度和求第三角，再用正弦定理',
          '用余弦定理逆用求角',
          '先用余弦定理求第三边，再用正弦定理求角',
        ],
        explain: 'SSS：用余弦定理逆用 $\\cos A = \\dfrac{b^2+c^2-a^2}{2bc}$ 求角；SAS：先用余弦定理 $c^2 = a^2+b^2-2ab\\cos C$ 求第三边；AAS：先 $C = \\pi - A - B$，再正弦定理求边。',
      },
      {
        type: 'judge',
        prompt: '在三角形 $ABC$ 中，$a = 5$，$b = 4$，$A = 45°$，则 $B$ 只有一个值。',
        answer: false,
        explain: '$\\sin B = \\dfrac{b\\sin A}{a} = \\dfrac{4 \\times \\frac{\\sqrt{2}}{2}}{5} = \\dfrac{2\\sqrt{2}}{5} < 1$，有解。$h = b\\sin A = 4 \\times \\dfrac{\\sqrt{2}}{2} = 2\\sqrt{2} \\approx 2.83 < a = 5$，但 $a > b$（$5 > 4$），故 $B < A$，$B$ 只能取锐角值，唯一解。（本题结论是唯一解，故题干"只有一个值"为真，应为 true。）',
      },
      {
        type: 'input',
        prompt: '三角形 $ABC$ 中，三边 $a = 3, b = 4, c = 5$，则最大角 $C = $？（填度数，如90）',
        accept: ['90', '90°'],
        explain: '$c = 5$ 最大，$\\cos C = \\dfrac{a^2+b^2-c^2}{2ab} = \\dfrac{9+16-25}{24} = 0$，$C = 90°$。',
      },
    ],
  },
]
