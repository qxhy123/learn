export const LESSONS = [
  {
    id: 'u1c6',
    title: '二次根式',
    subtitle: '有意义条件、三条核心性质、最简根式与同类根式',
    intro: [
      {
        title: '什么是二次根式？',
        body: '形如 $\\sqrt{a}$（$a \\geq 0$）的代数式叫做**二次根式**，$a$ 叫做**被开方数**。\n\n当被开方数中含有字母或代数式时，就进入了"二次根式"的领域。例如 $\\sqrt{3}$、$\\sqrt{2x-1}$、$\\sqrt{a^2+b^2}$ 都是二次根式。\n\n注意：根指数为 $2$ 时省略不写；$\\sqrt[3]{\\cdot}$ 是三次根式（根指数 $3$ 不省略）。',
      },
      {
        title: '有意义的条件——被开方数 $\\geq 0$',
        body: '**根本要求**：被开方数必须 $\\geq 0$，否则在实数范围内无意义。\n\n含字母时，令被开方数 $\\geq 0$，解出字母的取值范围：\n- $\\sqrt{x-1}$ 有意义 $\\Leftrightarrow$ $x - 1 \\geq 0$ $\\Leftrightarrow$ $x \\geq 1$\n- $\\sqrt{2-x}$ 有意义 $\\Leftrightarrow$ $2 - x \\geq 0$ $\\Leftrightarrow$ $x \\leq 2$',
        tip: '遇到含字母的根式，第一步：列不等式确认取值范围。这是解题的前提条件。',
      },
      {
        title: '性质 1：平方消根',
        body: '$（\\sqrt{a}）^2 = a \\quad (a \\geq 0)$\n\n平方与开方互为逆运算，两者叠加后相消。\n\n例：$(\\sqrt{7})^2 = 7$；$(\\sqrt{x+1})^2 = x+1$（需 $x \\geq -1$）。\n\n这条性质在证明题和化简中经常用到。',
        formula: '(\\sqrt{a})^2 = a \\quad (a \\geq 0)',
      },
      {
        title: '性质 2 & 3：积/商的根号拆合',
        body: '**性质 2（积）**：$\\sqrt{ab} = \\sqrt{a} \\cdot \\sqrt{b}$（$a \\geq 0$，$b \\geq 0$）\n\n用途：把完全平方因子提到根号外（化最简）。\n例：$\\sqrt{12} = \\sqrt{4 \\times 3} = 2\\sqrt{3}$\n\n**性质 3（商）**：$\\sqrt{\\dfrac{a}{b}} = \\dfrac{\\sqrt{a}}{\\sqrt{b}}$（$a \\geq 0$，$b > 0$）\n\n用途：消去被开方数中的分母。\n例：$\\sqrt{\\dfrac{3}{4}} = \\dfrac{\\sqrt{3}}{2}$',
        tip: '性质 2 和 3 只适用于被开方数非负的情形！$\\sqrt{(-2)(-3)}=\\sqrt{6}$，但 $\\sqrt{-2} \\cdot \\sqrt{-3}$ 无意义。',
      },
      {
        title: '最简二次根式',
        body: '一个二次根式是**最简二次根式**，当且仅当同时满足：\n1. 被开方数中**不含完全平方因子**（整数无平方因子；字母指数均为 $1$）\n2. 被开方数中**不含分母**\n\n化最简的两步：\n- **去平方因子**：用性质 2，把 $4, 9, x^2$ 等提出根号\n- **分母有理化**：用性质 3，消去被开方数中的分母',
        tip: '$\\sqrt{72}$ 化最简：$\\sqrt{4 \\times 18} = 2\\sqrt{18}$——$\\sqrt{18}$ 仍有平方因子 $9$，未完成。正确：$\\sqrt{72} = \\sqrt{36 \\times 2} = 6\\sqrt{2}$。',
      },
      {
        title: '同类二次根式',
        body: '两个二次根式化为最简后，若**被开方数相同**，则称为**同类二次根式**（类比整式中的同类项）。\n\n同类根式才能像同类项一样合并（加减系数）：\n$m\\sqrt{a} \\pm n\\sqrt{a} = (m \\pm n)\\sqrt{a}$\n\n**例**：$\\sqrt{18}$ 化简后是 $3\\sqrt{2}$，$5\\sqrt{2}$ 被开方数同为 $2$，它们是同类根式：\n$\\sqrt{18} + 5\\sqrt{2} = 3\\sqrt{2} + 5\\sqrt{2} = 8\\sqrt{2}$',
        reveal: {
          q: '$\\sqrt{18} + \\sqrt{8} - \\sqrt{2}$ 等于什么？',
          a: '$\\sqrt{18}=3\\sqrt{2}$，$\\sqrt{8}=2\\sqrt{2}$；$3\\sqrt{2}+2\\sqrt{2}-\\sqrt{2} = 4\\sqrt{2}$',
        },
      },
      {
        title: '例题：化为最简与取值范围',
        body: '化简：(1) $\\sqrt{50}$，(2) $\\sqrt{48}$，(3) $\\sqrt{8a^3}$（$a \\geq 0$）\n\n(1) $\\sqrt{50} = \\sqrt{25 \\times 2} = 5\\sqrt{2}$\n(2) $\\sqrt{48} = \\sqrt{16 \\times 3} = 4\\sqrt{3}$\n(3) $\\sqrt{8a^3} = \\sqrt{4a^2 \\cdot 2a} = \\sqrt{4a^2} \\cdot \\sqrt{2a} = 2a\\sqrt{2a}$\n\n求 $\\sqrt{x^2 - 4x + 4}$ 有意义的范围：注意 $x^2-4x+4 = (x-2)^2 \\geq 0$ 恒成立，故对所有实数 $x$ 均有意义，且 $\\sqrt{(x-2)^2} = |x-2|$。',
      },
      {
        title: '易错点总结',
        body: '1. 忘记有意义条件，直接把负数塞入根号。\n2. $\\sqrt{72}$ 未化到底，只做了 $2\\sqrt{18}$，$\\sqrt{18}$ 仍含平方因子。\n3. $\\sqrt{2}$ 与 $\\sqrt{8}$ 看起来不同，但 $\\sqrt{8} = 2\\sqrt{2}$，化简后是同类根式，可以合并。\n4. 性质 2 用在负数上：$\\sqrt{(-2)(-3)} = \\sqrt{6}$，但不能写成 $\\sqrt{-2} \\cdot \\sqrt{-3}$（无意义）。\n5. $\\sqrt{a^2}$ 写成 $a$ 而不写 $|a|$（当 $a$ 的正负未知时必须用绝对值）。',
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '$\\sqrt{3x+6}$ 有意义的条件是',
        options: ['$x \\geq 2$', '$x \\leq -2$', '$x \\geq -2$', '$x > -2$'],
        answer: 2,
        explain: '令 $3x + 6 \\geq 0$，解得 $3x \\geq -6$，即 $x \\geq -2$。注意边界 $x = -2$ 时 $3(-2)+6=0$，$\\sqrt{0}=0$ 有意义，故包含等号。',
      },
      {
        type: 'input',
        prompt: '化简 $\\sqrt{75} - \\sqrt{27} + \\sqrt{3}$ 的结果。',
        accept: ['3\\sqrt{3}', '$3\\sqrt{3}$'],
        explain: '$\\sqrt{75} = \\sqrt{25 \\times 3} = 5\\sqrt{3}$；$\\sqrt{27} = \\sqrt{9 \\times 3} = 3\\sqrt{3}$；$5\\sqrt{3} - 3\\sqrt{3} + \\sqrt{3} = (5-3+1)\\sqrt{3} = 3\\sqrt{3}$。',
      },
      {
        type: 'judge',
        prompt: '$\\sqrt{12}$ 是最简二次根式。',
        answer: false,
        explain: '$\\sqrt{12} = \\sqrt{4 \\times 3} = 2\\sqrt{3}$，被开方数中含完全平方因子 $4$，不是最简形式。$2\\sqrt{3}$ 才是最简二次根式。',
      },
      {
        type: 'choice',
        prompt: '下列各对根式中，是同类二次根式的是',
        options: [
          '$\\sqrt{2}$ 与 $\\sqrt{3}$',
          '$\\sqrt{18}$ 与 $3\\sqrt{2}$',
          '$\\sqrt{5}$ 与 $\\sqrt{10}$',
          '$2\\sqrt{6}$ 与 $\\sqrt{3}$',
        ],
        answer: 1,
        explain: '$\\sqrt{18} = 3\\sqrt{2}$，化简后被开方数均为 $2$，故 $\\sqrt{18}$ 与 $3\\sqrt{2}$ 是同类二次根式。其他选项化简后被开方数不同。',
      },
      {
        type: 'input',
        prompt: '化简 $\\sqrt{8a^3}$（$a \\geq 0$）的结果。',
        accept: ['2a\\sqrt{2a}', '$2a\\sqrt{2a}$'],
        explain: '$\\sqrt{8a^3} = \\sqrt{4a^2 \\cdot 2a} = \\sqrt{4a^2} \\cdot \\sqrt{2a} = 2a\\sqrt{2a}$（因 $a \\geq 0$，$\\sqrt{4a^2} = 2a$）。',
      },
      {
        type: 'match',
        prompt: '将各根式与其最简形式配对',
        left: ['$\\sqrt{50}$', '$\\sqrt{48}$', '$\\sqrt{\\dfrac{3}{4}}$', '$\\sqrt{12}$'],
        right: ['$5\\sqrt{2}$', '$4\\sqrt{3}$', '$\\dfrac{\\sqrt{3}}{2}$', '$2\\sqrt{3}$'],
        explain: '$\\sqrt{50}=\\sqrt{25\\times2}=5\\sqrt{2}$；$\\sqrt{48}=\\sqrt{16\\times3}=4\\sqrt{3}$；$\\sqrt{3/4}=\\sqrt{3}/2$；$\\sqrt{12}=\\sqrt{4\\times3}=2\\sqrt{3}$。',
      },
      {
        type: 'judge',
        prompt: '$\\sqrt{(-2)(-3)} = \\sqrt{-2} \\cdot \\sqrt{-3}$',
        answer: false,
        explain: '性质 $\\sqrt{ab} = \\sqrt{a} \\cdot \\sqrt{b}$ 要求 $a \\geq 0$ 且 $b \\geq 0$。$\\sqrt{-2}$ 和 $\\sqrt{-3}$ 在实数范围内无意义，不能拆开；而左边 $\\sqrt{(-2)(-3)} = \\sqrt{6}$ 是有意义的。',
      },
    ],
  },
]
