export const LESSONS = [
  {
    id: 'u1c5',
    title: '充分条件、必要条件与充要条件',
    subtitle: '逻辑推断与集合包含关系',
    intro: [
      {
        title: '引入：一道判断关系的题',
        body: '设 $p: x > 1$，$q: x > 0$，问 $p$ 是 $q$ 的什么条件？\n若 $x > 1$，则一定 $x > 0$（因为 $1 > 0$），所以 $p \\Rightarrow q$ 成立；\n但若 $x > 0$，不一定 $x > 1$（反例：$x = 0.5$），所以 $q \\not\\Rightarrow p$。\n结论：$p$ 是 $q$ 的充分不必要条件。这道题有两种解法，本关都会讲到。',
      },
      {
        title: '充分条件与必要条件的定义',
        body: '若 $p \\Rightarrow q$（从 $p$ 可以推出 $q$），则称 $p$ 是 $q$ 的充分条件，$q$ 是 $p$ 的必要条件。\n"充分"：有了 $p$ 就够了，$q$ 必然成立。\n"必要"：$q$ 是 $p$ 能成立的必要前提，没有 $q$ 就不可能有 $p$（逆否命题：$\\neg q \\Rightarrow \\neg p$）。\n若 $p \\Rightarrow q$ 且 $q \\Rightarrow p$（双向互推），则 $p$ 是 $q$ 的充要条件，记 $p \\Leftrightarrow q$。',
        formula: 'p \\Rightarrow q \\iff P \\subseteq Q',
        tip: '记忆锚点：箭头从 $p$ 出发（$p \\Rightarrow q$），所以 $p$ 是让 $q$ 成立的"充分"理由；$q$ 是 $p$ 能成立的"必要"前提。充分在前，必要在后，方向不能搞反。',
      },
      {
        title: '集合视角：最快的判断方法',
        body: '用 $P = \\{x \\mid p(x)\\}$，$Q = \\{x \\mid q(x)\\}$ 分别表示使 $p$、$q$ 成立的真值集，则：\n$P \\subsetneq Q$（$P$ 更小）$\\Rightarrow$ $p$ 充分不必要；\n$Q \\subsetneq P$（$Q$ 更小）$\\Rightarrow$ $p$ 必要不充分；\n$P = Q$ $\\Rightarrow$ $p$ 充要；\n$P$、$Q$ 互不包含 $\\Rightarrow$ 既不充分也不必要。\n引入题：$P = (1,+\\infty)$，$Q = (0,+\\infty)$，$P \\subsetneq Q$，故 $p$ 充分不必要。',
        tip: '画数轴！把 $P$ 和 $Q$ 画成区间，包含关系一眼看出，比逻辑推断更不容易出错，高考首选。',
      },
      {
        title: '逆命题、否命题与逆否命题',
        body: '给定命题"若 $p$ 则 $q$"：\n逆命题：若 $q$ 则 $p$（不一定等价）\n否命题：若 $\\neg p$ 则 $\\neg q$（不一定等价）\n逆否命题：若 $\\neg q$ 则 $\\neg p$（与原命题等价！）\n口诀：原命题与逆否命题等价；逆命题与否命题等价（互为逆否）。\n用途：若 $p \\Rightarrow q$ 难以直接证明，可改证逆否命题 $\\neg q \\Rightarrow \\neg p$，两者真假完全一致。',
        formula: 'p \\Rightarrow q \\iff \\neg q \\Rightarrow \\neg p',
      },
      {
        title: '含参题：已知充分/必要，求参数',
        body: '例题：设 $p: 0 < x < 3$，$q: 0 < x < a$（$a > 0$），若 $p$ 是 $q$ 的充分不必要条件，求 $a$ 的范围。\n分析：$P = (0,3)$，$Q = (0,a)$。$p$ 充分不必要 $\\iff P \\subsetneq Q$。\n$P \\subseteq Q$ 要求 $Q$ 的右端点 $a \\geq 3$；\n$P \\neq Q$（严格包含）要求 $a \\neq 3$，即 $a > 3$。\n解：$a > 3$。\n套路：充分条件 $\\to P \\subseteq Q$；必要条件 $\\to Q \\subseteq P$；充要 $\\to P = Q$。把集合写成区间，列端点不等式组求解。',
        formula: 'p \\text{ 充分于 } q \\iff P \\subseteq Q,\\quad p \\text{ 必要于 } q \\iff Q \\subseteq P',
      },
      {
        title: '充要条件的双向证明',
        body: '证明 $p \\Leftrightarrow q$ 必须写两步：\n①充分性（$p \\Rightarrow q$）：假设 $p$ 成立，推出 $q$ 成立；\n②必要性（$q \\Rightarrow p$）：假设 $q$ 成立，推出 $p$ 成立。\n例：证明"$a^2+b^2=0$（$a,b \\in \\mathbb{R}$）"是"$a=0$ 且 $b=0$"的充要条件。\n充分性：若 $a=0$ 且 $b=0$，则 $a^2+b^2=0$，显然成立。\n必要性：若 $a^2+b^2=0$，由 $a^2 \\geq 0$，$b^2 \\geq 0$，两非负数之和为 0，只能 $a^2=b^2=0$，故 $a=b=0$。两步完成，充要条件得证。',
        formula: 'p \\Leftrightarrow q \\iff (p \\Rightarrow q) \\text{ 且 } (q \\Rightarrow p)',
      },
      {
        title: '易错：充分和必要方向搞反',
        body: '高频混淆：$p \\Rightarrow q$ 时，$p$ 是充分条件，$q$ 是必要条件——新手常把充分和必要对应的条件说反。\n验证方法：若"$p$ 是 $q$ 的充分条件"，应能从 $p$ 推出 $q$（$p \\Rightarrow q$），即 $P \\subseteq Q$。若反了（$Q \\subseteq P$），那是必要不充分。\n另一个易错点：含参子集题中，$B = \\varnothing$ 的情形（第 3 章学过），在充要条件题里同样要单独讨论。',
        reveal: {
          q: '设 $p: x = 1$，$q: x^2 = 1$，判断 $p$ 是 $q$ 的什么条件。',
          a: '$P = \\{1\\}$，$Q = \\{-1, 1\\}$，$P \\subsetneq Q$，故 $p$ 是 $q$ 的充分不必要条件。（$x=1 \\Rightarrow x^2=1$ 成立；但 $x=-1$ 时 $x^2=1$ 成立而 $x \\neq 1$，故 $q \\not\\Rightarrow p$。）',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '设 $p: x > 2$，$q: x > 1$，则 $p$ 是 $q$ 的什么条件？',
        options: [
          '充要条件',
          '充分不必要条件',
          '必要不充分条件',
          '既不充分也不必要条件',
        ],
        answer: 1,
        explain: '$P = (2,+\\infty)$，$Q = (1,+\\infty)$，$P \\subsetneq Q$（$P$ 更小）。$p \\Rightarrow q$（$x>2$ 则 $x>1$，成立）；$q \\not\\Rightarrow p$（反例：$x = 1.5$，满足 $q$ 但不满足 $p$）。故 $p$ 是 $q$ 的充分不必要条件。',
      },
      {
        type: 'judge',
        prompt: '原命题"若 $p$ 则 $q$"与逆否命题"若 $\\neg q$ 则 $\\neg p$"等价。',
        answer: true,
        explain: '正确。原命题与逆否命题真假完全一致（互相等价）。当原命题难以直接证明时，可改证逆否命题，两者效果相同。逆命题和否命题是等价的一对，但与原命题不一定等价。',
      },
      {
        type: 'choice',
        prompt: '设 $p: 0 < x < 3$，$q: 0 < x < a$（$a > 0$），若 $p$ 是 $q$ 的充分不必要条件，则 $a$ 的范围是？',
        options: [
          '$0 < a < 3$',
          '$a = 3$',
          '$a > 3$',
          '$a \\geq 3$',
        ],
        answer: 2,
        explain: '$P = (0,3)$，$Q = (0,a)$。$p$ 充分不必要 $\\iff P \\subsetneq Q$，即 $Q$ 包含 $P$ 且严格更大，需要 $a > 3$（若 $a = 3$ 则 $P = Q$，不是"不必要"）。',
      },
      {
        type: 'match',
        prompt: '将左侧集合关系与右侧条件类型配对（$P = \\{x \\mid p(x)\\}$，$Q = \\{x \\mid q(x)\\}$）。',
        left: [
          '$P \\subsetneq Q$',
          '$Q \\subsetneq P$',
          '$P = Q$',
          '$P$ 与 $Q$ 互不包含',
        ],
        right: [
          '$p$ 充分不必要于 $q$',
          '$p$ 必要不充分于 $q$',
          '$p$ 是 $q$ 的充要条件',
          '$p$ 既不充分也不必要于 $q$',
        ],
        explain: '$P \\subsetneq Q$ 时，$P$ 的每个元素都在 $Q$ 里（$p \\Rightarrow q$），但 $Q$ 更大（$q \\not\\Rightarrow p$），充分不必要；$Q \\subsetneq P$ 反之；$P = Q$ 时双向等价，充要；互不包含时两个方向都有反例，既不充分也不必要。',
      },
      {
        type: 'choice',
        prompt: '设 $p: a > b$，$q: a^2 > b^2$（$a, b \\in \\mathbb{R}$），则 $p$ 是 $q$ 的什么条件？',
        options: [
          '充分不必要条件',
          '必要不充分条件',
          '充要条件',
          '既不充分也不必要条件',
        ],
        answer: 3,
        explain: '两个方向各有反例：$p \\not\\Rightarrow q$（反例：$a=1, b=-2$，$a>b$ 但 $a^2=1<4=b^2$）；$q \\not\\Rightarrow p$（反例：$a=-2, b=1$，$a^2=4>1=b^2$ 但 $a<b$）。两方向都不能推出，故既不充分也不必要。',
      },
      {
        type: 'judge',
        prompt: '"$a=0$ 且 $b=0$"是"$a^2+b^2=0$（$a,b \\in \\mathbb{R}$）"的充要条件。',
        answer: true,
        explain: '正确。充分性：若 $a=b=0$，则 $a^2+b^2=0$，成立。必要性：若 $a^2+b^2=0$，由 $a^2 \\geq 0$，$b^2 \\geq 0$，两非负数和为 0 只能各为 0，故 $a=b=0$，成立。两方向均成立，充要条件。',
      },
      {
        type: 'choice',
        prompt: '设 $p: -1 \\leq x \\leq 5$，$q: x \\leq a$（$a$ 为实数），若 $p$ 是 $q$ 的充分条件，则 $a$ 的范围是？',
        options: [
          '$a \\geq 5$',
          '$a \\leq -1$',
          '$-1 \\leq a \\leq 5$',
          '$a \\leq 5$',
        ],
        answer: 0,
        explain: '$p$ 充分于 $q$ $\\iff P \\subseteq Q$。$P = [-1,5]$，$Q = (-\\infty, a]$。$P \\subseteq Q$ 要求 $P$ 的右端点（最大值）$5 \\leq a$，即 $a \\geq 5$。',
      },
      {
        type: 'input',
        prompt: '设 $p: m-1 \\leq x \\leq m+1$，$q: -1 \\leq x \\leq 3$，若 $p$ 是 $q$ 的充分不必要条件（即 $P \\subsetneq Q$），求 $m$ 的范围（写成不等式，如 $0 \\leq m \\leq 2$）。',
        accept: ['0≤m≤2', '0 ≤ m ≤ 2', '0<=m<=2', '0\\leq m\\leq 2'],
        explain: '$P = [m-1, m+1]$，$Q = [-1,3]$。$P \\subseteq Q$ 要求左端点 $m-1 \\geq -1$（即 $m \\geq 0$）且右端点 $m+1 \\leq 3$（即 $m \\leq 2$）。$P \\neq Q$ 自动满足（$P$ 长度为 2，$Q$ 长度为 4，两者不等）。故 $0 \\leq m \\leq 2$。',
      },
    ],
  },
]
