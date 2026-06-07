export const LESSONS = [
  {
    id: 'u1c3',
    title: '子集、真子集与幂集',
    subtitle: '集合之间的包含关系',
    intro: [
      {
        title: '包含关系：从元素 ∈ 到集合 ⊆',
        body: '第 1 章学的 $\\in$ 描述"元素与集合"的关系；本章讲的 $\\subseteq$ 描述"集合与集合"的关系。如果集合 $A$ 里的每一个元素都出现在集合 $B$ 里，我们就说 $A$ 包含于 $B$，记作 $A \\subseteq B$。这就是子集关系，是含参集合讨论题和充要条件（第 5 章）的语言基础。',
      },
      {
        title: '子集的定义与基本性质',
        body: '定义：$A \\subseteq B$ 当且仅当"对任意 $x$，若 $x \\in A$，则 $x \\in B$"。\n三条基本性质：\n①自反性：$A \\subseteq A$，每个集合是自身的子集；\n②反对称性：若 $A \\subseteq B$ 且 $B \\subseteq A$，则 $A = B$；\n③传递性：若 $A \\subseteq B$ 且 $B \\subseteq C$，则 $A \\subseteq C$。\n此外：$A \\cap B \\subseteq A \\subseteq A \\cup B$（交集最小，并集最大）。',
        formula: 'A \\subseteq B \\iff \\forall x,\\; x \\in A \\Rightarrow x \\in B',
      },
      {
        title: '真子集：$A \\subsetneq B$',
        body: '真子集比子集多一个"严格"的要求：$A \\subsetneq B$ 指 $A \\subseteq B$ 且 $A \\neq B$，即 $B$ 里至少有一个元素不在 $A$ 里。$\\subseteq$ 好比 $\\leq$（允许等号），$\\subsetneq$ 好比 $<$（不允许等号）。\n例：$\\{1,2\\} \\subseteq \\{1,2,3\\}$（是子集，也是真子集）；$\\{1,2\\} \\subseteq \\{1,2\\}$（是子集，但不是真子集）。',
        formula: 'A \\subsetneq B \\iff A \\subseteq B \\text{ 且 } A \\neq B',
        tip: '题目问"真子集个数"时，要从子集总数 $2^n$ 中减去 $A$ 本身，得 $2^n - 1$。',
      },
      {
        title: '空集是任何集合的子集',
        body: '无论 $A$ 是什么集合，$\\varnothing \\subseteq A$ 都成立。\n为什么？子集定义是"$\\varnothing$ 的每个元素都在 $A$ 中"。但 $\\varnothing$ 没有任何元素，没有反例，这个命题空真成立。\n在含参集合题中，当参数值使某集合为空集时，$\\varnothing \\subseteq A$ 仍然成立，必须单独列入答案，这是含参题最常见的漏点。',
        formula: '\\varnothing \\subseteq A \\text{ 对一切集合 } A \\text{ 成立}',
      },
      {
        title: '集合相等：双向包含法',
        body: '两集合相等 $A = B$ 当且仅当 $A \\subseteq B$ 且 $B \\subseteq A$（互相包含）。这是证明两集合相等的标准方法：\n①取任意 $x \\in A$，证明 $x \\in B$（证 $A \\subseteq B$）；\n②取任意 $x \\in B$，证明 $x \\in A$（证 $B \\subseteq A$）。\n例：$A = \\{x \\mid x^2 - x - 2 = 0\\} = \\{-1, 2\\}$，$B = \\{x \\mid (x-2)(x+1)=0\\} = \\{-1,2\\}$，两个方程等价，故 $A = B$。',
        formula: 'A = B \\iff A \\subseteq B \\text{ 且 } B \\subseteq A',
      },
      {
        title: '幂集：所有子集的集合',
        body: '集合 $A$ 的所有子集拼成的新集合叫 $A$ 的幂集，记作 $2^A$。若 $|A| = n$，则 $|2^A| = 2^n$。\n原因：$A$ 的每个子集由"对每个元素独立决定选或不选"来确定，$n$ 个元素各自二选一，共 $2^n$ 种，对应 $2^n$ 个不同子集。\n例：$A = \\{a, b, c\\}$（3 个元素），有 $2^3 = 8$ 个子集，按层列举：$\\varnothing$，$\\{a\\}$，$\\{b\\}$，$\\{c\\}$，$\\{a,b\\}$，$\\{a,c\\}$，$\\{b,c\\}$，$\\{a,b,c\\}$。',
        formula: '|A| = n \\Rightarrow |2^A| = 2^n',
        tip: '幂集 $2^A$ 的元素是集合（子集），不是原集合 $A$ 中的元素。$1 \\in 2^{\\{1,2\\}}$ 是错的；正确写法是 $\\{1\\} \\in 2^{\\{1,2\\}}$。',
      },
      {
        title: '含参集合：$B \\subseteq A$ 的参数求解',
        body: '例题：设 $A = \\{-1, 1\\}$，$B = \\{x \\mid mx = 1\\}$，若 $B \\subseteq A$，求 $m$ 的所有可能值。\n分析：先讨论 $B$ 的元素个数。\n①当 $m = 0$ 时，$0 \\cdot x = 1$ 无解，$B = \\varnothing$；$\\varnothing \\subseteq A$ 成立，$m = 0$ 合法。\n②当 $m \\neq 0$ 时，$B = \\{\\frac{1}{m}\\}$，需 $\\frac{1}{m} \\in A = \\{-1,1\\}$，即 $m = -1$ 或 $m = 1$。\n综合：$m \\in \\{-1, 0, 1\\}$。',
        reveal: {
          q: '集合 $\\{1,2,3\\}$ 的子集个数和真子集个数各是多少？',
          a: '子集个数：$2^3 = 8$；真子集个数：$2^3 - 1 = 7$（去掉集合本身）。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '集合 $A = \\{1, 2\\}$ 的子集个数是多少？',
        options: ['2', '3', '4', '8'],
        answer: 2,
        explain: '$|A| = 2$，子集个数为 $2^2 = 4$，分别是 $\\varnothing$，$\\{1\\}$，$\\{2\\}$，$\\{1,2\\}$。',
      },
      {
        type: 'judge',
        prompt: '$\\varnothing \\subseteq \\{1, 2, 3\\}$（空集是任何集合的子集）。',
        answer: true,
        explain: '正确。空集没有任何元素，因此"空集的每个元素都在 $\\{1,2,3\\}$ 中"是空真命题，成立。$\\varnothing$ 是任何集合的子集。',
      },
      {
        type: 'choice',
        prompt: '设 $A = \\{-1, 1\\}$，$B = \\{x \\mid mx = 1\\}$，若 $B \\subseteq A$，则 $m$ 的所有可能值是？',
        options: [
          '$m = 1$ 或 $m = -1$',
          '$m = 0$',
          '$m \\in \\{-1, 0, 1\\}$',
          '$m$ 可以是任意实数',
        ],
        answer: 2,
        explain: '当 $m=0$ 时 $B = \\varnothing$，$\\varnothing \\subseteq A$ 成立；当 $m \\neq 0$ 时 $B = \\{\\frac{1}{m}\\}$，需 $\\frac{1}{m} = \\pm 1$，即 $m = \\pm 1$。综合：$m \\in \\{-1,0,1\\}$。注意不能漏掉 $m=0$ 的情形。',
      },
      {
        type: 'match',
        prompt: '将左侧描述与右侧含义配对（$A$、$B$ 为集合）。',
        left: [
          '$A \\subseteq B$ 且 $B \\subseteq A$',
          '$A \\subseteq B$ 且 $A \\neq B$',
          '$\\forall x,\\; x \\in A \\Rightarrow x \\in B$',
          '$A$ 的所有子集组成的集合',
        ],
        right: [
          '$A = B$',
          '$A \\subsetneq B$（真子集）',
          '$A \\subseteq B$（子集定义）',
          '$2^A$（幂集）',
        ],
        explain: '双向包含等价于集合相等；子集加"不等于"就是真子集；子集的定义即全称蕴含；幂集是所有子集的集合。',
      },
      {
        type: 'choice',
        prompt: '集合 $\\{1,2,3,4\\}$ 的真子集个数是多少？',
        options: ['15', '16', '8', '14'],
        answer: 0,
        explain: '$|A| = 4$，子集共 $2^4 = 16$ 个，真子集去掉 $A$ 本身，共 $16 - 1 = 15$ 个。',
      },
      {
        type: 'judge',
        prompt: '$\\{1\\} \\in 2^{\\{1,2\\}}$（"含 1 的单元素集"是 $\\{1,2\\}$ 的幂集的一个元素）。',
        answer: true,
        explain: '正确。$2^{\\{1,2\\}} = \\{\\varnothing, \\{1\\}, \\{2\\}, \\{1,2\\}\\}$，其中 $\\{1\\}$ 是它的一个元素。注意：$1 \\in 2^{\\{1,2\\}}$ 是错的，$\\{1\\} \\in 2^{\\{1,2\\}}$ 才是对的。',
      },
      {
        type: 'input',
        prompt: '设 $A = \\{x \\mid 1 \\leq x \\leq 4\\}$，$B = \\{x \\mid a \\leq x \\leq a+1\\}$，若 $B \\subseteq A$，则 $a$ 的范围为？（请写成不等式，如 $1 \\leq a \\leq 3$）',
        accept: ['1≤a≤3', '1<=a<=3', '1\\leq a\\leq 3', '1 ≤ a ≤ 3'],
        explain: '$B \\subseteq A$ 要求 $B$ 的左端点 $a \\geq 1$ 且右端点 $a+1 \\leq 4$，即 $1 \\leq a \\leq 3$。（$B$ 不为空集时 $a \\leq a+1$ 恒成立，无需额外讨论。）',
      },
      {
        type: 'choice',
        prompt: '下列关于子集、真子集的说法，正确的是？',
        options: [
          '任何集合都没有真子集',
          '$A \\subsetneq B$ 蕴含 $A \\subseteq B$，反之不一定',
          '$A \\subseteq B$ 与 $A \\subsetneq B$ 等价',
          '$\\varnothing$ 没有子集',
        ],
        answer: 1,
        explain: '真子集是子集的特殊情形，$A \\subsetneq B$ 一定有 $A \\subseteq B$；但 $A \\subseteq B$ 可能 $A = B$，不一定是真子集。$\\varnothing$ 有一个子集，就是 $\\varnothing$ 本身。',
      },
    ],
  },
]
