export const LESSONS = [
  {
    id: 'u1c4',
    title: '全称量词、存在量词与命题否定',
    subtitle: '精确描述"所有"与"存在"',
    intro: [
      {
        title: '引入：一道让人停下来的题',
        body: '请先凭直觉写出答案：命题"对所有 $x \\in \\mathbb{R}$，$x^2 + 1 > 0$"的否定是什么？\n许多同学写"对所有 $x \\in \\mathbb{R}$，$x^2 + 1 \\leq 0$"——量词没变；还有人写"对所有 $x \\notin \\mathbb{R}$，$x^2 + 1 \\leq 0$"——连范围也改了。这两种都是错的。正确答案是"存在某个 $x \\in \\mathbb{R}$，使 $x^2 + 1 \\leq 0$"。为什么？本关讲清楚这件事。',
      },
      {
        title: '全称量词 ∀ 与存在量词 ∃',
        body: '全称命题：$\\forall x \\in M,\\; p(x)$，意为"对集合 $M$ 中的所有 $x$，性质 $p(x)$ 都成立"。$\\forall$ 读作"对任意"或"对所有"。\n存在命题：$\\exists x \\in M,\\; p(x)$，意为"集合 $M$ 中存在某个 $x$，使性质 $p(x)$ 成立"。$\\exists$ 读作"存在"或"至少有一个"。\n关键区别：全称命题被一个反例推翻；存在命题被找到一个例子证真。',
        formula: '\\forall x \\in M,\\; p(x) \\qquad \\exists x \\in M,\\; p(x)',
      },
      {
        title: '三步否定法',
        body: '否定一个含量词的命题，只需三步，缺一不可：\n①变更量词：$\\forall$ 变 $\\exists$，$\\exists$ 变 $\\forall$；\n②否定结论：按反义符号表替换（$>$ 变 $\\leq$，$\\geq$ 变 $<$，$=$ 变 $\\neq$，依此类推）；\n③保留范围：$x \\in M$ 这一约束不动，否则讨论的是另一个问题。\n口诀：变量词、否结论、留范围。',
        formula: '\\neg(\\forall x \\in M,\\; p(x)) \\iff \\exists x \\in M,\\; \\neg p(x)',
        tip: '否定 $>$ 不是 $<$，而是 $\\leq$（包含等号）！否定 $\\geq$ 不是 $\\leq$，而是 $<$。漏掉等号是高频丢分点。',
      },
      {
        title: '反义符号表：结论怎么否定',
        body: '否定结论时按如下对应关系替换：\n$=$ 变 $\\neq$；$\\neq$ 变 $=$；$>$ 变 $\\leq$；$\\geq$ 变 $<$；$<$ 变 $\\geq$；$\\leq$ 变 $>$；$\\in$ 变 $\\notin$。\n若结论含"且"（$p$ 且 $q$），否定后变"或"（$\\neg p$ 或 $\\neg q$）——德摩根律 $\\neg(p \\wedge q) = \\neg p \\vee \\neg q$。\n若结论含"或"（$p$ 或 $q$），否定后变"且"（$\\neg p$ 且 $\\neg q$）——德摩根律 $\\neg(p \\vee q) = \\neg p \\wedge \\neg q$。',
        formula: '\\neg(\\exists x \\in M,\\; p(x)) \\iff \\forall x \\in M,\\; \\neg p(x)',
      },
      {
        title: '验证技巧：真假互换',
        body: '一个命题与它的否定真假相反：若原命题为真，则否定命题为假；若原命题为假，则否定命题为真。这是快速检验答案的方法。\n例：命题"$\\forall x \\in \\mathbb{R},\\; x^2+1>0$"为真（因为 $x^2 \\geq 0$，故 $x^2+1 \\geq 1 > 0$）。\n否定命题"$\\exists x \\in \\mathbb{R},\\; x^2+1 \\leq 0$"应为假——确实找不到这样的 $x$，验证正确。',
        tip: '做完否定题后，立刻判断原命题和否定命题的真假，看是否恰好相反，这是自我验证的最快方法。',
      },
      {
        title: '结论含"且"或"或"时的否定',
        body: '例：命题"$\\forall x \\in \\mathbb{R},\\; x^2 \\geq 0$ 且 $x^2+1>0$"的否定。\n量词：$\\forall \\to \\exists$；结论"$p$ 且 $q$"否定为"$\\neg p$ 或 $\\neg q$"（德摩根律）。\n故否定为"$\\exists x \\in \\mathbb{R},\\; x^2 < 0$ 或 $x^2+1 \\leq 0$"。\n常见错误：把"且"直接带过去，写"$x^2 < 0$ 且 $x^2+1 \\leq 0$"——忘记了德摩根律。',
        formula: '\\neg(p \\wedge q) = \\neg p \\vee \\neg q,\\qquad \\neg(p \\vee q) = \\neg p \\wedge \\neg q',
      },
      {
        title: '量词命题与恒成立/存在性问题',
        body: '全称命题 $\\forall x \\in D,\\; f(x) > k$ 等价于函数在 $D$ 上恒大于 $k$，即 $\\min_{x \\in D} f(x) > k$（转化为最小值问题）。\n存在命题 $\\exists x \\in D,\\; f(x) > k$ 等价于函数在 $D$ 上最大值大于 $k$，即 $\\max_{x \\in D} f(x) > k$（转化为最大值问题）。\n例：$\\forall x \\in \\mathbb{R},\\; x^2-2ax+a > 0$ 成立，要求二次函数恒正，即判别式 $\\Delta = 4a^2-4a < 0$，解得 $0 < a < 1$（注意还需 $a > 0$ 保证开口向上，实际 $a > 0$ 已包含在 $0 < a < 1$ 内）。',
        reveal: {
          q: '写出命题"$\\exists n \\in \\mathbb{N}^*,\\; n^2 < n$"的否定。',
          a: '$\\forall n \\in \\mathbb{N}^*,\\; n^2 \\geq n$。量词 $\\exists$ 变 $\\forall$，结论 $< n$ 否定为 $\\geq n$，范围 $n \\in \\mathbb{N}^*$ 保留。（原命题为假，否定命题为真——对正整数 $n$，$n^2 = n \\cdot n \\geq n$。）',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '命题"$\\forall x \\in \\mathbb{R},\\; x^2 + 1 > 0$"的否定是？',
        options: [
          '$\\forall x \\in \\mathbb{R},\\; x^2 + 1 \\leq 0$',
          '$\\exists x \\in \\mathbb{R},\\; x^2 + 1 \\leq 0$',
          '$\\exists x \\notin \\mathbb{R},\\; x^2 + 1 \\leq 0$',
          '$\\exists x \\in \\mathbb{R},\\; x^2 + 1 < 0$',
        ],
        answer: 1,
        explain: '三步：①量词 $\\forall \\to \\exists$；②结论 $> 0$ 否定为 $\\leq 0$（不是 $< 0$，要含等号）；③范围 $x \\in \\mathbb{R}$ 保留不变。故否定为"$\\exists x \\in \\mathbb{R},\\; x^2+1 \\leq 0$"。',
      },
      {
        type: 'judge',
        prompt: '命题"$\\forall x \\geq 0,\\; \\sqrt{x} \\geq 0$"的否定是"$\\exists x < 0,\\; \\sqrt{x} < 0$"。',
        answer: false,
        explain: '错误。范围 $x \\geq 0$ 是"讨论的舞台"，否定时不能改变它。正确否定是"$\\exists x \\geq 0,\\; \\sqrt{x} < 0$"（量词变、结论否定、范围保留）。',
      },
      {
        type: 'choice',
        prompt: '命题"$\\forall x \\in \\mathbb{R},\\; x^2 \\geq 0$ 且 $x^2+1>0$"的否定中，结论部分应写成？',
        options: [
          '$x^2 < 0$ 且 $x^2+1 \\leq 0$',
          '$x^2 < 0$ 或 $x^2+1 \\leq 0$',
          '$x^2 \\leq 0$ 或 $x^2+1 < 0$',
          '$x^2 \\leq 0$ 且 $x^2+1 < 0$',
        ],
        answer: 1,
        explain: '结论"$p$ 且 $q$"的否定是"$\\neg p$ 或 $\\neg q$"（德摩根律）。$x^2 \\geq 0$ 的否定是 $x^2 < 0$；$x^2+1 > 0$ 的否定是 $x^2+1 \\leq 0$。两者用"或"连接。',
      },
      {
        type: 'match',
        prompt: '将左侧命题与右侧其否定配对。',
        left: [
          '$\\forall x \\in \\mathbb{R},\\; x^3 > 0$',
          '$\\exists n \\in \\mathbb{N}^*,\\; n^2 < n$',
          '$\\forall x \\in [0,1],\\; x^2 \\leq x$',
          '$\\exists x \\in \\mathbb{R},\\; x^2+x+1 \\leq 0$',
        ],
        right: [
          '$\\exists x \\in \\mathbb{R},\\; x^3 \\leq 0$',
          '$\\forall n \\in \\mathbb{N}^*,\\; n^2 \\geq n$',
          '$\\exists x \\in [0,1],\\; x^2 > x$',
          '$\\forall x \\in \\mathbb{R},\\; x^2+x+1 > 0$',
        ],
        explain: '每条否定：①$\\forall \\to \\exists$，$> 0$ 否定为 $\\leq 0$；②$\\exists \\to \\forall$，$< n$ 否定为 $\\geq n$；③$\\forall \\to \\exists$，$\\leq x$ 否定为 $> x$；④$\\exists \\to \\forall$，$\\leq 0$ 否定为 $> 0$。范围均保留。',
      },
      {
        type: 'choice',
        prompt: '命题"$\\exists x \\in \\mathbb{R},\\; x^2+x+1 \\leq 0$"是真命题还是假命题？',
        options: [
          '真命题，因为存在这样的 $x$',
          '假命题，因为 $x^2+x+1 > 0$ 恒成立',
          '无法判断，需要具体 $x$ 的值',
          '真命题，取 $x = -1$ 即可',
        ],
        answer: 1,
        explain: '$x^2+x+1 = (x+\\frac{1}{2})^2 + \\frac{3}{4} \\geq \\frac{3}{4} > 0$ 对所有实数 $x$ 成立，找不到使 $x^2+x+1 \\leq 0$ 的实数，故原存在命题为假命题。',
      },
      {
        type: 'judge',
        prompt: '若原命题为真，则其否定命题一定为假。',
        answer: true,
        explain: '正确。一个命题与它的否定真假相反，这是逻辑的基本原则。若原命题真，否定命题假；若原命题假，否定命题真。这是验证否定是否写对的快速方法。',
      },
      {
        type: 'input',
        prompt: '命题"$\\forall x \\in \\mathbb{R},\\; x^2 > 0$ 或 $x = 0$"中，结论含"或"，写出它的否定（结论部分）。结论否定应写成"$x^2 \\leq 0$ 且 $x \\neq 0$"，请回答整个否定命题（用"存在"开头的中文描述）。',
        accept: ['存在x∈R，x²≤0且x≠0', '存在x属于R使x²≤0且x≠0', '∃x∈R，x²≤0且x≠0'],
        explain: '结论"$p$ 或 $q$"的否定是"$\\neg p$ 且 $\\neg q$"（德摩根律）。$x^2 > 0$ 否定为 $x^2 \\leq 0$；$x = 0$ 否定为 $x \\neq 0$。整个否定命题：$\\exists x \\in \\mathbb{R},\\; x^2 \\leq 0$ 且 $x \\neq 0$。（该否定命题为假，因为 $x^2 \\leq 0$ 仅在 $x=0$ 时取等，与 $x \\neq 0$ 矛盾。）',
      },
      {
        type: 'choice',
        prompt: '命题"$\\forall x \\in \\mathbb{R},\\; x^2-2ax+a > 0$"为假命题，则 $a$ 的范围是？',
        options: [
          '$0 < a < 1$',
          '$a \\leq 0$ 或 $a \\geq 1$',
          '$a < 0$ 或 $a > 1$',
          '$0 \\leq a \\leq 1$',
        ],
        answer: 1,
        explain: '原命题为假，等价于"$\\exists x \\in \\mathbb{R},\\; x^2-2ax+a \\leq 0$"为真，即二次函数不恒正。二次函数 $x^2-2ax+a$ 开口向上，恒正当且仅当判别式 $\\Delta < 0$，即 $4a^2-4a < 0$，即 $0 < a < 1$。原命题为假意味着恒正不成立，故 $\\Delta \\geq 0$，即 $a \\leq 0$ 或 $a \\geq 1$。',
      },
    ],
  },
]
