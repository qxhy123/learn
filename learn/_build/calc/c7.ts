export const LESSONS = [
  // ─────────────────────────────────────────
  // 关 c7-a  数列极限的 ε-N 定义
  // ─────────────────────────────────────────
  {
    id: 'c7-a',
    title: '数列极限的 ε-N 定义',
    subtitle: '直觉→严格定义→证明三步法',
    intro: [
      // 卡 1：动机——"趋近"到底是什么意思？
      {
        title: '为什么要给"趋近"下精确定义？',
        body:
          '考察数列 $a_n = \\dfrac{1}{n}$：$1,\\,\\dfrac{1}{2},\\,\\dfrac{1}{3},\\,\\dfrac{1}{4},\\ldots$\n\n' +
          '直觉告诉我们：随着 $n$ 增大，$a_n$ 越来越接近 $0$。但"越来越接近"是模糊的说法——' +
          '数列 $b_n = 1 + \\dfrac{1}{n}$ 也在"接近"，但接近的目标是 $1$，不是 $0$。\n\n' +
          '问题在于：多近才算"接近"？多大的 $n$ 才算"足够大"？\n\n' +
          '**不精确就无法证明**。如果你说"$a_n$ 趋近于 $L$"，但不能给出一个可量化的判据，' +
          '别人就可以反问："趋近到 $0.01$ 以内算不算？趋近到 $0.0000001$ 以内呢？"\n\n' +
          '$\\varepsilon$-$N$ 定义的核心思想：**对于任意精度要求（不管多苛刻），我总能找到一个起点，' +
          '使得从那里往后，数列项永远满足该精度。**',
        tip: '$\\varepsilon$（epsilon）代表误差精度，$N$ 代表起点项数。',
      },
      // 卡 2：ε-N 定义（分句逐步拆解）
      {
        title: 'ε-N 严格定义（逐句拆解）',
        body:
          '**定义**：称数列 $\\{a_n\\}$ 收敛于实数 $L$，若：\n\n' +
          '**① 对于任意给定的 $\\varepsilon > 0$**——精度要求由"挑战者"任意给出，可以极小。\n\n' +
          '**② 都存在正整数 $N$**——"答题者"根据 $\\varepsilon$ 找一个起点 $N$（$N$ 依赖 $\\varepsilon$）。\n\n' +
          '**③ 使得当 $n > N$ 时，有 $|a_n - L| < \\varepsilon$**——从第 $N+1$ 项起，所有项都落在 $(L-\\varepsilon,\\,L+\\varepsilon)$ 内。\n\n' +
          '记作 $\\displaystyle\\lim_{n\\to\\infty} a_n = L$，或 $a_n \\to L$（$n \\to \\infty$）。\n\n' +
          '**几何意义**：无论以 $L$ 为中心、半径为 $\\varepsilon$ 的区间多小，数列从某项起永远在区间内，至多**有限项**在外面。',
        formula:
          '\\lim_{n\\to\\infty} a_n = L \\iff \\forall\\,\\varepsilon>0,\\;\\exists\\,N\\in\\mathbb{N}^+,\\;\\forall\\,n>N:\\;|a_n-L|<\\varepsilon',
      },
      // 卡 3：证明模板（三步法）
      {
        title: '三步证明法：从 |aₙ − L| 反向解 N',
        body:
          '证明 $\\lim_{n\\to\\infty} a_n = L$ 的标准套路分三步：\n\n' +
          '**第一步：设 $\\varepsilon > 0$ 任意给定。**（不要忘记这句话！）\n\n' +
          '**第二步：反向估计。**从 $|a_n - L| < \\varepsilon$ 出发，反向解出"需要 $n$ 多大"，' +
          '得到条件 $n > f(\\varepsilon)$，于是取 $N = \\lceil f(\\varepsilon) \\rceil$（向上取整）。\n\n' +
          '**第三步：验证。**当 $n > N$ 时，推出 $|a_n - L| < \\varepsilon$，完成证明。\n\n' +
          '**核心口诀**：先看目标 $|a_n - L|$，化简后解出 $n$ 的下界，那就是 $N$。',
        steps: [
          '设 $\\varepsilon > 0$ 任意给定',
          '对 $|a_n - L|$ 化简，得 $|a_n - L| = g(n)$',
          '令 $g(n) < \\varepsilon$，解出 $n > f(\\varepsilon)$，取 $N = \\lceil f(\\varepsilon) \\rceil$',
          '当 $n > N$ 时验证 $|a_n - L| < \\varepsilon$ 成立',
        ],
      },
      // 卡 4：例题1——证明 lim(1/n) = 0
      {
        title: '例题1：证明 $\\lim_{n\\to\\infty}\\dfrac{1}{n}=0$',
        body:
          '**目标**：对任意 $\\varepsilon > 0$，找 $N$ 使得 $n > N$ 时 $|\\tfrac{1}{n} - 0| < \\varepsilon$。\n\n' +
          '**第一步**：设 $\\varepsilon > 0$ 任意给定。\n\n' +
          '**第二步**：$\\left|\\dfrac{1}{n} - 0\\right| = \\dfrac{1}{n}$。\n\n' +
          '令 $\\dfrac{1}{n} < \\varepsilon$，即需要 $n > \\dfrac{1}{\\varepsilon}$。\n\n' +
          '取 $N = \\left\\lfloor\\dfrac{1}{\\varepsilon}\\right\\rfloor$（向下取整，此时 $n > N$ 蕴含 $n \\geq N+1 > \\dfrac{1}{\\varepsilon}$）。\n\n' +
          '**第三步**：当 $n > N$ 时，$n > \\dfrac{1}{\\varepsilon}$，故 $\\dfrac{1}{n} < \\varepsilon$，即 $|a_n - 0| < \\varepsilon$。\n\n' +
          '**具体感受**：若 $\\varepsilon = 0.01$，则 $N = 100$；若 $\\varepsilon = 0.001$，则 $N = 1000$。$\\varepsilon$ 越小，$N$ 越大——这完全合理！',
      },
      // 卡 5：例题2——证明 lim((n+1)/n) = 1
      {
        title: '例题2：证明 $\\lim_{n\\to\\infty}\\dfrac{n+1}{n}=1$',
        body:
          '**第一步**：设 $\\varepsilon > 0$。\n\n' +
          '**第二步**：计算 $\\left|\\dfrac{n+1}{n} - 1\\right| = \\left|\\dfrac{n+1-n}{n}\\right| = \\dfrac{1}{n}$。\n\n' +
          '这和例题1 完全一样！令 $\\dfrac{1}{n} < \\varepsilon$，需要 $n > \\dfrac{1}{\\varepsilon}$。\n\n' +
          '取 $N = \\left\\lfloor\\dfrac{1}{\\varepsilon}\\right\\rfloor$。\n\n' +
          '**第三步**：当 $n > N$ 时，$\\left|\\dfrac{n+1}{n} - 1\\right| = \\dfrac{1}{n} < \\varepsilon$。$\\square$\n\n' +
          '**关键领悟**：$\\dfrac{n+1}{n}$ 和 $\\dfrac{1}{n}$ 看起来不同，但化简后差都是 $\\dfrac{1}{n}$——' +
          'ε-N 证明的艺术在于**把绝对值化简到尽可能简单的形式**，再解不等式。',
      },
      // 卡 6：常见陷阱与易错点
      {
        title: '易错点：$N$ 依赖 $\\varepsilon$，以及"不等于"与"小于"',
        body:
          '**易错点一：忘记 $N$ 依赖 $\\varepsilon$**。若有人问"$N$ 等于多少"，正确回答是"$N = N(\\varepsilon)$，' +
          '依赖精度要求"。每缩小一倍 $\\varepsilon$，$N$ 可能变大十倍——这是极限定义的核心。\n\n' +
          '**易错点二：$|a_n - L| < \\varepsilon$ 还是 $\\leq \\varepsilon$？** 定义用严格小于号 $<$，' +
          '这样更强，且实际证明时也更方便（绝大多数情况可同时满足两种）。\n\n' +
          '**易错点三：$\\varepsilon$ 必须是任意正数**。若你的证明只在 $\\varepsilon < 1$ 时成立，' +
          '那并不完整——$\\varepsilon = 100$ 的情形也要覆盖（虽然很容易满足）。\n\n' +
          '**易错点四：收敛 $\\neq$ 单调**。$(-1)^n/n \\to 0$，但数列来回震荡，不单调。' +
          '极限是关于"终于稳定在哪里"，与路径形状无关。',
        tip: '口诀：先写"设 $\\varepsilon > 0$"，后解 $|a_n - L| < \\varepsilon$，最后验证。三步缺一不可。',
      },
      // 卡 7：reveal 综合练习
      {
        title: '综合例题：证明 $\\lim_{n\\to\\infty}\\dfrac{2n-1}{n+1}=2$',
        body:
          '这道题是标准练习。先做化简：\n\n' +
          '$\\left|\\dfrac{2n-1}{n+1} - 2\\right| = \\left|\\dfrac{2n-1-2(n+1)}{n+1}\\right| = \\left|\\dfrac{-3}{n+1}\\right| = \\dfrac{3}{n+1}$\n\n' +
          '令 $\\dfrac{3}{n+1} < \\varepsilon$，即 $n+1 > \\dfrac{3}{\\varepsilon}$，即 $n > \\dfrac{3}{\\varepsilon} - 1$。\n\n' +
          '取 $N = \\left\\lfloor\\dfrac{3}{\\varepsilon}\\right\\rfloor$，则当 $n > N$ 时 $n+1 > \\dfrac{3}{\\varepsilon}$，故 $\\dfrac{3}{n+1} < \\varepsilon$。$\\square$\n\n' +
          '**规律**：多项式型数列 $\\dfrac{\\text{一次式}}{\\text{一次式}}$，化简后差的绝对值形如 $\\dfrac{c}{n+d}$，直接解即可。',
        reveal: {
          q: '证明 $\\lim_{n\\to\\infty}\\dfrac{2n-1}{n+1}=2$ 时，若 $\\varepsilon=0.03$，$N$ 至少取多少？',
          a: '$N = \\lfloor 3/\\varepsilon \\rfloor = \\lfloor 3/0.03 \\rfloor = \\lfloor 100 \\rfloor = 100$。即从第 101 项起，所有项与 2 的距离小于 0.03。',
        },
      },
      // 卡 8：极限的唯一性与有界性（性质直觉）
      {
        title: '两大基本性质：唯一性与有界性',
        body:
          '**唯一性**：收敛数列的极限唯一。若 $\\lim a_n = L_1$ 且 $\\lim a_n = L_2$，则 $L_1 = L_2$。\n\n' +
          '直觉：数列不可能"同时趋近"两个不同的地方。若 $L_1 \\neq L_2$，取误差 $\\varepsilon = |L_1-L_2|/2$，' +
          '以 $L_1$ 为中心的区间和以 $L_2$ 为中心的区间不重叠，数列不可能同时永远在两个不重叠区间内。\n\n' +
          '**有界性**：收敛数列必有界。若 $\\lim a_n = L$，则存在 $M>0$，对所有 $n$ 有 $|a_n| \\leq M$。\n\n' +
          '直觉：从某项起项目落在 $(L-1, L+1)$ 内，有界；有限项前缀也有界；两部分合并取最大即可。\n\n' +
          '**重要逆命题（假的！）**：有界不意味着收敛。反例：$a_n = (-1)^n$ 有界（$|a_n|=1$）但发散（来回震荡，没有极限）。',
        tip: '收敛 $\\Rightarrow$ 有界；有界 $\\not\\Rightarrow$ 收敛。逆命题不成立！',
      },
    ],
    questions: [
      // 题 1：judge — ε 是任意的
      {
        id: 'c7-a-q1',
        type: 'judge',
        prompt:
          '在 $\\varepsilon$-$N$ 定义中，$\\varepsilon$ 可以是任意正实数，包括非常大的数如 $\\varepsilon=1000$。',
        answer: true,
        explain:
          '$\\varepsilon > 0$ 是"任意给定的"，没有大小限制。$\\varepsilon = 1000$ 时条件很容易满足（$N$ 很小甚至取 $N=1$ 就行），' +
          '真正困难的是 $\\varepsilon$ 极小时。定义要求对**所有** $\\varepsilon > 0$ 都能找到 $N$。',
      },
      // 题 2：choice — 证明 lim(1/n²)=0 时取 N
      {
        id: 'c7-a-q2',
        type: 'choice',
        prompt:
          '证明 $\\lim_{n\\to\\infty}\\dfrac{1}{n^2}=0$ 时，第二步化简得 $\\left|\\dfrac{1}{n^2}-0\\right|=\\dfrac{1}{n^2}$。' +
          '令 $\\dfrac{1}{n^2}<\\varepsilon$，应取 $N$ 为下列哪项？',
        options: [
          '$N = \\left\\lfloor\\dfrac{1}{\\varepsilon}\\right\\rfloor$',
          '$N = \\left\\lfloor\\dfrac{1}{\\sqrt{\\varepsilon}}\\right\\rfloor$',
          '$N = \\left\\lfloor\\varepsilon\\right\\rfloor$',
          '$N = \\left\\lfloor\\sqrt{\\varepsilon}\\right\\rfloor$',
        ],
        answer: 1,
        explain:
          '$\\dfrac{1}{n^2}<\\varepsilon$ 等价于 $n^2 > \\dfrac{1}{\\varepsilon}$，即 $n > \\dfrac{1}{\\sqrt{\\varepsilon}}$。' +
          '故取 $N = \\left\\lfloor\\dfrac{1}{\\sqrt{\\varepsilon}}\\right\\rfloor$，当 $n > N$ 时自动满足。',
      },
      // 题 3：input — 计算绝对值差
      {
        id: 'c7-a-q3',
        type: 'input',
        prompt:
          '化简 $\\left|\\dfrac{3n+1}{n}-3\\right|$（填写结果，仅用 $n$ 表示，如 "1/n"）。',
        accept: ['1/n'],
        explain:
          '$\\dfrac{3n+1}{n} - 3 = \\dfrac{3n+1-3n}{n} = \\dfrac{1}{n}$，取绝对值仍为 $\\dfrac{1}{n}$（因 $n>0$）。' +
          '这种一次/一次型化简后差的绝对值形如 $\\dfrac{c}{n}$，是最常见的模式。',
      },
      // 题 4：judge — 有界数列一定收敛？
      {
        id: 'c7-a-q4',
        type: 'judge',
        prompt: '有界数列一定收敛。',
        answer: false,
        explain:
          '反例：$a_n = (-1)^n = -1, 1, -1, 1, \\ldots$，$|a_n| = 1$（有界），但数列在 $-1$ 和 $1$ 之间来回跳动，没有极限（发散）。' +
          '有界是收敛的必要条件，但不是充分条件。',
      },
      // 题 5：choice — 极限唯一性反证
      {
        id: 'c7-a-q5',
        type: 'choice',
        prompt:
          '若假设 $\\lim_{n\\to\\infty}a_n = L_1$ 且 $\\lim_{n\\to\\infty}a_n = L_2$，$L_1\\neq L_2$，' +
          '证明唯一性时取 $\\varepsilon$ 为哪个值来导出矛盾？',
        options: [
          '$\\varepsilon = |L_1 + L_2|$',
          '$\\varepsilon = |L_1 - L_2|$',
          '$\\varepsilon = \\dfrac{|L_1 - L_2|}{2}$',
          '$\\varepsilon = 1$',
        ],
        answer: 2,
        explain:
          '取 $\\varepsilon = \\dfrac{|L_1-L_2|}{2}$，则以 $L_1$ 为中心半径为 $\\varepsilon$ 的区间与以 $L_2$ 为中心半径为 $\\varepsilon$ 的区间**不相交**。' +
          '数列不可能同时"进入"两个不相交区间，导出矛盾。这是极限唯一性证明的标准套路。',
      },
      // 题 6：input — 给出 ε 求 N
      {
        id: 'c7-a-q6',
        type: 'input',
        prompt:
          '证明 $\\lim_{n\\to\\infty}\\dfrac{1}{n}=0$ 时，若 $\\varepsilon=0.001$，按 $N=\\lfloor 1/\\varepsilon\\rfloor$ 取，$N$ 等于多少？',
        accept: ['1000'],
        explain:
          '$N = \\lfloor 1/0.001 \\rfloor = \\lfloor 1000 \\rfloor = 1000$。即从第 1001 项起，$\\dfrac{1}{n} < 0.001$ 恒成立。',
      },
      // 题 7：match — 收敛数列性质
      {
        id: 'c7-a-q7',
        type: 'match',
        prompt: '将性质与正确描述对应。',
        left: ['极限唯一性', '收敛则有界', '有界则收敛'],
        right: [
          '收敛数列最多只有一个极限值',
          '收敛数列所有项的绝对值存在上界',
          '此命题为假，$(-1)^n$ 是反例',
        ],
      },
      // 题 8：choice — 从ε-N定义读懂含义
      {
        id: 'c7-a-q8',
        type: 'choice',
        prompt:
          '$\\lim_{n\\to\\infty}a_n=L$ 的几何含义是：无论以 $L$ 为中心的区间多小，数列中落在区间**外**的项有多少个？',
        options: ['无穷多个', '恰好 $N$ 个', '至多有限个', '零个'],
        answer: 2,
        explain:
          '只有**有限项**（至多 $N$ 项）可能落在区间 $(L-\\varepsilon,L+\\varepsilon)$ 外，从第 $N+1$ 项起全部在区间内。' +
          '"无穷"和"有限"是理解极限定义的关键对立概念。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c7-b  夹逼定理与单调有界定理
  // ─────────────────────────────────────────
  {
    id: 'c7-b',
    title: '夹逼定理与单调有界定理',
    subtitle: '两大存在性工具 + 递推数列求极限四步法',
    intro: [
      // 卡 1：动机——有些极限算不出来，只能"夹住"
      {
        title: '当四则运算不够用时：夹住它！',
        body:
          '考虑 $a_n = \\dfrac{\\sin n}{n}$。$\\sin n$ 没有通项公式，在 $[-1,1]$ 之间乱跳，无法直接用四则运算。\n\n' +
          '但我们知道：$-1 \\leq \\sin n \\leq 1$，两边除以 $n$（$n > 0$）：\n\n' +
          '$-\\dfrac{1}{n} \\leq \\dfrac{\\sin n}{n} \\leq \\dfrac{1}{n}$\n\n' +
          '左边 $\\to 0$，右边 $\\to 0$，中间的 $a_n$ 被"夹住"，只能 $\\to 0$！\n\n' +
          '这就是**夹逼定理**的核心思想：**找两条已知极限相等的边界，把目标数列夹在中间。**\n\n' +
          '单调有界定理解决的则是另一类问题：当极限不好直接算出来，但能证明数列"单调且有界"时，' +
          '极限**必然存在**（尽管需要再解方程才能求出具体值）。',
        tip: '夹逼定理用于"极限能算，但过程复杂"；单调有界定理用于"极限是否存在未知，但有单调性"。',
      },
      // 卡 2：夹逼定理定义与条件
      {
        title: '夹逼定理（三明治定理）',
        body:
          '**定理**：设数列 $\\{\\alpha_n\\}$、$\\{a_n\\}$、$\\{\\beta_n\\}$ 满足：\n\n' +
          '① 从某项起（$n > N_0$），有 $\\alpha_n \\leq a_n \\leq \\beta_n$；\n\n' +
          '② $\\displaystyle\\lim_{n\\to\\infty}\\alpha_n = \\lim_{n\\to\\infty}\\beta_n = L$（两侧极限相等）。\n\n' +
          '则 $\\displaystyle\\lim_{n\\to\\infty} a_n = L$。\n\n' +
          '**证明直觉**：对任意 $\\varepsilon > 0$，从某项起 $L - \\varepsilon < \\alpha_n \\leq a_n \\leq \\beta_n < L + \\varepsilon$，' +
          '故 $|a_n - L| < \\varepsilon$。\n\n' +
          '**关键条件**：两侧必须收敛到**同一**极限 $L$。若 $\\alpha_n \\to 0$ 而 $\\beta_n \\to 1$，夹逼定理**不适用**。',
        formula:
          '\\alpha_n \\leq a_n \\leq \\beta_n,\\;\\lim\\alpha_n=\\lim\\beta_n=L \\implies \\lim a_n=L',
      },
      // 卡 3：夹逼例题1——sin n / n
      {
        title: '夹逼例题1：$\\lim_{n\\to\\infty}\\dfrac{\\sin n}{n}$',
        body:
          '**构造不等式**：由 $|\\sin n| \\leq 1$，得 $-1 \\leq \\sin n \\leq 1$。除以 $n > 0$：\n\n' +
          '$-\\dfrac{1}{n} \\leq \\dfrac{\\sin n}{n} \\leq \\dfrac{1}{n}$\n\n' +
          '**取极限**：$\\displaystyle\\lim_{n\\to\\infty}\\left(-\\dfrac{1}{n}\\right) = 0$，$\\displaystyle\\lim_{n\\to\\infty}\\dfrac{1}{n} = 0$。\n\n' +
          '**结论**：由夹逼定理，$\\displaystyle\\lim_{n\\to\\infty}\\dfrac{\\sin n}{n} = 0$。\n\n' +
          '**关键步骤**：不等式的两侧都趋向同一个值 $0$，夹逼成立。如果 $\\sin n$ 直接"乱跳"而没有上下界，这个方法就失效了。',
      },
      // 卡 4：夹逼例题2——n项求和型
      {
        title: '夹逼例题2：$n$ 项求和型（放缩到相同分母）',
        body:
          '**题目**：求 $S_n = \\dfrac{1}{n^2+1} + \\dfrac{1}{n^2+2} + \\cdots + \\dfrac{1}{n^2+n}$。\n\n' +
          '**思路**：$n$ 项，每项介于最小值 $\\dfrac{1}{n^2+n}$ 和最大值 $\\dfrac{1}{n^2+1}$ 之间。\n\n' +
          '**下界**：$n$ 项之和 $\\geq n \\cdot \\dfrac{1}{n^2+n} = \\dfrac{n}{n^2+n} = \\dfrac{1}{n+1}$\n\n' +
          '**上界**：$n$ 项之和 $\\leq n \\cdot \\dfrac{1}{n^2+1} = \\dfrac{n}{n^2+1}$\n\n' +
          '故 $\\dfrac{1}{n+1} \\leq S_n \\leq \\dfrac{n}{n^2+1}$。\n\n' +
          '两侧极限：$\\displaystyle\\lim_{n\\to\\infty}\\dfrac{1}{n+1} = 0$，$\\displaystyle\\lim_{n\\to\\infty}\\dfrac{n}{n^2+1} = 0$。\n\n' +
          '由夹逼定理，$\\displaystyle\\lim_{n\\to\\infty} S_n = 0$。',
        tip: '$n$ 项求和放缩口诀：找最大项和最小项，分别乘以 $n$，得上下界，取极限。',
      },
      // 卡 5：单调有界定理
      {
        title: '单调有界定理：单调 + 有界 ⇒ 收敛',
        body:
          '**定理**：单调有界数列必收敛。具体地：\n\n' +
          '- 单调递增有上界 $\\Rightarrow$ 必收敛（极限 $\\leq$ 上界）\n' +
          '- 单调递减有下界 $\\Rightarrow$ 必收敛（极限 $\\geq$ 下界）\n\n' +
          '**直觉**：单调递增有上界的数列，像"爬楼梯但有天花板"——步步升高但永远碰不破天花板，' +
          '必然会越来越靠近某个值（上确界）。\n\n' +
          '**使用时机**：只有递推公式时，不知道极限是否存在。先用单调有界定理证明存在性，' +
          '再设 $\\lim a_n = L$ 代入递推公式解方程求极限值。\n\n' +
          '**重要提醒**：定理只保证极限**存在**，不直接给出极限值。要求值必须额外解方程。',
        formula:
          '\\{a_n\\}\\text{ 单调递增有上界} \\implies \\lim_{n\\to\\infty}a_n = \\sup\\{a_n\\}\\text{ 存在}',
      },
      // 卡 6：递推数列四步法（引入例）
      {
        title: '递推数列四步法：以 $a_1=1,\\,a_{n+1}=\\dfrac{a_n+2}{2}$ 为例',
        body:
          '**第一步：猜极限**（假设存在时）。设 $L = \\lim a_n$，两边取极限：$L = \\dfrac{L+2}{2} \\Rightarrow L = 2$。\n\n' +
          '**第二步：用归纳法证有界（上界为 2）**。$a_1 = 1 < 2$。设 $a_n < 2$，则 $a_{n+1} = \\dfrac{a_n+2}{2} < \\dfrac{2+2}{2} = 2$。归纳完毕，$a_n < 2$ 对所有 $n$ 成立。\n\n' +
          '**第三步：证单调**。$a_{n+1} - a_n = \\dfrac{a_n+2}{2} - a_n = \\dfrac{2-a_n}{2}$。' +
          '由 $a_n < 2$ 知 $2-a_n > 0$，故 $a_{n+1} - a_n > 0$，数列单调递增。\n\n' +
          '**第四步：由单调有界定理收敛，极限为 $L=2$**。\n\n' +
          '**陷阱警告**：绝不能跳过第二、三步直接说"设极限为 $L$"！若数列不收敛，解出的 $L$ 是假象。',
        steps: [
          '猜极限：设极限存在，代入递推式解方程',
          '证有界：数学归纳法',
          '证单调：计算 $a_{n+1}-a_n$ 或 $a_{n+1}/a_n$ 的符号',
          '由单调有界定理收敛，极限等于第一步的解',
        ],
      },
      // 卡 7：数 e 的定义——经典例子
      {
        title: '经典应用：$\\lim_{n\\to\\infty}\\left(1+\\dfrac{1}{n}\\right)^n = e$',
        body:
          '设 $a_n = \\left(1+\\dfrac{1}{n}\\right)^n$，可以证明（用均值不等式和二项式定理）：\n\n' +
          '① $\\{a_n\\}$ 单调递增；② $a_n < 3$（有上界）。\n\n' +
          '由单调有界定理，$\\{a_n\\}$ 必收敛。将其极限定义为自然常数：\n\n' +
          '$e \\approx 2.71828\\ldots$\n\n' +
          '**上界估计**（用二项式定理）：$a_n < 1 + 1 + \\dfrac{1}{2!} + \\dfrac{1}{3!} + \\cdots < 1 + 1 + \\dfrac{1}{2} + \\dfrac{1}{4} + \\cdots = 3$。\n\n' +
          '这里 $\\dfrac{1}{k!} \\leq \\dfrac{1}{2^{k-1}}$（$k \\geq 1$），几何级数求和得上界 $3$。\n\n' +
          '$e$ 是微积分最重要的常数，出现在指数函数导数、自然对数、微分方程的解中。',
        reveal: {
          q: '数列 $a_n = \\left(1+\\dfrac{1}{n}\\right)^n$ 收敛到 $e$，$a_{100}$ 与 $e$ 相差约多少（保留两位小数）？',
          a: '$a_{100} \\approx 2.7048$，$e \\approx 2.7183$，差约 $0.01$（百项时已非常接近）。单调有界定理保证收敛，但收敛速度可能很慢。',
        },
      },
      // 卡 8：易错汇总
      {
        title: '易错点汇总',
        body:
          '**易错点一：夹逼定理两侧极限必须相等**。若 $0 \\leq a_n \\leq 1$ 但两侧趋向不同极限，结论不成立。\n\n' +
          '**易错点二：递推型必须先证收敛再求极限**。"设极限为 $L$"只有在极限存在时才合法。' +
          '若先解出 $L$，需在结论中声明"因此极限存在且等于 $L$"——但这依赖单调有界定理。\n\n' +
          '**易错点三：证有界时选对归纳假设**。有时需要先证单调性，有时需要先证有界性，顺序不能搞反。' +
          '正确顺序通常是：先猜上界 $\\to$ 归纳证有界 $\\to$ 用有界性证单调。\n\n' +
          '**易错点四：单调有界定理对递减数列同样适用**。单调递减有下界同样保证收敛，极限 $\\geq$ 下界。',
        tip: '夹逼定理核心：找两侧 + 验证同极限。单调有界定理核心：证单调 + 证有界，顺序灵活，先找容易的那个。',
      },
    ],
    questions: [
      // 题 1：choice — 夹逼定理条件
      {
        id: 'c7-b-q1',
        type: 'choice',
        prompt:
          '夹逼定理中，若 $\\alpha_n \\leq a_n \\leq \\beta_n$，$\\lim\\alpha_n=0$，$\\lim\\beta_n=1$，能得出 $\\lim a_n$ 的结论吗？',
        options: [
          '$\\lim a_n = 0$',
          '$\\lim a_n = 1$',
          '$\\lim a_n = \\dfrac{1}{2}$',
          '不能确定，夹逼定理不适用',
        ],
        answer: 3,
        explain:
          '夹逼定理要求两侧极限**相等**。$\\lim\\alpha_n = 0 \\neq 1 = \\lim\\beta_n$，条件不满足，无法用夹逼定理得出结论。',
      },
      // 题 2：judge — 单调递增有上界
      {
        id: 'c7-b-q2',
        type: 'judge',
        prompt: '单调递增但无上界的数列一定发散（趋向 $+\\infty$）。',
        answer: true,
        explain:
          '单调递增无上界的数列，对任意 $M>0$，存在某项 $a_N > M$，之后所有项 $a_n > M$（因单调递增），故数列趋向 $+\\infty$，发散。' +
          '单调有界定理的逆向：有界是收敛的必要条件，无界则必发散。',
      },
      // 题 3：input — 夹逼计算 sin n / n
      {
        id: 'c7-b-q3',
        type: 'input',
        prompt:
          '用夹逼定理求 $\\lim_{n\\to\\infty}\\dfrac{\\sin n}{n}$，结果是多少？（填整数）',
        accept: ['0'],
        explain:
          '$-\\dfrac{1}{n} \\leq \\dfrac{\\sin n}{n} \\leq \\dfrac{1}{n}$，两侧均趋向 $0$，由夹逼定理 $\\lim_{n\\to\\infty}\\dfrac{\\sin n}{n} = 0$。',
      },
      // 题 4：match — 四步法对应
      {
        id: 'c7-b-q4',
        type: 'match',
        prompt: '将递推数列求极限四步法的步骤与内容对应。',
        left: ['第一步', '第二步', '第三步', '第四步'],
        right: [
          '设极限存在代入递推式猜出 $L$',
          '数学归纳法证明有界性',
          '计算 $a_{n+1}-a_n$ 证明单调性',
          '由单调有界定理得极限存在，极限为 $L$',
        ],
      },
      // 题 5：choice — 递推数列 a₁=1, aₙ₊₁=(aₙ+2)/2 的极限
      {
        id: 'c7-b-q5',
        type: 'choice',
        prompt:
          '设 $a_1 = 1$，$a_{n+1} = \\dfrac{a_n+2}{2}$，用单调有界定理证明收敛后，极限 $L$ 满足哪个方程？',
        options: [
          '$L = L + 2$',
          '$2L = L + 2$',
          '$L = 2L + 2$',
          '$L^2 = L + 2$',
        ],
        answer: 1,
        explain:
          '在 $a_{n+1} = \\dfrac{a_n+2}{2}$ 两边对 $n\\to\\infty$ 取极限，得 $L = \\dfrac{L+2}{2}$，即 $2L = L+2$，解得 $L=2$。',
      },
      // 题 6：judge — 先设L再证收敛是否正确？
      {
        id: 'c7-b-q6',
        type: 'judge',
        prompt:
          '对递推数列，可以直接假设 $\\lim a_n = L$ 并代入递推式解出 $L$，以此作为证明极限存在并等于 $L$ 的完整证明。',
        answer: false,
        explain:
          '"假设极限存在"和"证明极限存在"是两件事。解出 $L$ 只说明"若极限存在则等于 $L$"，' +
          '但极限可能根本不存在（数列发散）。必须先用单调有界定理证明存在性，再代入递推式求 $L$。',
      },
      // 题 7：input — n项和夹逼结果
      {
        id: 'c7-b-q7',
        type: 'input',
        prompt:
          '用夹逼定理求 $\\lim_{n\\to\\infty}\\left(\\dfrac{1}{n^2+1}+\\dfrac{1}{n^2+2}+\\cdots+\\dfrac{1}{n^2+n}\\right)$，结果是多少？（填整数）',
        accept: ['0'],
        explain:
          '下界 $\\dfrac{1}{n+1} \\to 0$，上界 $\\dfrac{n}{n^2+1} \\to 0$（分子分母最高次相差一次）。两侧同趋 $0$，由夹逼定理极限为 $0$。',
      },
      // 题 8：choice — e 的定义数列
      {
        id: 'c7-b-q8',
        type: 'choice',
        prompt:
          '数列 $a_n = \\left(1+\\dfrac{1}{n}\\right)^n$ 通过单调有界定理被证明收敛，下列关于该数列的说法正确的是？',
        options: [
          '$\\{a_n\\}$ 单调递减且有下界 $2$',
          '$\\{a_n\\}$ 单调递增且有上界 $3$',
          '$\\{a_n\\}$ 不单调，但有界',
          '$\\{a_n\\}$ 单调递增但无上界',
        ],
        answer: 1,
        explain:
          '用均值不等式可证 $a_n < a_{n+1}$（单调递增），用二项式定理可证 $a_n < 3$（有上界）。' +
          '由单调有界定理，极限存在，定义为 $e \\approx 2.718$。注意上界 $3$，而极限约为 $2.718$，满足"极限 $\\leq$ 上界"。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c7-c  极限四则运算与综合计算
  // ─────────────────────────────────────────
  {
    id: 'c7-c',
    title: '极限四则运算与综合计算',
    subtitle: '加减乘除法则 + 最高次法 + 指数型 + 易错辨析',
    intro: [
      // 卡 1：四则运算法则
      {
        title: '极限的四则运算法则',
        body:
          '**前提**：$\\lim_{n\\to\\infty}a_n = A$ 且 $\\lim_{n\\to\\infty}b_n = B$（两个极限都存在）。\n\n' +
          '① 加减法：$\\lim(a_n \\pm b_n) = A \\pm B$\n\n' +
          '② 乘法：$\\lim(a_n \\cdot b_n) = A \\cdot B$（含常数乘：$\\lim c\\cdot a_n = cA$）\n\n' +
          '③ 除法：$\\lim\\dfrac{a_n}{b_n} = \\dfrac{A}{B}$，**要求 $B \\neq 0$ 且 $b_n \\neq 0$**\n\n' +
          '**核心陷阱**：以上法则仅在两个极限**都存在**时成立。若 $a_n = n$（发散），' +
          '则 $a_n + (-a_n) = 0 \\to 0$，但不能写成 $\\lim a_n + \\lim(-a_n)$（两者均不存在）。',
        formula:
          '\\lim(a_n\\pm b_n)=A\\pm B,\\quad\\lim(a_nb_n)=AB,\\quad\\lim\\frac{a_n}{b_n}=\\frac{A}{B}\\;(B\\neq0)',
      },
      // 卡 2：最高次法（多项式型）
      {
        title: '多项式型极限：最高次法（分子分母同除）',
        body:
          '**题型**：$\\lim_{n\\to\\infty}\\dfrac{P(n)}{Q(n)}$，其中 $P,Q$ 是关于 $n$ 的多项式。\n\n' +
          '**方法**：分子分母同除以最高次 $n^k$（取分子、分母最高次中较大者），所有低次项趋向 $0$。\n\n' +
          '**结果模板**（设分子最高次为 $m$，分母为 $k$）：\n\n' +
          '$\\dfrac{a_mn^m+\\cdots}{b_kn^k+\\cdots} \\to \\begin{cases}a_m/b_k & m=k\\\\0 & m<k\\\\\\pm\\infty & m>k\\end{cases}$\n\n' +
          '**例**：$\\dfrac{3n^2+2n-1}{2n^2-n+5}$，同除 $n^2$：$\\dfrac{3+2/n-1/n^2}{2-1/n+5/n^2} \\to \\dfrac{3+0-0}{2-0+0} = \\dfrac{3}{2}$。',
        tip: '只看分子分母最高次项的系数之比（当次数相同时）。',
      },
      // 卡 3：指数型极限（除最大底）
      {
        title: '指数型极限：除以最大底的 $n$ 次方',
        body:
          '**题型**：$\\lim_{n\\to\\infty}\\dfrac{a^n+b^n}{c^n+d^n}$，其中 $a,b,c,d > 0$。\n\n' +
          '**方法**：设 $M = \\max\\{a,b,c,d\\}$，分子分母同除以 $M^n$，其余各项 $\\to 0$（因底数绝对值 $< 1$）。\n\n' +
          '**例**：$\\dfrac{2^n+3^n}{3^n+4^n}$，最大底为 $4$，同除 $4^n$：\n\n' +
          '$\\dfrac{(2/4)^n+(3/4)^n}{(3/4)^n+1} = \\dfrac{(1/2)^n+(3/4)^n}{(3/4)^n+1}$\n\n' +
          '$(1/2)^n \\to 0$，$(3/4)^n \\to 0$（绝对值 $< 1$ 的公比），故极限 $= \\dfrac{0+0}{0+1} = 0$。\n\n' +
          '**一般结论**：若各项中最大底只在分母，则极限为 $0$；若在分子，则 $\\to \\infty$（发散）。',
      },
      // 卡 4：$q^n$ 的收敛性汇总
      {
        title: '等比数列收敛性：$|q|$ 决定一切',
        body:
          '设 $a_n = q^n$，极限行为完全由 $|q|$ 决定：\n\n' +
          '- $|q| < 1$：$q^n \\to 0$（绝对值趋 $0$，收敛）。例：$(1/2)^n \\to 0$，$(-0.9)^n \\to 0$。\n\n' +
          '- $q = 1$：$q^n = 1 \\to 1$（常数列，收敛）。\n\n' +
          '- $q = -1$：$q^n = (-1)^n$，来回在 $-1$ 和 $1$ 之间跳，发散。\n\n' +
          '- $|q| > 1$：$|q^n| = |q|^n \\to +\\infty$，发散。例：$2^n \\to +\\infty$，$(-2)^n$ 绝对值 $\\to +\\infty$，发散。\n\n' +
          '**口诀**：$|q| < 1 \\Rightarrow q^n \\to 0$；$q = 1 \\Rightarrow$ 常数；其他情况发散。',
        formula:
          '\\lim_{n\\to\\infty}q^n = \\begin{cases}0 & |q|<1 \\\\ 1 & q=1 \\\\ \\text{发散} & q=-1\\text{ 或 }|q|>1\\end{cases}',
        tip: '指数底数的绝对值小于 1 时才收敛到 0，等于 1 收敛到 1，其他均发散。',
      },
      // 卡 5：$n^k / q^n$ 型（指数胜多项式）
      {
        title: '指数增长远超多项式：$n^k / q^n \\to 0$（$|q|>1$）',
        body:
          '**结论**：当 $|q| > 1$，无论 $k$ 多大，$\\dfrac{n^k}{q^n} \\to 0$。\n\n' +
          '**直觉**：多项式 $n^k$ 增长缓慢（幂次增长），而 $|q|^n$ 以指数速度爆炸。指数最终完胜。\n\n' +
          '**数字感受**：$n=100$ 时，$n^3 = 10^6$ 而 $2^{100} \\approx 1.27\\times10^{30}$，差距悬殊。\n\n' +
          '**常见题型**：$\\lim_{n\\to\\infty}\\dfrac{n^{10}}{2^n} = 0$；$\\lim_{n\\to\\infty}\\dfrac{n!}{n^n} = 0$（阶乘增长更慢）。\n\n' +
          '**应用**：神经网络训练中，若误差以公比 $|q| < 1$ 衰减，则 $e_k = q^k \\to 0$，训练收敛；' +
          '衰减速度（$\\log|q|$）决定收敛效率。',
      },
      // 卡 6：有理化——根式型化简
      {
        title: '根式型极限：有理化去掉根式',
        body:
          '**题型**：含 $\\sqrt{n+a}-\\sqrt{n+b}$ 型，直接算是 $\\infty - \\infty$ 的不定型。\n\n' +
          '**方法**：乘以共轭根式，分子有理化。\n\n' +
          '**例**：$\\lim_{n\\to\\infty}n(\\sqrt{n+1}-\\sqrt{n})$。\n\n' +
          '有理化：$\\sqrt{n+1}-\\sqrt{n} = \\dfrac{(n+1)-n}{\\sqrt{n+1}+\\sqrt{n}} = \\dfrac{1}{\\sqrt{n+1}+\\sqrt{n}}$\n\n' +
          '代入：$n \\cdot \\dfrac{1}{\\sqrt{n+1}+\\sqrt{n}} = \\dfrac{n}{\\sqrt{n+1}+\\sqrt{n}}$\n\n' +
          '分子分母同除 $\\sqrt{n}$：$\\dfrac{\\sqrt{n}}{\\sqrt{1+1/n}+1} \\to$ …\n\n' +
          '实际上分子 $n$，分母 $\\approx 2\\sqrt{n}$，故整体 $\\approx \\dfrac{n}{2\\sqrt{n}} = \\dfrac{\\sqrt{n}}{2} \\to \\dfrac{1}{2}$（精确极限为 $1/2$）。',
        tip: '遇到 $\\sqrt{A}-\\sqrt{B}$ 型，乘共轭 $\\dfrac{\\sqrt{A}+\\sqrt{B}}{\\sqrt{A}+\\sqrt{B}}$，分子变 $A-B$，消去根式。',
      },
      // 卡 7：易错汇总 + reveal
      {
        title: '易错点汇总与综合自测',
        body:
          '**易错点一：两极限不存在时四则运算失效**。$a_n = (-1)^n$ 不收敛，' +
          '$a_n + (-a_n) = 0$ 收敛，但这不能用 $\\lim a_n + \\lim(-a_n)$（因两者均不存在）。\n\n' +
          '**易错点二：除法时必须检查分母极限非零**。若 $\\lim b_n = 0$ 但 $b_n \\neq 0$，' +
          '不能直接用除法法则，需另行处理（如提公因子、洛必达等）。\n\n' +
          '**易错点三：最高次法只适用于多项式**。含 $\\sin n$、$n^n$、$n!$ 时不能直接套用。\n\n' +
          '**易错点四：$(1/2)^n \\to 0$ 但 $(-1)^n$ 不收敛**。负公比绝对值 $< 1$ 时 $q^n \\to 0$（结论对），' +
          '绝对值 $= 1$ 时发散（$(-1)^n$ 不收敛）。',
        reveal: {
          q: '$\\lim_{n\\to\\infty}\\dfrac{3n^3-2n+1}{n^3+5n^2}$ 等于多少？',
          a: '分子分母同除 $n^3$：$\\dfrac{3-2/n^2+1/n^3}{1+5/n} \\to \\dfrac{3}{1} = 3$。最高次相同，极限为最高次系数之比。',
        },
      },
      // 卡 8：重要数列极限速查
      {
        title: '重要数列极限速查表',
        body:
          '以下极限请熟记，是计算复杂极限的基础砖块：\n\n' +
          '$\\lim_{n\\to\\infty}\\dfrac{1}{n} = 0$；$\\lim_{n\\to\\infty}\\dfrac{1}{n^k} = 0$（$k > 0$）\n\n' +
          '$\\lim_{n\\to\\infty}q^n = 0$（$|q| < 1$）\n\n' +
          '$\\lim_{n\\to\\infty}\\sqrt[n]{n} = 1$；$\\lim_{n\\to\\infty}\\sqrt[n]{a} = 1$（$a > 0$）\n\n' +
          '$\\lim_{n\\to\\infty}\\left(1+\\dfrac{1}{n}\\right)^n = e$\n\n' +
          '$\\lim_{n\\to\\infty}\\dfrac{n^k}{q^n} = 0$（$|q| > 1$，任意 $k$）\n\n' +
          '**用法**：看到极限式，先识别属于哪种类型，套用对应结论；复杂式子拆成几个已知型的组合。',
        tip: '速查表要背熟，考试时识别题型的速度决定得分速度。',
      },
    ],
    questions: [
      // 题 1：choice — 多项式型极限
      {
        id: 'c7-c-q1',
        type: 'choice',
        prompt:
          '求 $\\lim_{n\\to\\infty}\\dfrac{n^2-2n+3}{2n^2+n-1}$。',
        options: ['$2$', '$\\dfrac{1}{2}$', '$\\dfrac{3}{2}$', '$0$'],
        answer: 1,
        explain:
          '分子分母最高次均为 $n^2$，同除 $n^2$：$\\dfrac{1-2/n+3/n^2}{2+1/n-1/n^2} \\to \\dfrac{1}{2}$。最高次相同时极限为系数比。',
      },
      // 题 2：judge — 两极限不存在四则运算
      {
        id: 'c7-c-q2',
        type: 'judge',
        prompt:
          '若 $\\lim_{n\\to\\infty}a_n$ 和 $\\lim_{n\\to\\infty}b_n$ 都不存在，则 $\\lim_{n\\to\\infty}(a_n+b_n)$ 也一定不存在。',
        answer: false,
        explain:
          '反例：$a_n = (-1)^n$，$b_n = (-1)^{n+1} = -(-1)^n$，两者均发散，但 $a_n + b_n = 0 \\to 0$，极限存在。' +
          '四则运算法则要求两个极限**都存在**时才能使用，不满足条件时结论不能反推。',
      },
      // 题 3：input — 指数型极限
      {
        id: 'c7-c-q3',
        type: 'input',
        prompt:
          '求 $\\lim_{n\\to\\infty}\\dfrac{2^n+3^n}{3^n+4^n}$，结果是多少？（填整数）',
        accept: ['0'],
        explain:
          '分子分母同除 $4^n$（最大底）：$\\dfrac{(1/2)^n+(3/4)^n}{(3/4)^n+1} \\to \\dfrac{0+0}{0+1} = 0$。因 $|1/2|<1$ 且 $|3/4|<1$，两项均趋 $0$。',
      },
      // 题 4：match — q^n 收敛情形
      {
        id: 'c7-c-q4',
        type: 'match',
        prompt: '将 $q$ 的范围与 $\\lim_{n\\to\\infty}q^n$ 的结论对应。',
        left: ['$|q|<1$', '$q=1$', '$|q|>1$'],
        right: [
          '极限为 $0$',
          '极限为 $1$',
          '数列发散',
        ],
      },
      // 题 5：choice — 根式有理化
      {
        id: 'c7-c-q5',
        type: 'choice',
        prompt:
          '$\\lim_{n\\to\\infty}n(\\sqrt{n+1}-\\sqrt{n})$ 等于多少？',
        options: ['$0$', '$\\dfrac{1}{2}$', '$1$', '$+\\infty$'],
        answer: 3,
        explain:
          '有理化：$\\sqrt{n+1}-\\sqrt{n} = \\dfrac{(n+1)-n}{\\sqrt{n+1}+\\sqrt{n}} = \\dfrac{1}{\\sqrt{n+1}+\\sqrt{n}}$。\n\n' +
          '故 $n(\\sqrt{n+1}-\\sqrt{n}) = \\dfrac{n}{\\sqrt{n+1}+\\sqrt{n}}$。\n\n' +
          '分母 $\\sqrt{n+1}+\\sqrt{n} \\approx 2\\sqrt{n}$，比值 $\\approx \\dfrac{n}{2\\sqrt{n}} = \\dfrac{\\sqrt{n}}{2} \\to +\\infty$。\n\n' +
          '故极限为 $+\\infty$（发散）。注意：与 $\\sqrt{n+1}-\\sqrt{n} \\to 0$ 不同，乘以 $n$ 后发散，是 $0 \\cdot \\infty$ 不定型。',
      },
      // 题 6：input — 最高次法（分子次>分母）
      {
        id: 'c7-c-q6',
        type: 'input',
        prompt:
          '$\\lim_{n\\to\\infty}\\dfrac{3n^3-2n+1}{n^3+5n^2}$ 等于多少？（填整数）',
        accept: ['3'],
        explain:
          '分子分母最高次均为 $n^3$，同除 $n^3$：$\\dfrac{3-2/n^2+1/n^3}{1+5/n} \\to \\dfrac{3}{1} = 3$。',
      },
      // 题 7：judge — 除法法则的条件
      {
        id: 'c7-c-q7',
        type: 'judge',
        prompt:
          '若 $\\lim_{n\\to\\infty}a_n = 1$ 且 $\\lim_{n\\to\\infty}b_n = 0$，则可用极限除法法则得 $\\lim_{n\\to\\infty}\\dfrac{a_n}{b_n}$。',
        answer: false,
        explain:
          '极限除法法则要求分母极限 $B \\neq 0$。这里 $\\lim b_n = 0$，条件不满足，不能直接使用除法法则。' +
          '此时 $\\dfrac{a_n}{b_n}$ 的行为需具体分析（可能 $\\to \\infty$，也可能不存在）。',
      },
      // 题 8：choice — 速查表考查
      {
        id: 'c7-c-q8',
        type: 'choice',
        prompt:
          '$\\lim_{n\\to\\infty}\\sqrt[n]{5}$ 等于多少？',
        options: ['$5$', '$0$', '$1$', '$\\sqrt{5}$'],
        answer: 2,
        explain:
          '$\\sqrt[n]{5} = 5^{1/n}$，由重要极限 $\\lim_{n\\to\\infty}\\sqrt[n]{a} = 1$（$a > 0$），故 $\\lim_{n\\to\\infty}\\sqrt[n]{5} = 1$。' +
          '直觉：$5^{1/n}$ 是"5 的 $n$ 次方根"，$n$ 越大开方次数越多，结果越接近 $1$。',
      },
    ],
  },
];
