export const LESSONS = [
  {
    id: 'u13c5',
    title: '新定义型代数题',
    subtitle: '读懂定义 → 代入 → 用老方法——新符号不过是熟悉式子换了个名字',
    intro: [
      {
        title: '引入：陌生符号 $\\heartsuit$ 是什么？',
        body: '定义运算 $a\\heartsuit b = 2a+b^2$（$a,b\\in\\mathbb{R}$）。\n\n(1) 计算 $3\\heartsuit(-2)$；\n(2) 解方程 $x\\heartsuit 1=10$；\n(3) 若 $2\\heartsuit m = m\\heartsuit 2$，求 $m$。\n\n看到 $\\heartsuit$ 会不会一愣？没关系——**题目已经告诉你它的含义了**：$a\\heartsuit b$ 就是 $2a+b^2$，左边的数对应 $a$，右边的数对应 $b$。新符号只是给一个表达式起了个名字，换掉之后全是你认识的式子。',
        image: 'alg-p14-06.svg',
        imageCaption: '分段函数图象示意（含新定义型分段规则）——左段 $y=-x$，右段 $y=(x-1)^2$',
      },
      {
        title: '三步策略：读懂 → 代入 → 用老方法',
        body: '**第 1 步：读懂定义**\n\n仔细读题，明确"左边对应谁、右边对应谁"。做 1-2 个简单数字的热身代入，确认理解正确。\n\n**第 2 步：直接代入**\n\n把具体数字或字母代入定义的"模板变量"，消去新符号，得到普通代数式。\n\n注意：$a\\ast b$ 通常不满足交换律，$a\\ast b\\neq b\\ast a$，顺序不能颠倒！\n\n**第 3 步：用已学方法解**\n\n代入后得到的是一次方程、二次方程还是不等式？用对应的已学方法解，**不需要发明新方法**。',
        steps: [
          '读定义：弄清楚"第一个输入""第二个输入"对应哪个变量',
          '代入：把题目给的数替换掉定义里的字母',
          '化简：得到普通代数式或方程',
          '求解：一次方程/二次方程/不等式，用熟悉套路',
        ],
      },
      {
        title: '引入题完整求解',
        body: '定义 $a\\heartsuit b=2a+b^2$。\n\n**(1)** $3\\heartsuit(-2)$：$a=3$，$b=-2$，代入：$2\\times 3+(-2)^2=6+4=10$。\n\n**(2)** $x\\heartsuit 1=10$：$a=x$，$b=1$，代入：$2x+1^2=10$，$2x+1=10$，$x=\\dfrac{9}{2}$。\n\n这就是一道普通的一次方程！\n\n**(3)** $2\\heartsuit m = m\\heartsuit 2$：\n\n左边：$a=2,b=m$，$2\\heartsuit m=4+m^2$\n\n右边：$a=m,b=2$，$m\\heartsuit 2=2m+4$\n\n令两者相等：$4+m^2=2m+4$，$m^2-2m=0$，$m(m-2)=0$\n\n$m=0$ 或 $m=2$。',
        formula: 'a \\heartsuit b = 2a + b^2 \\quad \\Rightarrow \\quad 3 \\heartsuit (-2) = 2\\times 3 + (-2)^2 = 10',
        tip: '关键陷阱：$2\\heartsuit m\\neq m\\heartsuit 2$！新运算通常不满足交换律，左右顺序一定要仔细对应。',
      },
      {
        title: '第 1 类：新符号运算',
        body: '**定义形式**：$a\\ast b=\\text{某个关于}a,b\\text{的表达式}$（$\\ast$ 可以是 $\\star,\\heartsuit,\\oplus,\\square$ 等）。\n\n**常见题型**：\n- 直接计算：代入两个已知数，算出结果\n- 解方程：其中一个输入含未知数 $x$，代入后得到 $x$ 的方程\n- 解不等式：条件是 $a\\ast b>c$，代入后得到不等式\n- 证明性质：如"$a\\ast a$ 恒大于某值"\n\n**嵌套运算**（组合题型）：$(a\\ast b)\\ast c$ —— 严格从**内向外**，先算括号内层，得到中间结果后再与外层输入运算。',
      },
      {
        title: '第 2 类：新函数与新概念',
        body: '**新函数**：定义 $f(x)$ 满足某规则（分段的、含取整的）。\n\n例：$f(x)=\\begin{cases}x+1,&x\\geq 0\\\\-x+1,&x<0\\end{cases}$\n\n求值：先判断 $x$ 属于哪一段，代入对应式子。解方程 $f(x)=c$：分段令等于 $c$，各段独立解，最后检验。\n\n**新概念（新数/新集合）**：题目定义满足某性质的数，例如"若两位数各位数字之差为 3，称为差数"。\n\n解法：把文字条件翻译为方程/不等式，逐步列举或代数求解，注意整数/正负/位数等约束。',
        reveal: {
          q: '含参数的新运算：$a\\triangle b=a^2-b$，若 $k\\triangle 3=(k-1)\\triangle 5$，求 $k$。',
          a: '$k\\triangle 3=k^2-3$；$(k-1)\\triangle 5=(k-1)^2-5=k^2-2k-4$。\n令两者相等：$k^2-3=k^2-2k-4$，$2k=-1$，$k=-\\dfrac{1}{2}$。',
        },
      },
      {
        title: '例题：分段新运算的解方程',
        body: '**题**：定义 $a\\circ b=\\begin{cases}a+b,&a\\geq b\\\\a\\times b,&a<b\\end{cases}$（$a,b$ 为实数）。解方程 $x\\circ 2=6$。\n\n**解**：分两种情形（因为 $a=x$ 的正负影响用哪个分支）：\n\n**情形 1：$x\\geq 2$**，则 $x\\circ 2=x+2=6$，$x=4$。\n验证：$4\\geq 2$ ✓，$x=4$ 合法。\n\n**情形 2：$x<2$**，则 $x\\circ 2=2x=6$，$x=3$。\n验证：$3<2$？矛盾 ✗，舍去。\n\n**答**：$x=4$。\n\n**教训**：含分段的新运算，解完后必须验证 $x$ 是否满足该情形的条件。',
        tip: '新符号涉及分段时，等同于分段函数处理——先找分界点，分情形，解完验证，合并有效解。',
      },
      {
        title: '新定义题的常见陷阱',
        body: '**陷阱 1：搞反左右顺序**\n\n$a\\ast b$ 中 $a$ 是第一个，$b$ 是第二个。题目给"$5\\ast 2$"就是 $a=5,b=2$；给"$2\\ast 5$"就是 $a=2,b=5$。两者不同！\n\n**陷阱 2：嵌套时漏了括号**\n\n$(1\\ast 2)\\ast 3$：先算 $1\\ast 2$ 得到中间值 $r$，再算 $r\\ast 3$，不能跳步。\n\n**陷阱 3：新函数忘记分段检验**\n\n分段新函数解方程后，解 $x$ 必须属于对应那段的定义域，否则舍去。\n\n**陷阱 4：新概念遗漏约束**\n\n"两位数"要求十位 $\\geq 1$，个位 $0\\sim 9$，别忘了这些数字范围约束。',
      },
    ],
    questions: [
      {
        type: 'input',
        prompt: '定义 $a\\heartsuit b=2a+b^2$，计算 $3\\heartsuit(-2)$。',
        accept: ['10', '结果为10'],
        explain: '$a=3,b=-2$，代入：$2\\times 3+(-2)^2=6+4=10$。',
      },
      {
        type: 'choice',
        prompt: '定义 $a\\heartsuit b=2a+b^2$，若 $x\\heartsuit 1=10$，则 $x=$？',
        options: ['$\\dfrac{9}{2}$', '$\\dfrac{11}{2}$', '$9$', '$4$'],
        answer: 0,
        explain: '$x\\heartsuit 1=2x+1=10$，$2x=9$，$x=\\dfrac{9}{2}$。',
      },
      {
        type: 'input',
        prompt: '定义 $a\\square b=a^2+ab-b$，计算 $2\\square 3$。',
        accept: ['7', '结果为7'],
        explain: '$a=2,b=3$：$2^2+2\\times 3-3=4+6-3=7$。',
      },
      {
        type: 'judge',
        prompt: '定义 $a\\heartsuit b=2a+b^2$，则 $2\\heartsuit m = m\\heartsuit 2$ 对所有实数 $m$ 成立。',
        answer: false,
        explain: '$2\\heartsuit m=4+m^2$，$m\\heartsuit 2=2m+4$。令两者相等：$m^2-2m=0$，$m=0$ 或 $m=2$。只有这两个值时等式成立，不是对所有 $m$ 成立。新运算一般不满足交换律。',
      },
      {
        type: 'choice',
        prompt: '定义 $a\\circ b=\\begin{cases}a+b,&a\\geq b\\\\ a\\times b,&a<b\\end{cases}$，则 $(-1)\\circ 2=$？',
        options: ['$-2$', '$1$', '$-1$', '$3$'],
        answer: 0,
        explain: '$a=-1<b=2$，用乘法分支：$(-1)\\times 2=-2$。',
      },
      {
        type: 'match',
        prompt: '将新定义类型与解题主要步骤配对',
        left: ['新符号运算（直接计算）', '新函数解方程', '新概念（如"差数"）'],
        right: ['翻译为代数条件，列举或解方程', '分段令 $f(x)=c$，逐段解并验证', '代入定义，化为普通代数式'],
        explain: '新符号：代入化简；新函数：分段处理；新概念：翻译条件后代数求解。三类方法不同但都回归已学工具。',
      },
      {
        type: 'input',
        prompt: '定义 $a\\square b=a^2+ab-b$，若 $m\\square m=5$，求 $m$ 的所有值（用"或"连接）。',
        accept: ['m=5/2或m=-1', 'm=\\frac{5}{2}或m=-1', '5/2或-1'],
        explain: '$m\\square m=m^2+m\\cdot m-m=2m^2-m$。令 $2m^2-m=5$：$2m^2-m-5=0$，$(2m-5)(m+1)=0$，$m=\\dfrac{5}{2}$ 或 $m=-1$。',
      },
      {
        type: 'judge',
        prompt: '嵌套运算 $(1\\ast 2)\\ast 3$（其中 $a\\ast b=a+b-ab$），应先算 $2\\ast 3$，再与 $1$ 运算。',
        answer: false,
        explain: '括号优先：先算括号内的 $1\\ast 2=1+2-1\\times 2=1$，得中间值 $1$；再算 $1\\ast 3=1+3-1\\times 3=1$。嵌套必须从内向外，按括号顺序。',
      },
      {
        type: 'choice',
        prompt: '定义"好数"：若正整数 $n$ 可以表示为两个连续正整数的乘积，则 $n$ 是好数。下列哪个是好数？',
        options: ['$6$', '$8$', '$9$', '$10$'],
        answer: 0,
        explain: '$6=2\\times 3$（连续正整数 $2,3$ 的乘积），故 $6$ 是好数。$8=2\\times 4$（不连续），$9=3\\times 3$（不连续），$10=2\\times 5$（不连续）。',
      },
    ],
  },
]
