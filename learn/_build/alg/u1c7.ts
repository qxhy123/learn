export const LESSONS = [
  {
    id: 'u1c7',
    title: '二次根式的运算',
    subtitle: '加减合并同类、乘除积商性质、分母有理化',
    intro: [
      {
        title: '四种运算的总览',
        body: '有了最简根式和同类根式的概念，就可以对二次根式进行加、减、乘、除四种运算。\n\n- **加减**：先化最简，再合并同类（加减系数）\n- **乘法**：用 $\\sqrt{a} \\cdot \\sqrt{b} = \\sqrt{ab}$，含括号时展开\n- **除法/有理化**：把分母中的根号消去，变成有理数\n\n乘除用积/商的根号性质，加减要先化简再"认同类"。',
      },
      {
        title: '加减法：先化简，再合并',
        body: '**三步走**：\n1. 将所有根式化为最简形式（去平方因子，消分母）\n2. 识别同类根式（被开方数相同的）\n3. 合并：系数相加减，根式不变\n\n$m\\sqrt{a} \\pm n\\sqrt{a} = (m \\pm n)\\sqrt{a}$\n\n**例**：$\\sqrt{18} - \\sqrt{8} + 3\\sqrt{2}$\n$= 3\\sqrt{2} - 2\\sqrt{2} + 3\\sqrt{2} = 4\\sqrt{2}$',
        tip: '头号错误：$\\sqrt{12} + \\sqrt{3} \\neq \\sqrt{15}$，根号内不能直接相加！必须先化简：$\\sqrt{12}=2\\sqrt{3}$，再合并：$2\\sqrt{3}+\\sqrt{3}=3\\sqrt{3}$。',
      },
      {
        title: '乘法：积的根号性质',
        body: '$\\sqrt{a} \\cdot \\sqrt{b} = \\sqrt{ab} \\quad (a \\geq 0,\\; b \\geq 0)$\n\n多项根式相乘时，先用分配律展开（类比多项式乘法），再化简。\n\n**例**：\n- $\\sqrt{3} \\cdot \\sqrt{6} = \\sqrt{18} = 3\\sqrt{2}$（结果还需化简）\n- $(\\sqrt{3}+\\sqrt{2})(\\sqrt{3}-\\sqrt{2}) = (\\sqrt{3})^2-(\\sqrt{2})^2 = 3-2 = 1$\n- $(\\sqrt{5}+1)^2 = 5 + 2\\sqrt{5} + 1 = 6 + 2\\sqrt{5}$',
        tip: '乘法结果出现完全平方因子时，必须继续化简；$\\sqrt{18}$ 不是最终答案，要写成 $3\\sqrt{2}$。',
      },
      {
        title: '分母有理化——单项分母',
        body: '**目标**：把分母中的根号消去，使分母变成有理数。\n\n**单项分母**（分母是 $\\sqrt{a}$ 型）：上下同乘 $\\sqrt{a}$\n$\\frac{b}{\\sqrt{a}} = \\frac{b \\cdot \\sqrt{a}}{\\sqrt{a} \\cdot \\sqrt{a}} = \\frac{b\\sqrt{a}}{a} \\quad (a > 0)$\n\n**例**：$\\dfrac{6}{\\sqrt{6}} = \\dfrac{6\\sqrt{6}}{6} = \\sqrt{6}$\n\n原理：$\\sqrt{a} \\cdot \\sqrt{a} = (\\sqrt{a})^2 = a$，分母变成有理数 $a$。',
        formula: '\\frac{b}{\\sqrt{a}} = \\frac{b\\sqrt{a}}{a} \\quad (a > 0)',
      },
      {
        title: '分母有理化——二项分母（乘共轭）',
        body: '**二项分母**（分母是 $\\sqrt{a} + \\sqrt{b}$ 型）：上下同乘其**共轭式** $\\sqrt{a} - \\sqrt{b}$，利用平方差公式：\n\n$(\\sqrt{a}+\\sqrt{b})(\\sqrt{a}-\\sqrt{b}) = a - b \\quad (\\text{有理数！})$\n\n**例**：$\\dfrac{2}{\\sqrt{3}-1}$\n$= \\frac{2(\\sqrt{3}+1)}{(\\sqrt{3}-1)(\\sqrt{3}+1)} = \\frac{2(\\sqrt{3}+1)}{3-1} = \\sqrt{3}+1$\n\n**为什么要乘共轭？** 乘以自身 $(\\sqrt{a}+\\sqrt{b})^2 = a+2\\sqrt{ab}+b$ 仍含根号，达不到目的；乘共轭才能用平方差消去根号。',
        tip: '分母是 $\\sqrt{a}-\\sqrt{b}$ 时，共轭式是 $\\sqrt{a}+\\sqrt{b}$（加号），不要搞错符号。',
      },
      {
        title: '综合运算：平方差与完全平方公式',
        body: '根式运算中经常用到乘法公式（和整式完全一样）：\n\n**平方差**：$(a+b)(a-b) = a^2 - b^2$\n令 $a = \\sqrt{3}$，$b = \\sqrt{2}$：$(\\sqrt{3}+\\sqrt{2})(\\sqrt{3}-\\sqrt{2}) = 3-2 = 1$\n\n**完全平方**：$(a+b)^2 = a^2 + 2ab + b^2$\n$(\\sqrt{5}+1)^2 = 5 + 2\\sqrt{5} + 1 = 6 + 2\\sqrt{5}$\n\n**关键**：展开时不要遗漏中间项 $2ab$！',
        reveal: {
          q: '$\\sqrt{5} \\cdot \\sqrt{20} - (\\sqrt{3}-1)^2$ 等于多少？',
          a: '$\\sqrt{5} \\cdot \\sqrt{20} = \\sqrt{100} = 10$；$(\\sqrt{3}-1)^2 = 3-2\\sqrt{3}+1=4-2\\sqrt{3}$；结果 $= 10-(4-2\\sqrt{3}) = 6+2\\sqrt{3}$',
        },
      },
      {
        title: '综合例题：$a^2 - b^2$ 的计算',
        body: '已知 $a = \\sqrt{3}+1$，$b = \\sqrt{3}-1$，求 $a^2 - b^2$。\n\n**思路**：用平方差公式 $a^2-b^2 = (a+b)(a-b)$，无需展开平方。\n\n$a + b = (\\sqrt{3}+1)+(\\sqrt{3}-1) = 2\\sqrt{3}$\n\n$a - b = (\\sqrt{3}+1)-(\\sqrt{3}-1) = 2$\n\n$a^2 - b^2 = 2\\sqrt{3} \\times 2 = 4\\sqrt{3}$\n\n这道题展示了公式法比逐项展开更高效。',
      },
      {
        title: '易错点总结',
        body: '1. $\\sqrt{a} + \\sqrt{b} \\neq \\sqrt{a+b}$（除非其中一个为 $0$），这是头号错误。\n2. 二项分母有理化要乘共轭式，乘以自身无法消去根号。\n3. 完全平方展开时不要漏掉中间项 $2ab$：$(\\sqrt{2}+\\sqrt{3})^2 = 5 + 2\\sqrt{6}$，不是 $5$。\n4. 乘法结果忘记再化简：$\\sqrt{6} \\cdot \\sqrt{3} = \\sqrt{18} = 3\\sqrt{2}$，不能停在 $\\sqrt{18}$。\n5. 分母是 $\\sqrt{a}-\\sqrt{b}$ 时，共轭是 $\\sqrt{a}+\\sqrt{b}$，不是 $\\sqrt{a}-\\sqrt{b}$ 自身。',
      },
    ],
    questions: [
      {
        type: 'input',
        prompt: '计算 $2\\sqrt{3} + \\sqrt{27} - \\sqrt{75}$ 的结果。',
        accept: ['0', '$0$'],
        explain: '$\\sqrt{27} = 3\\sqrt{3}$，$\\sqrt{75} = 5\\sqrt{3}$；$2\\sqrt{3} + 3\\sqrt{3} - 5\\sqrt{3} = (2+3-5)\\sqrt{3} = 0$。',
      },
      {
        type: 'choice',
        prompt: '$\\dfrac{2}{\\sqrt{3}-1}$ 有理化后等于',
        options: [
          '$\\dfrac{2(\\sqrt{3}+1)}{2}$',
          '$\\sqrt{3}+1$',
          '$\\sqrt{3}-1$',
          '$\\dfrac{\\sqrt{3}+1}{2}$',
        ],
        answer: 1,
        explain: '上下乘共轭式 $\\sqrt{3}+1$：分母 $(\\sqrt{3}-1)(\\sqrt{3}+1) = 3-1 = 2$；分子 $2(\\sqrt{3}+1)$；结果 $\\dfrac{2(\\sqrt{3}+1)}{2} = \\sqrt{3}+1$。',
      },
      {
        type: 'judge',
        prompt: '$\\sqrt{12} + \\sqrt{3} = \\sqrt{15}$',
        answer: false,
        explain: '$\\sqrt{12} = 2\\sqrt{3}$；$2\\sqrt{3} + \\sqrt{3} = 3\\sqrt{3}$。根号内不能直接相加，$\\sqrt{12}+\\sqrt{3} \\neq \\sqrt{15}$。',
      },
      {
        type: 'input',
        prompt: '化简 $\\dfrac{6}{\\sqrt{6}}$ 的结果。',
        accept: ['\\sqrt{6}', '$\\sqrt{6}$'],
        explain: '上下乘 $\\sqrt{6}$：$\\dfrac{6\\sqrt{6}}{\\sqrt{6} \\cdot \\sqrt{6}} = \\dfrac{6\\sqrt{6}}{6} = \\sqrt{6}$。',
      },
      {
        type: 'choice',
        prompt: '$(\\sqrt{3}+\\sqrt{2})(\\sqrt{3}-\\sqrt{2})$ 的结果是',
        options: ['$5$', '$1$', '$\\sqrt{6}$', '$3-2$'],
        answer: 1,
        explain: '平方差公式：$(\\sqrt{3})^2 - (\\sqrt{2})^2 = 3 - 2 = 1$。两个无理数相乘得到了有理数，这正是"有理化"思想的体现。',
      },
      {
        type: 'match',
        prompt: '将各运算与其结果配对',
        left: ['$(\\sqrt{5}+1)^2$', '$\\dfrac{3}{\\sqrt{7}+2}$', '$\\sqrt{5} \\cdot \\sqrt{20}$', '$2\\sqrt{3}+\\sqrt{27}$'],
        right: ['$6+2\\sqrt{5}$', '$\\sqrt{7}-2$', '$10$', '$5\\sqrt{3}$'],
        explain: '$(\\sqrt{5}+1)^2 = 5+2\\sqrt{5}+1=6+2\\sqrt{5}$；$\\dfrac{3}{\\sqrt{7}+2} = \\dfrac{3(\\sqrt{7}-2)}{7-4} = \\sqrt{7}-2$；$\\sqrt{5}\\cdot\\sqrt{20}=\\sqrt{100}=10$；$2\\sqrt{3}+3\\sqrt{3}=5\\sqrt{3}$。',
      },
      {
        type: 'input',
        prompt: '已知 $a=\\sqrt{3}+1$，$b=\\sqrt{3}-1$，求 $a^2-b^2$ 的值。',
        accept: ['4\\sqrt{3}', '$4\\sqrt{3}$'],
        explain: '$a^2-b^2=(a+b)(a-b) = 2\\sqrt{3} \\times 2 = 4\\sqrt{3}$。其中 $a+b = 2\\sqrt{3}$，$a-b=2$。',
      },
      {
        type: 'judge',
        prompt: '$(\\sqrt{2}+\\sqrt{3})^2 = 5$',
        answer: false,
        explain: '$(\\sqrt{2}+\\sqrt{3})^2 = (\\sqrt{2})^2 + 2\\sqrt{2}\\sqrt{3} + (\\sqrt{3})^2 = 2+2\\sqrt{6}+3 = 5+2\\sqrt{6} \\neq 5$。展开时必须有中间项 $2\\sqrt{2}\\sqrt{3} = 2\\sqrt{6}$。',
      },
    ],
  },
]
