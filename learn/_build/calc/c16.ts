import type { Lesson } from '../../src/types'

export const LESSONS: Lesson[] = [
  // ─────────────────────────────────────────
  // 关 c16-a  部分分式分解与有理函数积分
  // ─────────────────────────────────────────
  {
    id: 'c16-a',
    title: '部分分式：拆开有理函数',
    subtitle: '真分式→最简分式→逐项积分',
    intro: [
      // 卡 1：为什么要拆？动机
      {
        title: '为什么要"拆开"有理函数？',
        body:
          '遇到 $\\dfrac{x+1}{x^2-5x+6}$，你会直接积分吗？——很难，因为分母是乘积形式。\n\n' +
          '关键洞察：$\\dfrac{1}{x-2}$ 和 $\\dfrac{1}{x-3}$ 都可以直接积分（得 $\\ln|x-2|$ 和 $\\ln|x-3|$）。\n\n' +
          '**策略**：把复杂的有理函数"拆成"若干最简分式之和，每项单独积分，最后合并——这就是**部分分式分解**的核心思想。\n\n' +
          '类比：把 $\\dfrac{7}{6}$ 拆成 $\\dfrac{1}{2}+\\dfrac{1}{3}$，处理"最简"部分往往更方便。',
        tip: '部分分式分解只适用于**真分式**（分子次数 $<$ 分母次数）。若为假分式，先做多项式长除法化成"多项式 $+$ 真分式"。',
      },
      // 卡 2：核心流程
      {
        title: '三步走：拆→定系数→积分',
        body:
          '**第一步：检查是否为真分式。**\n\n' +
          '若 $\\deg P \\geq \\deg Q$，先做长除法：$\\dfrac{P(x)}{Q(x)}=S(x)+\\dfrac{R(x)}{Q(x)}$，其中 $S(x)$ 是商，$R(x)/Q(x)$ 是余下的真分式。\n\n' +
          '**第二步：对 $Q(x)$ 做因式分解，写出部分分式的模板。**\n\n' +
          '$(x-a)$ 型因子 → 对应 $\\dfrac{A}{x-a}$；$(x-a)^2$ 型 → 对应 $\\dfrac{A}{x-a}+\\dfrac{B}{(x-a)^2}$；不可约二次因子 $x^2+px+q$ → 对应 $\\dfrac{Bx+C}{x^2+px+q}$。\n\n' +
          '**第三步：通分，比较系数（或代入特殊值）求 $A,B,C,\\ldots$，逐项积分。**',
        formula:
          '\\frac{P(x)}{Q(x)} = \\frac{A_1}{x-a_1} + \\frac{A_2}{x-a_2} + \\cdots',
      },
      // 卡 3：例题——两个不同一次因子
      {
        title: '例题精讲①：不同一次因子',
        body:
          '**求 $\\displaystyle\\int\\frac{x+1}{(x-2)(x-3)}\\,dx$。**\n\n' +
          '**设模板**：$\\dfrac{x+1}{(x-2)(x-3)}=\\dfrac{A}{x-2}+\\dfrac{B}{x-3}$。\n\n' +
          '**通分**：$x+1=A(x-3)+B(x-2)$。\n\n' +
          '**代入 $x=2$**：$3=A(-1)$，得 $A=-3$。\n\n' +
          '**代入 $x=3$**：$4=B(1)$，得 $B=4$。\n\n' +
          '**积分**：$\\displaystyle\\int\\left(\\frac{-3}{x-2}+\\frac{4}{x-3}\\right)dx=-3\\ln|x-2|+4\\ln|x-3|+C$。\n\n' +
          '**验证**：对结果求导，应还原被积函数 $\\checkmark$。',
        steps: [
          '**识别因子**：分母 $x^2-5x+6=(x-2)(x-3)$，两个不同一次因子',
          '**写模板**：$\\dfrac{A}{x-2}+\\dfrac{B}{x-3}$，通分得恒等式',
          '**代入零点**：令 $x=2$ 得 $A$，令 $x=3$ 得 $B$（"掩盖法"）',
          '**逐项积分**：$\\int\\dfrac{A}{x-a}dx=A\\ln|x-a|+C$',
        ],
        tip: '"掩盖法"：求 $\\dfrac{A}{x-a}$ 中的 $A$，把分母 $(x-a)$ 掩住，在原式中令 $x=a$，直接读出 $A$。',
      },
      // 卡 4：重因子的处理
      {
        title: '重因子：$(x-a)^2$ 要拆两项',
        body:
          '若分母含 $(x-a)^2$，必须拆出两项：$\\dfrac{A}{x-a}+\\dfrac{B}{(x-a)^2}$。\n\n' +
          '**为什么？** 若只写 $\\dfrac{A}{x-a}$，通分后无法还原含 $(x-a)^2$ 的项，无法确定 $A$。\n\n' +
          '**例**：$\\dfrac{x^2}{(x-1)(x+1)^2}=\\dfrac{A}{x-1}+\\dfrac{B}{x+1}+\\dfrac{C}{(x+1)^2}$。\n\n' +
          '通分：$x^2=A(x+1)^2+B(x-1)(x+1)+C(x-1)$。\n\n' +
          '令 $x=1$：$1=4A$，$A=\\dfrac{1}{4}$；令 $x=-1$：$1=-2C$，$C=-\\dfrac{1}{2}$；比较 $x^2$ 系数：$1=A+B$，$B=\\dfrac{3}{4}$。\n\n' +
          '积分：$\\dfrac{1}{4}\\ln|x-1|+\\dfrac{3}{4}\\ln|x+1|+\\dfrac{1}{2(x+1)}+C$。',
        formula:
          '\\frac{B}{(x-a)^n} \\xrightarrow{\\int} \\frac{B}{(1-n)(x-a)^{n-1}}+C \\quad (n\\geq2)',
      },
      // 卡 5：不可约二次因子与配方
      {
        title: '不可约二次因子：配方后积分',
        body:
          '若分母含 $x^2+px+q$（判别式 $p^2-4q<0$，不可分解），对应项为 $\\dfrac{Bx+C}{x^2+px+q}$，积分时需配方。\n\n' +
          '**例**：$\\displaystyle\\int\\frac{1}{x^2+2x+5}\\,dx$。\n\n' +
          '配方：$x^2+2x+5=(x+1)^2+4$，令 $u=x+1$：\n\n' +
          '$\\displaystyle\\int\\frac{1}{u^2+4}\\,du=\\frac{1}{2}\\arctan\\frac{u}{2}+C=\\frac{1}{2}\\arctan\\frac{x+1}{2}+C$。\n\n' +
          '**公式记忆**：$\\displaystyle\\int\\frac{du}{u^2+a^2}=\\frac{1}{a}\\arctan\\frac{u}{a}+C$，配方后凑出这个标准型。',
        formula:
          '\\int\\frac{du}{u^2+a^2}=\\frac{1}{a}\\arctan\\frac{u}{a}+C',
        tip: '含分子 $Bx+C$ 时，先把分子拆成 $\\dfrac{B}{2}(2x+p)+(C-\\dfrac{Bp}{2})$，前半部分凑 $d(x^2+px+q)$ 得对数，后半部分配方得 $\\arctan$。',
      },
      // 卡 6：易错点
      {
        title: '易错点汇总',
        body:
          '**易错点①：忘记先化真分式。** 若分子次数 $\\geq$ 分母次数，直接写部分分式是错的，必须先做长除法。\n\n' +
          '例：$\\dfrac{x^3}{x^2-1}$，分子 3 次 $>$ 分母 2 次 → 先除法：$x^3=(x)(x^2-1)+x$，得 $x+\\dfrac{x}{x^2-1}$。\n\n' +
          '**易错点②：重因子漏拆项。** $(x-a)^2$ 对应两项：$\\dfrac{A}{x-a}+\\dfrac{B}{(x-a)^2}$，缺一不可。\n\n' +
          '**易错点③：不可约二次因子写成两个一次项。** $x^2+1$ 在实数范围内不可分解，不能写成 $\\dfrac{A}{x+i}+\\dfrac{B}{x-i}$（那是复数域）。\n\n' +
          '**易错点④：积分 $\\dfrac{A}{(x-a)^2}$ 时错用 $\\ln$。** 正确结果是 $\\dfrac{-A}{x-a}$，不是 $\\ln$。',
        reveal: {
          q: '对 $\\dfrac{1}{x^2-1}$ 做部分分式分解，设 $\\dfrac{A}{x-1}+\\dfrac{B}{x+1}$，求 $A$ 和 $B$。',
          a: '通分：$1=A(x+1)+B(x-1)$。令 $x=1$：$1=2A$，$A=\\dfrac{1}{2}$；令 $x=-1$：$1=-2B$，$B=-\\dfrac{1}{2}$。',
        },
      },
      // 卡 7：完整例题——综合运用
      {
        title: '完整例题：从头到尾',
        body:
          '**求 $\\displaystyle\\int\\frac{x^2+1}{x(x-1)^2}\\,dx$。**\n\n' +
          '**检查**：分子 2 次 $<$ 分母 3 次，是真分式，可直接分解。\n\n' +
          '**设模板**：$\\dfrac{A}{x}+\\dfrac{B}{x-1}+\\dfrac{C}{(x-1)^2}$，通分得 $x^2+1=A(x-1)^2+Bx(x-1)+Cx$。\n\n' +
          '**定系数**：令 $x=0$：$1=A$，$A=1$；令 $x=1$：$2=C$，$C=2$；令 $x^2$ 系数：$1=A+B$，$B=0$。\n\n' +
          '**积分**：$\\displaystyle\\int\\left(\\frac{1}{x}+\\frac{2}{(x-1)^2}\\right)dx=\\ln|x|-\\frac{2}{x-1}+C$。',
        steps: [
          '**第一步**：确认真分式，分母 $x(x-1)^2$ 已分解',
          '**第二步**：写模板 $A/x+B/(x-1)+C/(x-1)^2$',
          '**第三步**：代入 $x=0,1$ 得 $A,C$；比较系数得 $B$',
          '**第四步**：$\\int 1/x\\,dx=\\ln|x|$，$\\int 2/(x-1)^2\\,dx=-2/(x-1)$，合并',
        ],
      },
      // 卡 8：条件反射速查
      {
        title: '条件反射：看到什么，做什么',
        body:
          '**看到有理函数 $P(x)/Q(x)$：**\n\n' +
          '① 检查真假分式 → ② 分解 $Q(x)$ → ③ 写模板 → ④ 定系数 → ⑤ 逐项积分。\n\n' +
          '**分母因子类型速查：**\n\n' +
          '$(x-a)$ → 对应 $\\dfrac{A}{x-a}$，积分得 $A\\ln|x-a|$。\n\n' +
          '$(x-a)^2$ → 对应 $\\dfrac{A}{x-a}+\\dfrac{B}{(x-a)^2}$，积分得 $A\\ln|x-a|-\\dfrac{B}{x-a}$。\n\n' +
          '$(x^2+px+q)$（不可约）→ 对应 $\\dfrac{Bx+C}{x^2+px+q}$，积分需配方用 $\\arctan$。',
        formula:
          '\\int\\frac{A}{x-a}\\,dx=A\\ln|x-a|+C, \\quad \\int\\frac{A}{(x-a)^2}\\,dx=\\frac{-A}{x-a}+C',
      },
    ],
    questions: [
      // 题 1：choice — 真分式判断
      {
        id: 'c16-a-q1',
        type: 'choice',
        prompt:
          '下列哪个是真分式，可以直接做部分分式分解？',
        options: [
          '$\\dfrac{x^3+1}{x^2-1}$',
          '$\\dfrac{x^2}{x^3+1}$',
          '$\\dfrac{x^2+x}{x^2-1}$',
          '$\\dfrac{x^3}{x-1}$',
        ],
        answer: 1,
        explain:
          '$\\dfrac{x^2}{x^3+1}$ 中分子次数 $2<$ 分母次数 $3$，是真分式，可直接分解。其余选项分子次数 $\\geq$ 分母次数，须先做多项式长除法。',
      },
      // 题 2：input — 掩盖法求系数
      {
        id: 'c16-a-q2',
        type: 'input',
        prompt:
          '设 $\\dfrac{1}{(x-1)(x+1)}=\\dfrac{A}{x-1}+\\dfrac{B}{x+1}$，用掩盖法（令 $x=1$）求 $A$ 的值（化为最简分数，如 $1/2$）。',
        accept: ['1/2', '0.5'],
        placeholder: '输入分数或小数',
        explain:
          '通分：$1=A(x+1)+B(x-1)$。令 $x=1$：$1=A\\cdot2$，$A=\\dfrac{1}{2}$。类似地令 $x=-1$ 得 $B=-\\dfrac{1}{2}$。',
      },
      // 题 3：judge — 重因子拆法
      {
        id: 'c16-a-q3',
        type: 'judge',
        prompt:
          '对分母含 $(x-2)^2$ 的真分式，部分分式只需写一项 $\\dfrac{A}{(x-2)^2}$，无需写 $\\dfrac{B}{x-2}$。',
        answer: false,
        explain:
          '$(x-2)^2$ 对应必须拆出两项：$\\dfrac{A}{x-2}+\\dfrac{B}{(x-2)^2}$，缺少 $\\dfrac{A}{x-2}$ 会导致方程组无解，无法完整还原原分式。',
      },
      // 题 4：choice — 积分 1/(x-a)^2
      {
        id: 'c16-a-q4',
        type: 'choice',
        prompt:
          '$\\displaystyle\\int\\frac{2}{(x-3)^2}\\,dx$ 的结果是？',
        options: [
          '$2\\ln|x-3|+C$',
          '$\\dfrac{-2}{x-3}+C$',
          '$\\dfrac{2}{(x-3)^3}+C$',
          '$\\dfrac{1}{(x-3)^2}+C$',
        ],
        answer: 1,
        explain:
          '$\\displaystyle\\int\\frac{2}{(x-3)^2}\\,dx=2\\int(x-3)^{-2}\\,dx=2\\cdot\\dfrac{(x-3)^{-1}}{-1}+C=\\dfrac{-2}{x-3}+C$。注意不要错用 $\\ln$！',
      },
      // 题 5：input — 完整部分分式积分
      {
        id: 'c16-a-q5',
        type: 'input',
        prompt:
          '$\\displaystyle\\int\\frac{1}{x^2-1}\\,dx=\\frac{1}{2}\\ln|x-1|-\\frac{1}{2}\\ln|x+1|+C$，化简可写成 $\\dfrac{1}{2}\\ln\\left|\\dfrac{x-1}{x+1}\\right|+C$。原部分分式中 $B$（即 $\\dfrac{B}{x+1}$ 的系数）等于多少（输入分数，如 $-1/2$）？',
        accept: ['-1/2', '-0.5'],
        placeholder: '输入分数',
        explain:
          '设 $\\dfrac{1}{(x-1)(x+1)}=\\dfrac{A}{x-1}+\\dfrac{B}{x+1}$，通分：$1=A(x+1)+B(x-1)$。令 $x=-1$：$1=B(-2)$，$B=-\\dfrac{1}{2}$。',
      },
      // 题 6：match — 分母因子类型与积分结果
      {
        id: 'c16-a-q6',
        type: 'match',
        prompt: '将部分分式的项型与对应的积分结果配对。',
        left: [
          '$\\dfrac{A}{x-a}$',
          '$\\dfrac{B}{(x-a)^2}$',
          '$\\dfrac{C}{x^2+a^2}$',
        ],
        right: [
          '$A\\ln|x-a|+C$',
          '$\\dfrac{-B}{x-a}+C$',
          '$\\dfrac{C}{a}\\arctan\\dfrac{x}{a}+C$',
        ],
      },
      // 题 7：judge — 不可约二次式
      {
        id: 'c16-a-q7',
        type: 'judge',
        prompt:
          '$x^2+x+1$ 在实数范围内可分解为两个一次因子的乘积。',
        answer: false,
        explain:
          '判别式 $\\Delta=1^2-4\\cdot1\\cdot1=-3<0$，无实数根，$x^2+x+1$ 在实数范围内是**不可约**的，对应的部分分式写 $\\dfrac{Bx+C}{x^2+x+1}$，积分需配方用 $\\arctan$。',
      },
      // 题 8：choice — 综合识别
      {
        id: 'c16-a-q8',
        type: 'choice',
        prompt:
          '对 $\\displaystyle\\int\\frac{x^2+1}{x(x-1)^2}\\,dx$，正确的部分分式模板是？',
        options: [
          '$\\dfrac{A}{x}+\\dfrac{B}{x-1}$',
          '$\\dfrac{A}{x}+\\dfrac{B}{(x-1)^2}$',
          '$\\dfrac{A}{x}+\\dfrac{B}{x-1}+\\dfrac{C}{(x-1)^2}$',
          '$\\dfrac{Ax+B}{x}+\\dfrac{C}{x-1}$',
        ],
        answer: 2,
        explain:
          '分母 $x(x-1)^2$：$x$ 是一次因子对应 $\\dfrac{A}{x}$；$(x-1)^2$ 是二重因子对应**两项** $\\dfrac{B}{x-1}+\\dfrac{C}{(x-1)^2}$，共三项。实际上 $B=0$（本题特殊情形），但模板仍需写三项。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c16-b  三角代换：根号无理函数的积分
  // ─────────────────────────────────────────
  {
    id: 'c16-b',
    title: '三角代换：消灭根号',
    subtitle: '$\\sqrt{a^2-x^2}$·$\\sqrt{a^2+x^2}$·$\\sqrt{x^2-a^2}$',
    intro: [
      // 卡 1：动机——根号让直接积分失效
      {
        title: '为什么要用三角代换？',
        body:
          '遇到 $\\displaystyle\\int\\frac{x^2}{\\sqrt{4-x^2}}\\,dx$，换元 $u=4-x^2$ 得 $du=-2x\\,dx$，分子有 $x^2$ 而非 $x$，凑不进去，直接换元失败。\n\n' +
          '**核心洞察**：利用三角恒等式 $\\sin^2\\theta+\\cos^2\\theta=1$ 和 $1+\\tan^2\\theta=\\sec^2\\theta$，令 $x$ 等于某个三角函数，就能把根号内的"完全平方"提出来，**彻底消灭根号**。\n\n' +
          '这是一种"有代价的换元"——被积函数变成三角式，更复杂，但**根号消失**，变成我们会算的三角积分。',
        formula:
          '\\sqrt{a^2-x^2} \\xrightarrow{x=a\\sin\\theta} a\\cos\\theta',
      },
      // 卡 2：三种代换总表
      {
        title: '三种代换一览表',
        body:
          '**规律**：根号内的结构决定代换方法。\n\n' +
          '**$\\sqrt{a^2-x^2}$型**：令 $x=a\\sin\\theta$（$\\theta\\in[-\\pi/2,\\pi/2]$），则 $\\sqrt{a^2-x^2}=a\\cos\\theta$，$dx=a\\cos\\theta\\,d\\theta$。\n\n' +
          '**$\\sqrt{a^2+x^2}$型**：令 $x=a\\tan\\theta$（$\\theta\\in(-\\pi/2,\\pi/2)$），则 $\\sqrt{a^2+x^2}=a\\sec\\theta$，$dx=a\\sec^2\\theta\\,d\\theta$。\n\n' +
          '**$\\sqrt{x^2-a^2}$型**：令 $x=a\\sec\\theta$，则 $\\sqrt{x^2-a^2}=a|\\tan\\theta|$，$dx=a\\sec\\theta\\tan\\theta\\,d\\theta$。',
        formula:
          '\\begin{array}{ll}\\sqrt{a^2-x^2}: & x=a\\sin\\theta \\\\\\sqrt{a^2+x^2}: & x=a\\tan\\theta \\\\\\sqrt{x^2-a^2}: & x=a\\sec\\theta\\end{array}',
        tip: '记忆口诀：**减用正弦，加用正切，减左用正割**。识别根号内的正负号即可确定代换。',
      },
      // 卡 3：例题——$\sqrt{a^2-x^2}$型
      {
        title: '例题精讲①：$\\sqrt{a^2-x^2}$ 型',
        body:
          '**求 $\\displaystyle\\int\\frac{x^2}{\\sqrt{4-x^2}}\\,dx$。**\n\n' +
          '**识别**：$\\sqrt{4-x^2}=\\sqrt{2^2-x^2}$，令 $x=2\\sin\\theta$，$dx=2\\cos\\theta\\,d\\theta$，$\\sqrt{4-x^2}=2\\cos\\theta$（$\\theta\\in[-\\pi/2,\\pi/2]$，$\\cos\\theta\\geq0$）。\n\n' +
          '**代入**：$\\displaystyle\\int\\frac{4\\sin^2\\theta}{2\\cos\\theta}\\cdot2\\cos\\theta\\,d\\theta=4\\int\\sin^2\\theta\\,d\\theta=4\\int\\frac{1-\\cos2\\theta}{2}\\,d\\theta=2\\theta-\\sin2\\theta+C$。\n\n' +
          '**回代**：$\\theta=\\arcsin\\dfrac{x}{2}$，$\\sin2\\theta=2\\sin\\theta\\cos\\theta=2\\cdot\\dfrac{x}{2}\\cdot\\dfrac{\\sqrt{4-x^2}}{2}=\\dfrac{x\\sqrt{4-x^2}}{2}$。\n\n' +
          '**结果**：$2\\arcsin\\dfrac{x}{2}-\\dfrac{x\\sqrt{4-x^2}}{2}+C$。',
        steps: [
          '**识别型**：$\\sqrt{2^2-x^2}$，用正弦代换 $x=2\\sin\\theta$',
          '**换元**：$dx=2\\cos\\theta\\,d\\theta$，根号 $=2\\cos\\theta$，代入化简',
          '**降幂**：$\\sin^2\\theta=\\dfrac{1-\\cos2\\theta}{2}$，积分得 $\\theta$ 和 $\\sin2\\theta$',
          '**回代**：$\\theta=\\arcsin(x/2)$，$\\sin2\\theta=2\\sin\\theta\\cos\\theta$ 用 $x$ 和根号表示',
        ],
      },
      // 卡 4：例题——$\sqrt{a^2+x^2}$型
      {
        title: '例题精讲②：$\\sqrt{a^2+x^2}$ 型',
        body:
          '**求 $\\displaystyle\\int\\frac{1}{\\sqrt{x^2+9}}\\,dx$。**\n\n' +
          '**识别**：$\\sqrt{x^2+9}=\\sqrt{x^2+3^2}$，令 $x=3\\tan\\theta$，$dx=3\\sec^2\\theta\\,d\\theta$，$\\sqrt{x^2+9}=3\\sec\\theta$。\n\n' +
          '**代入**：$\\displaystyle\\int\\frac{3\\sec^2\\theta}{3\\sec\\theta}\\,d\\theta=\\int\\sec\\theta\\,d\\theta=\\ln|\\sec\\theta+\\tan\\theta|+C$。\n\n' +
          '**回代**：$\\tan\\theta=\\dfrac{x}{3}$，$\\sec\\theta=\\dfrac{\\sqrt{x^2+9}}{3}$（由 $\\sec^2\\theta=1+\\tan^2\\theta=1+x^2/9$）。\n\n' +
          '**结果**：$\\ln\\left|\\dfrac{\\sqrt{x^2+9}}{3}+\\dfrac{x}{3}\\right|+C=\\ln|x+\\sqrt{x^2+9}|+C$（吸收常数 $\\ln3$ 进 $C$）。',
        tip: '回代时，用直角三角形图辅助记忆：若 $\\tan\\theta=x/a$，则斜边 $=\\sqrt{x^2+a^2}$，$\\sec\\theta=\\sqrt{x^2+a^2}/a$。',
      },
      // 卡 5：回代的技巧
      {
        title: '回代技巧：直角三角形大法',
        body:
          '三角代换后回代，最可靠的方法是**画直角三角形**。\n\n' +
          '**以 $x=2\\sin\\theta$ 为例**：$\\sin\\theta=x/2$，画三角形，对边 $x$，斜边 $2$，则邻边 $=\\sqrt{4-x^2}$，$\\cos\\theta=\\sqrt{4-x^2}/2$，$\\tan\\theta=x/\\sqrt{4-x^2}$。\n\n' +
          '**以 $x=3\\tan\\theta$ 为例**：$\\tan\\theta=x/3$，对边 $x$，邻边 $3$，斜边 $=\\sqrt{x^2+9}$，$\\sec\\theta=\\sqrt{x^2+9}/3$。\n\n' +
          '**关键**：三角代换的范围使得 $\\cos\\theta>0$（或 $\\sec\\theta>0$），所以开根号不用加绝对值符号（已知正号）。\n\n' +
          '**特别注意** $\\sqrt{x^2-a^2}$ 型（$x=a\\sec\\theta$）：$\\sqrt{x^2-a^2}=a|\\tan\\theta|$，具体正负取决于 $x>a$ 还是 $x<-a$。',
        formula:
          '\\tan\\theta=\\frac{x}{a} \\Rightarrow \\sec\\theta=\\frac{\\sqrt{x^2+a^2}}{a},\\quad \\sin\\theta=\\frac{x}{\\sqrt{x^2+a^2}}',
      },
      // 卡 6：易错点
      {
        title: '易错点汇总',
        body:
          '**易错点①：忘记换 $dx$。** 令 $x=a\\sin\\theta$，$dx=a\\cos\\theta\\,d\\theta$，$dx$ 必须一起换，否则积分错误。\n\n' +
          '**易错点②：根号范围问题。** $\\sqrt{a^2-x^2}=a\\cos\\theta$（当 $\\theta\\in[-\\pi/2,\\pi/2]$ 时 $\\cos\\theta\\geq0$，直接去根号），不是 $a|\\cos\\theta|$。取错范围会导致符号错误。\n\n' +
          '**易错点③：回代忘记把 $\\theta$ 全换成 $x$。** 所有的 $\\sin\\theta$、$\\cos\\theta$、$\\theta$ 都需要用 $x$ 表示完毕。\n\n' +
          '**易错点④：识别型号错误。** $\\sqrt{x^2+4}$ 是加号（用正切），$\\sqrt{4-x^2}$ 是减号（用正弦），两者绝对不可混淆。',
        reveal: {
          q: '对 $\\displaystyle\\int\\sqrt{1-x^2}\\,dx$，令 $x=\\sin\\theta$，化简后被积函数变成什么？',
          a: '$x=\\sin\\theta$，$dx=\\cos\\theta\\,d\\theta$，$\\sqrt{1-x^2}=\\cos\\theta$，被积函数变成 $\\cos^2\\theta$，再用降幂公式 $\\cos^2\\theta=\\dfrac{1+\\cos2\\theta}{2}$ 积分。',
        },
      },
      // 卡 7：与换元法的对比
      {
        title: '三角代换 vs 直接换元：怎么选？',
        body:
          '**先试直接换元**：若 $\\sqrt{a^2-x^2}$ 前有 $x\\,dx$（即"配套的微分"），令 $u=a^2-x^2$ 就能凑 $du$，不需要三角代换。\n\n' +
          '**例**：$\\displaystyle\\int x\\sqrt{1-x^2}\\,dx$，令 $u=1-x^2$，$du=-2x\\,dx$，化为 $-\\dfrac{1}{2}\\int\\sqrt{u}\\,du=-\\dfrac{u^{3/2}}{3}+C=-\\dfrac{(1-x^2)^{3/2}}{3}+C$，无需三角代换！\n\n' +
          '**三角代换的信号**：分母含根号，或根号无法通过直接换元消去（分子"配不上"）。\n\n' +
          '**判断口诀**：有根号 + 没有配套微分 → 考虑三角代换；有配套微分 → 优先直接换元。',
        formula:
          '\\int x\\sqrt{1-x^2}\\,dx\\xrightarrow{u=1-x^2}-\\frac{(1-x^2)^{3/2}}{3}+C',
      },
      // 卡 8：小结
      {
        title: '本关小结与条件反射',
        body:
          '**三种代换决策树：**\n\n' +
          '看到 $\\sqrt{a^2-x^2}$ → 令 $x=a\\sin\\theta$ → 根号变 $a\\cos\\theta$。\n\n' +
          '看到 $\\sqrt{a^2+x^2}$ → 令 $x=a\\tan\\theta$ → 根号变 $a\\sec\\theta$。\n\n' +
          '看到 $\\sqrt{x^2-a^2}$ → 令 $x=a\\sec\\theta$ → 根号变 $a|\\tan\\theta|$。\n\n' +
          '**回代三步**：① 用直角三角形画出所有三角函数与 $x$ 的关系；② 把 $\\theta$ 换成 $\\arcsin/\\arctan$；③ 把其他三角函数换成 $x$ 和根号的代数式。\n\n' +
          '**最终验证**：对结果求导，应还原被积函数。',
        formula:
          '\\frac{d}{dx}\\left[2\\arcsin\\frac{x}{2}-\\frac{x\\sqrt{4-x^2}}{2}\\right]=\\frac{x^2}{\\sqrt{4-x^2}}',
      },
    ],
    questions: [
      // 题 1：match — 根号型与对应代换
      {
        id: 'c16-b-q1',
        type: 'match',
        prompt: '将根号结构与对应的三角代换配对。',
        left: [
          '$\\sqrt{a^2-x^2}$',
          '$\\sqrt{a^2+x^2}$',
          '$\\sqrt{x^2-a^2}$',
        ],
        right: [
          '$x=a\\sin\\theta$',
          '$x=a\\tan\\theta$',
          '$x=a\\sec\\theta$',
        ],
      },
      // 题 2：choice — 代换后化简
      {
        id: 'c16-b-q2',
        type: 'choice',
        prompt:
          '令 $x=3\\sin\\theta$，则 $\\sqrt{9-x^2}$ 化简为（$\\theta\\in[-\\pi/2,\\pi/2]$）？',
        options: [
          '$3|\\cos\\theta|$',
          '$3\\cos\\theta$',
          '$3\\sin\\theta$',
          '$\\cos\\theta$',
        ],
        answer: 1,
        explain:
          '$x=3\\sin\\theta$，$9-x^2=9-9\\sin^2\\theta=9\\cos^2\\theta$，$\\sqrt{9-x^2}=3|\\cos\\theta|$。由于 $\\theta\\in[-\\pi/2,\\pi/2]$，$\\cos\\theta\\geq0$，所以 $|\\cos\\theta|=\\cos\\theta$，结果为 $3\\cos\\theta$。',
      },
      // 题 3：judge — 换 dx 的重要性
      {
        id: 'c16-b-q3',
        type: 'judge',
        prompt:
          '在三角代换 $x=2\\tan\\theta$ 中，$dx=2\\,d\\theta$（即 $dx$ 等于 $2$ 乘以 $d\\theta$）。',
        answer: false,
        explain:
          '$x=2\\tan\\theta$，对 $\\theta$ 求导：$\\dfrac{dx}{d\\theta}=2\\sec^2\\theta$，所以 $dx=2\\sec^2\\theta\\,d\\theta$，不是 $2\\,d\\theta$。忘记 $\\sec^2\\theta$ 因子是常见错误。',
      },
      // 题 4：choice — 选择代换类型
      {
        id: 'c16-b-q4',
        type: 'choice',
        prompt:
          '求 $\\displaystyle\\int\\frac{1}{\\sqrt{x^2-16}}\\,dx$ 时，应选哪种三角代换？',
        options: [
          '$x=4\\sin\\theta$',
          '$x=4\\tan\\theta$',
          '$x=4\\sec\\theta$',
          '$x=4\\cos\\theta$',
        ],
        answer: 2,
        explain:
          '$\\sqrt{x^2-16}=\\sqrt{x^2-4^2}$，属于 $\\sqrt{x^2-a^2}$ 型，应令 $x=4\\sec\\theta$，则 $x^2-16=16\\sec^2\\theta-16=16\\tan^2\\theta$，根号化为 $4|\\tan\\theta|$。',
      },
      // 题 5：input — 代换后的 dx
      {
        id: 'c16-b-q5',
        type: 'input',
        prompt:
          '令 $x=3\\tan\\theta$，则 $dx$ 等于 $k\\sec^2\\theta\\,d\\theta$，其中 $k$ 等于多少？',
        accept: ['3'],
        placeholder: '输入整数',
        explain:
          '$x=3\\tan\\theta$，$\\dfrac{dx}{d\\theta}=3\\sec^2\\theta$，故 $dx=3\\sec^2\\theta\\,d\\theta$，$k=3$。',
      },
      // 题 6：judge — 直接换元 vs 三角代换
      {
        id: 'c16-b-q6',
        type: 'judge',
        prompt:
          '计算 $\\displaystyle\\int x\\sqrt{1-x^2}\\,dx$ 时，用直接换元 $u=1-x^2$ 比三角代换更简便。',
        answer: true,
        explain:
          '$u=1-x^2$，$du=-2x\\,dx$，被积函数 $x\\,dx=-\\dfrac{1}{2}du$，积分化为 $-\\dfrac{1}{2}\\int u^{1/2}\\,du=-\\dfrac{u^{3/2}}{3}+C=-\\dfrac{(1-x^2)^{3/2}}{3}+C$。有"配套的 $x\\,dx$"，直接换元即可，无需三角代换。',
      },
      // 题 7：choice — 回代结果
      {
        id: 'c16-b-q7',
        type: 'choice',
        prompt:
          '令 $x=a\\sin\\theta$，已知 $\\sin\\theta=x/a$，则 $\\cos\\theta$（设 $\\theta\\in[-\\pi/2,\\pi/2]$）用 $x$ 和 $a$ 表示是？',
        options: [
          '$\\dfrac{a}{\\sqrt{a^2-x^2}}$',
          '$\\dfrac{\\sqrt{a^2-x^2}}{a}$',
          '$\\dfrac{x}{\\sqrt{a^2-x^2}}$',
          '$\\dfrac{x}{a}$',
        ],
        answer: 1,
        explain:
          '$\\cos\\theta=\\sqrt{1-\\sin^2\\theta}=\\sqrt{1-x^2/a^2}=\\dfrac{\\sqrt{a^2-x^2}}{a}$（$\\theta\\in[-\\pi/2,\\pi/2]$，$\\cos\\theta\\geq0$，取正值）。这是回代时最常用的关系。',
      },
      // 题 8：input — 对 sqrt(9-x^2) 积分结果
      {
        id: 'c16-b-q8',
        type: 'input',
        prompt:
          '$\\displaystyle\\int\\frac{dx}{\\sqrt{9-x^2}}=\\arcsin\\frac{x}{k}+C$，其中 $k$ 等于多少？',
        accept: ['3'],
        placeholder: '输入整数',
        explain:
          '令 $x=3\\sin\\theta$，$dx=3\\cos\\theta\\,d\\theta$，$\\sqrt{9-x^2}=3\\cos\\theta$。积分化为 $\\int\\dfrac{3\\cos\\theta}{3\\cos\\theta}\\,d\\theta=\\int d\\theta=\\theta+C=\\arcsin\\dfrac{x}{3}+C$，故 $k=3$。',
      },
    ],
  },

  // ─────────────────────────────────────────
  // 关 c16-c  定积分特殊技巧：华里士公式与区间再现
  // ─────────────────────────────────────────
  {
    id: 'c16-c',
    title: '定积分的特殊技巧',
    subtitle: '奇偶对称·华里士公式·区间再现',
    intro: [
      // 卡 1：奇偶对称性——最快的一招
      {
        title: '最快一招：奇偶函数的对称积分',
        body:
          '对称区间 $[-a,a]$ 上的定积分，先看奇偶性——往往不用计算直接得答案！\n\n' +
          '**偶函数** $f(-x)=f(x)$：$\\displaystyle\\int_{-a}^{a}f(x)\\,dx=2\\int_{0}^{a}f(x)\\,dx$（左右对称，翻倍）。\n\n' +
          '**奇函数** $f(-x)=-f(x)$：$\\displaystyle\\int_{-a}^{a}f(x)\\,dx=0$（正负抵消）。\n\n' +
          '**例**：$\\displaystyle\\int_{-\\pi}^{\\pi}\\frac{x^2\\sin x}{1+x^4}\\,dx$，令 $g(x)=\\dfrac{x^2\\sin x}{1+x^4}$，检验 $g(-x)=-g(x)$（奇函数），答案直接是 $0$。\n\n' +
          '**三步验证奇偶性**：① 写出 $f(-x)$；② 化简；③ 与 $f(x)$ 或 $-f(x)$ 比较。',
        formula:
          '\\int_{-a}^{a}f(x)\\,dx=\\begin{cases}2\\displaystyle\\int_{0}^{a}f(x)\\,dx, & f\\text{ 为偶函数}\\\\0, & f\\text{ 为奇函数}\\end{cases}',
        tip: '奇函数的特征：含奇数个 $x$ 的因子（或 $\\sin x$、$x^3$ 等）且不含偶函数部分。偶函数的特征：只含 $x^2$、$\\cos x$ 等。乘积"奇×偶=奇，奇×奇=偶"。',
      },
      // 卡 2：华里士公式推导与记忆
      {
        title: '华里士公式：$\\sin^n x$ 的定积分捷径',
        body:
          '$I_n=\\displaystyle\\int_0^{\\pi/2}\\sin^n x\\,dx$ 的计算有递推关系：$I_n=\\dfrac{n-1}{n}I_{n-2}$（$n\\geq2$），$I_0=\\dfrac{\\pi}{2}$，$I_1=1$。\n\n' +
          '**推导**（分部积分）：$I_n=-\\int_0^{\\pi/2}\\sin^{n-1}x\\,d(\\cos x)=(n-1)\\int_0^{\\pi/2}\\sin^{n-2}x\\cos^2x\\,dx=(n-1)(I_{n-2}-I_n)$，解出 $I_n=\\dfrac{n-1}{n}I_{n-2}$。\n\n' +
          '**展开后的闭合公式：**\n\n' +
          '$n$ 为偶数：$I_n=\\dfrac{(n-1)!!}{n!!}\\cdot\\dfrac{\\pi}{2}$（如 $I_6=\\dfrac{5\\cdot3\\cdot1}{6\\cdot4\\cdot2}\\cdot\\dfrac{\\pi}{2}=\\dfrac{5\\pi}{32}$）。\n\n' +
          '$n$ 为奇数：$I_n=\\dfrac{(n-1)!!}{n!!}$（如 $I_5=\\dfrac{4\\cdot2}{5\\cdot3\\cdot1}=\\dfrac{8}{15}$）。\n\n' +
          '**同样适用于 $\\int_0^{\\pi/2}\\cos^n x\\,dx$**（结果相同）。',
        formula:
          'I_n=\\int_0^{\\pi/2}\\sin^n x\\,dx=\\begin{cases}\\dfrac{(n-1)!!}{n!!}\\cdot\\dfrac{\\pi}{2}, & n\\text{ 偶}\\\\[4pt]\\dfrac{(n-1)!!}{n!!}, & n\\text{ 奇}\\end{cases}',
        tip: '双阶乘 $n!!=n(n-2)(n-4)\\cdots$，偶数降到 $2$，奇数降到 $1$。例：$5!!=5\\cdot3\\cdot1=15$，$6!!=6\\cdot4\\cdot2=48$。',
      },
      // 卡 3：华里士公式例题
      {
        title: '华里士公式实战',
        body:
          '**例①**：$\\displaystyle\\int_0^{\\pi/2}\\sin^4 x\\,dx$（$n=4$，偶数）。\n\n' +
          '$I_4=\\dfrac{3!!}{4!!}\\cdot\\dfrac{\\pi}{2}=\\dfrac{3\\cdot1}{4\\cdot2}\\cdot\\dfrac{\\pi}{2}=\\dfrac{3}{8}\\cdot\\dfrac{\\pi}{2}=\\dfrac{3\\pi}{16}$。\n\n' +
          '**例②**：$\\displaystyle\\int_0^{\\pi/2}\\cos^5 x\\,dx$（$n=5$，奇数，与 $\\sin$ 结果相同）。\n\n' +
          '$I_5=\\dfrac{4!!}{5!!}=\\dfrac{4\\cdot2}{5\\cdot3\\cdot1}=\\dfrac{8}{15}$。\n\n' +
          '**例③**：$\\displaystyle\\int_0^{\\pi/2}\\sin^4 x\\cos^2 x\\,dx=\\int_0^{\\pi/2}(\\sin^4 x-\\sin^6 x)\\,dx=I_4-I_6=\\dfrac{3\\pi}{16}-\\dfrac{5\\pi}{32}=\\dfrac{\\pi}{32}$。\n\n' +
          '（先用 $\\cos^2 x=1-\\sin^2 x$ 拆开，再分别用华里士公式。）',
        steps: [
          '**第一步**：判断 $n$ 的奇偶性',
          '**偶数**：$\\dfrac{(n-1)!!}{n!!}\\cdot\\dfrac{\\pi}{2}$，注意乘 $\\pi/2$',
          '**奇数**：$\\dfrac{(n-1)!!}{n!!}$，不乘 $\\pi/2$',
          '**混合型**：用 $\\cos^2=1-\\sin^2$ 或 $\\sin^2=1-\\cos^2$ 拆开，再分项套公式',
        ],
      },
      // 卡 4：区间再现公式
      {
        title: '区间再现公式：换元后加自身',
        body:
          '**公式**：$\\displaystyle\\int_a^b f(x)\\,dx=\\int_a^b f(a+b-x)\\,dx$。\n\n' +
          '**证明**（令 $t=a+b-x$，$dt=-dx$，上下限互换两次）：\n\n' +
          '$\\displaystyle\\int_a^b f(a+b-x)\\,dx=-\\int_b^a f(t)\\,dt=\\int_a^b f(t)\\,dt$。$\\checkmark$\n\n' +
          '**妙用**：设原积分 $=I$，令 $t=a+b-x$ 换元得另一个形式 $I\'$。若 $I+I\'$ 能化简为易算的积分，则两边除以 2 求出 $I$。\n\n' +
          '**经典应用**：$\\displaystyle\\int_0^{\\pi/2}\\frac{\\sin x}{\\sin x+\\cos x}\\,dx$，换元后得 $\\int_0^{\\pi/2}\\frac{\\cos x}{\\cos x+\\sin x}\\,dx$，两者之和 $=\\int_0^{\\pi/2}1\\,dx=\\dfrac{\\pi}{2}$，故原积分 $=\\dfrac{\\pi}{4}$。',
        formula:
          '\\int_a^b f(x)\\,dx=\\int_a^b f(a+b-x)\\,dx',
        tip: '区间再现的信号：被积函数含 $x$ 且难直接积，但换元 $t=a+b-x$ 后，$f(a+b-x)$ 与 $f(x)$ 的和或差形式更简单（常出现 $\\sin+\\cos$、$1+e^x$ 等可配对的结构）。',
      },
      // 卡 5：区间再现例题详解
      {
        title: '例题精讲：$\\int_0^{\\pi}\\frac{x\\sin x}{1+\\cos^2 x}\\,dx$',
        body:
          '**被积函数含 $x$ 因子，是区间再现的强信号。**\n\n' +
          '设 $I=\\displaystyle\\int_0^{\\pi}\\frac{x\\sin x}{1+\\cos^2 x}\\,dx$，令 $t=\\pi-x$：\n\n' +
          '$I=\\displaystyle\\int_0^{\\pi}\\frac{(\\pi-t)\\sin t}{1+\\cos^2 t}\\,dt=\\pi\\int_0^{\\pi}\\frac{\\sin t}{1+\\cos^2 t}\\,dt-I$。\n\n' +
          '因此 $2I=\\pi\\displaystyle\\int_0^{\\pi}\\frac{\\sin x}{1+\\cos^2 x}\\,dx$，令 $u=\\cos x$，$du=-\\sin x\\,dx$：\n\n' +
          '$\\displaystyle\\int_0^{\\pi}\\frac{\\sin x}{1+\\cos^2 x}\\,dx=-\\int_1^{-1}\\frac{du}{1+u^2}=\\int_{-1}^{1}\\frac{du}{1+u^2}=\\arctan u\\Big|_{-1}^{1}=\\frac{\\pi}{4}-(-\\frac{\\pi}{4})=\\frac{\\pi}{2}$。\n\n' +
          '故 $I=\\dfrac{\\pi}{2}\\cdot\\dfrac{\\pi}{2}\\cdot\\dfrac{1}{1}=\\dfrac{\\pi^2}{4}$（即 $2I=\\pi\\cdot\\dfrac{\\pi}{2}$，$I=\\dfrac{\\pi^2}{4}$）。',
        steps: [
          '**识别信号**：含 $x$ 因子难积，区间 $[0,\\pi]$，考虑区间再现',
          '**换元**：$t=\\pi-x$，$\\sin(\\pi-t)=\\sin t$，$\\cos(\\pi-t)=-\\cos t$，$\\cos^2$ 不变',
          '**合并**：$2I=\\pi\\int_0^\\pi\\dfrac{\\sin x}{1+\\cos^2 x}dx$，消去难算的 $x$ 因子',
          '**剩余积分**：换元 $u=\\cos x$ 化为 $\\arctan$',
        ],
      },
      // 卡 6：易错点与综合辨析
      {
        title: '易错点与辨析',
        body:
          '**易错点①：华里士公式偶奇混淆。** 偶数时**要乘** $\\pi/2$，奇数时**不乘** $\\pi/2$。口诀：偶数用到 $I_0=\\pi/2$，奇数到 $I_1=1$。\n\n' +
          '**易错点②：混合型 $\\int_0^{\\pi/2}\\sin^m x\\cos^n x\\,dx$ 有专门公式。** 对于一般的 $m,n$，需先用降次化单项，再用递推——不要直接套只有一个变量的华里士公式。\n\n' +
          '**易错点③：区间再现换元后忘记把 $x$ 换回。** 令 $t=a+b-x$ 后，结果是关于 $t$ 的积分，但由于上下限不变，变量名可以换回 $x$，这一步合理性来自定积分与变量名无关。\n\n' +
          '**易错点④：奇偶性判断时漏看整体。** $\\dfrac{x^4}{1+e^x}$ 看起来分子偶分母非对称，不是简单的奇或偶，但可用 $f(x)+f(-x)=x^4$ 的技巧（类型详见§13.4.1例18）。',
        reveal: {
          q: '计算 $\\displaystyle\\int_{-2}^{2}(x^3\\cos x+5x^2)\\,dx$ 时，哪部分利用奇函数为零，哪部分利用偶函数翻倍？',
          a: '$x^3\\cos x$ 是奇函数（奇×偶=奇），积分为 $0$；$5x^2$ 是偶函数，$\\int_{-2}^{2}5x^2\\,dx=2\\int_0^25x^2\\,dx=2\\cdot[5x^3/3]_0^2=2\\cdot40/3=80/3$。',
        },
      },
      // 卡 7：技巧选择的判断流程
      {
        title: '定积分技巧：选哪个？判断流程',
        body:
          '**步骤一：看区间是否对称** $[-a,a]$ → 检验奇偶性，奇函数直接得 0，偶函数翻倍到 $[0,a]$。\n\n' +
          '**步骤二：被积函数含 $\\sin^n$ 或 $\\cos^n$，积分区间是 $[0,\\pi/2]$** → 华里士公式（或递推）。\n\n' +
          '**步骤三：积分含 $xf(\\sin x)$，区间 $[0,\\pi]$** → 区间再现 $t=\\pi-x$，可把 $x$ 换成 $\\pi-t$，消去线性因子。\n\n' +
          '**步骤四：$[0,\\pi/2]$ 上 $\\sin/\\cos$ 互换型**（如 $\\dfrac{\\sin x}{\\sin x+\\cos x}$）→ 换元 $t=\\pi/2-x$ 得互补形式，两式之和 $=1$，各为 $\\pi/4$。\n\n' +
          '以上都不适用则回到常规换元或分部积分。',
        formula:
          '\\int_0^\\pi x\\cdot f(\\sin x)\\,dx = \\frac{\\pi}{2}\\int_0^\\pi f(\\sin x)\\,dx',
      },
    ],
    questions: [
      // 题 1：judge — 奇函数积分
      {
        id: 'c16-c-q1',
        type: 'judge',
        prompt:
          '$\\displaystyle\\int_{-3}^{3}x^3\\sin^2 x\\,dx=0$，因为 $x^3\\sin^2 x$ 是奇函数。',
        answer: true,
        explain:
          '$f(x)=x^3\\sin^2 x$，$f(-x)=(-x)^3\\sin^2(-x)=-x^3\\sin^2 x=-f(x)$，确实是奇函数。在对称区间 $[-3,3]$ 上积分为 $0$。',
      },
      // 题 2：input — 华里士公式（偶数）
      {
        id: 'c16-c-q2',
        type: 'input',
        prompt:
          '利用华里士公式计算 $\\displaystyle\\int_0^{\\pi/2}\\sin^4 x\\,dx$，结果是 $\\dfrac{k\\pi}{16}$，$k$ 等于多少？',
        accept: ['3'],
        placeholder: '输入整数',
        explain:
          '$n=4$（偶数），$I_4=\\dfrac{3!!}{4!!}\\cdot\\dfrac{\\pi}{2}=\\dfrac{3\\cdot1}{4\\cdot2}\\cdot\\dfrac{\\pi}{2}=\\dfrac{3}{8}\\cdot\\dfrac{\\pi}{2}=\\dfrac{3\\pi}{16}$，故 $k=3$。',
      },
      // 题 3：choice — 华里士公式（奇数）
      {
        id: 'c16-c-q3',
        type: 'choice',
        prompt:
          '$\\displaystyle\\int_0^{\\pi/2}\\cos^5 x\\,dx$ 等于？',
        options: [
          '$\\dfrac{8}{15}$',
          '$\\dfrac{8\\pi}{15}$',
          '$\\dfrac{4}{15}$',
          '$\\dfrac{15}{8}$',
        ],
        answer: 0,
        explain:
          '$n=5$（奇数），$I_5=\\dfrac{4!!}{5!!}=\\dfrac{4\\cdot2}{5\\cdot3\\cdot1}=\\dfrac{8}{15}$（奇数情形**不乘** $\\pi/2$）。$\\cos^n$ 和 $\\sin^n$ 在 $[0,\\pi/2]$ 的积分相等。',
      },
      // 题 4：match — 奇偶性与积分结果
      {
        id: 'c16-c-q4',
        type: 'match',
        prompt: '将被积函数的奇偶性与 $[-a,a]$ 上积分的结论配对。',
        left: [
          '$f(x)$ 是奇函数',
          '$f(x)$ 是偶函数',
          '$f(x)$ 既不是奇函数也不是偶函数',
        ],
        right: [
          '$\\displaystyle\\int_{-a}^{a}f(x)\\,dx=0$',
          '$\\displaystyle\\int_{-a}^{a}f(x)\\,dx=2\\displaystyle\\int_{0}^{a}f(x)\\,dx$',
          '需正常计算，不能直接化简',
        ],
      },
      // 题 5：judge — 区间再现公式方向
      {
        id: 'c16-c-q5',
        type: 'judge',
        prompt:
          '区间再现公式 $\\displaystyle\\int_a^b f(x)\\,dx=\\int_a^b f(a+b-x)\\,dx$ 要求被积函数必须是对称函数（$f(a+b-x)=f(x)$）才成立。',
        answer: false,
        explain:
          '区间再现公式对**任意**连续函数 $f$ 成立，无需 $f(a+b-x)=f(x)$。公式是换元 $t=a+b-x$ 的结果，与被积函数的奇偶性无关。其妙用在于 $f(x)$ 与 $f(a+b-x)$ 之**和**能化简，不要求两者相等。',
      },
      // 题 6：choice — 区间再现计算
      {
        id: 'c16-c-q6',
        type: 'choice',
        prompt:
          '$\\displaystyle\\int_0^{\\pi/2}\\frac{\\sin x}{\\sin x+\\cos x}\\,dx$ 等于？',
        options: [
          '$\\dfrac{\\pi}{2}$',
          '$\\dfrac{\\pi}{4}$',
          '$1$',
          '$\\dfrac{\\pi}{8}$',
        ],
        answer: 1,
        explain:
          '设 $I=\\int_0^{\\pi/2}\\dfrac{\\sin x}{\\sin x+\\cos x}\\,dx$，令 $t=\\pi/2-x$，得 $I=\\int_0^{\\pi/2}\\dfrac{\\cos t}{\\cos t+\\sin t}\\,dt$。两式相加：$2I=\\int_0^{\\pi/2}1\\,dx=\\dfrac{\\pi}{2}$，故 $I=\\dfrac{\\pi}{4}$。',
      },
      // 题 7：input — 华里士公式综合
      {
        id: 'c16-c-q7',
        type: 'input',
        prompt:
          '利用华里士公式计算 $\\displaystyle\\int_0^{\\pi/2}\\sin^5 x\\,dx$（化为最简分数，如 $8/15$）。',
        accept: ['8/15'],
        placeholder: '输入分数',
        explain:
          '$n=5$（奇数），$I_5=\\dfrac{4!!}{5!!}=\\dfrac{4\\cdot2}{5\\cdot3\\cdot1}=\\dfrac{8}{15}$。注意：奇数情形不乘 $\\pi/2$，这是最易错的地方。',
      },
      // 题 8：choice — 综合选择技巧
      {
        id: 'c16-c-q8',
        type: 'choice',
        prompt:
          '计算 $\\displaystyle\\int_0^{\\pi}\\frac{x\\sin x}{1+\\cos^2 x}\\,dx$ 时，最适合的技巧是？',
        options: [
          '利用奇函数积分为零',
          '华里士公式',
          '区间再现（令 $t=\\pi-x$）',
          '分部积分（令 $u=x$）',
        ],
        answer: 2,
        explain:
          '被积函数含线性因子 $x$，区间 $[0,\\pi]$，这是区间再现的典型信号。令 $t=\\pi-x$，$\\sin(\\pi-t)=\\sin t$，把 $x$ 换成 $\\pi-t$，原积分 $I$ 满足 $2I=\\pi\\int_0^\\pi\\dfrac{\\sin x}{1+\\cos^2 x}\\,dx$，再换元 $u=\\cos x$ 得 $\\pi^2/4$，故 $I=\\pi^2/4$。',
      },
    ],
  },
]
