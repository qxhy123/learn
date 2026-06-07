export const LESSONS = [
  {
    id: 'u2c4',
    title: '基本不等式',
    subtitle: '算术-几何平均 · 一正二定三相等 · 三大配凑技巧',
    intro: [
      {
        title: '从一道最值题出发',
        body: '已知 $x > 0$，求 $f(x) = x + \\dfrac{1}{x}$ 的最小值。用导数能解，但有一种更快更漂亮的方法——只需一眼看出"两项之积是常数"，然后一步写出答案。这就是基本不等式（AM-GM 不等式）的威力。',
      },
      {
        title: '基本不等式及其证明',
        body: '对任意正实数 $a, b > 0$，算术平均数 $\\geq$ 几何平均数：\n\n$\\dfrac{a+b}{2} \\geq \\sqrt{ab}$，即 $a + b \\geq 2\\sqrt{ab}$，**等号当且仅当 $a = b$ 时成立**。\n\n证明：$a + b - 2\\sqrt{ab} = (\\sqrt{a})^2 - 2\\sqrt{a}\\sqrt{b} + (\\sqrt{b})^2 = (\\sqrt{a} - \\sqrt{b})^2 \\geq 0$，等号当 $\\sqrt{a} = \\sqrt{b}$，即 $a = b$。',
        formula: 'a + b \\geq 2\\sqrt{ab} \\quad (a > 0,\\ b > 0)',
        tip: '无需正数约束的推论：$a^2 + b^2 \\geq 2ab$（等价于 $(a-b)^2 \\geq 0$），对**任意实数** $a, b$ 成立。',
      },
      {
        title: '三原则——用基本不等式前必须自查',
        body: '每次使用基本不等式前，心里默念三条：\n\n**一正**：两项都必须大于零（$a > 0, b > 0$）。若有负项，$\\sqrt{ab}$ 无意义，不等式不成立。\n\n**二定**：使用不等式是为了求最值，所以"积 $ab$"（或"和 $a+b$"）必须是常数——否则下界随变量变化，无法确定最小值。\n\n**三相等**：取等条件 $a = b$ 必须在约束范围内**可以达到**——若取等点不满足约束（如 $x > 0$ 而取等需 $x = -1$），下界取不到，结论错误。',
        steps: [
          '① 一正：$a > 0$ 且 $b > 0$ ✓',
          '② 二定：$ab = $ 常数（或 $a + b = $ 常数）✓',
          '③ 三相等：解方程 $a = b$，验证解满足所有约束条件 ✓',
        ],
        tip: '三个条件缺一不可。最常犯的错误是忘记验证取等条件是否可达。',
      },
      {
        title: '直接型：积为常数求和的最值',
        body: '看到 $x + \\dfrac{c}{x}$（$x > 0$，$c > 0$），两项之积 $= x \\cdot \\dfrac{c}{x} = c$（常数），自查三原则后直接用基本不等式：\n\n$x + \\dfrac{c}{x} \\geq 2\\sqrt{x \\cdot \\dfrac{c}{x}} = 2\\sqrt{c}$\n\n取等条件：$x = \\dfrac{c}{x}$，即 $x = \\sqrt{c}$（$x > 0$ 时可达）。',
        formula: 'x + \\frac{c}{x} \\geq 2\\sqrt{c} \\quad (x > 0,\\ c > 0)',
      },
      {
        title: '配凑型：平移后积为常数',
        body: '看到 $x + \\dfrac{c}{x - k}$（$x > k$，$c > 0$），分母带平移 $x - k$，两项之积不是常数，需要配凑。\n\n将 $x$ 拆分为 $(x - k) + k$：$x + \\dfrac{c}{x-k} = \\underbrace{(x-k)}_{t} + \\dfrac{c}{x-k} + k$。\n\n令 $t = x - k > 0$，转化为 $t + \\dfrac{c}{t} + k \\geq 2\\sqrt{c} + k$，取等条件 $t = \\sqrt{c}$，即 $x = k + \\sqrt{c}$。',
        formula: 'x + \\frac{c}{x-k} \\geq 2\\sqrt{c} + k \\quad (x > k,\\ c > 0)',
      },
      {
        title: '"1" 代换型：条件约束求最值',
        body: '已知 $\\dfrac{p}{x} + \\dfrac{q}{y} = 1$（$x, y > 0$），求含 $x, y$ 的线性式的最值，将条件中的 $1$ 代入：\n\n$(x + y) \\cdot 1 = (x + y)\\left(\\dfrac{p}{x} + \\dfrac{q}{y}\\right)$\n\n展开后对出现的 $\\dfrac{x}{y} + \\dfrac{qy}{x}$（或类似项）再用基本不等式，得到下界。',
        steps: [
          '将 $1$ 替换为已知的条件式 $\\dfrac{p}{x} + \\dfrac{q}{y}$',
          '展开乘积，提取 $\\dfrac{x}{y}$ 和 $\\dfrac{y}{x}$ 形式的项',
          '对这两项用基本不等式，注意两项之积是常数',
          '解取等方程，验证满足原始约束',
        ],
      },
      {
        title: '三大技巧综合例题',
        body: '例一（直接型）：$x > 0$，$f(x) = x + \\dfrac{4}{x}$，积 $= 4$，最小值 $= 2\\sqrt{4} = 4$，等号 $x = 2$。\n\n例二（配凑型）：$x > 1$，$f(x) = x + \\dfrac{4}{x-1} = (x-1) + \\dfrac{4}{x-1} + 1 \\geq 2\\sqrt{4} + 1 = 5$，等号 $x - 1 = 2$，$x = 3$。\n\n例三（$1$ 代换）：$x, y > 0$，$\\dfrac{2}{x} + \\dfrac{1}{y} = 1$，求 $x + 2y$ 最小值：$(x+2y) \\cdot 1$ 展开后 $\\geq 4 + 2\\sqrt{4} = 8$，等号 $x = 4, y = 2$。',
        reveal: {
          q: '$x > 3$ 时，$x + \\dfrac{9}{x-3}$ 的最小值是多少？',
          a: '配凑：$x + \\dfrac{9}{x-3} = (x-3) + \\dfrac{9}{x-3} + 3 \\geq 2\\sqrt{9} + 3 = 6 + 3 = 9$，等号 $x - 3 = 3$，$x = 6$，最小值为 $9$。',
        },
      },
      {
        title: '反向应用：和为常数求积的最大值',
        body: '已知 $a + b = S$（$a, b > 0$，$S > 0$ 为常数），求 $ab$ 的最大值：\n\n由基本不等式 $ab \\leq \\left(\\dfrac{a+b}{2}\\right)^2 = \\dfrac{S^2}{4}$，等号当 $a = b = \\dfrac{S}{2}$ 时成立。\n\n**口诀**：和一定，积最大时两数相等；积一定，和最小时两数相等。',
        formula: 'ab \\leq \\left(\\frac{a+b}{2}\\right)^2',
        tip: '已知 $2x + 8y = 1$（$x, y > 0$），求 $xy$ 最大值：$1 = 2x + 8y \\geq 2\\sqrt{16xy} = 8\\sqrt{xy}$，故 $\\sqrt{xy} \\leq \\dfrac{1}{8}$，$xy \\leq \\dfrac{1}{64}$，等号 $2x = 8y$，$x = 4y$，$y = \\dfrac{1}{16}$，$x = \\dfrac{1}{4}$。',
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '$x > 0$ 时，$f(x) = x + \\dfrac{4}{x}$ 的最小值是？',
        options: ['$2$', '$4$', '$3$', '$2\\sqrt{2}$'],
        answer: 1,
        explain: '两项之积 $= x \\cdot \\dfrac{4}{x} = 4$（常数），$x, \\dfrac{4}{x} > 0$，三原则满足。$x + \\dfrac{4}{x} \\geq 2\\sqrt{4} = 4$，取等 $x = 2$（$x > 0$ 可达）。最小值为 $4$，选B。',
      },
      {
        type: 'judge',
        prompt: '使用基本不等式 $a + b \\geq 2\\sqrt{ab}$ 时，只需要保证两项之积是常数，不需要验证取等条件是否可达。',
        answer: false,
        explain: '三原则中"三相等"要求取等条件在约束范围内可达。若取等点不满足约束，不等式只给出一个取不到的下界，不是最小值。例如：在 $x > 2$ 的约束下，若取等需 $x = 1$，则最小值并非该下界。',
      },
      {
        type: 'input',
        prompt: '$x > 3$ 时，$h(x) = x + \\dfrac{9}{x-3}$ 的最小值是多少（填数字）？',
        accept: ['9'],
        explain: '配凑：$h = (x-3) + \\dfrac{9}{x-3} + 3$。令 $t = x-3 > 0$，则 $t + \\dfrac{9}{t} \\geq 2\\sqrt{9} = 6$，故 $h \\geq 6 + 3 = 9$。取等 $t = 3$，$x = 6 > 3$，可达。最小值为 $9$。',
      },
      {
        type: 'choice',
        prompt: '已知 $a, b > 0$，$a + b = 4$，则 $ab$ 的最大值是？',
        options: ['$2$', '$4$', '$8$', '$16$'],
        answer: 1,
        explain: '$ab \\leq \\left(\\dfrac{a+b}{2}\\right)^2 = \\left(\\dfrac{4}{2}\\right)^2 = 4$，等号当 $a = b = 2$ 时（满足 $a, b > 0$ 和 $a + b = 4$）。最大值为 $4$，选B。',
      },
      {
        type: 'judge',
        prompt: '$a^2 + b^2 \\geq 2ab$ 对任意实数 $a, b$ 都成立，等号当 $a = b$ 时取得。',
        answer: true,
        explain: '$a^2 + b^2 - 2ab = (a-b)^2 \\geq 0$ 对任意实数成立，等号当 $a = b$。与基本不等式 $a + b \\geq 2\\sqrt{ab}$ 不同，此式无需 $a, b > 0$ 的限制。',
      },
      {
        type: 'match',
        prompt: '将每种题型与对应的处理技巧配对。',
        left: [
          '$x + \\dfrac{c}{x}$（$x > 0$）',
          '$x + \\dfrac{c}{x-k}$（$x > k$）',
          '$\\dfrac{p}{x} + \\dfrac{q}{y} = 1$，求 $x+y$ 最值',
          '$a + b = S$，求 $ab$ 最大值',
        ],
        right: [
          '直接用基本不等式，最小值 $2\\sqrt{c}$',
          '配凑分离 $(x-k)$，令 $t = x-k$',
          '"$1$" 代换，将条件式乘入目标式',
          '反向应用 $ab \\leq (S/2)^2$',
        ],
        explain: '四种技巧覆盖了基本不等式应用的所有主要题型：直接型、配凑型、"1"代换型、反向应用型。',
      },
      {
        type: 'choice',
        prompt: '已知 $x, y > 0$，$\\dfrac{2}{x} + \\dfrac{1}{y} = 1$，则 $x + 2y$ 的最小值是？',
        options: ['$6$', '$8$', '$9$', '$10$'],
        answer: 1,
        explain: '$(x+2y) \\cdot 1 = (x+2y)(\\dfrac{2}{x}+\\dfrac{1}{y}) = 2 + \\dfrac{x}{y} + \\dfrac{4y}{x} + 2 = 4 + \\dfrac{x}{y} + \\dfrac{4y}{x} \\geq 4 + 2\\sqrt{4} = 8$。取等：$\\dfrac{x}{y} = \\dfrac{4y}{x}$，$x = 2y$，代入得 $y=2, x=4$，满足 $x,y>0$。最小值为 $8$，选B。',
      },
      {
        type: 'input',
        prompt: '$x > 0$ 时，$g(x) = 3x + \\dfrac{12}{x}$ 的最小值是多少（填数字）？',
        accept: ['12'],
        explain: '两项之积 $= 3x \\cdot \\dfrac{12}{x} = 36$，均为正数（$x > 0$）。$3x + \\dfrac{12}{x} \\geq 2\\sqrt{36} = 12$，取等 $3x = \\dfrac{12}{x}$，$x^2 = 4$，$x = 2$（$x > 0$ 可达）。最小值为 $12$。',
      },
    ],
  },
]
