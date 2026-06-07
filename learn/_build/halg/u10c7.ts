export const LESSONS = [
  {
    id: 'u10c7',
    title: '导数综合应用',
    subtitle: '不等式证明 + 恒成立 vs 存在性 + 三类综合',
    intro: [
      {
        title: '综合题的三大类型',
        body: '导数综合大题是高考压轴题的核心。三大类型：①用导数证明不等式（构造辅助函数法）；②恒成立与存在性问题（参数分离 + 最值对应）；③函数、不等式、数列三者综合（放缩 + 裂项 + 单调性）。题面可能看起来不像导数题，但解法骨架几乎全靠导数工具。',
      },
      {
        title: '不等式证明：构造辅助函数法',
        body: '要证 f(x) > g(x)，构造 h(x) = f(x) − g(x)，目标变成证 h(x) > 0。两条路：①最值法：求 h 的极小值，证极小值 > 0；②单调 + 端点法：若 h(x₀) = 0（端点），且 h 在 x > x₀ 上严格递增，则 h(x) > 0。一阶导看不出符号时，再求一次导（"二阶导突破"）分析 h′ 的单调性。',
        formula: 'h(x) = f(x) - g(x),\\quad h_{\\min} > 0 \\Rightarrow f(x) > g(x)',
        steps: [
          '构造 h = f − g',
          '求 h(0)（或边界值）',
          '求 h′(x)，若符号不定再求 h″',
          '找 h′ 的极小值点，验证 h′ > 0（则 h 单调增）或 h 极小值 > 0',
          '由端点 h(x₀)=0 + 单调递增 → h(x) > 0',
        ],
      },
      {
        title: '四大"砖头"不等式（必会导数证法）',
        body: '① ln(1+x) < x（x > 0）：h = x − ln(1+x)，h(0)=0，h′=x/(1+x)>0，h 递增，h(x)>0。\n② eˣ ≥ 1+x（x∈ℝ）：h = eˣ−1−x，h′=eˣ−1，x=0 极小值 h(0)=0，h≥0。\n③ sin x < x（x>0）：h = x−sin x，h′=1−cos x≥0，h 单调增，h(0)=0，h(x)>0。\n④ ln x ≤ x−1（x>0）：h=x−1−ln x，极小值 h(1)=0，h≥0。\n这四个是构造不等式的"母本"，考试中常出现其代换变形。',
        tip: '遇到 eˣ > x²+1（x>0）一类，h 的一阶导符号不定，要再求 h″ 找 h′ 的极小值（二阶导突破）。',
      },
      {
        title: '恒成立与存在性：最值的两个方向',
        body: '参数分离后得 a ≥ f(x) 或 a ≤ f(x)。\n恒成立（对所有 x 成立）：a ≥ f(x) 恒成立 ⟺ a ≥ f_max；a ≤ f(x) 恒成立 ⟺ a ≤ f_min。口诀：恒成立"卡极端"。\n存在性（存在某 x 成立）：a ≥ f(x) 存在解 ⟺ a ≥ f_min；a ≤ f(x) 存在解 ⟺ a ≤ f_max。口诀：存在性"反向卡极端"。',
        formula: 'a \\geq f(x)\\text{ 恒成立} \\Leftrightarrow a \\geq f_{\\max};\\quad a \\geq f(x)\\text{ 存在解} \\Leftrightarrow a \\geq f_{\\min}',
        tip: '几何直觉：画出 y=f(x) 的图象，水平线 y=a。"恒成立"要水平线在所有图象之上（达到最高点）；"存在解"只要碰到图象一次（达到最低点）。',
      },
      {
        title: '例题：恒成立求参数范围',
        body: '若 x − ln x ≥ a 对所有 x > 0 恒成立，求 a 的范围。\n分析：已是 a ≤ f(x) 形式（f=x−ln x），恒成立 ⟺ a ≤ f_min。\nf′(x) = 1−1/x = (x−1)/x，f′=0 得 x=1，x=1 是极小值点，f(1)=1。\n故 a ≤ 1，即 a ∈ (−∞, 1]。',
        formula: 'f(x) = x - \\ln x,\\quad f_{\\min} = f(1) = 1 \\Rightarrow a \\leq 1',
      },
      {
        title: '综合：导数 + 数列 + 放缩',
        body: '证明 1 + 1/4 + 1/9 + … + 1/n² < 2。关键放缩：对 k ≥ 2，1/k² < 1/(k(k−1)) = 1/(k−1) − 1/k（裂项）。\n级数 = 1 + ∑(k=2 到 n) 1/k² < 1 + ∑(k=2 到 n) [1/(k−1)−1/k] = 1 + (1−1/n) = 2−1/n < 2。\n放缩来自 f(x) = 1/x² 的单调性（x 增大 f 减小），是导数思维渗入数列的体现。',
        formula: '\\frac{1}{k^2} < \\frac{1}{k(k-1)} = \\frac{1}{k-1} - \\frac{1}{k}\\quad (k \\geq 2)',
        reveal: {
          q: '证明：x > 0 时，e^x > x²+1。',
          a: '构造 h(x)=eˣ−x²−1，h(0)=0。h′=eˣ−2x，h′(0)=1>0。h″=eˣ−2，h″=0 得 x=ln2。h′ 在 (0,ln2) 递减后递增，极小值 h′(ln2)=2−2ln2>0（因 ln2<1）。故 h′(x)>0 对所有 x>0，h 递增，h(x)>h(0)=0，得证。',
        },
      },
    ],
    questions: [
      {
        type: 'choice' as const,
        stem: '要证 f(x) > g(x)（x∈D），最常用的方法是',
        options: [
          'A. 直接比较 f 和 g 各点的大小',
          'B. 构造 h(x) = f(x) − g(x)，证 h(x) > 0',
          'C. 求 f 和 g 的交点',
          'D. 用反证法假设 f(x) ≤ g(x)',
        ],
        answer: 'B',
        explanation: '构造辅助函数法是导数证不等式的标准套路：令 h=f−g，转化为证 h>0，再用单调性和极值工具分析。选 B。',
      },
      {
        type: 'judge' as const,
        stem: '若"a ≥ f(x) 对所有 x∈D 恒成立"，则 a ≥ f_min（f 的最小值）。',
        answer: '错误',
        explanation: '恒成立条件是 a ≥ f_max（最大值），不是 f_min。"a ≥ f(x) 恒成立"要求 a 超过所有 f 值中最大的那个。只需 a ≥ f_min 的是"存在性"条件。',
      },
      {
        type: 'input' as const,
        stem: '若 x − ln x ≥ a 对所有 x > 0 恒成立，则 a 的最大值是多少？',
        answer: '1',
        explanation: 'f(x)=x−ln x，f′=(x−1)/x，极小值 f(1)=1。a ≤ f_min=1，最大值 a=1。',
      },
      {
        type: 'choice' as const,
        stem: '"存在 x ∈ [1, e] 使 ax ≥ ln x 成立"，a 的取值范围是',
        options: [
          'A. a ≥ 1/e',
          'B. a ≥ 0',
          'C. a ≥ 1',
          'D. a > 0',
        ],
        answer: 'B',
        explanation: '分离 a ≥ (ln x)/x，存在解条件 a ≥ min[(ln x)/x，x∈[1,e]]。h(x)=(ln x)/x 在 [1,e] 递增（h′=(1−ln x)/x²≥0 在 x≤e），最小值 h(1)=0。故 a≥0，选 B。',
      },
      {
        type: 'input' as const,
        stem: '已知 f(x) = x³ − 3x，若 f(x) ≥ m 对所有 x ∈ [−2, 1] 恒成立，则 m 的最大值是多少？',
        answer: '−2',
        explanation: 'm ≤ f_min。f′=3x²−3=3(x−1)(x+1)，临界点 x=±1。f(−2)=−2，f(−1)=2，f(1)=−2。[−2,1] 上最小值 −2，m_max=−2。',
      },
      {
        type: 'judge' as const,
        stem: '对 k ≥ 2，有 1/k² < 1/(k(k−1))，故 ∑_{k=1}^{n} 1/k² < 2。',
        answer: '正确',
        explanation: '1/k²<1/(k(k−1))=1/(k−1)−1/k（k≥2）。求和：1+∑(k=2到n)[1/(k−1)−1/k] = 1+(1−1/n) = 2−1/n < 2。正确。',
      },
      {
        type: 'choice' as const,
        stem: '已知砖头不等式 eˣ ≥ 1+x，将 x 换成 −t 可以得到',
        options: [
          'A. e^{−t} ≥ 1−t（对所有 t 成立）',
          'B. e^{−t} ≤ 1−t',
          'C. e^t ≥ 1+t',
          'D. t ≤ 1−e^{−t}',
        ],
        answer: 'A',
        explanation: '令 x=−t 代入 eˣ≥1+x：e^{−t}≥1+(−t)=1−t，即 e^{−t}≥1−t，选 A。',
      },
      {
        type: 'match' as const,
        stem: '将不等式类型与等价条件正确连线',
        left: ['a ≥ f(x) 对所有 x 恒成立', 'a ≤ f(x) 对所有 x 恒成立', 'a ≥ f(x) 存在 x 成立', 'a ≤ f(x) 存在 x 成立'],
        right: ['a ≤ f_max', 'a ≥ f_min', 'a ≤ f_min', 'a ≥ f_max'],
        answer: ['a ≥ f_max', 'a ≤ f_min', 'a ≥ f_min', 'a ≤ f_max'],
      },
    ],
  },
]
