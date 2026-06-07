export const LESSONS = [
  {
    id: 'u1c2',
    title: '向量的加法与减法',
    subtitle: '首尾相接与平行四边形——向量运算的几何语言',
    intro: [
      {
        title: '引入：位移的合成',
        body: '一个人先向东走 $3\\,\\text{m}$（位移向量 $\\vec{a}$），再向北走 $4\\,\\text{m}$（位移向量 $\\vec{b}$），最终的总位移是什么？\n\n总位移就是 $\\vec{a}$ 与 $\\vec{b}$ 的**和向量** $\\vec{a} + \\vec{b}$。这给了我们加法的物理动机：向量加法描述「依次执行两段位移后的合效果」。',
        tip: '加法的核心问题：两段有方向的量叠加，合效果是什么？',
      },
      {
        title: '三角形法则（首尾相接法）',
        body: '将 $\\vec{b}$ 的起点移到 $\\vec{a}$ 的终点（保持大小方向不变），连接 $\\vec{a}$ 的起点到 $\\vec{b}$ 的终点，得到的向量即为 $\\vec{a} + \\vec{b}$。\n\n用有向线段表述：$\\overrightarrow{AB} + \\overrightarrow{BC} = \\overrightarrow{AC}$。$A$ 是起点，$B$ 是中间过渡点，$C$ 是终点，和向量「跳过」中间点直达终点。',
        formula: '\\overrightarrow{AB} + \\overrightarrow{BC} = \\overrightarrow{AC}',
        image: 'geo-p1-02-1.svg',
        imageCaption: '三角形法则：$\\overrightarrow{AB}$ 与 $\\overrightarrow{BC}$ 首尾相接，和为 $\\overrightarrow{AC}$',
        steps: [
          '第一步：将第二个向量的起点移到第一个向量的终点',
          '第二步：从第一个向量的起点画箭头指向第二个向量的终点',
          '第三步：这条新有向线段就是和向量',
        ],
      },
      {
        title: '三角形法则的推广与封闭链',
        body: '多个向量首尾依次相接，和向量永远是「第一个起点到最后一个终点」：\n\n$$\\overrightarrow{AB} + \\overrightarrow{BC} + \\overrightarrow{CD} + \\overrightarrow{DE} = \\overrightarrow{AE}$$\n\n**特别情况**：若最后的终点与第一个起点重合，构成封闭折线，则和为零向量：\n\n$$\\overrightarrow{AB} + \\overrightarrow{BC} + \\overrightarrow{CA} = \\vec{0}$$',
        tip: '封闭折线的向量和为零向量，因为「绕了一圈」回到原点，净位移为零。',
      },
      {
        title: '平行四边形法则（共起点加法）',
        body: '当两向量 $\\vec{a} = \\overrightarrow{AB}$，$\\vec{b} = \\overrightarrow{AD}$ 从**同一起点** $A$ 出发时，以它们为邻边构造平行四边形 $ABCD$，则对角线 $\\overrightarrow{AC}$ 即为 $\\vec{a} + \\vec{b}$：\n\n$$\\overrightarrow{AB} + \\overrightarrow{AD} = \\overrightarrow{AC}$$\n\n**与三角形法则的关系**：$\\overrightarrow{AD} = \\overrightarrow{BC}$（对边相等），故 $\\overrightarrow{AB} + \\overrightarrow{AD} = \\overrightarrow{AB} + \\overrightarrow{BC} = \\overrightarrow{AC}$，两法等价。',
        formula: '\\overrightarrow{AB} + \\overrightarrow{AD} = \\overrightarrow{AC}',
        image: 'geo-p1-02-2.svg',
        imageCaption: '平行四边形法则：以 $\\overrightarrow{AB}$、$\\overrightarrow{AD}$ 为邻边，对角线 $\\overrightarrow{AC}$ 为和',
      },
      {
        title: '加法运算律',
        body: '向量加法满足交换律和结合律：\n\n**交换律**：$\\vec{a} + \\vec{b} = \\vec{b} + \\vec{a}$（几何上，两条对角线是同一条，所以顺序无关）\n\n**结合律**：$(\\vec{a} + \\vec{b}) + \\vec{c} = \\vec{a} + (\\vec{b} + \\vec{c})$（先加哪两个，终点不变）\n\n因此多个向量相加可去掉括号、任意顺序：$\\vec{a}_1 + \\vec{a}_2 + \\cdots + \\vec{a}_n$。',
        formula: '\\vec{a} + \\vec{b} = \\vec{b} + \\vec{a},\\quad (\\vec{a}+\\vec{b})+\\vec{c} = \\vec{a}+(\\vec{b}+\\vec{c})',
      },
      {
        title: '向量减法——加相反向量',
        body: '**定义**：$\\vec{a} - \\vec{b} = \\vec{a} + (-\\vec{b})$，即减去 $\\vec{b}$ 等同于加上 $\\vec{b}$ 的相反向量。\n\n**几何意义（共起点）**：设 $\\overrightarrow{OA} = \\vec{a}$，$\\overrightarrow{OB} = \\vec{b}$（共起点 $O$），则：\n\n$$\\vec{a} - \\vec{b} = \\overrightarrow{OA} - \\overrightarrow{OB} = \\overrightarrow{BA}$$\n\n**记忆口诀**：共起点两向量相减，结果从**减数的终点**指向**被减数的终点**。\n\n**推导**：$\\overrightarrow{OA} - \\overrightarrow{OB} = \\overrightarrow{OA} + (-\\overrightarrow{OB}) = \\overrightarrow{OA} + \\overrightarrow{BO} = \\overrightarrow{BO} + \\overrightarrow{OA} = \\overrightarrow{BA}$',
        formula: '\\overrightarrow{OA} - \\overrightarrow{OB} = \\overrightarrow{BA}',
        tip: '减法方向口诀：从减数终点（$B$）指向被减数终点（$A$）。注意：结果是 $\\overrightarrow{BA}$，不是 $\\overrightarrow{AB}$！',
      },
      {
        title: '模的三角不等式',
        body: '对任意向量 $\\vec{a}$ 与 $\\vec{b}$：\n\n$$\\bigl|\\,|\\vec{a}| - |\\vec{b}|\\,\\bigr| \\leq |\\vec{a} + \\vec{b}| \\leq |\\vec{a}| + |\\vec{b}|$$\n\n**右侧**（上界）：三角形两边之和 $\\geq$ 第三边——和向量的模不超过两模之和。取等条件：$\\vec{a}$ 与 $\\vec{b}$ **同向**（三角形退化为线段）。\n\n**左侧**（下界）：和向量的模不小于两模之差的绝对值。取等条件：$\\vec{a}$ 与 $\\vec{b}$ **反向**。',
        formula: "\\bigl||\\vec{a}| - |\\vec{b}|\\bigr| \\leq |\\vec{a}+\\vec{b}| \\leq |\\vec{a}|+|\\vec{b}|",
        tip: '同向时和向量最长（取上界），反向时和向量最短（取下界）。',
      },
      {
        title: '例题与易错点',
        body: '**例题**：已知 $|\\vec{a}| = 3$，$|\\vec{b}| = 5$，求 $|\\vec{a} + \\vec{b}|$ 的取值范围。\n\n**解**：由三角不等式，$|3-5| \\leq |\\vec{a}+\\vec{b}| \\leq 3+5$，即 $2 \\leq |\\vec{a}+\\vec{b}| \\ leq 8$。同向取 $8$，反向取 $2$。\n\n**易错1**：$\\overrightarrow{OA} - \\overrightarrow{OB} = \\overrightarrow{BA}$ 不是 $\\overrightarrow{AB}$，方向容易弄反。\n**易错2**：$\\vec{AB} + \\vec{CD}$，若 $B \\neq C$ 不能直接得 $\\vec{AD}$，需先把 $\\vec{CD}$ 平移使其起点为 $B$。\n**易错3**：$|\\vec{a}+\\vec{b}| \\neq |\\vec{a}|+|\\vec{b}|$（一般情况），只有同向时等号成立。',
        reveal: {
          q: '已知 $|\\vec{a}| = 4$，$|\\vec{b}| = 3$，化简 $\\overrightarrow{AB} + \\overrightarrow{CD} + \\overrightarrow{BC}$。',
          a: '先合并首尾相接的向量：$\\overrightarrow{BC} + \\overrightarrow{CD} = \\overrightarrow{BD}$，再 $\\overrightarrow{AB} + \\overrightarrow{BD} = \\overrightarrow{AD}$。答：$\\overrightarrow{AD}$。注意要先调整顺序使首尾相接。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '在六边形 $ABCDEF$ 中，$\\overrightarrow{AB} + \\overrightarrow{BC} + \\overrightarrow{CD} + \\overrightarrow{DE} + \\overrightarrow{EF}$ 等于（　　）',
        options: [
          '$\\overrightarrow{AF}$',
          '$\\overrightarrow{FA}$',
          '$\\overrightarrow{AE}$',
          '$\\vec{0}$',
        ],
        answer: 0,
        explain: '选 A。多个向量首尾相接，和向量等于从第一个向量的起点（$A$）到最后一个向量的终点（$F$），即 $\\overrightarrow{AF}$。',
      },
      {
        type: 'judge',
        prompt: '对任意向量 $\\vec{a}$ 与 $\\vec{b}$，都有 $|\\vec{a} + \\vec{b}| = |\\vec{a}| + |\\vec{b}|$。',
        answer: false,
        explain: '错误。等号成立的条件是 $\\vec{a}$ 与 $\\vec{b}$ 同向（或有零向量）。一般情况下 $|\\vec{a}+\\vec{b}| < |\\vec{a}|+|\\vec{b}|$，如 $\\vec{a}$ 与 $\\vec{b}$ 反向时 $|\\vec{a}+\\vec{b}| = \\bigl||\\vec{a}|-|\\vec{b}|\\bigr|$，远小于 $|\\vec{a}|+|\\vec{b}|$。',
      },
      {
        type: 'choice',
        prompt: '已知平面上三点 $O, A, B$，则 $\\overrightarrow{AB}$ 用 $\\overrightarrow{OA}$ 和 $\\overrightarrow{OB}$ 表示为（　　）',
        options: [
          '$\\overrightarrow{OA} + \\overrightarrow{OB}$',
          '$\\overrightarrow{OA} - \\overrightarrow{OB}$',
          '$\\overrightarrow{OB} - \\overrightarrow{OA}$',
          '$-(\\overrightarrow{OA} + \\overrightarrow{OB})$',
        ],
        answer: 2,
        explain: '选 C。$\\overrightarrow{AB} = \\overrightarrow{AO} + \\overrightarrow{OB} = -\\overrightarrow{OA} + \\overrightarrow{OB} = \\overrightarrow{OB} - \\overrightarrow{OA}$。也可用共起点减法：$\\overrightarrow{OB} - \\overrightarrow{OA}$，被减数终点为 $B$，减数终点为 $A$，结果从 $A$ 指向 $B$，即 $\\overrightarrow{AB}$。',
      },
      {
        type: 'input',
        prompt: '已知 $|\\vec{a}| = 3$，$|\\vec{b}| = 5$，则 $|\\vec{a} + \\vec{b}|$ 的最大值是多少？',
        accept: ['8'],
        explain: '由三角不等式 $|\\vec{a}+\\vec{b}| \\leq |\\vec{a}|+|\\vec{b}| = 3+5 = 8$。当 $\\vec{a}$ 与 $\\vec{b}$ 同向时取等，最大值为 $8$。',
      },
      {
        type: 'choice',
        prompt: '设 $\\overrightarrow{OA} = \\vec{a}$，$\\overrightarrow{OB} = \\vec{b}$，$M$ 是 $AB$ 的中点，则 $\\overrightarrow{OM}$ 等于（　　）',
        options: [
          '$\\vec{a} + \\vec{b}$',
          '$\\dfrac{1}{2}(\\vec{a} - \\vec{b})$',
          '$\\dfrac{1}{2}(\\vec{a} + \\vec{b})$',
          '$\\vec{b} - \\vec{a}$',
        ],
        answer: 2,
        explain: '选 C。$\\overrightarrow{OM} = \\overrightarrow{OA} + \\overrightarrow{AM} = \\vec{a} + \\dfrac{1}{2}\\overrightarrow{AB} = \\vec{a} + \\dfrac{1}{2}(\\vec{b}-\\vec{a}) = \\dfrac{1}{2}\\vec{a} + \\dfrac{1}{2}\\vec{b} = \\dfrac{1}{2}(\\vec{a}+\\vec{b})$。这是向量中点公式。',
      },
      {
        type: 'judge',
        prompt: '在三角形 $ABC$ 中，$G$ 是重心，则 $\\overrightarrow{GA} + \\overrightarrow{GB} + \\overrightarrow{GC} = \\vec{0}$。',
        answer: true,
        explain: '正确。设 $M$ 为 $BC$ 中点，则 $G$ 在 $AM$ 上且 $AG = 2GM$，故 $\\overrightarrow{GA} = -2\\overrightarrow{GM}$。又 $\\overrightarrow{GB} + \\overrightarrow{GC} = (\\overrightarrow{GM}+\\overrightarrow{MB})+(\\overrightarrow{GM}+\\overrightarrow{MC}) = 2\\overrightarrow{GM}+\\vec{0} = 2\\overrightarrow{GM}$（$M$ 是 $BC$ 中点）。故三者之和为 $-2\\overrightarrow{GM}+2\\overrightarrow{GM} = \\vec{0}$。',
      },
      {
        type: 'match',
        prompt: '将左侧条件与右侧对应的取等情形配对',
        left: [
          '$|\\vec{a}+\\vec{b}| = |\\vec{a}|+|\\vec{b}|$',
          '$|\\vec{a}+\\vec{b}| = \\bigl||\\vec{a}|-|\\vec{b}|\\bigr|$',
          '$\\overrightarrow{AB}+\\overrightarrow{BC}+\\overrightarrow{CA} = \\vec{0}$',
        ],
        right: [
          '向量首尾相接构成封闭折线',
          '$\\vec{a}$ 与 $\\vec{b}$ 同向（或有零向量）',
          '$\\vec{a}$ 与 $\\vec{b}$ 反向（或有零向量）',
        ],
        explain: '上界等号 $\\Leftrightarrow$ 同向；下界等号 $\\Leftrightarrow$ 反向；封闭折线的向量和为零向量，是三角形法则的直接推论。',
      },
      {
        type: 'input',
        prompt: '化简：$\\overrightarrow{AB} + \\overrightarrow{CD} + \\overrightarrow{BC}$，结果写成 $\\overrightarrow{XY}$ 形式。',
        accept: ['\\overrightarrow{AD}', 'AD', '向量AD'],
        explain: '调整顺序利用首尾相接：$\\overrightarrow{AB} + \\overrightarrow{BC} + \\overrightarrow{CD}$（交换律），$\\overrightarrow{AB}+\\overrightarrow{BC} = \\overrightarrow{AC}$，再 $\\overrightarrow{AC}+\\overrightarrow{CD} = \\overrightarrow{AD}$。',
      },
    ],
  },
]
