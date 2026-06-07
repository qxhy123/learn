export const LESSONS = [
  {
    id: 'u1c4',
    title: '余角与补角',
    subtitle: '两角的"和为 90° 或 180°"关系',
    intro: [
      {
        title: '什么是余角？什么是补角？',
        body: '**互余（余角）：** 两个角的度数之和恰好等于 $90^\\circ$，则称这两个角**互为余角**。互余的两角可以拼成一个**直角**。\n\n**互补（补角）：** 两个角的度数之和恰好等于 $180^\\circ$，则称这两个角**互为补角**。互补的两角可以拼成一个**平角**。\n\n**重要：** 互余、互补是**两个角之间的相互关系**，不能孤立地说"某角是余角"，必须说"$\\angle 1$ 是 $\\angle 2$ 的余角"或"$\\angle 1$ 与 $\\angle 2$ 互余"。',
        tip: '互余 $\\leftrightarrow$ 直角（$90^\\circ$）；互补 $\\leftrightarrow$ 平角（$180^\\circ$）。拼在一起刚好"填满"就是互余或互补。',
      },
      {
        title: '图形直觉：拼出直角和平角',
        body: '互余的两角拼在一起，恰好组成一个直角（$90^\\circ$）。互补的两角拼在一起，恰好组成一个平角（$180^\\circ$，即一条直线）。\n\n**注意：** 互余、互补只看度数之和，与两角是否相邻、是否共顶点、是否在同一图形中**无关**。课本里常画相邻的图形只是为了直观，并非定义的要求。',
        image: 'def-complementary.svg',
        imageCaption: '互余：两角拼成直角 $90^\\circ$',
      },
      {
        title: '互补的图形示意',
        body: '互补的两角拼在一起恰好成平角。平角对应一条直线，因此互补的两角"合成一条直线"是常见的几何情形（如相邻两角组成平角）。\n\n同样地，互补只要求度数之和为 $180^\\circ$，两角可以不相邻、不共顶点。',
        image: 'def-supplementary.svg',
        imageCaption: '互补：两角拼成平角 $180^\\circ$',
      },
      {
        title: '关键性质：同角的余角（补角）相等',
        body: '**性质 1（同角或等角的余角相等）：** 如果 $\\angle 1$ 和 $\\angle 2$ 都是 $\\angle 3$ 的余角（即都与 $\\angle 3$ 互余），那么 $\\angle 1 = \\angle 2$。\n\n**性质 2（同角或等角的补角相等）：** 如果 $\\angle 1$ 和 $\\angle 2$ 都是 $\\angle 3$ 的补角，那么 $\\angle 1 = \\angle 2$。\n\n**证明思路（以余角为例）：** 由 $\\angle 1 + \\angle 3 = 90^\\circ$ 和 $\\angle 2 + \\angle 3 = 90^\\circ$，各自解出 $\\angle 1 = 90^\\circ - \\angle 3$，$\\angle 2 = 90^\\circ - \\angle 3$，两式右侧相同，故 $\\angle 1 = \\angle 2$。',
        formula: '\\angle 1 = 90^\\circ - \\angle 3 = \\angle 2',
        tip: '这是"等量代换"（等式的传递性）的几何应用，也是后续证明两角相等最常用的"隐藏工具"。',
      },
      {
        title: '例题 1：基本计算',
        body: '**例 1：** 已知 $\\angle A = 35^\\circ$，求 $\\angle A$ 的余角和补角。\n\n**解：** 余角 $= 90^\\circ - 35^\\circ = 55^\\circ$；补角 $= 180^\\circ - 35^\\circ = 145^\\circ$。\n\n**例 2：** 一个角的补角比它的余角大多少？\n\n**解：** 设角为 $\\alpha$，补角 $= 180^\\circ - \\alpha$，余角 $= 90^\\circ - \\alpha$。\n\n$(180^\\circ - \\alpha) - (90^\\circ - \\alpha) = 90^\\circ$\n\n结论：**任何一个角的补角都比余角大 $90^\\circ$**，与 $\\alpha$ 的具体数值无关！',
        steps: [
          '设角为 $\\alpha$',
          '补角 $= 180^\\circ - \\alpha$，余角 $= 90^\\circ - \\alpha$',
          '差值 $= (180^\\circ - \\alpha) - (90^\\circ - \\alpha) = 90^\\circ$',
          '结论：差值恒为 $90^\\circ$，与 $\\alpha$ 无关',
        ],
      },
      {
        title: '例题 2：链式推理',
        body: '**问题：** 已知 $\\angle 1$ 与 $\\angle 2$ 互余，$\\angle 2$ 与 $\\angle 3$ 互补，$\\angle 1 = 30^\\circ$，求 $\\angle 3$。\n\n**思路：** 顺着已知条件"链式"推进——先由互余求 $\\angle 2$，再由互补求 $\\angle 3$。\n\n**解：**\n$\\angle 1 + \\angle 2 = 90^\\circ \\Rightarrow \\angle 2 = 90^\\circ - 30^\\circ = 60^\\circ$\n$\\angle 2 + \\angle 3 = 180^\\circ \\Rightarrow \\angle 3 = 180^\\circ - 60^\\circ = 120^\\circ$',
        image: 'ex-p1-04-4.svg',
        imageCaption: '链条推理：$\\angle 1+\\angle 2=90^\\circ$ 与 $\\angle 2+\\angle 3=180^\\circ$',
        formula: '\\angle 3 = 180^\\circ - \\angle 2 = 180^\\circ - (90^\\circ - \\angle 1)',
      },
      {
        title: '例题 3：在直角三角形中的应用',
        body: '**命题：** 在 $\\triangle ABC$ 中，$\\angle C = 90^\\circ$，求证 $\\angle A$ 与 $\\angle B$ 互余。\n\n**证明：** 由三角形内角和定理，\n$\\angle A + \\angle B + \\angle C = 180^\\circ$\n代入 $\\angle C = 90^\\circ$，得\n$\\angle A + \\angle B = 90^\\circ$\n故 $\\angle A$ 与 $\\angle B$ 互为余角。$\\blacksquare$\n\n这就是"直角三角形两锐角互余"，是解直角三角形的基础，后续章节会反复用到。',
        tip: '直角三角形中两锐角之和等于 $90^\\circ$，这是互余定义的典型应用场景。',
      },
      {
        title: '易错点全解',
        body: '1. **互余必为两锐角：** 若有一个角 $\\geq 90^\\circ$，两角之和必 $> 90^\\circ$，不可能互余。**钝角和直角没有余角（初中范围）。**\n2. **互补可以一锐一钝：** $30^\\circ + 150^\\circ = 180^\\circ$；也可以两直角：$90^\\circ + 90^\\circ = 180^\\circ$；但不能两锐或两钝。\n3. **只看度数，不看位置：** 互余/互补与两角是否相邻、是否共顶点无关。\n4. **互余、互补只限两个角：** 不存在"三个角互余"的说法。\n5. **"同角的余角相等"是隐藏工具：** 看到两角分别与同一角互余/互补，立即断定这两角相等，无需列方程。',
        reveal: {
          q: '已知 $\\angle A$ 与 $\\angle B$ 互补，$\\angle B$ 与 $\\angle C$ 互补，能否断定 $\\angle A = \\angle C$？',
          a: '能。$\\angle A$ 和 $\\angle C$ 都是 $\\angle B$ 的补角，由"同角的补角相等"，$\\angle A = \\angle C$。这正是性质 2 的直接应用。',
        },
      },
    ],
    questions: [
      {
        type: 'input',
        prompt: '已知 $\\angle \\alpha = 72^\\circ$，求 $\\angle \\alpha$ 的余角。',
        accept: ['18°', '18'],
        explain: '余角 $= 90^\\circ - 72^\\circ = 18^\\circ$。',
      },
      {
        type: 'choice',
        prompt: '一个角的余角是这个角的 $2$ 倍，这个角的度数是',
        options: ['$20^\\circ$', '$30^\\circ$', '$40^\\circ$', '$45^\\circ$'],
        answer: 1,
        explain: '设角为 $x$，则余角为 $90^\\circ - x$，由题意 $90^\\circ - x = 2x$，解得 $x = 30^\\circ$。',
      },
      {
        type: 'judge',
        prompt: '钝角没有余角（在初中范围内）。',
        answer: true,
        explain: '互余要求两角之和为 $90^\\circ$。若一个角已经大于 $90^\\circ$，则其"余角"将为负数，在初中范围内（只讨论正角）钝角没有余角。直角同理（$90^\\circ$ 的余角为 $0^\\circ$，一般也不视为有效角）。',
      },
      {
        type: 'input',
        prompt: '已知 $\\angle 1$ 与 $\\angle 2$ 互余，$\\angle 2$ 与 $\\angle 3$ 互补，$\\angle 1 = 30^\\circ$，求 $\\angle 3$。',
        accept: ['120°', '120'],
        explain: '$\\angle 2 = 90^\\circ - 30^\\circ = 60^\\circ$；$\\angle 3 = 180^\\circ - 60^\\circ = 120^\\circ$。',
      },
      {
        type: 'choice',
        prompt: '一个角的补角比它的余角大多少度？',
        options: ['$45^\\circ$', '$90^\\circ$', '$135^\\circ$', '与角的大小有关，不确定'],
        answer: 1,
        explain: '设角为 $\\alpha$，补角 $= 180^\\circ - \\alpha$，余角 $= 90^\\circ - \\alpha$，差 $= (180^\\circ - \\alpha) - (90^\\circ - \\alpha) = 90^\\circ$，与 $\\alpha$ 无关，恒为 $90^\\circ$。',
      },
      {
        type: 'judge',
        prompt: '互余的两角一定相邻（有公共顶点和公共边）。',
        answer: false,
        explain: '互余只要求两角度数之和为 $90^\\circ$，与它们是否相邻、是否共顶点、是否在同一图形中均无关。只看度数，不看位置。',
      },
      {
        type: 'match',
        prompt: '将左边的关系与右边的"和"正确连线',
        left: ['互余', '互补', '周角'],
        right: ['$= 360^\\circ$', '和为 $90^\\circ$', '和为 $180^\\circ$'],
        explain: '互余：两角和为 $90^\\circ$（直角）；互补：两角和为 $180^\\circ$（平角）；周角 $= 360^\\circ$（一圈）。',
      },
      {
        type: 'choice',
        prompt: '如图，$\\angle AOB = 90^\\circ$，$\\angle COD = 90^\\circ$，且射线 $OC$ 在 $\\angle AOB$ 内部。下列结论正确的是',
        options: [
          '$\\angle AOC = \\angle COD$',
          '$\\angle AOC = \\angle BOD$',
          '$\\angle AOC = \\angle AOD$',
          '$\\angle BOC = \\angle BOD$',
        ],
        answer: 1,
        explain: '$\\angle AOC$ 与 $\\angle BOC$ 互余（因为 $\\angle AOB = 90^\\circ$）；$\\angle BOD$ 与 $\\angle BOC$ 也互余（因为 $\\angle COD = 90^\\circ$）。$\\angle AOC$ 和 $\\angle BOD$ 都是 $\\angle BOC$ 的余角，由"同角的余角相等"，$\\angle AOC = \\angle BOD$。',
      },
    ],
  },
]
