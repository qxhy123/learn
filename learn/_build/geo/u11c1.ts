export const LESSONS = [
  {
    id: 'u11c1',
    title: '正弦、余弦、正切',
    subtitle: '用边的比值度量角——锐角三角函数的由来',
    intro: [
      {
        title: '直角三角形中的三条边有"身份"',
        body: '当我们盯住直角三角形中的某个锐角 α 时，三条边各有名字：斜边是直角所对的边（最长）；对边是 α 所对的边；邻边是 α 旁边的那条直角边。注意：对边、邻边的身份是相对所选锐角而言的——看不同的锐角，对边和邻边会互换。',
        image: 'thm-trig-ratio-def.svg',
        imageCaption: '直角三角形中相对锐角 α 的对边 a、邻边 b、斜边 c',
        tip: '先固定"看谁"，再分清"对、邻、斜"，这是写出正确三角函数的第一步。',
      },
      {
        title: '三种比值的定义',
        body: '设锐角 α 的对边为 a、邻边为 b、斜边为 c，三角函数就是用这三条边的比值来精确描述 α 的"形状"：',
        formula: '\\sin\\alpha = \\dfrac{a}{c},\\quad \\cos\\alpha = \\dfrac{b}{c},\\quad \\tan\\alpha = \\dfrac{a}{b}',
        tip: '口诀：正弦 = 对÷斜；余弦 = 邻÷斜；正切 = 对÷邻。英文缩写记法：S 对斜、C 邻斜、T 对邻。',
      },
      {
        title: '为什么比值只与角的大小有关？',
        body: '两个直角三角形只要有一个锐角相等（∠A = ∠A\' = α），由 AA 相似可知两三角形相似，对应边成比例，从而对应的边比（如 a/c 和 a\'/c\'）完全相等。这说明无论三角形画得多大多小，只要 α 相同，sin α、cos α、tan α 就是同一个值。',
        formula: '\\dfrac{a}{a\'} = \\dfrac{b}{b\'} = \\dfrac{c}{c\'} \\;\\Longrightarrow\\; \\dfrac{a}{c} = \\dfrac{a\'}{c\'}',
        tip: '三角函数是角的固有属性，与三角形大小无关。这是用比值度量角的合法性基础。',
      },
      {
        title: '取值范围与单调性',
        body: '对锐角 α（0° < α < 90°）：sin α 和 cos α 都在 0 到 1 之间（对边、邻边都小于斜边）；tan α 大于 0，且没有上界。单调性：把斜边长固定，让 α 慢慢增大——对边变长、邻边变短，所以 sin α 随 α 增大而增大，cos α 随 α 增大而减小，tan α（对÷邻）分子增分母减，增大得更快。',
        formula: '0 < \\sin\\alpha < 1,\\quad 0 < \\cos\\alpha < 1,\\quad \\tan\\alpha > 0',
      },
      {
        title: '三个重要恒等式',
        body: '以下三个等式随时可用，需要牢记：平方和恒等式由勾股定理直接推出；商关系把 sin 和 cos 联系到 tan；互余关系说明两个互补为 90° 的锐角，正弦和余弦互换身份。',
        steps: [
          '平方和：sin²α + cos²α = 1（由 a² + b² = c² 两边除以 c² 得到）',
          '商关系：tan α = sin α / cos α（将 a/c 除以 b/c 化简）',
          '互余：sin(90° - α) = cos α，cos(90° - α) = sin α，tan(90° - α) = 1 / tan α',
        ],
        formula: '\\sin^2\\alpha + \\cos^2\\alpha = 1,\\quad \\tan\\alpha = \\dfrac{\\sin\\alpha}{\\cos\\alpha}',
        image: 'thm-trig-complementary.svg',
        imageCaption: '互余关系：α 的对边恰好是 (90° - α) 的邻边',
      },
      {
        title: '例题：三边求三角函数',
        body: '直角三角形 ABC 中，∠C = 90°，AC = 3，BC = 4。先由勾股定理求斜边 AB = √(9 + 16) = 5。对锐角 A：对边 = BC = 4，邻边 = AC = 3，斜边 = AB = 5；对锐角 B：对边 = AC = 3，邻边 = BC = 4，斜边 = AB = 5。',
        image: 'ex-p7-01-1.svg',
        imageCaption: 'Rt△ABC：AC = 3，BC = 4，AB = 5',
        formula: '\\sin A = \\dfrac{4}{5},\\;\\cos A = \\dfrac{3}{5},\\;\\tan A = \\dfrac{4}{3};\\quad \\sin B = \\dfrac{3}{5},\\;\\cos B = \\dfrac{4}{5},\\;\\tan B = \\dfrac{3}{4}',
        tip: '观察：sin A = cos B，cos A = sin B，tan A · tan B = 1，完全符合互余关系（A + B = 90°）。',
      },
      {
        title: '例题：已知一个三角函数值，求其他两个',
        body: '锐角 α 满足 sin α = 3/5，求 cos α 与 tan α。方法一（恒等式）：cos²α = 1 - 9/25 = 16/25，因 α 为锐角故 cos α = 4/5，再由商关系 tan α = (3/5)÷(4/5) = 3/4。方法二（构造直角三角形）：设对边 = 3，斜边 = 5，勾股得邻边 = 4，直接读出 cos α = 4/5，tan α = 3/4。',
        formula: '\\cos\\alpha = \\dfrac{4}{5},\\quad \\tan\\alpha = \\dfrac{3}{4}',
        tip: '"构造一个直角三角形 + 勾股"几乎总能绕过恒等式的繁琐计算，是中考首选方法。',
      },
      {
        title: '易错点与总结',
        body: '最常见的四个错误：①对边、邻边搞反——计算前先确认看的是哪个锐角；②平方和等式中两角必须相同，sin²30° + cos²60° ≠ 1；③三角函数只在直角三角形中这样定义，非直角三角形要先作高；④tan α 没有上界，sin α 和 cos α 始终在 0 到 1 之间。',
        reveal: {
          q: '已知锐角 α 满足 tan α = 1，利用商关系说明 sin α 与 cos α 各等于多少？',
          a: 'tan α = sin α / cos α = 1 说明 sin α = cos α。再由 sin²α + cos²α = 1 得 2sin²α = 1，sin α = √2/2，cos α = √2/2。（即 α = 45°）',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '在 Rt△ABC 中，∠C = 90°，AC = 5，BC = 12，则 sin A 等于（  ）。',
        options: ['5/13', '12/13', '5/12', '12/5'],
        answer: 1,
        explain: '先求斜边：AB = √(5² + 12²) = √169 = 13。sin A = 对边/斜边 = BC/AB = 12/13。',
      },
      {
        type: 'choice',
        prompt: '锐角 α 满足 cos α = 3/5，则 tan α 等于（  ）。',
        options: ['3/4', '4/3', '4/5', '3/5'],
        answer: 1,
        explain: '由 sin²α = 1 - cos²α = 1 - 9/25 = 16/25，得 sin α = 4/5（锐角正值）。tan α = sin α / cos α = (4/5)/(3/5) = 4/3。',
      },
      {
        type: 'input',
        prompt: '在 Rt△ABC 中，∠C = 90°，AC = 5，AB = 13。请计算 cos A 的值（以分数形式作答，如 5/13）。',
        accept: ['5/13'],
        explain: '由勾股定理：BC = √(AB² - AC²) = √(169 - 25) = √144 = 12。cos A = 邻边/斜边 = AC/AB = 5/13。',
      },
      {
        type: 'judge',
        prompt: '在直角三角形中，sin²30° + cos²60° = 1。',
        answer: false,
        explain: '平方和恒等式要求两个函数的角相同：sin²α + cos²α = 1。这里 30° ≠ 60°，两项分别等于 1/4 和 1/4，相加等于 1/2，不等于 1。',
      },
      {
        type: 'judge',
        prompt: '对于锐角 α，cos α 随 α 的增大而增大。',
        answer: false,
        explain: '在 0° < α < 90° 范围内，cos α 随 α 的增大而减小。把斜边固定，α 变大则邻边变短，邻边/斜边减小。',
      },
      {
        type: 'input',
        prompt: '锐角 α 满足 sin α = √5/5，利用恒等式求 cos α 的值（以最简根式分数形式作答，如 2√5/5）。',
        accept: ['2√5/5', '2\\sqrt{5}/5'],
        explain: 'sin²α = 1/5，cos²α = 1 - 1/5 = 4/5，cos α = √(4/5) = 2/√5 = 2√5/5（锐角故取正值）。',
      },
      {
        type: 'match',
        prompt: '将下列三角函数表达式与其含义配对。',
        left: ['sin α', 'cos α', 'tan α', 'sin(90° - α)'],
        right: ['邻边除以斜边', '对边除以邻边', '对边除以斜边', '等于 cos α'],
        explain: 'sin α = 对边/斜边；cos α = 邻边/斜边；tan α = 对边/邻边；sin(90° - α) = cos α（互余关系）。',
      },
    ],
  },
]
