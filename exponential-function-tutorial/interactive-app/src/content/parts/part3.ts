import type { Unit } from '../../types'

// 第三部分：指数方程与模型（第 7–9 章）
// u7 指数方程 / u8 指数不等式·参数 / u9 数列·复利·离散模型
export const PART3_UNITS: Unit[] = [
  // ===================== 第 7 章：指数方程 =====================
  {
    id: 'u7',
    title: '指数方程',
    color: '#1cb0f6',
    icon: '🟰',
    blurb: '同底化、换元、取对数：先认结构，再动手算。',
    lessons: [
      {
        id: 'u7-l1',
        title: '同底化与共同指数块',
        subtitle: '能统一底数就别取对数',
        questions: [
          {
            id: 'u7-l1-q1',
            type: 'choice',
            prompt: '解方程 $2^{x+3}=16$，先把 $16$ 写成以 $2$ 为底的幂，再比较指数，得 $x=?$',
            options: ['$x=1$', '$x=4$', '$x=7$', '$x=13$'],
            answer: 0,
            explain: '$16=2^4$，故 $x+3=4$，$x=1$。',
          },
          {
            id: 'u7-l1-q2',
            type: 'input',
            prompt: '解方程 $3^{x-2}=27$，填 $x$ 的值。',
            accept: ['5', 'x=5'],
            explain: '$27=3^3$，故 $x-2=3$，$x=5$。',
          },
          {
            id: 'u7-l1-q3',
            type: 'choice',
            prompt: '解方程 $2^x+2^{x+1}=12$。第一步应该做什么？',
            options: [
              '直接两边取对数',
              '把左边提取公因式 $2^x$，得 $3\\cdot 2^x=12$',
              '把 $12$ 写成 $2$ 的幂',
              '令 $t=2^{2x}$',
            ],
            answer: 1,
            explain:
              '$2^{x+1}=2\\cdot 2^x$，故左边 $=2^x+2\\cdot 2^x=3\\cdot 2^x$。再 $2^x=4$，$x=2$。',
          },
          {
            id: 'u7-l1-q4',
            type: 'judge',
            prompt: '判断：方程 $5^{2x}=125$ 的解是 $x=\\dfrac{3}{2}$。',
            answer: true,
            explain: '$125=5^3$，故 $2x=3$，$x=\\dfrac{3}{2}$。',
          },
          {
            id: 'u7-l1-q5',
            type: 'choice',
            prompt: '解方程 $3^{x+1}=\\dfrac{1}{27}$，得 $x=?$',
            options: ['$x=-4$', '$x=-2$', '$x=2$', '$x=4$'],
            answer: 0,
            explain: '$\\dfrac{1}{27}=3^{-3}$，故 $x+1=-3$，$x=-4$。分数倒数对应指数变号。',
          },
          {
            id: 'u7-l1-q6',
            type: 'match',
            prompt: '把每个数写成以 $2$ 为底的幂，左右配对。',
            left: ['$8$', '$\\dfrac{1}{4}$', '$1$', '$\\sqrt{2}$'],
            right: ['$2^{3}$', '$2^{-2}$', '$2^{0}$', '$2^{1/2}$'],
            explain: '同底化的基本功：任何 $2$ 的幂都要能一眼写成 $2^{k}$。',
          },
        ],
      },
      {
        id: 'u7-l2',
        title: '换元降阶：令 $t=a^x>0$',
        subtitle: '把指数方程变成二次方程，记得舍负根',
        questions: [
          {
            id: 'u7-l2-q1',
            type: 'choice',
            prompt:
              '解 $2^{2x}-5\\cdot 2^x+6=0$。令 $t=2^x$ 后，原方程化为哪个二次方程？',
            options: [
              '$t^2-5t+6=0$',
              '$t^2+5t+6=0$',
              '$2t^2-5t+6=0$',
              '$t^2-5t-6=0$',
            ],
            answer: 0,
            explain: '$2^{2x}=(2^x)^2=t^2$，故 $t^2-5t+6=0$，$(t-2)(t-3)=0$。',
          },
          {
            id: 'u7-l2-q2',
            type: 'judge',
            prompt:
              '判断：解 $4^x-2^x-6=0$ 时令 $t=2^x$ 得 $t=3$ 或 $t=-2$，两个根都要回代求 $x$。',
            answer: false,
            explain:
              '$t=2^x>0$，故 $t=-2$ 必须舍去。只有 $t=3$ 合法，$x=\\log_2 3$。',
          },
          {
            id: 'u7-l2-q3',
            type: 'choice',
            prompt: '承上题，$4^x-2^x-6=0$ 的实数解是？',
            options: ['$x=\\log_2 3$', '$x=3$', '$x=\\log_2 3$ 或 $x=\\log_2(-2)$', '无解'],
            answer: 0,
            explain: '$t=2^x=3\\Rightarrow x=\\log_2 3$；$t=-2$ 舍去（$2^x=-2$ 无实数解）。',
          },
          {
            id: 'u7-l2-q4',
            type: 'choice',
            prompt:
              '解 $9^x-4\\cdot 3^x+3=0$。令 $t=3^x$ 得 $t^2-4t+3=0$，两根回代后 $x=?$',
            options: ['$x=0$ 或 $x=1$', '只有 $x=1$', '$x=1$ 或 $x=3$', '无解'],
            answer: 0,
            explain:
              '$(t-1)(t-3)=0$，$t=1$ 或 $t=3$（均 $>0$ 保留）。$3^x=1\\Rightarrow x=0$；$3^x=3\\Rightarrow x=1$。',
          },
          {
            id: 'u7-l2-q5',
            type: 'input',
            prompt:
              '令 $t=a^x$ 后必须立刻补上的取值条件是什么？请填一个不等式（形如 t>0）。',
            accept: ['t>0', 't > 0'],
            explain: '$a^x$ 恒为正，故 $t>0$，这是舍掉伪根的依据。',
          },
          {
            id: 'u7-l2-q6',
            type: 'match',
            prompt: '把结构信号与对应的换元代换配对。',
            left: [
              '$2^{2x}$ 与 $2^x$ 同时出现',
              '$3^{2x}$ 与 $3^x$ 同时出现',
              '$9^x$ 与 $3^x$ 同时出现',
            ],
            right: ['令 $t=2^x$', '令 $t=3^x$', '令 $t=3^x$（因 $9^x=(3^x)^2$）'],
            explain: '换元不是死记，而是找那个反复出现的核心指数块。',
          },
        ],
      },
      {
        id: 'u7-l3',
        title: '取对数与单调性判解',
        subtitle: '底数无法统一时把未知数从指数上拉下来',
        questions: [
          {
            id: 'u7-l3-q1',
            type: 'choice',
            prompt: '解方程 $5^x=12$，两边取自然对数后 $x=?$',
            options: [
              '$x=\\dfrac{\\ln 12}{\\ln 5}$',
              '$x=\\ln 12-\\ln 5$',
              '$x=\\dfrac{\\ln 5}{\\ln 12}$',
              '$x=12-5$',
            ],
            answer: 0,
            explain: '$x\\ln 5=\\ln 12\\Rightarrow x=\\dfrac{\\ln 12}{\\ln 5}$。',
          },
          {
            id: 'u7-l3-q2',
            type: 'input',
            prompt: '解方程 $2^{x+1}=7$，把整个指数块当整体取对数，$x=$ ？（用 ln 表示）',
            accept: ['ln7/ln2-1', '(ln7)/(ln2)-1', '\\frac{\\ln 7}{\\ln 2}-1', 'ln(7)/ln(2)-1'],
            explain: '$(x+1)\\ln 2=\\ln 7\\Rightarrow x=\\dfrac{\\ln 7}{\\ln 2}-1$。',
          },
          {
            id: 'u7-l3-q3',
            type: 'choice',
            prompt:
              '解 $2^x=5^{x-1}$（结果用对数表示）。两边取对数整理后 $x=?$',
            options: [
              '$x=\\dfrac{\\ln 5}{\\ln 5-\\ln 2}$',
              '$x=\\dfrac{\\ln 2}{\\ln 5}$',
              '$x=\\dfrac{\\ln 5}{\\ln 2}$',
              '$x=\\ln\\dfrac{5}{2}$',
            ],
            answer: 0,
            explain:
              '$x\\ln 2=(x-1)\\ln 5\\Rightarrow x(\\ln 5-\\ln 2)=\\ln 5\\Rightarrow x=\\dfrac{\\ln 5}{\\ln 5-\\ln 2}$。',
          },
          {
            id: 'u7-l3-q4',
            type: 'judge',
            prompt:
              '判断：当 $a>0,a\\ne 1,c>0$ 时，方程 $a^x=c$ 恒有唯一实数解。',
            answer: true,
            explain: '指数函数单调，水平线 $y=c$ 与曲线 $y=a^x$ 恰交一次，故唯一解。',
          },
          {
            id: 'u7-l3-q5',
            type: 'judge',
            prompt:
              '判断：要解 $a^{f(x)}=g(x)$ 时可以直接两边取对数，不必先检查 $g(x)$ 是否为正。',
            answer: false,
            explain: '左边恒正但右边 $g(x)$ 未必恒正；取对数前须先确认右侧大于 0。',
          },
          {
            id: 'u7-l3-q6',
            type: 'match',
            prompt: '把每个方程与最适合它的解法配对。',
            left: ['$2^{x+3}=16$', '$4^x-2^x-6=0$', '$5^x=12$', '$2^x=x+3$'],
            right: ['同底化', '换元 $t=2^x$', '两边取对数', '图像/单调性判解的个数'],
            explain:
              '先认结构再选方法：能同底就同底，有重复块就换元，底数不同取对数，和直线比就看图像。',
          },
        ],
      },
    ],
  },

  // ============== 第 8 章：指数不等式与参数分析 ==============
  {
    id: 'u8',
    title: '指数不等式·参数',
    color: '#ce82ff',
    icon: '⚖️',
    blurb: '按底数定方向，换元降阶，含参讨论有解与恒成立。',
    lessons: [
      {
        id: 'u8-l1',
        title: '同底比较与不等号方向',
        subtitle: '$a>1$ 不变号，$0<a<1$ 反向',
        questions: [
          {
            id: 'u8-l1-q1',
            type: 'choice',
            prompt: '解不等式 $2^x>8$。',
            options: ['$x>3$', '$x<3$', '$x>8$', '$x\\ge 3$'],
            answer: 0,
            explain: '$8=2^3$，底数 $2>1$ 递增，方向不变：$x>3$。',
          },
          {
            id: 'u8-l1-q2',
            type: 'choice',
            prompt: '解不等式 $\\left(\\dfrac13\\right)^{x}\\ge 9$。',
            options: ['$x\\le -2$', '$x\\ge -2$', '$x\\le 2$', '$x\\ge 2$'],
            answer: 0,
            explain:
              '$9=\\left(\\tfrac13\\right)^{-2}$，底数 $\\tfrac13\\in(0,1)$ 递减，方向反转：$x\\le -2$。',
          },
          {
            id: 'u8-l1-q3',
            type: 'judge',
            prompt:
              '判断：因为底数 $\\tfrac12<1$，由 $\\left(\\tfrac12\\right)^x<8$ 可得 $x>-3$。',
            answer: true,
            explain:
              '$8=\\left(\\tfrac12\\right)^{-3}$，递减函数比较指数反向：$x>-3$。',
          },
          {
            id: 'u8-l1-q4',
            type: 'input',
            prompt: '解不等式 $2^{x+2}\\ge 16$，填解集（形如 x>=2）。',
            accept: ['x>=2', 'x≥2', 'x >= 2'],
            explain: '$16=2^4$，$x+2\\ge 4$，$x\\ge 2$。',
          },
          {
            id: 'u8-l1-q5',
            type: 'choice',
            prompt:
              '解关于 $x$ 的 $a^{2x-1}>a^{x+2}$（$a>0,a\\ne 1$）。当 $0<a<1$ 时，解集是？',
            options: ['$x<3$', '$x>3$', '$x=3$', '$x>-3$'],
            answer: 0,
            explain:
              '临界点 $2x-1=x+2\\Rightarrow x=3$。$0<a<1$ 递减反向，故 $x<3$。',
          },
          {
            id: 'u8-l1-q6',
            type: 'match',
            prompt: '把底数情形与“比较指数时的不等号方向”配对。',
            left: ['$a>1$', '$0<a<1$'],
            right: ['方向不变', '方向反转'],
            explain: '递增不变号、递减反向，这是所有指数不等式的总开关。',
          },
        ],
      },
      {
        id: 'u8-l2',
        title: '换元处理高次指数不等式',
        subtitle: '令 $t=a^x>0$ 降成二次不等式再回代',
        questions: [
          {
            id: 'u8-l2-q1',
            type: 'choice',
            prompt:
              '解 $4^x-5\\cdot 2^x+4<0$。令 $t=2^x$ 后得 $t^2-5t+4<0$，解出 $t$ 的范围是？',
            options: ['$1<t<4$', '$t<1$ 或 $t>4$', '$-1<t<4$', '$t>4$'],
            answer: 0,
            explain: '$(t-1)(t-4)<0\\Rightarrow 1<t<4$。',
          },
          {
            id: 'u8-l2-q2',
            type: 'choice',
            prompt: '承上：由 $1<2^x<4$ 回代（底数 $2>1$），$x$ 的范围是？',
            options: ['$0<x<2$', '$x<0$ 或 $x>2$', '$1<x<4$', '$0\\le x\\le 2$'],
            answer: 0,
            explain: '$2^0<2^x<2^2$，递增不变号：$0<x<2$。',
          },
          {
            id: 'u8-l2-q3',
            type: 'choice',
            prompt:
              '解 $3^{2x}-10\\cdot 3^{x}+9\\le 0$。令 $t=3^x>0$ 解出 $t$ 范围后回代，$x$ 的范围是？',
            options: ['$0\\le x\\le 2$', '$1\\le x\\le 9$', '$x\\le 0$ 或 $x\\ge 2$', '$0<x<2$'],
            answer: 0,
            explain:
              '$(t-1)(t-9)\\le 0\\Rightarrow 1\\le t\\le 9$，即 $3^0\\le 3^x\\le 3^2$，$0\\le x\\le 2$。',
          },
          {
            id: 'u8-l2-q4',
            type: 'judge',
            prompt:
              '判断：换元 $t=2^x$ 后解二次不等式，可以不写 $t>0$ 这个条件。',
            answer: false,
            explain:
              '$t=2^x$ 始终为正；漏掉 $t>0$ 可能保留根本不可能出现的负 $t$ 分支。',
          },
          {
            id: 'u8-l2-q5',
            type: 'input',
            prompt:
              '解 $4^x-5\\cdot 2^x+4<0$ 时令 $t=2^x$，换元后必须立即补上的取值条件是什么？（形如 t>0）',
            accept: ['t>0', 't > 0'],
            explain: '$t=2^x$ 恒为正，漏掉 $t>0$ 会保留不可能出现的负 $t$ 分支。',
          },
          {
            id: 'u8-l2-q6',
            type: 'match',
            prompt: '把指数不等式的特征与“首选策略”配对。',
            left: ['两边可统一底数', '含 $a^{2x}$ 与 $a^x$', '右侧是参数常数', '与直线/另一指数比'],
            right: ['同底比较', '换元降阶', '先判是否恒正再分段', '看交点与图像上下'],
            explain: '决策树：同底→换元→参数分段→交点视角。',
          },
        ],
      },
      {
        id: 'u8-l3',
        title: '含参不等式：有解与恒成立',
        subtitle: '先找边界与临界值，再分区间讨论',
        questions: [
          {
            id: 'u8-l3-q1',
            type: 'choice',
            prompt: '使 $2^x>m$ 对所有 $x>0$ 都成立的参数条件是？',
            options: ['$m\\le 1$', '$m<0$', '$m\\ge 1$', '$m>1$'],
            answer: 0,
            explain:
              '$x>0$ 时 $2^x>1$，且 $x\\to 0^+$ 时 $2^x\\to 1$。要恒成立须 $m\\le 1$。',
          },
          {
            id: 'u8-l3-q2',
            type: 'input',
            prompt:
              '讨论 $a^x>1$（$a>0,a\\ne 1$），不论底数大小，所有情形共享的“不变边界点”在哪个 $x$？（因 $a^0=1$，填一个数）',
            accept: ['0', 'x=0'],
            explain:
              '$a^0=1$ 对任何底数都成立，故边界恒在 $x=0$。$a>1$ 时解为 $x>0$，$0<a<1$ 时解为 $x<0$。',
          },
          {
            id: 'u8-l3-q3',
            type: 'choice',
            prompt:
              '若 $4^x+2^x\\cdot a+1>0$ 对一切实数 $x$ 恒成立，求 $a$ 的范围。（令 $t=2^x>0$，分离参数）',
            options: ['$a>-2$', '$a\\ge -2$', '$a>2$', '$a<-2$'],
            answer: 0,
            explain:
              '$t^2+at+1>0$ 对 $t>0$ 恒成立 $\\iff a>-(t+\\tfrac1t)$。$t+\\tfrac1t\\ge 2$，取等 $t=1$，故 $a>-2$（严格）。',
          },
          {
            id: 'u8-l3-q4',
            type: 'judge',
            prompt:
              '判断：$2^{x-1}>a$ 中若 $a\\le 0$，则不等式对所有实数 $x$ 成立。',
            answer: true,
            explain: '左边 $2^{x-1}>0$ 恒成立；$a\\le 0$ 时右边非正，故恒成立。',
          },
          {
            id: 'u8-l3-q5',
            type: 'choice',
            prompt:
              '设 $f(x)=\\left(\\tfrac12\\right)^x$。方程 $f(x)=x+k$ 在 $(0,+\\infty)$ 上恰有一解，$k$ 的范围是？',
            options: ['$k<1$', '$k\\le 1$', '$k>1$', '$0<k<1$'],
            answer: 0,
            explain:
              '分离参数 $k=\\left(\\tfrac12\\right)^x-x$ 在 $(0,+\\infty)$ 严格递减，值域 $(-\\infty,1)$（开区间端点取不到），故 $k<1$。',
          },
          {
            id: 'u8-l3-q6',
            type: 'match',
            prompt: '不等式 $2^{x-1}>a$ 按参数 $a$ 的取值，把情形与解集配对。',
            left: ['$a\\le 0$', '$a>0$'],
            right: ['对一切实数 $x$ 成立', '$x>1+\\log_2 a$'],
            explain:
              '左边恒正：$a\\le 0$ 时无条件成立；$a>0$ 时由 $2^{x-1}=a$ 得临界 $x=1+\\log_2 a$，解在其右侧。',
          },
        ],
      },
    ],
  },

  // ======= 第 9 章：数列、复利与离散增长模型 =======
  {
    id: 'u9',
    title: '数列·复利·离散模型',
    color: '#58cc02',
    icon: '💰',
    blurb: '等比数列即离散指数；复利按年/连续计息，反推翻倍期数。',
    lessons: [
      {
        id: 'u9-l1',
        title: '等比数列：离散指数',
        subtitle: '从递推到通项，再到求和',
        questions: [
          {
            id: 'u9-l1-q1',
            type: 'choice',
            prompt: '等比数列首项 $a_1=2$，公比 $q=3$，通项 $a_n=?$',
            options: [
              '$a_n=2\\cdot 3^{\\,n-1}$',
              '$a_n=2\\cdot 3^{\\,n}$',
              '$a_n=3\\cdot 2^{\\,n-1}$',
              '$a_n=2+3(n-1)$',
            ],
            answer: 0,
            explain: '每前进一步乘一个 $q$，到第 $n$ 项共乘 $n-1$ 次：$a_n=a_1 q^{n-1}$。',
          },
          {
            id: 'u9-l1-q2',
            type: 'input',
            prompt: '承上（$a_1=2,q=3$），求 $a_4$。',
            accept: ['54'],
            explain: '$a_4=2\\cdot 3^3=2\\cdot 27=54$。',
          },
          {
            id: 'u9-l1-q3',
            type: 'judge',
            prompt:
              '判断：等比数列通项的指数是 $n-1$ 而不是 $n$，因为第 $1$ 项还没乘过公比。',
            answer: true,
            explain: '$a_1$ 乘了 $0$ 次 $q$，到第 $n$ 项乘 $n-1$ 次，故指数为 $n-1$。',
          },
          {
            id: 'u9-l1-q4',
            type: 'choice',
            prompt:
              '$a_1=1,q=2$ 的等比数列前 $n$ 项和 $S_n=?$（用求和公式 $S_n=a_1\\dfrac{1-q^n}{1-q}$）',
            options: ['$S_n=2^n-1$', '$S_n=2^n$', '$S_n=2^{n-1}$', '$S_n=n\\cdot 2$'],
            answer: 0,
            explain: '$S_n=\\dfrac{1-2^n}{1-2}=2^n-1$。',
          },
          {
            id: 'u9-l1-q5',
            type: 'input',
            prompt: '承上 $S_n=2^n-1$，求使 $S_n>1000$ 的最小正整数 $n$。',
            accept: ['10', 'n=10'],
            explain: '$2^n>1001$；$2^9=512<1001<1024=2^{10}$，故最小 $n=10$。',
          },
          {
            id: 'u9-l1-q6',
            type: 'match',
            prompt: '把等比数列的“公比情形”与对应的前 $n$ 项和公式配对。',
            left: ['$q\\ne 1$', '$q=1$', '$|q|<1$（无穷和）'],
            right: ['$S_n=a_1\\dfrac{1-q^n}{1-q}$', '$S_n=na_1$', '$S_\\infty=\\dfrac{a_1}{1-q}$'],
            explain:
              '用 $\\dfrac{1-q^n}{1-q}$ 前必须 $q\\ne 1$（否则分母为 $0$）；$q=1$ 时各项相同得 $na_1$；$|q|<1$ 时 $q^n\\to 0$，无穷和收敛。',
          },
        ],
      },
      {
        id: 'u9-l2',
        title: '离散增长、衰减与复利',
        subtitle: '比例因子 $q=1+r$，复利分多次计息',
        questions: [
          {
            id: 'u9-l2-q1',
            type: 'choice',
            prompt:
              '某量每步增长 $20\\%$，初值 $A_0$，则 $A_n=?$（增长率写法）',
            options: [
              '$A_n=A_0(1.2)^n$',
              '$A_n=A_0(0.2)^n$',
              '$A_n=A_0(0.8)^n$',
              '$A_n=A_0+0.2n$',
            ],
            answer: 0,
            explain: '比例因子 $q=1+r=1.2$，故 $A_n=A_0(1.2)^n$。',
          },
          {
            id: 'u9-l2-q2',
            type: 'choice',
            prompt: '某设备每年折旧 $15\\%$，每年的比例因子 $q$ 是多少？',
            options: ['$0.85$', '$0.15$', '$1.15$', '$1.5$'],
            answer: 0,
            explain: '折旧 $15\\%$ 即变为原来的 $1-0.15=0.85$ 倍，$q=0.85$。',
          },
          {
            id: 'u9-l2-q3',
            type: 'choice',
            prompt:
              '本金 $10000$ 元，年利率 $5\\%$，按月复利，$3$ 年后本利和表达式是？',
            options: [
              '$10000\\left(1+\\dfrac{0.05}{12}\\right)^{36}$',
              '$10000\\left(1+0.05\\right)^{3}$',
              '$10000\\left(1+\\dfrac{0.05}{12}\\right)^{3}$',
              '$10000\\left(1+0.05\\cdot 12\\right)^{3}$',
            ],
            answer: 0,
            explain:
              '$A=P\\left(1+\\dfrac{r}{m}\\right)^{mt}$，$m=12,t=3$，更新 $36$ 次：$10000\\left(1+\\tfrac{0.05}{12}\\right)^{36}$。',
          },
          {
            id: 'u9-l2-q4',
            type: 'input',
            prompt:
              '复利公式 $A=P\\left(1+\\dfrac{r}{m}\\right)^{mt}$ 中，指数 $mt$ 表示什么？请填一个数：本金 $8000$、年利率 $4\\%$、按年复利（$m=1$）、存 $5$ 年时，指数 $mt=$ ？',
            accept: ['5'],
            explain: '$m=1,t=5$，更新 $mt=5$ 次：$A=8000(1.04)^5$。',
          },
          {
            id: 'u9-l2-q5',
            type: 'judge',
            prompt:
              '判断：$q$（比例因子）与 $1+r$（增长率写法）描述的是同一件事，满足 $q=1+r$。',
            answer: true,
            explain: '$q$ 强调“每步乘的因子”，$1+r$ 强调“增加了多少比例”，二者等价。',
          },
          {
            id: 'u9-l2-q6',
            type: 'match',
            prompt: '把模型形式与含义配对。',
            left: [
              '$A_n=A_0(1+r)^n$',
              '$A_n=A_0(1-r)^n$',
              '$A=P\\left(1+\\tfrac{r}{m}\\right)^{mt}$',
            ],
            right: ['离散增长', '离散衰减', '分 $m$ 次计息的复利'],
            explain: '三式同源，只是每步比例与更新次数不同。',
          },
        ],
      },
      {
        id: 'u9-l3',
        title: '翻倍期数与连续复利',
        subtitle: '反推增长率，对数算步数，趋向 $e^{rt}$',
        questions: [
          {
            id: 'u9-l3-q1',
            type: 'choice',
            prompt: '模型 $A_n=A_0 q^n$（$q>1$）的倍增步数 $N$ 满足 $q^N=2$，则 $N=?$',
            options: [
              '$N=\\dfrac{\\ln 2}{\\ln q}$',
              '$N=\\dfrac{\\ln q}{\\ln 2}$',
              '$N=2q$',
              '$N=\\ln 2\\cdot \\ln q$',
            ],
            answer: 0,
            explain: '$q^N=2\\Rightarrow N\\ln q=\\ln 2\\Rightarrow N=\\dfrac{\\ln 2}{\\ln q}$。',
          },
          {
            id: 'u9-l3-q2',
            type: 'input',
            prompt:
              '希望 $10$ 年内本金翻倍（按年复利），由 $(1+r)^{10}=2$ 解出 $r$。请填表达式（如 2^(1/10)-1）。',
            accept: [
              '2^(1/10)-1',
              '\\sqrt[10]{2}-1',
              '2^{1/10}-1',
              'root(10,2)-1',
              '2**(1/10)-1',
            ],
            explain:
              '$(1+r)^{10}=2\\Rightarrow 1+r=2^{1/10}\\Rightarrow r=\\sqrt[10]{2}-1\\approx 7.18\\%$，与“$72$ 法则”吻合。',
          },
          {
            id: 'u9-l3-q3',
            type: 'judge',
            prompt:
              '判断：离散复利 $\\left(1+\\dfrac{r}{m}\\right)^{mt}$ 当 $m\\to\\infty$ 时趋于 $e^{rt}$，对应连续复利 $A=Pe^{rt}$。',
            answer: true,
            explain: '复利频率越来越细，离散更新趋向连续流动，极限即连续复利。',
          },
          {
            id: 'u9-l3-q4',
            type: 'choice',
            prompt:
              '同名义年利率 $r=12\\%$、存 $1$ 年。按月复利与连续复利比较，哪个本利和更高？',
            options: [
              '连续复利略高（$m\\to\\infty$ 为上确界）',
              '按月复利更高',
              '两者完全相等',
              '无法比较',
            ],
            answer: 0,
            explain:
              '$\\left(1+\\tfrac{r}{m}\\right)^m$ 随 $m$ 单调递增，趋于 $e^r$，故连续复利是上确界，略高于按月。',
          },
          {
            id: 'u9-l3-q5',
            type: 'judge',
            prompt:
              '判断：递推 $A_{n+1}=qA_n+b$（$b\\ne 0$）仍是纯指数结构。',
            answer: false,
            explain:
              '每步除按比例更新外还加固定增量 $b$，结构不再是纯指数。只有 $A_{n+1}=qA_n$ 才是。',
          },
          {
            id: 'u9-l3-q6',
            type: 'match',
            prompt: '把三种增长写法与其强调重点配对。',
            left: ['$A_0 q^n$', '$P\\left(1+\\tfrac{r}{m}\\right)^{mt}$', '$Pe^{kt}$'],
            right: ['每步乘同一比例因子', '每年分多次离散更新', '时间细到可视为连续流动'],
            explain: '同一主线在离散、分次、连续三种语境下的写法。',
          },
        ],
      },
    ],
  },
]
