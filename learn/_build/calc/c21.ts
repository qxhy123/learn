export const LESSONS = [
  // ─────────────────────────────────────────────────────────────────────────────
  // 关 c21-a  偏导数与全微分
  // ─────────────────────────────────────────────────────────────────────────────
  {
    id: 'c21-a',
    title: '偏导数与全微分',
    subtitle: '冻结变量·逐个求导·线性化误差',
    intro: [
      // 卡 1：动机——为什么一元求导不够？
      {
        title: '为什么需要偏导数？',
        body:
          '一元函数 $f(x)$ 只有一个自变量，导数 $f\'(x)$ 就是唯一的变化率。\n\n' +
          '但现实问题往往有多个变量：气温 $T$ 同时受海拔 $h$ 和纬度 $\\varphi$ 影响；' +
          '神经网络损失 $L$ 依赖数百万个参数 $\\theta_1,\\theta_2,\\dots$。\n\n' +
          '我们想回答："**固定其他变量不动，只改变 $x$，函数变化有多快？**"——这就是偏导数的问题。\n\n' +
          '偏导数是一元导数的直接推广：把其余变量当常数，剩下的就是一元求导。',
        tip: '口诀：**求哪个变量的偏导，其余变量全当常数**。',
      },
      // 卡 2：偏导数的定义与记号
      {
        title: '偏导数的定义',
        body:
          '设 $z = f(x,y)$，在点 $(x_0, y_0)$ 附近有定义。**对 $x$ 的偏导数**定义为：\n\n' +
          '把 $y$ 固定在 $y_0$，让 $x$ 从 $x_0$ 变化，记录增量比的极限。\n\n' +
          '**记号**：$f_x$、$\\dfrac{\\partial f}{\\partial x}$、$\\dfrac{\\partial z}{\\partial x}$ 三种写法等价，都读作"$f$ 对 $x$ 的偏导"。\n\n' +
          '**计算方法**：求 $f_x$ 时，将 $y$ 视为常数，对 $x$ 用所有一元求导规则（乘积、链式、指数……）；求 $f_y$ 时，将 $x$ 视为常数，对 $y$ 求导。',
        formula:
          'f_x(x_0,y_0)=\\lim_{\\Delta x\\to 0}\\frac{f(x_0+\\Delta x,\\,y_0)-f(x_0,y_0)}{\\Delta x}',
        tip: '偏导符号 $\\partial$（读作"partial"）区别于全导数 $d$，提醒我们还有其他变量被固定了。',
      },
      // 卡 3：具体例题——多项式
      {
        title: '例题：多项式的偏导',
        body:
          '设 $f(x,y) = x^2 + 3xy + y^2$，求 $f_x$ 和 $f_y$。\n\n' +
          '**求 $f_x$（把 $y$ 当常数）**：\n\n' +
          '- $x^2$ 对 $x$ 求导 $\\to 2x$\n' +
          '- $3xy$ 对 $x$ 求导，$y$ 是常数 $\\to 3y$\n' +
          '- $y^2$ 对 $x$ 求导，$y^2$ 是常数 $\\to 0$\n\n' +
          '所以 $f_x = 2x + 3y$。\n\n' +
          '**求 $f_y$（把 $x$ 当常数）**：\n\n' +
          '- $x^2 \\to 0$；$3xy \\to 3x$；$y^2 \\to 2y$\n\n' +
          '所以 $f_y = 3x + 2y$。\n\n' +
          '**验证**：代入 $(1,2)$，$f_x(1,2)=2+6=8$，$f_y(1,2)=3+4=7$——这两个数分别是沿 $x$ 方向和 $y$ 方向的变化率。',
        formula:
          'f_x = 2x+3y,\\quad f_y = 3x+2y',
      },
      // 卡 4：例题——乘积与链式法则
      {
        title: '例题：乘积法则与链式法则',
        body:
          '设 $f(x,y) = e^{xy}\\sin(x+y)$，求 $f_x$。\n\n' +
          '把 $y$ 当常数，$f$ 是 $e^{xy}$ 和 $\\sin(x+y)$ 的**乘积**，用乘积法则：\n\n' +
          '第一步：$\\dfrac{\\partial}{\\partial x}(e^{xy}) = y e^{xy}$（链式：指数内层 $xy$ 对 $x$ 求导得 $y$）\n\n' +
          '第二步：$\\dfrac{\\partial}{\\partial x}(\\sin(x+y)) = \\cos(x+y)$（链式：$x+y$ 对 $x$ 求导得 1）\n\n' +
          '合并：$f_x = ye^{xy}\\sin(x+y) + e^{xy}\\cos(x+y) = e^{xy}\\bigl[y\\sin(x+y)+\\cos(x+y)\\bigr]$\n\n' +
          '类似地，对称可得 $f_y = e^{xy}\\bigl[x\\sin(x+y)+\\cos(x+y)\\bigr]$。',
        formula:
          'f_x = e^{xy}\\bigl[y\\sin(x+y)+\\cos(x+y)\\bigr]',
        tip: '复合函数求偏导：先写乘积/商/链式的框架，再逐项把"对目标变量求导"填进去，其余当常数。',
      },
      // 卡 5：全微分——线性化
      {
        title: '全微分：增量的线性近似',
        body:
          '偏导数告诉我们沿单个坐标轴方向的变化率。当 $x$ 和 $y$ **同时**有微小变化 $dx$、$dy$ 时，函数增量近似为：\n\n' +
          '$\\Delta z \\approx f_x\\,dx + f_y\\,dy$\n\n' +
          '右边叫做**全微分** $dz$，是实际增量 $\\Delta z$ 的**线性主部**（误差为比 $\\rho=\\sqrt{(dx)^2+(dy)^2}$ 更高阶的无穷小）。\n\n' +
          '**具体例子**：$z = x^2 y$，$f_x=2xy$，$f_y=x^2$，全微分 $dz = 2xy\\,dx + x^2\\,dy$。\n\n' +
          '在点 $(2,3)$ 附近，$x$ 变 $0.1$，$y$ 变 $-0.05$：\n\n' +
          '$\\Delta z \\approx 2\\cdot2\\cdot3\\cdot0.1 + 4\\cdot(-0.05) = 1.2 - 0.2 = 1.0$',
        formula:
          'dz = f_x\\,dx + f_y\\,dy = \\frac{\\partial z}{\\partial x}dx + \\frac{\\partial z}{\\partial y}dy',
        tip: '全微分是曲面 $z=f(x,y)$ 切平面上的增量——切平面是曲面在该点的最佳线性近似。',
      },
      // 卡 6：可微与偏导的关系——重要易错点
      {
        title: '易错点：偏导存在不等于可微！',
        body:
          '一元函数：导数存在 $\\Leftrightarrow$ 可微。**多元函数这个等价不再成立！**\n\n' +
          '经典反例：$f(x,y) = \\dfrac{xy}{x^2+y^2}$（原点处定义为 0）。\n\n' +
          '- $f_x(0,0) = 0$，$f_y(0,0) = 0$（用定义直接算，分子含 $y$ 或 $x$，代 0 后为 0）\n' +
          '- 但沿 $y = x$ 趋近原点：$f(x,x) = \\dfrac{x^2}{2x^2} = \\dfrac{1}{2}$，极限 $\\neq f(0,0)=0$\n' +
          '- 函数在原点**不连续**，更不可微！\n\n' +
          '**充分条件（实用判断）**：若偏导数 $f_x$、$f_y$ **连续**，则 $f$ 可微。对常见初等函数组合，这个条件几乎总成立。',
        tip: '三层递进：偏导连续 $\\Rightarrow$ 可微 $\\Rightarrow$ 偏导存在。反过来全都不能推！',
        reveal: {
          q: '已知 $z = x^3 + 2xy^2$，写出全微分 $dz$。',
          a: '$f_x = 3x^2+2y^2$，$f_y = 4xy$，故 $dz = (3x^2+2y^2)dx + 4xy\\,dy$。',
        },
      },
      // 卡 7：高阶偏导与混合偏导
      {
        title: '高阶偏导与混合偏导相等',
        body:
          '对偏导数再次求偏导，得到**二阶偏导数**。有四个：\n\n' +
          '$f_{xx}$（先 $x$ 后 $x$）、$f_{yy}$（先 $y$ 后 $y$）、$f_{xy}$（先 $x$ 后 $y$）、$f_{yx}$（先 $y$ 后 $x$）\n\n' +
          '其中 $f_{xy}$、$f_{yx}$ 叫**混合偏导数**，顺序不同。\n\n' +
          '**Schwarz 定理**：若 $f_{xy}$ 和 $f_{yx}$ 都连续，则 $f_{xy} = f_{yx}$（混合偏导可以交换顺序）。\n\n' +
          '**例**：$f(x,y) = x^3y + xy^3$。$f_x = 3x^2y+y^3$，对 $y$ 再求导：$f_{xy} = 3x^2+3y^2$。$f_y = x^3+3xy^2$，对 $x$ 再求导：$f_{yx} = 3x^2+3y^2$。确实相等。\n\n' +
          '**实用意义**：对连续函数，两种混合偏导顺序随意——选更简单的那个算。',
        formula:
          'f_{xy} = \\frac{\\partial^2 f}{\\partial y\\,\\partial x},\\quad f_{yx} = \\frac{\\partial^2 f}{\\partial x\\,\\partial y};\\quad f_{xy}=f_{yx}\\text{ 若连续}',
      },
      // 卡 8：综合例题——二阶偏导
      {
        title: '综合例题：计算全部二阶偏导',
        body:
          '设 $f(x,y) = \\ln(x^2+y^2)$，求四个二阶偏导。\n\n' +
          '**一阶**：$f_x = \\dfrac{2x}{x^2+y^2}$，$f_y = \\dfrac{2y}{x^2+y^2}$\n\n' +
          '**$f_{xx}$**（对 $f_x$ 再对 $x$ 求导，用商的求导法则）：\n\n' +
          '$f_{xx} = \\dfrac{2(x^2+y^2) - 2x\\cdot2x}{(x^2+y^2)^2} = \\dfrac{2y^2-2x^2}{(x^2+y^2)^2}$\n\n' +
          '**$f_{yy}$**（对称，换 $x\\leftrightarrow y$）：$f_{yy} = \\dfrac{2x^2-2y^2}{(x^2+y^2)^2}$\n\n' +
          '**$f_{xy}$**（对 $f_x$ 关于 $y$ 求导）：$f_{xy} = \\dfrac{-4xy}{(x^2+y^2)^2}$\n\n' +
          '**验证**：$f_{xx}+f_{yy} = 0$（即 $\\ln(x^2+y^2)$ 是调和函数！）',
        formula:
          'f_{xx}+f_{yy} = \\frac{2y^2-2x^2}{(x^2+y^2)^2}+\\frac{2x^2-2y^2}{(x^2+y^2)^2} = 0',
      },
    ],
    questions: [
      // 题 1：choice — 偏导计算
      {
        id: 'c21-a-q1',
        type: 'choice',
        prompt: '设 $f(x,y) = x^3y^2 + \\sin(y)$，则 $f_x$ 等于？',
        options: [
          '$3x^2y^2$',
          '$3x^2y^2 + \\cos(y)$',
          '$x^3 \\cdot 2y$',
          '$3x^2y^2 + x^3\\cdot 2y$',
        ],
        answer: 0,
        explain:
          '对 $x$ 求偏导时，$y$ 是常数。$\\dfrac{\\partial}{\\partial x}(x^3y^2) = 3x^2y^2$（$y^2$ 是常数因子），$\\dfrac{\\partial}{\\partial x}(\\sin y) = 0$（$\\sin y$ 对 $x$ 是常数）。故 $f_x = 3x^2y^2$。',
      },
      // 题 2：input — 偏导数值
      {
        id: 'c21-a-q2',
        type: 'input',
        prompt:
          '设 $f(x,y) = x^2 + 3xy + y^2$，求 $f_y(1,2)$（即在点 $(1,2)$ 处对 $y$ 的偏导数值）。',
        accept: ['7'],
        explain:
          '$f_y = 3x + 2y$。代入 $(1,2)$：$f_y(1,2) = 3\\times1 + 2\\times2 = 3+4 = 7$。',
      },
      // 题 3：judge — 偏导与可微
      {
        id: 'c21-a-q3',
        type: 'judge',
        prompt:
          '若二元函数在某点的两个偏导数 $f_x$、$f_y$ 都存在，则该函数在该点一定可微。',
        answer: false,
        explain:
          '这是多元微积分的经典易错点。偏导数存在**不能**保证可微。反例：$f(x,y)=xy/(x^2+y^2)$（原点处为 0），在原点偏导均为 0，但函数在原点不连续，更不可微。充分条件是：偏导数**连续**则可微。',
      },
      // 题 4：choice — 全微分
      {
        id: 'c21-a-q4',
        type: 'choice',
        prompt: '设 $z = x^2y + y^3$，全微分 $dz$ 等于？',
        options: [
          '$2xy\\,dx + (x^2+3y^2)\\,dy$',
          '$2xy\\,dx + x^2\\,dy$',
          '$(2xy+y^3)\\,dx + (x^2+3y^2)\\,dy$',
          '$2x\\,dx + 3y^2\\,dy$',
        ],
        answer: 0,
        explain:
          '$f_x = \\dfrac{\\partial}{\\partial x}(x^2y+y^3) = 2xy$（$y$、$y^3$ 相对 $x$ 是常数）。' +
          '$f_y = \\dfrac{\\partial}{\\partial y}(x^2y+y^3) = x^2+3y^2$。' +
          '故 $dz = 2xy\\,dx + (x^2+3y^2)\\,dy$。',
      },
      // 题 5：judge — 混合偏导相等
      {
        id: 'c21-a-q5',
        type: 'judge',
        prompt:
          '对任意二元函数 $f(x,y)$，混合偏导数 $f_{xy}$ 与 $f_{yx}$ 一定相等。',
        answer: false,
        explain:
          'Schwarz 定理要求混合偏导数**连续**才能保证 $f_{xy}=f_{yx}$。存在反例：某些分段定义的函数，混合偏导数在原点处 $f_{xy}(0,0)=-1$，$f_{yx}(0,0)=1$，两者不等。对连续初等函数则安全可交换。',
      },
      // 题 6：match — 偏导计算规则
      {
        id: 'c21-a-q6',
        type: 'match',
        prompt: '将函数与其对 $x$ 的偏导数 $f_x$ 对应（$y$ 视为常数）。',
        left: ['$f=e^{xy}$', '$f=y\\sin x$', '$f=x^2+y^2$'],
        right: ['$ye^{xy}$', '$y\\cos x$', '$2x$'],
      },
      // 题 7：input — 混合偏导
      {
        id: 'c21-a-q7',
        type: 'input',
        prompt:
          '设 $f(x,y) = x^3y + xy^3$，求 $f_{xy}$（先对 $x$ 求导，再对 $y$ 求导）的表达式在点 $(1,1)$ 处的值。',
        accept: ['6'],
        explain:
          '$f_x = 3x^2y + y^3$。再对 $y$ 求导：$f_{xy} = 3x^2 + 3y^2$。代入 $(1,1)$：$f_{xy}(1,1) = 3+3 = 6$。',
      },
      // 题 8：choice — 全微分近似
      {
        id: 'c21-a-q8',
        type: 'choice',
        prompt:
          '已知 $z = x^2y$ 在点 $(2,3)$ 处，$x$ 变化 $dx=0.1$，$y$ 变化 $dy=-0.1$，用全微分近似 $\\Delta z$ 约等于？',
        options: ['$0.8$', '$1.2$', '$0$', '$-0.4$'],
        answer: 0,
        explain:
          '$f_x = 2xy = 2\\times2\\times3=12$，$f_y = x^2 = 4$。' +
          '$\\Delta z \\approx f_x\\,dx + f_y\\,dy = 12\\times0.1 + 4\\times(-0.1) = 1.2 - 0.4 = 0.8$。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────────────────
  // 关 c21-b  梯度、方向导数与链式法则
  // ─────────────────────────────────────────────────────────────────────────────
  {
    id: 'c21-b',
    title: '梯度、方向导数与链式法则',
    subtitle: '最速上升方向·任意方向变化率·复合函数求导',
    intro: [
      // 卡 1：动机——偏导数只给了坐标方向
      {
        title: '偏导数的局限：只看坐标轴方向',
        body:
          '偏导数 $f_x$ 是沿 $x$ 轴正方向的变化率，$f_y$ 是沿 $y$ 轴正方向的变化率。\n\n' +
          '但实际问题中，我们常常想知道**沿任意方向**函数变化有多快，比如：\n\n' +
          '- 地形图上，向东北方向爬坡的坡度？\n' +
          '- 梯度下降时，应该选哪个方向让损失**下降最快**？\n\n' +
          '这引出两个新概念：**方向导数**（沿任意方向的变化率）和**梯度**（收集了所有方向信息的向量）。',
        tip: '把函数 $f(x,y)$ 想象成地形的海拔图：等高线是高度相同的曲线，梯度垂直等高线，指向海拔上升最快的方向。',
      },
      // 卡 2：梯度的定义
      {
        title: '梯度：把偏导数拼成向量',
        body:
          '函数 $f(x,y)$ 在点 $P_0(x_0,y_0)$ 处的**梯度**，是以两个偏导数为分量构成的向量：\n\n' +
          '$\\nabla f = (f_x,\\, f_y)$\n\n' +
          '读作"nabla $f$"或"grad $f$"。\n\n' +
          '**具体例子**：$f(x,y) = x^2 + y^2$，$f_x = 2x$，$f_y = 2y$，故 $\\nabla f = (2x, 2y)$。\n\n' +
          '在点 $(1,2)$ 处，$\\nabla f = (2,4)$，方向指向右上方，模 $|\\nabla f| = \\sqrt{4+16} = \\sqrt{20} = 2\\sqrt{5}$。\n\n' +
          '梯度是一个**向量**，有方向也有大小。',
        formula:
          '\\nabla f = \\left(\\frac{\\partial f}{\\partial x},\\,\\frac{\\partial f}{\\partial y}\\right) = f_x\\mathbf{i}+f_y\\mathbf{j}',
      },
      // 卡 3：方向导数——沿任意方向
      {
        title: '方向导数：沿指定方向的变化率',
        body:
          '给定单位方向向量 $\\mathbf{l} = (\\cos\\alpha, \\cos\\beta)$（$|\\mathbf{l}|=1$），**方向导数**定义为：\n\n' +
          '$\\dfrac{\\partial f}{\\partial \\mathbf{l}} = $ 沿 $\\mathbf{l}$ 移动单位长度时 $f$ 的增量\n\n' +
          '**关键公式**（当 $f$ 可微时）：\n\n' +
          '$\\dfrac{\\partial f}{\\partial \\mathbf{l}} = f_x\\cos\\alpha + f_y\\cos\\beta = \\nabla f \\cdot \\mathbf{l}$\n\n' +
          '**注意**：方向向量 $\\mathbf{l}$ 必须是**单位向量**！若给定的方向向量 $\\mathbf{v}$ 不是单位向量，必须先归一化：$\\mathbf{l} = \\dfrac{\\mathbf{v}}{|\\mathbf{v}|}$。\n\n' +
          '**例**：$f(x,y)=x^2+y^2$，在点 $(1,2)$ 处，沿 $\\mathbf{v}=(3,4)$ 方向：$|\\mathbf{v}|=5$，$\\mathbf{l}=(3/5,4/5)$，$\\nabla f=(2,4)$，方向导数 $= 2\\cdot(3/5)+4\\cdot(4/5)=6/5+16/5=22/5$。',
        formula:
          '\\frac{\\partial f}{\\partial \\mathbf{l}} = \\nabla f \\cdot \\mathbf{l} = f_x\\cos\\alpha + f_y\\cos\\beta',
        tip: '方向导数 $=$ 梯度与方向向量的**点积**，所以方向向量必须单位化，否则结果差一个缩放因子。',
      },
      // 卡 4：梯度的几何意义——最速上升
      {
        title: '梯度方向：函数增长最快的方向',
        body:
          '方向导数公式 $\\dfrac{\\partial f}{\\partial \\mathbf{l}} = |\\nabla f|\\cos\\theta$，其中 $\\theta$ 是梯度与 $\\mathbf{l}$ 的夹角。\n\n' +
          '- $\\theta = 0$（沿梯度方向）：$\\cos\\theta=1$，方向导数 $= |\\nabla f|$（最大值）\n' +
          '- $\\theta = \\pi/2$（垂直梯度）：$\\cos\\theta=0$，方向导数 $= 0$（沿等高线）\n' +
          '- $\\theta = \\pi$（逆梯度）：$\\cos\\theta=-1$，方向导数 $= -|\\nabla f|$（下降最快）\n\n' +
          '**结论**：**梯度方向是函数值增长最快的方向，模是最大方向导数**。\n\n' +
          '**梯度下降法**：每次沿**负梯度**方向移动一小步，让损失下降最快。$\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)$。',
        formula:
          '\\frac{\\partial f}{\\partial \\mathbf{l}} = |\\nabla f|\\cos\\theta \\leq |\\nabla f|,\\quad \\text{等号当且仅当}\\; \\mathbf{l}\\parallel\\nabla f',
      },
      // 卡 5：链式法则——多元复合函数
      {
        title: '链式法则：沿每条路径求和',
        body:
          '设 $z = f(u, v)$，$u = u(x,y)$，$v = v(x,y)$，则复合函数 $z$ 对 $x$ 的偏导：\n\n' +
          '$\\dfrac{\\partial z}{\\partial x} = \\dfrac{\\partial z}{\\partial u}\\dfrac{\\partial u}{\\partial x} + \\dfrac{\\partial z}{\\partial v}\\dfrac{\\partial v}{\\partial x}$\n\n' +
          '**直觉**：$x$ 改变后，$z$ 受两条路径影响——\n\n' +
          '- 路径一：$x \\to u \\to z$，贡献 $f_u \\cdot u_x$\n' +
          '- 路径二：$x \\to v \\to z$，贡献 $f_v \\cdot v_x$\n' +
          '- 总变化 $=$ 两条路径之和\n\n' +
          '**口诀**："有几条路，加几项"。\n\n' +
          '**例**：$z = e^{uv}$，$u = x+y$，$v = xy$。求 $\\partial z/\\partial x$：\n\n' +
          '$z_u = ve^{uv}$，$u_x = 1$；$z_v = ue^{uv}$，$v_x = y$。\n\n' +
          '故 $\\dfrac{\\partial z}{\\partial x} = ve^{uv}\\cdot1 + ue^{uv}\\cdot y = e^{(x+y)xy}(xy + (x+y)y) = e^{(x+y)xy}(2xy+y^2)$。',
        formula:
          '\\frac{\\partial z}{\\partial x} = \\frac{\\partial z}{\\partial u}\\frac{\\partial u}{\\partial x} + \\frac{\\partial z}{\\partial v}\\frac{\\partial v}{\\partial x}',
      },
      // 卡 6：隐函数求导
      {
        title: '隐函数求导：$F(x,y)=0$ 时的 $y\'$',
        body:
          '有时函数关系以 $F(x,y)=0$ 的隐式给出，求 $y\'=dy/dx$。\n\n' +
          '**方法**：对 $F(x,y)=0$ 两边对 $x$ 求导（$y$ 是 $x$ 的函数，用链式法则）：\n\n' +
          '$F_x + F_y \\cdot y\' = 0 \\Rightarrow y\' = -\\dfrac{F_x}{F_y}$（$F_y \\neq 0$）\n\n' +
          '**例**：圆 $x^2+y^2=1$，$F = x^2+y^2-1$，$F_x=2x$，$F_y=2y$，故 $y\' = -\\dfrac{x}{y}$。\n\n' +
          '几何验证：圆上点 $(x,y)$ 处切线斜率确实是 $-x/y$（向量 $(x,y)$ 是半径方向，切线垂直）。\n\n' +
          '**易错点**：公式里是 $-F_x/F_y$，分子是 $x$ 的偏导，分母是 $y$ 的偏导——"谁在分母，谁是因变量"。',
        formula:
          'F(x,y)=0\\;\\Rightarrow\\;\\frac{dy}{dx}=-\\frac{F_x}{F_y}\\quad(F_y\\neq 0)',
        reveal: {
          q: '设 $f(x,y) = x^2 + xy + y^2$，在点 $(1,1)$ 处的梯度 $\\nabla f$ 是什么？',
          a: '$f_x = 2x+y=3$，$f_y = x+2y=3$，故 $\\nabla f(1,1) = (3,3)$。',
        },
      },
      // 卡 7：综合例题——梯度 + 方向导数
      {
        title: '综合例题：梯度与方向导数的完整流程',
        body:
          '**题目**：$f(x,y) = x^2 + xy + y^2$，求在点 $(1,1)$ 处沿 $\\mathbf{v}=(3,4)$ 方向的方向导数，并指出增长最快的方向。\n\n' +
          '**第一步：求梯度**\n\n' +
          '$f_x = 2x+y$，$f_y = x+2y$。代入 $(1,1)$：$\\nabla f = (3, 3)$。\n\n' +
          '**第二步：归一化方向向量**\n\n' +
          '$|\\mathbf{v}|=\\sqrt{9+16}=5$，$\\mathbf{l} = (3/5, 4/5)$。\n\n' +
          '**第三步：计算方向导数**\n\n' +
          '$\\dfrac{\\partial f}{\\partial \\mathbf{l}} = \\nabla f \\cdot \\mathbf{l} = 3\\cdot\\dfrac{3}{5}+3\\cdot\\dfrac{4}{5} = \\dfrac{9+12}{5} = \\dfrac{21}{5}$\n\n' +
          '**第四步：增长最快方向**\n\n' +
          '梯度方向 $(3,3)$ 单位化：$(\\frac{1}{\\sqrt{2}}, \\frac{1}{\\sqrt{2}})$，最大方向导数 $= |\\nabla f| = 3\\sqrt{2}$。',
        formula:
          '\\frac{\\partial f}{\\partial \\mathbf{l}}\\bigg|_{(1,1)} = (3,3)\\cdot\\left(\\frac{3}{5},\\frac{4}{5}\\right) = \\frac{21}{5}',
      },
    ],
    questions: [
      // 题 1：choice — 梯度计算
      {
        id: 'c21-b-q1',
        type: 'choice',
        prompt: '设 $f(x,y) = x^2 + y^2$，在点 $(1,2)$ 处的梯度 $\\nabla f$ 是？',
        options: ['$(1,2)$', '$(2,4)$', '$(2,2)$', '$(4,2)$'],
        answer: 1,
        explain:
          '$f_x = 2x$，$f_y = 2y$。代入 $(1,2)$：$\\nabla f = (2\\times1,\\ 2\\times2) = (2, 4)$。',
      },
      // 题 2：judge — 方向向量单位化
      {
        id: 'c21-b-q2',
        type: 'judge',
        prompt: '计算方向导数 $\\dfrac{\\partial f}{\\partial \\mathbf{l}} = \\nabla f \\cdot \\mathbf{l}$ 时，$\\mathbf{l}$ 可以是任意方向向量，不必单位化。',
        answer: false,
        explain:
          '公式 $\\dfrac{\\partial f}{\\partial \\mathbf{l}} = \\nabla f\\cdot\\mathbf{l}$ 成立的前提是 $|\\mathbf{l}|=1$（单位向量）。若 $\\mathbf{l}$ 未归一化，结果会差一个缩放因子，得到错误答案。',
      },
      // 题 3：input — 方向导数计算
      {
        id: 'c21-b-q3',
        type: 'input',
        prompt:
          '$f(x,y)=x^2+y^2$，在点 $(1,2)$ 处，沿方向 $\\mathbf{v}=(3,4)$ 的方向导数是多少？（$|\\mathbf{v}|=5$）',
        accept: ['22/5', '4.4'],
        explain:
          '$\\nabla f(1,2) = (2,4)$，$\\mathbf{l} = (3/5, 4/5)$。方向导数 $= 2\\times(3/5)+4\\times(4/5) = 6/5+16/5 = 22/5$。',
      },
      // 题 4：choice — 梯度方向
      {
        id: 'c21-b-q4',
        type: 'choice',
        prompt:
          '在某点处，沿哪个方向移动，函数值增长最快？',
        options: [
          '沿 $x$ 轴正方向',
          '沿梯度 $\\nabla f$ 的方向',
          '沿 $y$ 轴正方向',
          '沿负梯度 $-\\nabla f$ 的方向',
        ],
        answer: 1,
        explain:
          '方向导数 $= |\\nabla f|\\cos\\theta$，$\\theta=0$ 时（沿梯度方向）取最大值 $|\\nabla f|$。梯度方向是函数值增长最快的方向，负梯度方向是下降最快的方向（梯度下降法的依据）。',
      },
      // 题 5：choice — 链式法则
      {
        id: 'c21-b-q5',
        type: 'choice',
        prompt:
          '设 $z = f(u,v)$，$u = x^2$，$v = \\sin x$，则 $\\dfrac{dz}{dx}$ 等于？',
        options: [
          '$f_u \\cdot 2x + f_v \\cdot \\cos x$',
          '$f_u \\cdot 2x$',
          '$f_u + f_v$',
          '$f_u \\cdot x + f_v \\cdot \\sin x$',
        ],
        answer: 0,
        explain:
          '链式法则："有几条路，加几项"。$z$ 通过 $u$ 和 $v$ 依赖 $x$，两条路径：路径一 $x\\to u\\to z$，贡献 $f_u\\cdot u\'=f_u\\cdot2x$；路径二 $x\\to v\\to z$，贡献 $f_v\\cdot v\'=f_v\\cdot\\cos x$。总计 $dz/dx = f_u\\cdot2x + f_v\\cdot\\cos x$。',
      },
      // 题 6：input — 隐函数求导
      {
        id: 'c21-b-q6',
        type: 'input',
        prompt:
          '设 $x^2 + y^2 - 4 = 0$ 确定隐函数 $y=y(x)$，在点 $(1, \\sqrt{3})$ 处 $dy/dx$ 等于多少？（结果用分数或小数表示，保留合理精度）',
        accept: ['-1/sqrt(3)', '-\\sqrt{3}/3', '-0.577', '-1/1.732'],
        explain:
          '令 $F=x^2+y^2-4$，$F_x=2x$，$F_y=2y$。$dy/dx = -F_x/F_y = -x/y$。代入 $(1,\\sqrt{3})$：$dy/dx = -1/\\sqrt{3} = -\\sqrt{3}/3 \\approx -0.577$。',
      },
      // 题 7：match — 梯度几何含义
      {
        id: 'c21-b-q7',
        type: 'match',
        prompt: '将梯度与方向的关系和对应的方向导数大小对应。',
        left: ['沿梯度 $\\nabla f$ 方向', '沿负梯度 $-\\nabla f$ 方向', '垂直梯度方向（沿等高线）'],
        right: ['方向导数最大，值为 $|\\nabla f|$', '方向导数最小，值为 $-|\\nabla f|$', '方向导数为 $0$'],
      },
      // 题 8：judge — 隐函数公式
      {
        id: 'c21-b-q8',
        type: 'judge',
        prompt:
          '对方程 $F(x,y)=0$ 确定的隐函数，有 $dy/dx = -F_y/F_x$（分子是 $F_y$，分母是 $F_x$）。',
        answer: false,
        explain:
          '正确公式是 $dy/dx = -F_x/F_y$（分子是 $F_x$，分母是 $F_y$）。记忆口诀："谁是因变量，谁在分母"——$y$ 是因变量，$F_y$ 在分母。',
      },
    ],
  },

  // ─────────────────────────────────────────────────────────────────────────────
  // 关 c21-c  多元函数的极值与拉格朗日乘数法
  // ─────────────────────────────────────────────────────────────────────────────
  {
    id: 'c21-c',
    title: '多元函数的极值与条件极值',
    subtitle: 'Hessian 判别·驻点分类·拉格朗日乘数法',
    intro: [
      // 卡 1：动机——从山峰到损失曲面
      {
        title: '为什么研究多元极值？',
        body:
          '一元函数求极值：令 $f\'(x)=0$ 找驻点，再用 $f\'\'$ 判断。\n\n' +
          '多元函数有更丰富的结构。曲面 $z=f(x,y)$ 的临界点可能是：\n\n' +
          '- **极小值点（谷底）**：沿任意方向走都会上升\n' +
          '- **极大值点（山顶）**：沿任意方向走都会下降\n' +
          '- **鞍点（马鞍）**：某方向上升，某方向下降\n\n' +
          '深度学习中，损失曲面的鞍点比极小值点多得多——这是优化研究的重要课题。',
        tip: '把 $f(x,y)$ 的图像想象成地图上的地形：驻点就是"水平切平面"处，可能是山顶、谷底、也可能是鞍形山口。',
      },
      // 卡 2：必要条件——驻点
      {
        title: '极值的必要条件：驻点',
        body:
          '若 $f(x_0,y_0)$ 是极值（且偏导存在），则两个偏导数同时为零：\n\n' +
          '$f_x(x_0,y_0) = 0$ 且 $f_y(x_0,y_0) = 0$\n\n' +
          '满足这两个条件的点 $(x_0,y_0)$ 称为**驻点**（稳定点）。\n\n' +
          '**几何理解**：驻点处切平面水平（平行于 $xOy$ 平面），就像山顶或谷底——地势"暂时不变"。\n\n' +
          '**注意**：驻点不一定是极值点！$f(x,y)=x^3$ 的类似结构在二元情形给出鞍点。\n\n' +
          '**求驻点步骤**：联立 $f_x=0$ 和 $f_y=0$，解这个方程组。',
        formula:
          '\\text{极值点}\\;(x_0,y_0) \\Rightarrow f_x(x_0,y_0)=0 \\;\\text{且}\\; f_y(x_0,y_0)=0',
        tip: '驻点是"候选极值点"，还需要充分条件（Hessian）来最终判断类型。',
      },
      // 卡 3：充分条件——Hessian 判别法
      {
        title: 'Hessian 判别法：确定极值类型',
        body:
          '在驻点 $(x_0,y_0)$ 处，计算三个二阶偏导数：\n\n' +
          '$A = f_{xx}(x_0,y_0)$，$B = f_{xy}(x_0,y_0)$，$C = f_{yy}(x_0,y_0)$\n\n' +
          '令 $\\Delta = AC - B^2$（Hessian 矩阵的行列式）：\n\n' +
          '- $\\Delta > 0$ 且 $A > 0$：**极小值点**（碗口朝上）\n' +
          '- $\\Delta > 0$ 且 $A < 0$：**极大值点**（碗口朝下）\n' +
          '- $\\Delta < 0$：**鞍点**（马鞍形，不是极值）\n' +
          '- $\\Delta = 0$：**无法判断**，需要更高阶分析\n\n' +
          '**物理直觉**：$\\Delta>0$ 表示 Hessian 正定/负定，曲面在所有方向同侧弯曲（碗形）；$\\Delta<0$ 表示 Hessian 不定，不同方向弯曲方向相反（马鞍形）。',
        formula:
          '\\Delta = AC-B^2 = f_{xx}f_{yy} - f_{xy}^2;\\quad \\Delta>0,A>0\\Rightarrow\\text{极小};\\quad \\Delta<0\\Rightarrow\\text{鞍点}',
      },
      // 卡 4：完整例题——找极值
      {
        title: '例题：找驻点并判断类型',
        body:
          '设 $f(x,y) = x^3 + y^3 - 3xy$，求极值。\n\n' +
          '**第一步：找驻点**\n\n' +
          '$f_x = 3x^2-3y=0 \\Rightarrow y=x^2$\n\n' +
          '$f_y = 3y^2-3x=0 \\Rightarrow y^2=x$\n\n' +
          '由 $y=x^2$ 代入 $y^2=x$：$x^4=x$，$x(x^3-1)=0$，解得 $x=0$ 或 $x=1$。\n\n' +
          '驻点：$(0,0)$ 和 $(1,1)$。\n\n' +
          '**第二步：计算二阶偏导**：$A=f_{xx}=6x$，$B=f_{xy}=-3$，$C=f_{yy}=6y$\n\n' +
          '- 在 $(0,0)$：$A=0$，$B=-3$，$C=0$，$\\Delta=0-9=-9<0$，**鞍点**\n' +
          '- 在 $(1,1)$：$A=6>0$，$B=-3$，$C=6$，$\\Delta=36-9=27>0$，**极小值点**\n\n' +
          '极小值 $f(1,1) = 1+1-3 = -1$。',
        formula:
          'f(1,1) = 1+1-3(-1)(1) = \\cdots \\text{ 实际代入：} 1+1-3 = -1',
      },
      // 卡 5：条件极值——为什么需要约束？
      {
        title: '条件极值：带约束的最优化',
        body:
          '许多实际问题要在**满足约束**的前提下求最值，例如：\n\n' +
          '- 周长固定为 $4$，面积最大的矩形是什么形状？\n' +
          '- 在曲面上，距某点最近的点在哪里？\n\n' +
          '这类问题叫**条件极值**：在约束 $g(x,y)=0$ 下，求 $f(x,y)$ 的极值。\n\n' +
          '**几何视角**：$f$ 的等高线是一族曲线，约束 $g=0$ 是另一条曲线。极值发生在等高线与约束曲线**相切**时（否则沿约束曲线移动可以让 $f$ 继续变大或变小）。\n\n' +
          '相切意味着两条曲线的法向量平行，即梯度平行：$\\nabla f = \\lambda \\nabla g$。',
        tip: '极值条件 $\\nabla f = \\lambda \\nabla g$ 的几何含义：$f$ 的等高线与约束曲线 $g=0$ 在极值点处相切。',
      },
      // 卡 6：拉格朗日乘数法
      {
        title: '拉格朗日乘数法：构造 $L = f + \\lambda g$',
        body:
          '**方法**：构造拉格朗日函数 $L(x,y,\\lambda) = f(x,y) + \\lambda g(x,y)$，令 $L_x=L_y=L_\\lambda=0$：\n\n' +
          '$\\begin{cases} f_x + \\lambda g_x = 0 \\\\ f_y + \\lambda g_y = 0 \\\\ g(x,y) = 0 \\end{cases}$\n\n' +
          '解这个方程组，得到候选极值点。\n\n' +
          '**例**：在约束 $x+y=1$ 下，求 $f=x^2+y^2$ 的极小值。\n\n' +
          '$L=x^2+y^2+\\lambda(x+y-1)$。$L_x=2x+\\lambda=0$，$L_y=2y+\\lambda=0$，$L_\\lambda=x+y-1=0$。\n\n' +
          '由前两式：$x=y=-\\lambda/2$。代入约束：$2x=1$，$x=y=1/2$，$\\lambda=-1$。\n\n' +
          '极小值 $f(1/2,1/2) = 1/4+1/4 = 1/2$。\n\n' +
          '几何验证：圆心在原点的圆 $x^2+y^2=r^2$，与直线 $x+y=1$ 相切，切点在 $(1/2,1/2)$ 处，此时圆的半径最小，$r^2=1/2$。',
        formula:
          'L(x,y,\\lambda) = f(x,y)+\\lambda g(x,y);\\quad \\nabla_x L=0,\\;\\nabla_y L=0,\\; g=0',
        reveal: {
          q: '在约束 $x^2+y^2=1$ 下，求 $f(x,y)=x+2y$ 的最大值。',
          a: '$L=x+2y+\\lambda(x^2+y^2-1)$，$L_x=1+2\\lambda x=0$，$L_y=2+2\\lambda y=0$，得 $x=-1/(2\\lambda)$，$y=-1/\\lambda$。代入约束：$1/(4\\lambda^2)+1/\\lambda^2=1$，$5/(4\\lambda^2)=1$，$\\lambda=\\pm\\sqrt{5}/2$。取 $\\lambda=-\\sqrt{5}/2$，$x=1/\\sqrt{5}$，$y=2/\\sqrt{5}$，$f_{max}=\\sqrt{5}$。',
        },
      },
      // 卡 7：闭区域上的最值——三步法
      {
        title: '闭区域最值：内部驻点 + 边界极值',
        body:
          '有界闭区域上的连续函数必取最大值和最小值（有界闭集上的连续函数定理）。\n\n' +
          '**求法三步**：\n\n' +
          '① 求内部驻点（令 $f_x=f_y=0$），计算这些点的函数值\n\n' +
          '② 求边界上的极值（边界是曲线，用参数化或条件极值方法转为一元问题）\n\n' +
          '③ 比较所有函数值，最大者是最大值，最小者是最小值\n\n' +
          '**例**：$f(x,y)=x^2+y^2-x$ 在圆盘 $x^2+y^2 \\leq 4$ 上的最值。\n\n' +
          '内部驻点：$f_x=2x-1=0$，$f_y=2y=0$，驻点 $(1/2,0)$，值 $f=-1/4$。\n\n' +
          '边界 $x^2+y^2=4$，参数化 $x=2\\cos t$，$y=2\\sin t$：$f=4-2\\cos t$，最小 $2$（$t=0$），最大 $6$（$t=\\pi$）。\n\n' +
          '比较：$-1/4 < 2 < 6$，最大值 $6$，最小值 $-1/4$。',
      },
      // 卡 8：易错点汇总
      {
        title: '易错点汇总与速查',
        body:
          '**易错点一**：忘记验证 $\\Delta\\neq0$ 才能用 Hessian 判别。$\\Delta=0$ 时判别法失效，要用高阶分析或几何直觉。\n\n' +
          '**易错点二**：驻点 $\\neq$ 极值点。必须用 Hessian 判别，$\\Delta<0$ 是鞍点（不是极值）。\n\n' +
          '**易错点三**：拉格朗日乘数法只给出候选极值点，不能直接判断是极大还是极小——需结合实际背景或二阶条件判断。\n\n' +
          '**速查表**：\n' +
          '- 求驻点：令 $f_x=f_y=0$，解方程组\n' +
          '- 判类型：$\\Delta=f_{xx}f_{yy}-f_{xy}^2$；$\\Delta>0,A>0\\to$极小；$\\Delta>0,A<0\\to$极大；$\\Delta<0\\to$鞍点\n' +
          '- 条件极值：拉格朗日函数 $L=f+\\lambda g$，解 $\\nabla L=0$',
      },
    ],
    questions: [
      // 题 1：choice — 找驻点
      {
        id: 'c21-c-q1',
        type: 'choice',
        prompt: '函数 $f(x,y) = x^2 + y^2 - 2x - 4y + 8$ 的驻点是？',
        options: ['$(0,0)$', '$(1,2)$', '$(2,4)$', '$(1,0)$'],
        answer: 1,
        explain:
          '$f_x = 2x-2=0 \\Rightarrow x=1$；$f_y = 2y-4=0 \\Rightarrow y=2$。驻点为 $(1,2)$。',
      },
      // 题 2：input — Hessian 判别
      {
        id: 'c21-c-q2',
        type: 'input',
        prompt:
          '在驻点 $(1,2)$ 处，$f(x,y) = x^2+y^2-2x-4y+8$ 的 Hessian 判别式 $\\Delta = AC-B^2$ 等于多少？（$A=f_{xx}$，$B=f_{xy}$，$C=f_{yy}$）',
        accept: ['4'],
        explain:
          '$f_{xx}=2=A$，$f_{xy}=0=B$，$f_{yy}=2=C$。$\\Delta = 2\\times2 - 0^2 = 4$。',
      },
      // 题 3：judge — 鞍点判断
      {
        id: 'c21-c-q3',
        type: 'judge',
        prompt: '若在驻点处 $\\Delta = f_{xx}f_{yy}-f_{xy}^2 < 0$，则该点一定是鞍点（不是极值点）。',
        answer: true,
        explain:
          '$\\Delta<0$ 表示 Hessian 矩阵不定（既有正特征值又有负特征值），曲面在不同方向的弯曲方向相反，形成马鞍形。这种点不是极大也不是极小，而是鞍点。',
      },
      // 题 4：choice — 极值类型判断
      {
        id: 'c21-c-q4',
        type: 'choice',
        prompt:
          '在驻点 $(0,0)$ 处，$f=xy-x^2-y^2$ 满足 $A=f_{xx}=-2$，$B=f_{xy}=1$，$C=f_{yy}=-2$，$\\Delta=AC-B^2=3>0$，$A=-2<0$。该点是？',
        options: ['极小值点', '极大值点', '鞍点', '无法判断'],
        answer: 1,
        explain:
          '$\\Delta = (-2)(-2)-1^2 = 4-1=3>0$，且 $A=-2<0$，满足"$\\Delta>0$ 且 $A<0$"，结论是**极大值点**。极大值 $f(0,0)=0$。',
      },
      // 题 5：input — 条件极值
      {
        id: 'c21-c-q5',
        type: 'input',
        prompt:
          '在约束 $x+y=1$ 下，$f(x,y)=x^2+y^2$ 的极小值是多少？',
        accept: ['1/2', '0.5'],
        explain:
          '拉格朗日法得 $x=y=1/2$，极小值 $f(1/2,1/2) = 1/4+1/4 = 1/2$。几何上，以原点为圆心的圆与直线 $x+y=1$ 相切于 $(1/2,1/2)$，最小圆半径平方为 $1/2$。',
      },
      // 题 6：match — 驻点类型判断规则
      {
        id: 'c21-c-q6',
        type: 'match',
        prompt: '将 Hessian 判别条件与驻点类型对应。',
        left: ['$\\Delta>0$，$A>0$', '$\\Delta>0$，$A<0$', '$\\Delta<0$'],
        right: ['极小值点', '极大值点', '鞍点'],
      },
      // 题 7：judge — 驻点与极值
      {
        id: 'c21-c-q7',
        type: 'judge',
        prompt: '二元函数的每个驻点都是极值点。',
        answer: false,
        explain:
          '驻点只是极值点的必要条件，不是充分条件。驻点可能是极大、极小，也可能是鞍点（非极值点）。判断需要用 Hessian 判别法：$\\Delta<0$ 时驻点是鞍点，不是极值。',
      },
      // 题 8：choice — 拉格朗日乘数法方程
      {
        id: 'c21-c-q8',
        type: 'choice',
        prompt:
          '在约束 $g(x,y)=x^2+y^2-1=0$ 下求 $f(x,y)=x+2y$ 极值，拉格朗日函数 $L=f+\\lambda g$ 对 $y$ 求导令为零，得到的方程是？',
        options: [
          '$2+2\\lambda y=0$',
          '$1+2\\lambda y=0$',
          '$2+\\lambda=0$',
          '$2\\lambda y=0$',
        ],
        answer: 0,
        explain:
          '$L = x+2y+\\lambda(x^2+y^2-1)$，对 $y$ 求偏导：$L_y = 2+2\\lambda y=0$。（$f_y=2$，$g_y=2y$，故 $L_y=2+\\lambda\\cdot2y=0$。）',
      },
    ],
  },
];
