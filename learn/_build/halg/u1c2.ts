export const LESSONS = [
  {
    id: 'u1c2',
    title: '集合的基本运算',
    subtitle: '交集、并集与补集',
    intro: [
      {
        title: '为什么需要集合运算？',
        body: '单个集合只能描述"哪些对象在一起"，现实问题往往需要组合或比较两个集合：两个班同时喜欢某项目的人（公共部分）、至少满足一个条件的学生（合并）、全体学生中不满足某条件的人（排除）。这三类需求分别对应交集、并集、补集三种运算。',
      },
      {
        title: '交集：同时属于两个集合',
        body: '交集 $A \\cap B$ 是同时属于 $A$ 和 $B$ 的所有元素构成的集合。关键词是"且""同时""公共"。\n例：$A = \\{1,2,3\\}$，$B = \\{2,3,4\\}$，则 $A \\cap B = \\{2,3\\}$。\n重要性质：$A \\cap B \\subseteq A$，$A \\cap B \\subseteq B$；$A \\cap \\varnothing = \\varnothing$；$A \\cap A = A$；$A \\cap B = B \\cap A$（交换律）。',
        formula: 'A \\cap B = \\{x \\mid x \\in A \\text{ 且 } x \\in B\\}',
      },
      {
        title: '并集：属于两者之一即可',
        body: '并集 $A \\cup B$ 是属于 $A$ 或属于 $B$（含两者都属于）的所有元素构成的集合。关键词是"或""至少一个""合并"。注意：这里的"或"是数学意义上的相容或，两者都满足也包含在内。\n例：$A = \\{1,2,3\\}$，$B = \\{2,3,4\\}$，则 $A \\cup B = \\{1,2,3,4\\}$（公共元素不重复写）。\n重要性质：$A \\subseteq A \\cup B$；$A \\cup \\varnothing = A$；$A \\cup A = A$；$A \\cup B = B \\cup A$。',
        formula: 'A \\cup B = \\{x \\mid x \\in A \\text{ 或 } x \\in B\\}',
        tip: '并集结果中元素不能重复！$\\{1,2,3\\} \\cup \\{2,3,4\\} = \\{1,2,3,4\\}$，不要写成 $\\{1,2,3,2,3,4\\}$。',
      },
      {
        title: '补集：全集中排除某集合',
        body: '讨论补集必须先指明全集 $U$（问题范围内最大的集合）。$A$ 在 $U$ 中的补集 $\\complement_U A$ 是 $U$ 中所有不属于 $A$ 的元素构成的集合。\n例：$U = \\{1,2,3,4,5\\}$，$A = \\{1,2,3\\}$，则 $\\complement_U A = \\{4,5\\}$。\n重要性质：$A \\cup \\complement_U A = U$；$A \\cap \\complement_U A = \\varnothing$；$\\complement_U(\\complement_U A) = A$（双补律）；$\\complement_U U = \\varnothing$；$\\complement_U \\varnothing = U$。',
        formula: '\\complement_U A = \\{x \\mid x \\in U \\text{ 且 } x \\notin A\\}',
        tip: '没有明确全集就写补集，是本章最常见的失分点。全集不同，同一个集合的补集也不同。',
      },
      {
        title: '德摩根律：补集与交并的桥梁',
        body: '德摩根律是高考重点，连接了补集与交、并运算：\n并的补等于补的交：$\\complement_U(A \\cup B) = \\complement_U A \\cap \\complement_U B$\n交的补等于补的并：$\\complement_U(A \\cap B) = \\complement_U A \\cup \\complement_U B$\n口诀：补集分配时，并变交，交变并。\n直觉：$A \\cup B$ 的补集是"既不在 $A$ 又不在 $B$ 里"的部分，即同时属于 $\\complement_U A$ 和 $\\complement_U B$，所以是它们的交集。',
        formula: '\\complement_U(A \\cup B) = \\complement_U A \\cap \\complement_U B,\\quad \\complement_U(A \\cap B) = \\complement_U A \\cup \\complement_U B',
      },
      {
        title: '综合例题：区间集合的运算',
        body: '设全集 $U = \\mathbb{R}$，$A = \\{x \\mid x < -1 \\text{ 或 } x > 2\\}$，$B = \\{x \\mid -2 < x < 3\\}$，求 $\\complement_U(A \\cup B)$。\n方法一（直接法）：$A = (-\\infty,-1) \\cup (2,+\\infty)$，$B = (-2,3)$。合并后 $A \\cup B$ 覆盖了整个实数轴，故 $A \\cup B = \\mathbb{R}$，$\\complement_U(A \\cup B) = \\varnothing$。\n方法二（德摩根律）：$\\complement_U A = [-1,2]$，$\\complement_U B = (-\\infty,-2] \\cup [3,+\\infty)$，两者无公共部分，交集为 $\\varnothing$。两种方法结果一致。',
        tip: '做区间运算时建议在数轴上画出各集合，直观避免遗漏端点或方向的错误。',
        reveal: {
          q: '已知 $U = \\{1,2,3,4,5,6\\}$，$A = \\{1,3,5\\}$，$B = \\{2,3,4\\}$，求 $A \\cap B$、$A \\cup B$、$\\complement_U A$。',
          a: '$A \\cap B = \\{3\\}$；$A \\cup B = \\{1,2,3,4,5\\}$；$\\complement_U A = \\{2,4,6\\}$。',
        },
      },
    ],
    questions: [
      {
        type: 'choice',
        prompt: '设 $A = \\{1,2,3,4\\}$，$B = \\{3,4,5,6\\}$，则 $A \\cap B$ 等于？',
        options: [
          '\\{1,2,3,4,5,6\\}',
          '\\{3,4\\}',
          '\\{1,2\\}',
          '\\{5,6\\}',
        ],
        answer: 1,
        explain: '交集取两个集合的公共元素。$A$ 和 $B$ 的公共元素是 3 和 4，故 $A \\cap B = \\{3,4\\}$。',
      },
      {
        type: 'judge',
        prompt: '若全集 $U = \\{1,2,3,4,5\\}$，$A = \\{1,2,3\\}$，则 $\\complement_U A = \\{4,5\\}$。',
        answer: true,
        explain: '正确。补集是全集中所有不属于 $A$ 的元素，$U$ 中除去 1、2、3 剩下 4 和 5，故 $\\complement_U A = \\{4,5\\}$。',
      },
      {
        type: 'choice',
        prompt: '设全集 $U = \\mathbb{R}$，$A = \\{x \\mid x \\leq -1 \\text{ 或 } x \\geq 3\\}$，则 $\\complement_U A$ 用区间表示是？',
        options: [
          '(-\\infty,-1] \\cup [3,+\\infty)',
          '(-1,3)',
          '[-1,3]',
          '(-\\infty,3)',
        ],
        answer: 1,
        explain: '$A$ 是"$x \\leq -1$ 或 $x \\geq 3$"，其补集是"$-1 < x < 3$"，即开区间 $(-1,3)$。注意端点：$A$ 包含 $-1$ 和 $3$（$\\leq$ 和 $\\geq$），所以补集不含这两个端点。',
      },
      {
        type: 'choice',
        prompt: '关于德摩根律，下列哪个等式正确？',
        options: [
          '\\complement_U(A \\cup B) = \\complement_U A \\cup \\complement_U B',
          '\\complement_U(A \\cap B) = \\complement_U A \\cap \\complement_U B',
          '\\complement_U(A \\cup B) = \\complement_U A \\cap \\complement_U B',
          '\\complement_U(A \\cup B) = A \\cap B',
        ],
        answer: 2,
        explain: '德摩根律：并的补等于补的交，即 $\\complement_U(A \\cup B) = \\complement_U A \\cap \\complement_U B$。口诀：补集分配时，并变交，交变并。',
      },
      {
        type: 'input',
        prompt: '已知 $U = \\{1,2,3,4,5,6\\}$，$\\complement_U A = \\{2,4,6\\}$，$\\complement_U B = \\{1,3,5\\}$。先还原 $A$、$B$，再求 $A \\cap B$（用列举法，如 $\\{1,3\\}$）。',
        accept: ['{1,3,5}', '{5,3,1}', '{3,5,1}', '{1,5,3}', '{3,1,5}', '{5,1,3}'],
        explain: '$A = U \\setminus \\{2,4,6\\} = \\{1,3,5\\}$，$B = U \\setminus \\{1,3,5\\} = \\{2,4,6\\}$。$A \\cap B = \\{1,3,5\\} \\cap \\{2,4,6\\} = \\varnothing$。注意：答案是 $\\varnothing$，即空集，因为两个集合的元素完全不同。若接受 $\\varnothing$，答案为空集。',
      },
      {
        type: 'judge',
        prompt: '对任意集合 $A$，有 $A \\cup \\complement_U A = U$ 且 $A \\cap \\complement_U A = \\varnothing$。',
        answer: true,
        explain: '正确。$A$ 和它的补集 $\\complement_U A$ 刚好"平分"全集 $U$，两者合并等于 $U$，两者交集为空集（没有元素既属于 $A$ 又不属于 $A$）。',
      },
      {
        type: 'match',
        prompt: '将左侧运算结果与右侧答案配对（$A = \\{1,2,3\\}$，$B = \\{2,3,4\\}$，$U = \\{1,2,3,4,5\\}$）。',
        left: [
          'A \\cap B',
          'A \\cup B',
          '\\complement_U A',
          '\\complement_U(A \\cap B)',
        ],
        right: [
          '\\{2,3\\}',
          '\\{1,2,3,4\\}',
          '\\{4,5\\}',
          '\\{1,4,5\\}',
        ],
        explain: '$A \\cap B = \\{2,3\\}$（公共元素）；$A \\cup B = \\{1,2,3,4\\}$（合并不重复）；$\\complement_U A = \\{4,5\\}$（$U$ 中不属于 $A$ 的元素）；$\\complement_U(A \\cap B) = \\complement_U\\{2,3\\} = \\{1,4,5\\}$，也可用德摩根律验证：$\\complement_U A \\cup \\complement_U B = \\{4,5\\} \\cup \\{1,5\\} = \\{1,4,5\\}$。',
      },
      {
        type: 'choice',
        prompt: '设 $A = \\{x \\mid a \\leq x \\leq a+2\\}$，$B = \\{x \\mid 1 \\leq x \\leq 4\\}$，若 $A \\cap B = A$，则实数 $a$ 的范围是？',
        options: [
          '1 \\leq a \\leq 4',
          'a \\geq 1',
          '1 \\leq a \\leq 2',
          'a \\leq 2',
        ],
        answer: 2,
        explain: '$A \\cap B = A$ 等价于 $A \\subseteq B$，即 $A$ 的左端点 $\\geq B$ 的左端点且右端点 $\\leq B$ 的右端点：$a \\geq 1$ 且 $a+2 \\leq 4$，即 $a \\geq 1$ 且 $a \\leq 2$，故 $1 \\leq a \\leq 2$。',
      },
    ],
  },
]
