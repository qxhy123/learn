export const LESSONS = [
  {
    id: 'u9c3',
    title: '圆周角定理',
    subtitle: '圆周角是圆心角的一半·同弧等角·直径直角·对角互补',
    intro: [
      {
        title: '圆周角是什么',
        body:
          '**圆周角**：顶点在**圆上**、两边都是**弦**的角。设圆 $\\odot O$ 上有三点 $A, B, C$，则 $\\angle ACB$（顶点 $C$ 在圆上，两边 $CA, CB$ 都是弦）是圆周角，它**所对的弧**是 $\\overset{\\frown}{AB}$（不含 $C$ 那一侧）。\n\n**圆心角**：顶点在**圆心**、两边是半径的角 $\\angle AOB$。同一段弧 $\\overset{\\frown}{AB}$ 对应一个圆心角，却对应无数个圆周角（顶点 $C$ 在另一侧弧上滑动）。',
        image: 'sample-inscribed-angle.svg',
        imageCaption: '圆周角 $\\angle ACB$ 与圆心角 $\\angle AOB$ 对同一弧 $\\overset{\\frown}{AB}$',
      },
      {
        title: '圆周角定理',
        body:
          '**定理**：同一个圆中，**一条弧所对的圆周角等于它所对的圆心角的一半**，即 $\\angle ACB = \\frac{1}{2}\\angle AOB$。\n\n**证明思路（三种情形）**：根据圆心 $O$ 与角 $\\angle ACB$ 的相对位置分三种情形。每种情形都利用"$OA = OC = OB = r$ 构成等腰三角形 + 外角定理"。\n- 情形 1（$O$ 在一边上）：$\\angle AOB = \\angle OAC + \\angle OCA = 2\\angle ACB$。\n- 情形 2（$O$ 在角内）：连辅助线 $CO$ 延长，分成两个情形 1 相加。\n- 情形 3（$O$ 在角外）：同样连辅助线，两个情形 1 相减。',
        formula: '\\angle ACB = \\dfrac{1}{2}\\angle AOB',
        image: 'thm-p5-03-inscribed-cases.svg',
        imageCaption: '圆周角定理三种证明情形：圆心在角边上 / 角内 / 角外',
      },
      {
        title: '推论 1：同弧圆周角相等',
        body:
          '**推论 1**：同一个圆中，同一条弧所对的所有圆周角彼此相等；相等的弧所对的圆周角也相等。\n\n理由：同一段弧对应的圆心角是确定的，所有圆周角都等于它的一半，因此两两相等。\n\n**注意**：弦 $AB$ 把圆分成两段弧，两侧的圆周角并**不相等**，而是**互补**（两段弧之和是整圆 $360^\\circ$，两圆周角各取一半，和为 $180^\\circ$）。',
        tip: '看到"两个角顶点都在圆上、张同一条弦的两个端点"——立刻判断是同弧（相等）还是不同侧（互补）。',
      },
      {
        title: '推论 2：直径所对的圆周角是直角',
        body:
          '**推论 2**：直径所对的圆周角等于 $90^\\circ$；反之，圆周角为 $90^\\circ$ 时，它所对的弦是直径。\n\n理由：直径所对圆心角 $= 180^\\circ$（平角），圆周角 $= 90^\\circ$。\n\n推论：**直角三角形的外接圆圆心就是斜边的中点，斜边就是直径**。已知 $\\angle C = 90^\\circ$，则 $AB$ 是直径，外接圆半径 $= AB/2$。',
        formula: '\\angle ACB = 90^\\circ \\iff AB \\text{ 是直径}',
        image: 'ex-p5-03-2.svg',
        imageCaption: '直径 $AB = 10$，$AC = 6$，$BC = 8$，$\\angle ACB = 90^\\circ$',
      },
      {
        title: '推论 3：圆内接四边形对角互补',
        body:
          '**推论 3**：若四边形 $ABCD$ 四个顶点都在同一圆上（**圆内接四边形**），则对角互补：$\\angle A + \\angle C = 180^\\circ$，$\\angle B + \\angle D = 180^\\circ$。\n\n理由：$\\angle A$ 与 $\\angle C$ 分别对弧 $\\overset{\\frown}{BCD}$ 与 $\\overset{\\frown}{BAD}$，两弧合成整圆 $360^\\circ$，两圆周角之和 $= \\frac{1}{2}\\times 360^\\circ = 180^\\circ$。\n\n推论：圆内接四边形的**外角等于内对角**——这在综合题中极为常用。',
        formula: '\\angle A + \\angle C = 180^\\circ,\\quad \\angle B + \\angle D = 180^\\circ',
        image: 'ex-p5-03-3.svg',
        imageCaption: '圆内接四边形 $ABCD$：$\\angle A + \\angle C = 180^\\circ$',
      },
      {
        title: '例题：综合运用',
        body:
          '**例**：$\\triangle ABC$ 内接于圆 $O$，$AB$ 是直径，$AC = 6$，$BC = 8$。求半径 $r$。\n\n由推论 2，$\\angle ACB = 90^\\circ$，$\\triangle ACB$ 是直角三角形，直角顶点为 $C$。由勾股定理：$AB = \\sqrt{AC^2 + BC^2} = \\sqrt{36 + 64} = 10$。$AB$ 是直径，故 $r = AB/2 = 5$。',
        reveal: {
          q: '若反过来已知 $\\angle C = 90^\\circ$，能求外接圆半径吗？',
          a: '可以。由推论 2 的逆命题，$\\angle C = 90^\\circ$ 说明 $AB$ 是外接圆的直径，外接圆半径 $= AB/2$。这是直角三角形外接圆半径的通用结论。',
        },
      },
      {
        title: '易错点',
        body:
          '1. **圆周角顶点必须在圆上**：顶点在圆内或圆外，不是圆周角，不能用定理。\n2. **只有直径所对的圆周角是 $90^\\circ$**：一般弦所对的圆周角不是 $90^\\circ$。\n3. **同弧圆周角相等的前提是"在同侧"**：弦两侧的圆周角互补，而非相等。\n4. **二倍关系只在"对同一段弧"时成立**：圆心角与圆周角必须对同一段弧才能用。\n5. **圆内接四边形是"对角"互补**：邻角之和无固定关系，只有 $\\angle A$ 与 $\\angle C$、$\\angle B$ 与 $\\angle D$ 这两对互补。',
      },
    ],
    questions: [
      {
        type: 'input',
        prompt:
          '圆 $O$ 中，圆心角 $\\angle AOB = 120^\\circ$。求弧 $\\overset{\\frown}{AB}$ 所对圆周角 $\\angle ACB$ 的度数。',
        accept: ['60', '60°', '60度'],
        explain:
          '由圆周角定理：$\\angle ACB = \\dfrac{1}{2}\\angle AOB = \\dfrac{1}{2}\\times 120^\\circ = 60^\\circ$。',
      },
      {
        type: 'input',
        prompt:
          '$\\triangle ABC$ 内接于圆 $O$，$\\angle BAC = 30^\\circ$。求圆心角 $\\angle BOC$ 的度数。',
        accept: ['60', '60°', '60度'],
        explain:
          '$\\angle BAC$ 与 $\\angle BOC$ 对同一段弧 $\\overset{\\frown}{BC}$，由圆周角定理：$\\angle BOC = 2\\angle BAC = 60^\\circ$。',
      },
      {
        type: 'choice',
        prompt:
          '圆 $O$ 的直径 $AB = 10$，$C$ 是圆上一点，$AC = 6$。则 $BC$ 的长为',
        options: ['$6$', '$7$', '$8$', '$10$'],
        answer: 2,
        explain:
          '由推论 2，直径所对圆周角 $\\angle ACB = 90^\\circ$。在直角 $\\triangle ACB$ 中：$BC = \\sqrt{AB^2 - AC^2} = \\sqrt{100 - 36} = \\sqrt{64} = 8$。',
      },
      {
        type: 'judge',
        prompt:
          '圆内接四边形 $ABCD$ 中，$\\angle A = 75^\\circ$，$\\angle B = 95^\\circ$，则 $\\angle C = 105^\\circ$，$\\angle D = 85^\\circ$。',
        answer: true,
        explain:
          '由推论 3，对角互补：$\\angle C = 180^\\circ - \\angle A = 105^\\circ$，$\\angle D = 180^\\circ - \\angle B = 85^\\circ$。命题正确。',
      },
      {
        type: 'choice',
        prompt:
          '弦 $AB$ 把圆 $O$ 分成两段弧，$C$ 在其中一段弧上且 $\\angle ACB = 50^\\circ$。则 $D$ 在另一段弧上时，$\\angle ADB$ 等于',
        options: ['$50^\\circ$', '$90^\\circ$', '$130^\\circ$', '$180^\\circ$'],
        answer: 2,
        explain:
          '$\\angle ACB + \\angle ADB = 180^\\circ$（两角分别对两段弧，两段弧合成整圆，对应圆心角和为 $360^\\circ$，圆周角各取一半之和为 $180^\\circ$）。故 $\\angle ADB = 180^\\circ - 50^\\circ = 130^\\circ$。',
      },
      {
        type: 'input',
        prompt:
          '圆内接四边形 $ABCD$ 中，$\\angle A = 80^\\circ$，$\\angle B = 110^\\circ$。求 $\\angle D$ 的度数。',
        accept: ['70', '70°', '70度'],
        explain:
          '$\\angle B + \\angle D = 180^\\circ$，故 $\\angle D = 180^\\circ - 110^\\circ = 70^\\circ$。（同理 $\\angle C = 180^\\circ - 80^\\circ = 100^\\circ$）',
      },
      {
        type: 'match',
        prompt: '将圆周角定理的各推论与对应结论配对。',
        left: ['推论 1（同弧等角）', '推论 2（直径直角）', '推论 3（对角互补）'],
        right: [
          '圆内接四边形对角之和等于 $180^\\circ$',
          '同一段弧所对的所有圆周角相等',
          '直角三角形斜边是外接圆直径',
        ],
        explain:
          '推论 1：同弧所对圆周角相等；推论 2：直径所对圆周角 $= 90^\\circ$，等价于直角三角形斜边是外接圆直径；推论 3：圆内接四边形对角互补（之和为 $180^\\circ$）。',
      },
    ],
  },
]
