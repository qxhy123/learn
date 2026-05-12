// p10-02 例 2：B(3,0) C(0,-3)，x 轴上 Q 使 △BCQ 等腰，4 个 Q
// Q1=(0,0), Q2=(3+3√2,0)≈(7.24,0), Q3=(3-3√2,0)≈(-1.24,0), Q4=(-3,0)
size(11cm);
import graph;

draw((-5,0)--(8,0), arrow=Arrow(TeXHead));
draw((0,-4)--(0,1.5), arrow=Arrow(TeXHead));
label("$x$", (8,0), E);
label("$y$", (0,1.5), N);
label("$O$", (0,0), NE);

pair B = (3,0);
pair C = (0,-3);
pair Q1 = (0,0);
pair Q2 = (3 + 3*sqrt(2), 0);
pair Q3 = (3 - 3*sqrt(2), 0);
pair Q4 = (-3, 0);

// BC 主线
draw(B--C, black+linewidth(1pt));

// 每个 Q 与 B, C 连线（淡色，避免拥挤）
draw(Q2--B, red+dashed); draw(Q2--C, red+dashed);
draw(Q3--B, red+dashed); draw(Q3--C, red+dashed);
draw(Q4--B, red+dashed); draw(Q4--C, red+dashed);
draw(Q1--C, red+dashed);
// Q1=O 与 B 在 x 轴上，QB=3 这条与 x 轴重合

dot(B); label("$B(3,0)$", B, N);
dot(C); label("$C(0,-3)$", C, W);
dot(Q1); label("$Q_1$", Q1, NW);
dot(Q2); label("$Q_2$", Q2, N);
dot(Q3); label("$Q_3$", Q3, N);
dot(Q4); label("$Q_4$", Q4, N);
