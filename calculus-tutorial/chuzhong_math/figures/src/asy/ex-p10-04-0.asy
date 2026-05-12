// p10-04 引入题：y=-x^2+2x+3，A(-1,0) B(3,0) C(0,3)；x 轴上 P 使 △APC 等腰
// 4 个解：P=(4,0), (-1+√10, 0)≈(2.16,0), (-1-√10,0)≈(-4.16,0), (1,0)
size(11cm);
import graph;

real f(real x) { return -x*x + 2*x + 3; }

draw((-5,0)--(5,0), arrow=Arrow(TeXHead));
draw((0,-0.5)--(0,4.5), arrow=Arrow(TeXHead));
label("$x$", (5,0), E);
label("$y$", (0,4.5), N);
label("$O$", (0,0), SW);

draw(graph(f, -1.6, 3.6), blue);

pair A = (-1,0);
pair B = (3,0);
pair C = (0,3);
pair P1 = (4, 0);
pair P2 = (-1 + sqrt(10), 0);
pair P3 = (-1 - sqrt(10), 0);
pair P4 = (1, 0);

dot(A); label("$A$", A, SW);
dot(B); label("$B$", B, NE);
dot(C); label("$C$", C, W);

// 用淡红画 △APC 的多种位置
draw(A--C, black+linewidth(1pt));
draw(A--P1--C--cycle, red+dashed);
draw(A--P2--C--cycle, red+dashed);
draw(A--P3--C--cycle, red+dashed);
draw(A--P4--C--cycle, red+dashed);

dot(P1); label("$P_1$", P1, S);
dot(P2); label("$P_2$", P2, S);
dot(P3); label("$P_3$", P3, S);
dot(P4); label("$P_4$", P4, S);
