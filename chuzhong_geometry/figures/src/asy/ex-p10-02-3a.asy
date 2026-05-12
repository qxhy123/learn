// p10-02 例 3 情形 A（配对甲）：AC 与 PQ 为对角线
// P=(2,3), Q=(-3,0)
settings.tex = "xelatex";
texpreamble("\usepackage{ctex}");
size(9cm);
import graph;

real f(real x) { return -x*x + 2*x + 3; }

draw((-4,0)--(4,0), arrow=Arrow(TeXHead));
draw((0,-1)--(0,4.2), arrow=Arrow(TeXHead));
label("$x$", (4,0), E);
label("$y$", (0,4.2), N);
label("$O$", (0,0), SW);

draw(graph(f, -1.5, 3.5), blue);

pair A = (-1,0);
pair B = (3,0);
pair C = (0,3);
pair P = (2, 3);
pair Q = (-3, 0);

dot(A); label("$A$", A, S);
dot(B); label("$B$", B, SE);
dot(C); label("$C$", C, NW);
dot(P); label("$P$", P, N);
dot(Q); label("$Q$", Q, S);

// 平行四边形顶点环序 A→P→C→Q（AC、PQ 是对角线）
draw(A--P--C--Q--cycle, red+linewidth(1pt));
// 对角线
draw(A--C, gray+dashed);
draw(P--Q, gray+dashed);

label("情形 A：$AC, PQ$ 为对角线", (0, -0.7), S, fontsize(9pt));
