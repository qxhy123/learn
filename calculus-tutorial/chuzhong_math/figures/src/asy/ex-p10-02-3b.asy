// p10-02 例 3 情形 B（配对乙）：AP 与 CQ 为对角线
// P=(2,3), Q=(1,0)
settings.tex = "xelatex";
texpreamble("\usepackage{ctex}");
size(9cm);
import graph;

real f(real x) { return -x*x + 2*x + 3; }

draw((-2.5,0)--(4,0), arrow=Arrow(TeXHead));
draw((0,-1)--(0,4.2), arrow=Arrow(TeXHead));
label("$x$", (4,0), E);
label("$y$", (0,4.2), N);
label("$O$", (0,0), SW);

draw(graph(f, -1.5, 3.5), blue);

pair A = (-1,0);
pair B = (3,0);
pair C = (0,3);
pair P = (2, 3);
pair Q = (1, 0);

dot(A); label("$A$", A, SW);
dot(B); label("$B$", B, SE);
dot(C); label("$C$", C, NW);
dot(P); label("$P$", P, NE);
dot(Q); label("$Q$", Q, S);

// 顶点环序 A→C→P→Q（A 与 P 相对，C 与 Q 相对，故 AP、CQ 为对角线）
draw(A--C--P--Q--cycle, magenta+linewidth(1pt));
// 对角线 AP, CQ
draw(A--P, gray+dashed);
draw(C--Q, gray+dashed);

label("情形 B：$AP, CQ$ 为对角线", (0.5, -0.7), S, fontsize(9pt));
