// p10-02 例 3 情形 C（配对丙）：AQ 与 CP 为对角线，两组解
// 由 mid(AQ)=mid(CP) 解得 x = 1±√7, P=(x, -3), Q=(x+1, 0)
// 组1: P=(1+√7, -3) ≈ (3.65, -3), Q=(2+√7, 0) ≈ (4.65, 0)
// 组2: P=(1-√7, -3) ≈ (-1.65, -3), Q=(2-√7, 0) ≈ (-0.65, 0)
settings.tex = "xelatex";
texpreamble("\usepackage{ctex}");
size(12cm);
import graph;

real f(real x) { return -x*x + 2*x + 3; }

draw((-3,0)--(6,0), arrow=Arrow(TeXHead));
draw((0,-4)--(0,4.2), arrow=Arrow(TeXHead));
label("$x$", (6,0), E);
label("$y$", (0,4.2), N);
label("$O$", (0,0), NE);

draw(graph(f, -2.0, 4.0), blue);

pair A = (-1,0);
pair B = (3,0);
pair C = (0,3);
pair P1 = (1 + sqrt(7), -3);
pair Q1 = (2 + sqrt(7), 0);
pair P2 = (1 - sqrt(7), -3);
pair Q2 = (2 - sqrt(7), 0);

dot(A); label("$A$", A, NW);
dot(B); label("$B$", B, NE);
dot(C); label("$C$", C, NW);

// 组1 (绿)：AQ、CP 为对角线 → 顶点环序 A-C-Q-P
draw(A--C--Q1--P1--cycle, deepgreen+linewidth(1pt));
draw(A--Q1, gray+dashed);
draw(C--P1, gray+dashed);
dot(P1); label("$P_1$", P1, S);
dot(Q1); label("$Q_1$", Q1, NE);

// 组2 (橙)
draw(A--C--Q2--P2--cycle, orange+linewidth(1pt));
draw(A--Q2, gray+dashed);
draw(C--P2, gray+dashed);
dot(P2); label("$P_2$", P2, S);
dot(Q2); label("$Q_2$", Q2, N);

label("情形 C：$AQ, CP$ 为对角线（两组解）", (1.5, -3.7), S, fontsize(9pt));
