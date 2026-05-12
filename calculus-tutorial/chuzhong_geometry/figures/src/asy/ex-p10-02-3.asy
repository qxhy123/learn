// p10-02 例 3：A(-1,0) B(3,0) C(0,3)；P 在抛物线 y=-x^2+2x+3，Q 在 x 轴。
// 平行四边形 ACPQ 共 4 组解。
// 配对甲: P=(2,3), Q=(-3,0)
// 配对乙: P=(2,3), Q=(1,0)
// 配对丙(2): P=(1+√7,-2)≈(3.65,-2), Q=(√7,0)≈(2.65,0); P=(1-√7,-2)≈(-1.65,-2), Q=(-√7,0)≈(-2.65,0)
size(11cm);
import graph;

real f(real x) { return -x*x + 2*x + 3; }

draw((-3.5,0)--(4.5,0), arrow=Arrow(TeXHead));
draw((0,-3)--(0,4.2), arrow=Arrow(TeXHead));
label("$x$", (4.5,0), E);
label("$y$", (0,4.2), N);
label("$O$", (0,0), SW);

draw(graph(f, -1.9, 3.9), blue);

pair A = (-1,0);
pair B = (3,0);
pair C = (0,3);
pair P_a = (2, 3);
pair Q_a = (-3, 0);
pair P_b = (2, 3);   // 同 P
pair Q_b = (1, 0);
pair P_c1 = (1 + sqrt(7), -2);
pair Q_c1 = (sqrt(7), 0);
pair P_c2 = (1 - sqrt(7), -2);
pair Q_c2 = (-sqrt(7), 0);

dot(A); label("$A$", A, NE);
dot(B); label("$B$", B, NE);
dot(C); label("$C$", C, NW);

// 配对甲：A, C, P, Q (AC, PQ 对角线)
draw(A--C--P_a--Q_a--cycle, red+dashed);
dot(P_a); label("$P_{1,2}$", P_a, N);
dot(Q_a); label("$Q_1$", Q_a, S);

// 配对乙：A, P, C, Q (AP, CQ 对角线)
draw(A--P_b--C--Q_b--cycle, magenta+dashed);
dot(Q_b); label("$Q_2$", Q_b, S);

// 配对丙
draw(A--Q_c1--C--P_c1--cycle, deepgreen+dashed);
draw(A--Q_c2--C--P_c2--cycle, deepgreen+dashed);
dot(P_c1); label("$P_3$", P_c1, S);
dot(Q_c1); label("$Q_3$", Q_c1, N);
dot(P_c2); label("$P_4$", P_c2, S);
dot(Q_c2); label("$Q_4$", Q_c2, N);
