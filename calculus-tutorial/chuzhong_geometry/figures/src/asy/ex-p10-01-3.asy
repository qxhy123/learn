// p10-01 例 3：抛物线 y=x^2-4x+3 与 x 轴交 A(1,0), B(3,0)，与 y 轴交 C(0,3)
// P=(t,0) 在 AB 上，过 P 作 x 轴垂线交抛物线于 Q（在 x 轴下方）
// 取 t=2（最大值时）
size(8cm);
import graph;

real f(real x) { return x*x - 4*x + 3; }

// 坐标轴
draw((-0.7,0)--(4,0), arrow=Arrow(TeXHead));
draw((0,-1.4)--(0,3.5), arrow=Arrow(TeXHead));
label("$x$", (4,0), E);
label("$y$", (0,3.5), N);
label("$O$", (0,0), SW);

// 抛物线
draw(graph(f, -0.4, 3.8), blue);

pair A = (1,0);
pair B = (3,0);
pair C = (0,3);
pair P = (2,0);
pair Q = (2, f(2));  // (2,-1)

dot(A); label("$A$", A, N);
dot(B); label("$B$", B, NE);
dot(C); label("$C$", C, W);
dot(P); label("$P$", P, N);
dot(Q); label("$Q$", Q, S);

draw(P--Q, red+linewidth(1pt));
label("$PQ$", (P+Q)/2, E, red);

// P 在 AB 上的运动范围用粗线表示
draw(A--B, black+linewidth(1.2pt));
