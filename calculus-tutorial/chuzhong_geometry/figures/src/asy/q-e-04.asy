// E.4 y=-x^2+2x+3. A(-1,0), B(3,0), C(0,3). 对称轴 x=1.
// P 在第一象限抛物线上, PD⊥x 轴交 BC 于 E. 示例 P(1.5, 3.75): D(1.5,0)
// 直线 BC: y = -x+3, E(1.5, 1.5)
size(10cm);
import graph;

real f(real x) { return -x*x + 2*x + 3; }

draw((-2,0)--(4,0), arrow=Arrow(TeXHead));
draw((0,-1)--(0,5), arrow=Arrow(TeXHead));
label("$x$", (4,0), E);
label("$y$", (0,5), N);
label("$O$", (0,0), SW);

draw(graph(f, -1.3, 3.3), blue);

pair A = (-1,0);
pair B = (3,0);
pair C = (0,3);
pair P = (1.5, 3.75);
pair D = (1.5, 0);
pair E = (1.5, 1.5);

draw(B--C, black);
draw(P--D, red+dashed);

dot(A); label("$A$", A, SW);
dot(B); label("$B$", B, SE);
dot(C); label("$C$", C, NW);
dot(P); label("$P$", P, NE);
dot(D); label("$D$", D, S);
dot(E); label("$E$", E, NE);

// 对称轴 x=1
draw((1,-0.5)--(1,4.5), gray+dashed);
label("$x=1$", (1,4.5), NE, fontsize(8));
