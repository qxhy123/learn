// E.11 y=ax²+bx+3, A(-1,0), B(3,0). 代入: a-b+3=0, 9a+3b+3=0 → a=-1, b=2.
// y=-x²+2x+3. C(0,3). 对称轴 x=1.
size(10cm);
import graph;

real f(real x) { return -x*x + 2*x + 3; }

draw((-2,0)--(4,0), arrow=Arrow(TeXHead));
draw((0,-1)--(0,5), arrow=Arrow(TeXHead));
label("$x$", (4,0), E);
label("$y$", (0,5), N);
label("$O$", (0,0), SW);

draw(graph(f, -1.4, 3.4), blue);

pair A = (-1,0);
pair B = (3,0);
pair C = (0,3);

draw(B--C, black);
draw(A--C, gray);
draw((1,-0.5)--(1,4.5), gray+dashed);

dot(A); label("$A$", A, SW);
dot(B); label("$B$", B, SE);
dot(C); label("$C$", C, NW);
label("$x=1$", (1,4.5), NE, fontsize(8));
