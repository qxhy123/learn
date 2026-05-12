// E.10 y=-x²+4x = -(x-2)²+4. 与 x 轴: O(0,0), A(4,0). 顶点 B(2,4).
size(10cm);
import graph;

real f(real x) { return -x*x + 4*x; }

draw((-1,0)--(5.5,0), arrow=Arrow(TeXHead));
draw((0,-1)--(0,5.5), arrow=Arrow(TeXHead));
label("$x$", (5.5,0), E);
label("$y$", (0,5.5), N);

draw(graph(f, -0.5, 4.5), blue);

pair O = (0,0);
pair A = (4,0);
pair B = (2,4);

draw((2,-0.5)--(2,5), gray+dashed);

dot(O); label("$O$", O, SW);
dot(A); label("$A$", A, SE);
dot(B); label("$B$", B, N);
label("$x=2$", (2,5), NE, fontsize(8));
