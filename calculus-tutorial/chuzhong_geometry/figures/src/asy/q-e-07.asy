// E.7 ⊙O r=4. OA=8, OB=6, ∠AOB=90°. AB=10.
// O(0,0), A(8,0), B(0,6). P 在 ⊙O 上.
size(9cm);
import graph;

pair O = (0,0);
pair A = (8,0);
pair B = (0,6);
real r = 4;

// 示例 P
pair P = r*dir(60);

draw(circle(O, r), blue);
draw(A--B, gray+dashed);
draw(O--A, gray);
draw(O--B, gray);
draw(P--A, red);
draw(P--B, red);

// 直角符号
draw((0.4,0)--(0.4,0.4)--(0,0.4));

dot(O); label("$O$", O, SW);
dot(A); label("$A$", A, SE);
dot(B); label("$B$", B, NW);
dot(P); label("$P$", P, NE);

label("$8$", (4,0), S, fontsize(9));
label("$6$", (0,3), W, fontsize(9));
label("$r=4$", r*dir(140)+0.3*dir(140), NW, fontsize(8));
