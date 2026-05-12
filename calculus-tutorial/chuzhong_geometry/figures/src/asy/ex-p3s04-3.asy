// 共角共边 例3：圆内两弦 AB, CD 相交于 P，证 △PAC~△PDB
size(8cm);
import geometry;

pair O = (0, 0);
real r = 3;
draw(circle(O, r));

pair A = r * dir(150);
pair B = r * dir(-20);
pair C = r * dir(80);
pair D = r * dir(-110);

// 交点 P
pair P = extension(A, B, C, D);

draw(A -- B);
draw(C -- D);
draw(A -- C, red);
draw(D -- B, red);

dot(O); label("$O$", O, S);
dot(A); label("$A$", A, W);
dot(B); label("$B$", B, E);
dot(C); label("$C$", C, N);
dot(D); label("$D$", D, SW);
dot(P); label("$P$", P, NE);

label("$PA \cdot PB = PC \cdot PD$", (0, -3.6));
