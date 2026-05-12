// 例1: 半径=5, AB=6, 求弦心距 OD
size(7cm);
import geometry;

pair O = (0, 0);
real r = 2.5;  // visual scale
draw(circle(O, r));

// AB horizontal, half=1.5 (since 3/5 of r), OD=2 (4/5 of r)
real halfAB = 1.5;   // scaled 3
real OD = 2.0;       // scaled 4
pair D = (0, -OD);
pair A = D + (-halfAB, 0);
pair B = D + ( halfAB, 0);

draw(A -- B);
draw(O -- A);
draw(O -- B);
draw(O -- D, red);

markrightangle(O, D, A, size=0.18cm);

dot(O); label("$O$", O, N);
dot(A); label("$A$", A, SW);
dot(B); label("$B$", B, SE);
dot(D); label("$D$", D, S);

label("$5$", (O+A)/2, NW);
label("$3$", (D+A)/2, S);
label("$3$", (D+B)/2, S);
label("$OD=?$", (O+D)/2 + (0.3, 0), E, red);
