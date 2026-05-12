// 例1: r=5, AB=8, 求 OD=3
size(7cm);
import geometry;

pair O = (0, 0);
real r = 2.5;
draw(circle(O, r));

real halfAB = 2.0;   // 4 of 5
real OD = 1.5;       // 3 of 5
pair D = (0, -OD);
pair A = D + (-halfAB, 0);
pair B = D + ( halfAB, 0);

draw(A -- B);
draw(O -- A);
draw(O -- D, red);

markrightangle(O, D, A, size=0.18cm);

dot(O); label("$O$", O, N);
dot(A); label("$A$", A, SW);
dot(B); label("$B$", B, SE);
dot(D); label("$D$", D, S);

label("$5$", (O+A)/2, NW);
label("$4$", (D+A)/2, S);
label("$3$", (O+D)/2, E, red);
