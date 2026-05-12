// 母子相似 例3：圆 O 直径 AB=10, C 在圆上, CD⊥AB 于 D, AD=2, 求 CD=4
size(8cm);
import geometry;

pair O = (0, 0);
real r = 5;
draw(circle(O, r));

pair A = (-5, 0);
pair B = (5, 0);
pair D = A + (2, 0); // AD=2 -> D=(-3,0)
real h = sqrt(2 * 8); // CD^2 = AD * DB = 2*8 = 16, h=4
pair C = D + (0, h);

draw(A -- B);
draw(A -- C -- B);
draw(C -- D, blue+dashed);

// 直角 at C (圆周角)
markrightangle(A, C, B, size=0.18cm);
// 直角 at D
markrightangle(C, D, A, size=0.18cm);

dot(O); label("$O$", O, S);
dot(A); label("$A$", A, SW);
dot(B); label("$B$", B, SE);
dot(C); label("$C$", C, N);
dot(D); label("$D$", D, S);

label("$2$", (A+D)/2, S);
label("$8$", (D+B)/2, S);
label("$CD=4$", (C+D)/2, E);
