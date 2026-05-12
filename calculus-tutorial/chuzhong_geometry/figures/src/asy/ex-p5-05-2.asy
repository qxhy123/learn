// 例2 干净版本：等腰 ABC, AD⊥BC, 以 A 为圆心 AD 为半径 → BC 切 ⊙A 于 D
size(8cm);
import geometry;

pair A = (0, 2.2);
pair B = (-1.6, 0);
pair Cpt = (1.6, 0);
pair D = (0, 0);

real rad = length(A - D);  // = 2.2
draw(circle(A, rad));

draw(A--B); draw(A--Cpt); draw(B--Cpt);
draw(A--D, red);

markrightangle(B, D, A, size=0.18cm);

dot(A); label("$A$", A, N);
dot(B); label("$B$", B, SW);
dot(Cpt); label("$C$", Cpt, SE);
dot(D); label("$D$", D, S);

label("$r=AD$", (A+D)/2, E, red);
