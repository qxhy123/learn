// 例3: 等腰 AB=AC=5, BC=6, 内切圆 r=3/2
size(8cm);
import geometry;

real s = 0.5;
pair B = (-3*s, 0);
pair Cpt = (3*s, 0);
pair A = (0, 4*s);   // height = 4

real a = length(B - Cpt);   // 6s
real b = length(A - Cpt);   // 5s
real c = length(A - B);     // 5s
pair I = (a*A + b*B + c*Cpt) / (a+b+c);
real r = I.y;

draw(circle(I, r));
draw(A--B--Cpt--cycle);

// altitude
pair D = (0, 0);
draw(A--D, gray+dashed);
markrightangle(B, D, A, size=0.12cm);

dot(A); label("$A$", A, N);
dot(B); label("$B$", B, SW);
dot(Cpt); label("$C$", Cpt, SE);
dot(I); label("$I$", I, E);
dot(D); label("$D$", D, S);

label("$5$", (A+B)/2, W);
label("$5$", (A+Cpt)/2, E);
label("$6$", (B+Cpt)/2, S);
label("$r=\frac{3}{2}$", I + (0.4, 0), E);
label("$4$", (A+D)/2, E, gray);
