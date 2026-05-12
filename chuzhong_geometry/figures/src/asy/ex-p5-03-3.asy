// 例3: 圆内接四边形 ABCD, ∠A=80°, ∠B=110°, ∠C=100°, ∠D=70°
size(8cm);
import geometry;

pair O = (0, 0);
real r = 2.3;
draw(circle(O, r));

pair A = r * dir(150);
pair B = r * dir(60);
pair Cpt = r * dir(-30);
pair D = r * dir(230);

draw(A--B--Cpt--D--cycle);

dot(O); label("$O$", O, E);
dot(A); label("$A$", A, NW);
dot(B); label("$B$", B, NE);
dot(Cpt); label("$C$", Cpt, E);
dot(D); label("$D$", D, SW);

label("$80^\circ$", A, 1.2*dir(-30));
label("$110^\circ$", B, 1.5*dir(-110));
label("$100^\circ$", Cpt, 1.5*dir(150));
label("$70^\circ$", D, 1.5*dir(50));
