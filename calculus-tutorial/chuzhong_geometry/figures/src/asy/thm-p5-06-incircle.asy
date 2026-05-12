// 三角形内切圆: 切点 D (BC), E (CA), F (AB)
size(8cm);
import geometry;

pair A = (0, 3);
pair B = (-2.4, 0);
pair Cpt = (2.4, 0);

// incenter: weighted by opposite side lengths
real a = length(B - Cpt);   // BC
real b = length(A - Cpt);   // CA
real c = length(A - B);     // AB
pair I = (a*A + b*B + c*Cpt) / (a+b+c);

// inradius: distance from I to BC (which is x-axis)
real r = I.y;

draw(circle(I, r));
draw(A--B--Cpt--cycle);

// touch points
pair D = (I.x, 0);                              // on BC
pair E = Cpt + dot(I - Cpt, unit(A - Cpt))*unit(A - Cpt);  // foot on CA
pair F = B   + dot(I - B,   unit(A - B  ))*unit(A - B);    // foot on AB

draw(I--D, gray+dashed);
draw(I--E, gray+dashed);
draw(I--F, gray+dashed);

markrightangle(B, D, I, size=0.12cm);

dot(A); label("$A$", A, N);
dot(B); label("$B$", B, SW);
dot(Cpt); label("$C$", Cpt, SE);
dot(I); label("$I$", I, NE);
dot(D); label("$D$", D, S);
dot(E); label("$E$", E, NE);
dot(F); label("$F$", F, NW);
label("$r$", (I+D)/2, E, gray);
