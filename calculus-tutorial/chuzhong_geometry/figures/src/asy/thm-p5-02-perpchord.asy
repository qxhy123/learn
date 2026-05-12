// 垂径定理：过圆心 O 的直径 CD 垂直于弦 AB 于 M，则 AM=BM，弧AC=弧BC，弧AD=弧BD
size(8cm);
import geometry;

pair O = (0, 0);
real r = 2.2;
draw(circle(O, r));

// chord AB horizontal at y = -0.9 (below center)
real y0 = -0.9;
real halfAB = sqrt(r*r - y0*y0);
pair A = (-halfAB, y0);
pair B = ( halfAB, y0);
pair M = (0, y0);

// diameter CD vertical
pair C = (0,  r);
pair D = (0, -r);

draw(A -- B);
draw(C -- D);
draw(O -- A, gray+dashed);
draw(O -- B, gray+dashed);

markrightangle(O, M, B, size=0.18cm);

// equal-mark on AM and BM
draw((A+M)/2 + (0,0.08) -- (A+M)/2 + (0,-0.08));
draw((B+M)/2 + (0,0.08) -- (B+M)/2 + (0,-0.08));

dot(O); label("$O$", O, NE);
dot(A); label("$A$", A, W);
dot(B); label("$B$", B, E);
dot(C); label("$C$", C, N);
dot(D); label("$D$", D, S);
dot(M); label("$M$", M, NE);
