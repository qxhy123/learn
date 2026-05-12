// 切线长定理：P 外, PA=PB, OP 平分 ∠APB
size(9cm);
import geometry;

pair O = (0, 0);
real r = 1.8;
draw(circle(O, r));

pair P = (4.0, 0);

// A, B are tangent points: on circle with PA ⊥ OA. Use circle on OP as diameter
pair M = (O + P)/2;
real rM = length(P - O)/2;
pair[] inters = intersectionpoints(circle(O, r), circle(M, rM));
pair A = (inters[0].y > 0) ? inters[0] : inters[1];
pair B = (inters[0].y < 0) ? inters[0] : inters[1];

draw(P--A); draw(P--B);
draw(O--A, gray); draw(O--B, gray);
draw(O--P, gray+dashed);
draw(A--B, red);

markrightangle(P, A, O, size=0.15cm);
markrightangle(O, B, P, size=0.15cm);

// equal tick marks on PA and PB
pair mA = (P+A)/2;
pair mB = (P+B)/2;
draw(mA + (-0.06, 0.06) -- mA + (0.06, -0.06));
draw(mB + (-0.06, -0.06) -- mB + (0.06, 0.06));

dot(O); label("$O$", O, W);
dot(P); label("$P$", P, E);
dot(A); label("$A$", A, NW);
dot(B); label("$B$", B, SW);

markangle(Label("", Relative(0.5)), O, P, A, radius=0.45cm);
markangle(Label("", Relative(0.5)), B, P, O, radius=0.45cm);
