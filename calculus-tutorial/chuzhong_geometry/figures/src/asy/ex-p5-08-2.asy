// 四点共圆 例2：△ABC, BD⊥AC, CE⊥AB. 证 B,C,D,E 共圆 (以 BC 为直径)
size(9cm);
import geometry;

pair A = (1, 4);
pair B = (-2, 0);
pair C = (3, 0);

// D = foot of perpendicular from B to AC; E = foot from C to AB
pair footPt(pair P, pair X, pair Y) {
  pair v = Y - X;
  real t = dot(P - X, v) / dot(v, v);
  return X + t * v;
}
pair D = footPt(B, A, C);
pair E = footPt(C, A, B);

draw(A -- B -- C -- cycle);
draw(B -- D, red);
draw(C -- E, red);

// 圆 with diameter BC
pair Mbc = (B + C)/2;
real rBC = length(B - C)/2;
draw(circle(Mbc, rBC), gray+dashed);

markrightangle(B, D, A, size=0.18cm);
markrightangle(C, E, A, size=0.18cm);

dot(A); label("$A$", A, N);
dot(B); label("$B$", B, SW);
dot(C); label("$C$", C, SE);
dot(D); label("$D$", D, NE);
dot(E); label("$E$", E, W);
