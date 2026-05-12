// 圆周角定理 三种证明情形：O在角边上 / O在角内 / O在角外
size(15cm);
import geometry;

real r = 1.6;

void case1(pair shift, string ttl) {
  pair O = shift;
  draw(shift(shift)*scale(r)*unitcircle);
  // C on top, CB is diameter (O on edge CB)
  pair C = shift + r*dir(90);
  pair B = shift + r*dir(270);
  pair A = shift + r*dir(200);
  draw(C--A); draw(C--B); draw(A--B);
  draw(O--A, gray);
  markangle(Label("$\theta$", Relative(0.5)), B, C, A, radius=0.4cm);
  markangle(Label("$2\theta$", Relative(0.5)), B, O, A, radius=0.35cm);
  dot(O); label("$O$", O, E);
  dot(A); label("$A$", A, W);
  dot(B); label("$B$", B, S);
  dot(C); label("$C$", C, N);
  label(ttl, shift + (0, -r-0.5), S);
}

void case2(pair shift, string ttl) {
  pair O = shift;
  draw(shift(shift)*scale(r)*unitcircle);
  pair C = shift + r*dir(90);
  pair A = shift + r*dir(210);
  pair B = shift + r*dir(330);
  draw(C--A); draw(C--B); draw(A--B);
  draw(O--A, gray); draw(O--B, gray);
  markangle(Label("$\theta$", Relative(0.5)), B, C, A, radius=0.4cm);
  markangle(Label("$2\theta$", Relative(0.5)), A, O, B, radius=0.4cm);
  dot(O); label("$O$", O, N);
  dot(A); label("$A$", A, SW);
  dot(B); label("$B$", B, SE);
  dot(C); label("$C$", C, N);
  label(ttl, shift + (0, -r-0.5), S);
}

void case3(pair shift, string ttl) {
  pair O = shift;
  draw(shift(shift)*scale(r)*unitcircle);
  // C at top; A and B both on one side (both on right)
  pair C = shift + r*dir(110);
  pair A = shift + r*dir(40);
  pair B = shift + r*dir(70);
  draw(C--A); draw(C--B); draw(A--B);
  draw(O--A, gray); draw(O--B, gray);
  markangle(Label("$\theta$", Relative(0.5)), A, C, B, radius=0.6cm);
  markangle(Label("$2\theta$", Relative(0.5)), B, O, A, radius=0.35cm);
  dot(O); label("$O$", O, S);
  dot(A); label("$A$", A, E);
  dot(B); label("$B$", B, NE);
  dot(C); label("$C$", C, NW);
  label(ttl, shift + (0, -r-0.5), S);
}

case1((0,0),  "case 1: $O$ on edge");
case2((5,0),  "case 2: $O$ inside");
case3((10,0), "case 3: $O$ outside");
