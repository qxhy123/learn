// 例3: 两平行弦 AB=24 CD=10 在 r=13 圆内 — 同侧/异侧两种情形并列
size(13cm);
import geometry;

real r = 2.2;  // visual

void drawCase(pair shift, bool sameSide, string title) {
  pair O = shift;
  draw(shift(shift)*scale(r)*unitcircle);

  // y-coords of two chords; AB=24 (halfAB=12, OM=5); CD=10 (halfCD=5, ON=12)
  // visual: OM = 5/13 * r, halfAB = 12/13 * r
  real OM = 5.0/13 * r;
  real halfAB = 12.0/13 * r;
  real ON = 12.0/13 * r;
  real halfCD = 5.0/13 * r;

  real yAB, yCD;
  if (sameSide) {
    yAB = OM;   // above O
    yCD = ON;   // above O
  } else {
    yAB = OM;   // above
    yCD = -ON;  // below
  }

  pair A = shift + (-halfAB, yAB);
  pair B = shift + ( halfAB, yAB);
  pair C = shift + (-halfCD, yCD);
  pair D = shift + ( halfCD, yCD);
  pair M = shift + (0, yAB);
  pair N = shift + (0, yCD);

  draw(A--B);
  draw(C--D);
  draw(M--N, red);
  draw(shift--M, gray);
  draw(shift--N, gray);

  dot(shift); label("$O$", shift, E);
  dot(A); label("$A$", A, W);
  dot(B); label("$B$", B, E);
  dot(C); label("$C$", C, W);
  dot(D); label("$D$", D, E);
  dot(M); label("$M$", M, NE);
  dot(N); label("$N$", N, SE);

  label(title, shift + (0, -r-0.4), S);
}

drawCase((0, 0), true,  "same side: $7$");
drawCase((6, 0), false, "opposite side: $17$");
