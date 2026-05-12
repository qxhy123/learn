// 直线与圆三种位置关系：相离 / 相切 / 相交
size(13cm);
import geometry;

real r = 1.4;

void scene(pair shift, real d, string ttl) {
  pair O = shift;
  draw(shift(shift)*scale(r)*unitcircle);
  dot(O); label("$O$", O, S);
  // horizontal line at y = O.y + d (above)
  real xL = -2.2, xR = 2.2;
  pair P1 = shift + (xL, d);
  pair P2 = shift + (xR, d);
  draw(P1 -- P2);
  // perpendicular from O
  pair H = shift + (0, d);
  draw(O -- H, gray+dashed);
  markrightangle(P2, H, O, size=0.15cm);
  dot(H); label("$H$", H, N);
  label(ttl, shift + (0, -r-0.45), S);
  // intersection points if any
  if (d < r) {
    real dx = sqrt(r*r - d*d);
    dot(shift + (-dx, d));
    dot(shift + ( dx, d));
  } else if (abs(d - r) < 0.01) {
    // tangent — H itself is the contact point (already dotted)
  }
}

scene((0,0),    1.9, "$d>r$: disjoint");
scene((5,0),    r,   "$d=r$: tangent");
scene((10,0),   0.6, "$d<r$: intersect");
