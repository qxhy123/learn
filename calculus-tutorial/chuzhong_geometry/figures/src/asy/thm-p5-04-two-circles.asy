// 两圆五种位置关系：外离 外切 相交 内切 内含
size(16cm);
import geometry;

real r1 = 0.85;
real r2 = 0.55;

void scene(pair shift, real d, string ttl) {
  pair O1 = shift;
  pair O2 = shift + (d, 0);
  draw(shift(O1)*scale(r1)*unitcircle);
  draw(shift(O2)*scale(r2)*unitcircle);
  draw(O1 -- O2, gray+dashed);
  dot(O1); label("$O_1$", O1, SW);
  dot(O2); label("$O_2$", O2, SE);
  label(ttl, shift + (d/2, -r1-0.35), S);
}

real sum = r1 + r2;     // 1.4
real diff = r1 - r2;    // 0.3

scene((0,0),    1.8,  "disjoint");                 // d>sum
scene((3.5,0),  sum,  "ext. tangent");             // d=sum
scene((7,0),    1.0,  "intersect");                // diff<d<sum
scene((10.5,0), diff, "int. tangent");             // d=diff
scene((13.5,0), 0.15, "contained");                // d<diff
