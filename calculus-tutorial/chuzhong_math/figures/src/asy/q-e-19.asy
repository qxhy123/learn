// E.19 ⊙O r=2, OA=4. P 在 ⊙O 上, Q=PA 中点. B=OA 中点.
// O(0,0), A(4,0), B(2,0). Q 轨迹: 以 OA 中点 B 为圆心, 半径 1 的圆.
// 示例 P=(2cos60°, 2sin60°)=(1, √3). Q=((1+4)/2, √3/2)=(2.5, 0.866)
import graph;
size(10cm);

pair O = (0,0);
pair A = (4,0);
pair B = (2,0);
pair P = (1, sqrt(3));
pair Q = (2.5, sqrt(3)/2);

draw(circle(O, 2), blue);
draw(circle(B, 1), red+dashed);
draw(O--A, gray);
draw(P--A, black);

dot(O); label("$O$", O, SW);
dot(A); label("$A$", A, SE);
dot(B); label("$B$", B, S);
dot(P); label("$P$", P, N);
dot(Q); label("$Q$", Q, NE);

label("$2$", (0,1), W, fontsize(9));
label("$4$", (2,0)+(0,-0.1), S, fontsize(9));
