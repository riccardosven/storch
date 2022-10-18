#include "common.h"
#include "scorch/scorch.h"

int
main(void)
{

  /* a = [ 1, 2] b = [0.7]
   *     [ 3, 4]     [0.9]
   *     [ 5, 6]
   * c = [1 -2 3]
   */

  GRAPH_CTX ctx = G_CTX_New();

  T_eltype v_a[] = {1,3,5,2,4,6};
  T_eltype v_b[] = {0.7, 0.9};
  T_eltype v_c[] = {1, -2, 3};

  Tensor* t_a = T_Wrap(3, 2, v_a);
  Tensor* t_b = T_Wrap(2, 1, v_b);
  Tensor* t_c = T_Wrap(1, 3, v_c);

  GraphNode* a = G_Parameter(ctx, t_a);
  GraphNode* b = G_Parameter(ctx, t_b);
  GraphNode* c = G_Parameter(ctx, t_c);

  GraphNode* t = G_MatMul(ctx, a, b);

  GraphNode* g = G_MatMul(ctx, c, t);

  forward(g);
  backward(g);

  int retval = almost_eq(value(g)->data[0], 17.8) &&

     almost_eq(grad(c)->data[0], 2.5) &&
     almost_eq(grad(c)->data[1], 5.7) &&
     almost_eq(grad(c)->data[2], 8.9) &&

     almost_eq(grad(a)->data[0], 0.7) &&
     almost_eq(grad(a)->data[1], -1.4) &&
     almost_eq(grad(a)->data[2], 2.1) &&
     almost_eq(grad(a)->data[3], 0.9) &&
     almost_eq(grad(a)->data[4], -1.8) &&
     almost_eq(grad(a)->data[5], 2.7) &&

     almost_eq(grad(b)->data[0], 10.) &&
     almost_eq(grad(b)->data[1], 12.);

  G_CTX_Destroy(ctx);
  T_Destroy(t_a);
  T_Destroy(t_b);
  T_Destroy(t_c);

  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
