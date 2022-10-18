#include "common.h"
#include "scorch/scorch.h"

int
main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  T_eltype a_v = 14;
  T_eltype b_v = 7;

  Tensor* t_a = T_Scalar(a_v);
  Tensor* t_b = T_Scalar(b_v);

  GraphNode* a = G_Parameter(ctx, t_a);
  GraphNode* b = G_Parameter(ctx, t_b);

  GraphNode* g = G_Div(ctx, a, b);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a)->data[0], 1.0 / b_v);
  retval += check_almost_eq(grad(b)->data[0], -a_v / (b_v * b_v));

  G_CTX_Destroy(ctx);
  T_Destroy(t_a);
  T_Destroy(t_b);

  return retval;
}
