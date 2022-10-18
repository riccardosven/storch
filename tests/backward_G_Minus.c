#include "common.h"
#include "scorch/scorch.h"

int
main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  T_eltype x_v = 9.91;
  Tensor* t_x = T_Scalar(x_v);

  GraphNode* x = G_Parameter(ctx, t_x);

  GraphNode* g = G_Minus(ctx, x);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(x)->data[0], -1);

  G_CTX_Destroy(ctx);

  T_Destroy(t_x);

  return retval;
}
