#include "common.h"
#include "scorch/scorch.h"

int
main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  Tensor* t_13 = T_Scalar(13);
  Tensor* t_24 = T_Scalar(24);
  GraphNode* a = G_Parameter(ctx, t_13);
  GraphNode* b = G_Parameter(ctx, t_24);
  GraphNode* g = G_Sum(ctx, a, b);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a)->data[0], 1);
  retval += check_almost_eq(grad(b)->data[0], 1);

  G_CTX_Destroy(ctx);
  T_Destroy(t_13);
  T_Destroy(t_24);

  return retval;
}
