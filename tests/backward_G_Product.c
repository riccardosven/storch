#include "common.h"
#include "scorch/scorch.h"

int
main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  Tensor* t_5 = T_Scalar(5);
  Tensor* t_3 = T_Scalar(3);

  GraphNode* a = G_Parameter(ctx, t_5);
  GraphNode* b = G_Parameter(ctx, t_3);

  GraphNode* g = G_Product(ctx, a, b);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a)->data[0], 3);
  retval = check_almost_eq(grad(b)->data[0], 5);

  G_CTX_Destroy(ctx);
  T_Destroy(t_5);
  T_Destroy(t_3);

  return retval;
}
