#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

int
main(void)
{
  GRAPH_CTX ctx = G_CTX_New();
  Tensor* t_a = T_Scalar(13);
  Tensor* t_b = T_Scalar(24);
  GraphNode* a = G_Parameter(ctx, t_a);
  GraphNode* b = G_Parameter(ctx, t_b);
  GraphNode* g = G_Diff(ctx, a, b);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a)->data[0], 1);
  retval += check_almost_eq(grad(b)->data[0], -1);

  G_CTX_Destroy(ctx);
  T_Destroy(t_a);
  T_Destroy(t_b);

  return retval;
}
