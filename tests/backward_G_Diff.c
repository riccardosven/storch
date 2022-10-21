#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"

int
main(void)
{
  STORCH_CTX ctx = STORCH_CTX_New();
  Tensor* t_a = T_Scalar(ctx, 13);
  Tensor* t_b = T_Scalar(ctx, 24);
  GraphNode* a = G_Parameter(ctx, t_a);
  GraphNode* b = G_Parameter(ctx, t_b);
  GraphNode* g = G_Diff(ctx, a, b);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a)->data[0], 1);
  retval += check_almost_eq(grad(b)->data[0], -1);

  STORCH_CTX_Destroy(ctx);

  return retval;
}
