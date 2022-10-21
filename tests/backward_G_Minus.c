#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"

int
main(void)
{
  STORCH_CTX ctx = STORCH_CTX_New();

  T_eltype x_v = 9.91;
  Tensor* t_x = T_Scalar(ctx, x_v);

  GraphNode* x = G_Parameter(ctx, t_x);

  GraphNode* g = G_Minus(ctx, x);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(x)->data[0], -1);

  STORCH_CTX_Destroy(ctx);

  return retval;
}
