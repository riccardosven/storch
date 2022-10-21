#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"

int
main(void)
{
  STORCH_CTX ctx = STORCH_CTX_New();

  Tensor* t = T_Scalar(ctx, 5);

  GraphNode* g = G_Parameter(ctx, t);

  forward(g);
  backward(g);

  int retval = check_almost_eq(value(g)->data[0], 5);
  retval += check_almost_eq(grad(g)->data[0], 1);

  STORCH_CTX_Destroy(ctx);

  return retval;
}
