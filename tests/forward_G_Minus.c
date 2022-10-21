#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"

int
main(void)
{

  STORCH_CTX ctx = STORCH_CTX_New();
  Tensor* t = T_Scalar(ctx, 8.88);
  GraphNode* g = G_Minus(ctx, G_Value(ctx, t));

  forward(g);
  int retval = check_almost_eq(value(g)->data[0], -8.88);

  STORCH_CTX_Destroy(ctx);

  return retval;
}
