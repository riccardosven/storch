#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"

int
main(void)
{
  STORCH_CTX ctx = STORCH_CTX_New();

  Tensor* t = T_Scalar(ctx, 1.234);
  GraphNode* g = G_Exp(ctx, G_Value(ctx, t));

  forward(g);

  int retval = check_almost_eq(value(g)->data[0], 3.43494186080076);

  STORCH_CTX_Destroy(ctx);

  return retval;
}
