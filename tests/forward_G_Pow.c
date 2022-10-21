#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"

int
main(void)
{
  STORCH_CTX ctx = STORCH_CTX_New();

  Tensor* a = T_Scalar(ctx, 13.2);
  Tensor* b = T_Scalar(ctx, 2.2);

  GraphNode* g = G_Pow(ctx, G_Value(ctx, a), G_Value(ctx, b));

  forward(g);

  int retval = check_almost_eq(value(g)->data[0], pow(13.2, 2.2));

  STORCH_CTX_Destroy(ctx);

  return retval;
}
