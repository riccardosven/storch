#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"

int
main(void)
{
  STORCH_CTX ctx = STORCH_CTX_New();

  Tensor* t_5 = T_Scalar(ctx, 5);
  Tensor* t_3 = T_Scalar(ctx, 3);

  GraphNode* g = G_Product(ctx, G_Value(ctx, t_5), G_Value(ctx, t_3));

  forward(g);

  int retval = check_almost_eq(value(g)->data[0], 15);

  STORCH_CTX_Destroy(ctx);

  return retval;
}
