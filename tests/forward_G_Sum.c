#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"

int
main(void)
{
  STORCH_CTX ctx = STORCH_CTX_New();
  GraphNode* g = G_Sum(ctx,
        G_Value(ctx, T_Scalar(ctx, 13)),
        G_Value(ctx, T_Scalar(ctx, 24))
      );

  forward(g);

  int retval = check_almost_eq(value(g)->data[0], 13 + 24);
  STORCH_CTX_Destroy(ctx);

  return retval;
}
