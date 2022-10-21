#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"

int
main(void)
{
  STORCH_CTX ctx = STORCH_CTX_New();

  Tensor* t = T_Scalar(ctx, 1.234);
  GraphNode* g = G_Log(ctx, G_Value(ctx, t));

  forward(g);

  int retval = check_almost_eq(value(g)->data[0], 0.21026092548319605);

  STORCH_CTX_Destroy(ctx);

  return retval;
}
