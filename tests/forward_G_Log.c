#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

int
main(void)
{
  SCORCH_CTX ctx = SCORCH_CTX_New();

  Tensor* t = T_Scalar(ctx, 1.234);
  GraphNode* g = G_Log(ctx, G_Value(ctx, t));

  forward(g);

  int retval = check_almost_eq(value(g)->data[0], 0.21026092548319605);

  SCORCH_CTX_Destroy(ctx);

  return retval;
}
