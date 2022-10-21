#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"

int
main(void)
{
  STORCH_CTX ctx = STORCH_CTX_New();

  T_eltype a_v = 7.3;

  GraphNode* a = G_Parameter(ctx, T_Scalar(ctx, a_v));
  GraphNode* g = G_Log(ctx, a);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a)->data[0], 1/a_v);

  STORCH_CTX_Destroy(ctx);

  return retval;
}
