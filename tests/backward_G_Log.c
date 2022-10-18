#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

int
main(void)
{
  SCORCH_CTX ctx = SCORCH_CTX_New();

  T_eltype a_v = 7.3;

  GraphNode* a = G_Parameter(ctx, T_Scalar(ctx, a_v));
  GraphNode* g = G_Log(ctx, a);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a)->data[0], 1/a_v);

  SCORCH_CTX_Destroy(ctx);

  return retval;
}
