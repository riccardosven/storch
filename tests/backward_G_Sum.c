#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

int
main(void)
{
  SCORCH_CTX ctx = SCORCH_CTX_New();

  GraphNode* a = G_Parameter(ctx, T_Scalar(ctx, 13));
  GraphNode* b = G_Parameter(ctx, T_Scalar(ctx, 24));
  GraphNode* g = G_Sum(ctx, a, b);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a)->data[0], 1);
  retval += check_almost_eq(grad(b)->data[0], 1);

  SCORCH_CTX_Destroy(ctx);

  return retval;
}
