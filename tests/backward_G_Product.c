#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

int
main(void)
{
  SCORCH_CTX ctx = SCORCH_CTX_New();

  Tensor* t_5 = T_Scalar(ctx, 5);
  Tensor* t_3 = T_Scalar(ctx, 3);

  GraphNode* a = G_Parameter(ctx, t_5);
  GraphNode* b = G_Parameter(ctx, t_3);

  GraphNode* g = G_Product(ctx, a, b);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a)->data[0], 3);
  retval = check_almost_eq(grad(b)->data[0], 5);

  SCORCH_CTX_Destroy(ctx);

  return retval;
}
