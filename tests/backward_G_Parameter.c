#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

int
main(void)
{
  SCORCH_CTX ctx = SCORCH_CTX_New();

  Tensor* t = T_Scalar(ctx, 5);

  GraphNode* g = G_Parameter(ctx, t);

  forward(g);
  backward(g);

  int retval = check_almost_eq(value(g)->data[0], 5);
  retval += check_almost_eq(grad(g)->data[0], 1);

  SCORCH_CTX_Destroy(ctx);

  return retval;
}
