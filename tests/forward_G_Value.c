#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

int
main(void)
{
  SCORCH_CTX ctx = SCORCH_CTX_New();

  GraphNode* g = G_Value(ctx, T_Scalar(ctx, 2));

  forward(g);

  int retval = check_almost_eq(value(g)->data[0], 2);

  SCORCH_CTX_Destroy(ctx);

  return retval;
}
