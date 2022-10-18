#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

int
main(void)
{
  GRAPH_CTX ctx = G_CTX_New();
  Tensor* t = T_Scalar(2);

  GraphNode* g = G_Value(ctx, t);

  forward(g);

  int retval = check_almost_eq(value(g)->data[0], 2);

  T_Destroy(t);
  G_CTX_Destroy(ctx);

  return retval;
}
