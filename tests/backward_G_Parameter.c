#include "common.h"
#include "scorch/scorch.h"

int
main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  Tensor* t = T_Scalar(5);

  GraphNode* g = G_Parameter(ctx, t);

  forward(g);
  backward(g);

  int retval = check_almost_eq(value(g)->data[0], 5);
  retval += check_almost_eq(grad(g)->data[0], 1);

  G_CTX_Destroy(ctx);
  T_Destroy(t);

  return retval;
}
