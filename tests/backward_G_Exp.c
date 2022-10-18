#include "common.h"
#include "scorch/scorch.h"

int
main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  T_eltype a_v = 7.3;
  Tensor* t = T_Scalar(a_v);

  GraphNode* a = G_Parameter(ctx, t);
  GraphNode* g = G_Exp(ctx, a);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a)->data[0], exp(a_v));

  G_CTX_Destroy(ctx);
  T_Destroy(t);

  return retval;
}
