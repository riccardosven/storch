#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

int
main(void)
{
  GRAPH_CTX ctx = G_CTX_New();
  Tensor* t_13 = T_Scalar(13);
  Tensor* t_24 = T_Scalar(24);
  GraphNode* g = G_Diff(ctx, G_Value(ctx, t_13), G_Value(ctx, t_24));

  forward(g);

  int retval = check_almost_eq(value(g)->data[0], 13 - 24);
  G_CTX_Destroy(ctx);
  T_Destroy(t_13);
  T_Destroy(t_24);

  return retval;
}
