#include "common.h"
#include "scorch/scorch.h"

int
main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  Tensor* a = T_Scalar(13.2);
  Tensor* b = T_Scalar(2.2);

  GraphNode* g = G_Pow(ctx, G_Value(ctx, a), G_Value(ctx, b));

  forward(g);

  int retval = check_almost_eq(value(g)->data[0], pow(13.2, 2.2));

  G_CTX_Destroy(ctx);

  T_Destroy(a);
  T_Destroy(b);

  return retval;
}
