#include "common.h"
#include "scorch/scorch.h"

int
main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  Tensor* t_5 = T_Scalar(5);
  Tensor* t_3 = T_Scalar(3);

  GraphNode* g = G_Product(ctx, G_Value(ctx, t_5), G_Value(ctx, t_3));

  forward(g);

  int retval = check_almost_eq(value(g)->data[0], 15);

  G_CTX_Destroy(ctx);
  T_Destroy(t_5);
  T_Destroy(t_3);

  return retval;
}
