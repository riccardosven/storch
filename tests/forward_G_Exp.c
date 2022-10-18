#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

int
main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  Tensor* t = T_Scalar(1.234);
  GraphNode* g = G_Exp(ctx, G_Value(ctx, t));

  forward(g);

  int retval = check_almost_eq(value(g)->data[0], 3.43494186080076);

  G_CTX_Destroy(ctx);
  T_Destroy(t);

  return retval;
}
