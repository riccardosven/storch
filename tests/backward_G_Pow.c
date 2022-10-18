#include "common.h"
#include "scorch/scorch.h"

int
main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  Tensor* t_a = T_Scalar(4.4);
  Tensor* t_b = T_Scalar(3.2);

  GraphNode* a = G_Parameter(ctx, t_a);
  GraphNode* b = G_Parameter(ctx, t_b);

  GraphNode* g = G_Pow(ctx, a, b);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a)->data[0], 3.2 * pow(4.4, 2.2));
  retval += check_almost_eq(grad(b)->data[0], pow(4.4, 3.2) * log(4.4));

  G_CTX_Destroy(ctx);
  T_Destroy(t_a);
  T_Destroy(t_b);

  return retval;
}
