#include "scorch/scorch.h"
#include "common.h"


int main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* x = G_Parameter(ctx, 9.91);

  GraphNode* g = G_Minus(ctx, x);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(x), -1);

  G_CTX_Destroy(ctx);

  return retval;
}
