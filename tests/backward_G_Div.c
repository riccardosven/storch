#include "scorch/scorch.h"
#include "common.h"


int main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* a = G_Parameter(ctx, 14);
  GraphNode* b = G_Parameter(ctx, 7);

  GraphNode* g = G_Div(ctx, a, b);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a), 1.0 / 7.0);
  retval += check_almost_eq(grad(b), -14.0 / (7.0 * 7.0));

  G_CTX_Destroy(ctx);

  return retval;
}
