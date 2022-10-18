#include "scorch/scorch.h"
#include "common.h"


int main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* a = G_Parameter(ctx, 4.4);
  GraphNode* b = G_Parameter(ctx, 3.2);

  GraphNode* g = G_Pow(ctx, a, b);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a), 3.2 * pow(4.4, 2.2));
  retval += check_almost_eq(grad(b), pow(4.4, 3.2) * log(4.4));

  G_CTX_Destroy(ctx);

  return retval;
}
