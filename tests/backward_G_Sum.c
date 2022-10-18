#include "scorch/scorch.h"
#include "common.h"


int main(void)
{
  GRAPH_CTX ctx = G_CTX_New();
  GraphNode* a = G_Parameter(ctx, 13);
  GraphNode* b = G_Parameter(ctx, 24);
  GraphNode* g = G_Sum(ctx, a, b);

  forward(g);
  backward(g);

  assert_almost_eq(grad(a), 1);
  assert_almost_eq(grad(b), 1);

  G_CTX_Destroy(ctx);
}
