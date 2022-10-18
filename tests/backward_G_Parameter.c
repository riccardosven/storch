#include "scorch/scorch.h"
#include "common.h"

int main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* a = G_Parameter(ctx, 5);
  GraphNode* b = G_Parameter(ctx, 3);

  GraphNode* g = G_Product(ctx, a, b);

  forward(g);
  backward(g);

  assert_almost_eq(grad(a), 3);
  assert_almost_eq(grad(b), 5);

  G_CTX_Destroy(ctx);
}
