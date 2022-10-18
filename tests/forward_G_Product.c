#include "scorch/scorch.h"
#include "common.h"

int main(void)
{
  GRAPH_CTX ctx = G_CTX_New();
  GraphNode* g = G_Product(ctx, G_Value(ctx, 5), G_Value(ctx, 3));

  forward(g);

  assert_almost_eq(value(g), 15);

  G_CTX_Destroy(ctx);

  return 0;
}
