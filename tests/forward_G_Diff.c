#include "common.h"
#include "scorch/scorch.h"

int main(void)
{
  GRAPH_CTX ctx = G_CTX_New();
  GraphNode* g = G_Diff(ctx, G_Value(ctx, 13), G_Value(ctx, 24));

  forward(g);

  assert_almost_eq(value(g), 13 - 24);
  G_CTX_Destroy(ctx);

  return 0;
}
