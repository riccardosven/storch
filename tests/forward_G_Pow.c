#include "scorch/scorch.h"
#include "common.h"

int main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* g = G_Pow(ctx, G_Value(ctx, 13.2), G_Value(ctx, 2.2));

  forward(g);

  int retval = check_almost_eq(value(g), pow(13.2, 2.2));

  G_CTX_Destroy(ctx);

  return retval;
}
