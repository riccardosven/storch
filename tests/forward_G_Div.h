#include "scorch/scorch.h"
#include "common.h"

int main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* g = G_Div(ctx, G_Value(ctx, 14), G_Value(ctx, 7));

  forward(g);

  int retval = check_almost_eq(value(g), 2.0);

  G_CTX_Destroy(ctx);

  return retval;
}

