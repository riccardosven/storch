#include "scorch/scorch.h"
#include "common.h"

int main(void)
{

  GRAPH_CTX ctx = G_CTX_New();
  GraphNode* g = G_Minus(ctx, G_Value(ctx, 8.88));

  forward(g);
  int retval = check_almost_eq(value(g), -8.88);

  G_CTX_Destroy(ctx);

  return retval;
}
