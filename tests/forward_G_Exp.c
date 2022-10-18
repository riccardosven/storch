#include "scorch/scorch.h"
#include "common.h"


int main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* g = G_Exp(ctx, G_Value(ctx, 1.234));

  forward(g);

  int retval = check_almost_eq(value(g), 3.43494186080076);

  G_CTX_Destroy(ctx);

  return retval;
}
