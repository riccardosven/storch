#include "scorch/scorch.h"
#include "common.h"


int main(void)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* a = G_Parameter(ctx, 7.3);
  GraphNode* g = G_Exp(ctx, a);

  forward(g);
  backward(g);

  assert_almost_eq(grad(a), exp(7.3));

  G_CTX_Destroy(ctx);
}
