#include "scorch/scorch.h"
#include "common.h"


int main(void)
{
  GRAPH_CTX ctx = G_CTX_New();
  GraphNode* g = G_Value(ctx, 2);

  forward(g);
  backward(g); // Expected to fail

  G_CTX_Destroy(ctx);
}
