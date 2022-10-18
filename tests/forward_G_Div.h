#include "common.h"
#include "scorch/scorch.h"

int
main(void)
{
  SCORCH_CTX ctx = SCORCH_CTX_New();

  GraphNode* g = G_Div(ctx, G_Value(ctx, 14), G_Value(ctx, 7));

  forward(g);

  int retval = check_almost_eq(value(g), 2.0);

  SCORCH_CTX_Destroy(ctx);

  return retval;
}
