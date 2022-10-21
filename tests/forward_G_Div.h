#include "common.h"
#include "storch/storch.h"

int
main(void)
{
  STORCH_CTX ctx = STORCH_CTX_New();

  GraphNode* g = G_Div(ctx, G_Value(ctx, 14), G_Value(ctx, 7));

  forward(g);

  int retval = check_almost_eq(value(g), 2.0);

  STORCH_CTX_Destroy(ctx);

  return retval;
}
