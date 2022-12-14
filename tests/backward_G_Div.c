#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"

int
main(void)
{
  STORCH_CTX ctx = STORCH_CTX_New();

  T_eltype a_v = 14;
  T_eltype b_v = 7;

  GraphNode* a = G_Parameter(ctx, T_Scalar(ctx, a_v));
  GraphNode* b = G_Parameter(ctx, T_Scalar(ctx, b_v));

  GraphNode* g = G_Div(ctx, a, b);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(a)->data[0], 1.0 / b_v);
  retval += check_almost_eq(grad(b)->data[0], -a_v / (b_v * b_v));

  STORCH_CTX_Destroy(ctx);

  return retval;
}
