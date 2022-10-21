#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"

int
main(void)
{

  STORCH_CTX ctx = STORCH_CTX_New();

  T_eltype x_t[] = { 1, 2, 3, 4 };

  GraphNode* p = G_Parameter(ctx, T_Wrap(ctx, 1, 4, x_t));

  GraphNode* g = G_SumReduce(ctx, p);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(p)->data[0], 1);
  retval += check_almost_eq(grad(p)->data[1], 1);
  retval += check_almost_eq(grad(p)->data[2], 1);
  retval += check_almost_eq(grad(p)->data[3], 1);

  STORCH_CTX_Destroy(ctx);

  return retval;
}
