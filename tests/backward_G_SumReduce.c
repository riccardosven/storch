#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

int
main(void)
{

  SCORCH_CTX ctx = SCORCH_CTX_New();

  T_eltype x_t[] = {1, 2, 3, 4};

  GraphNode* p = G_Parameter(ctx, T_Wrap(ctx, 1, 4, x_t));

  GraphNode* g = G_SumReduce(ctx, p);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(p)->data[0], 1);
  retval += check_almost_eq(grad(p)->data[1], 1);
  retval += check_almost_eq(grad(p)->data[2], 1);
  retval += check_almost_eq(grad(p)->data[3], 1);

  SCORCH_CTX_Destroy(ctx);

  return retval;
}
