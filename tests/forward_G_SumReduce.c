#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

int
main(void)
{

  SCORCH_CTX ctx = SCORCH_CTX_New();

  T_eltype x_t[] = {1, 2, 3};

  Tensor *x = T_Wrap(ctx, 1, 3, x_t);

  GraphNode* g = G_SumReduce(ctx,
        G_Value(ctx, x)
      );

  forward(g);

  int retval = check_almost_eq(value(g)->data[0], 1+2+3);
  SCORCH_CTX_Destroy(ctx);

  return retval;
}
