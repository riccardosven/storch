#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <assert.h>

static int
by_row()
{
  STORCH_CTX ctx = STORCH_CTX_New();

  Tensor* x = T_Build(ctx, 2, 3, 6, 1.0, 2.0, 3.0, 4.0, -5.0, -6.0);

  GraphNode* g = G_SumReduce0(ctx, G_Value(ctx, x));

  forward(g);

  assert(value(g)->n == 1);
  assert(value(g)->m == 3);

  int retval = check_almost_eq(value(g)->data[0], 1.0 + 2.0) +
               check_almost_eq(value(g)->data[1], 3.0 + 4.0) +
               check_almost_eq(value(g)->data[2], -5.0 - 6.0);

  STORCH_CTX_Destroy(ctx);

  return retval;
}

static int
by_col()
{
  STORCH_CTX ctx = STORCH_CTX_New();

  Tensor* x = T_Build(ctx, 2, 3, 6, 1.0, 2.0, 3.0, 4.0, -5.0, -6.0);

  GraphNode* g = G_SumReduce1(ctx, G_Value(ctx, x));

  forward(g);

  assert(value(g)->n == 2);
  assert(value(g)->m == 1);

  int retval = check_almost_eq(value(g)->data[0], 1.0 + 3.0 - 5.0) +
               check_almost_eq(value(g)->data[1], 2.0 + 4.0 - 6.0);

  STORCH_CTX_Destroy(ctx);

  return retval;
}

int
main(void)
{

  return by_row() + by_col();
}
