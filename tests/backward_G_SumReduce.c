#include "common.h"
#include <assert.h>
#include "storch/storch.h"
#include "storch/tensor.h"

static int
by_row() {
  STORCH_CTX ctx = STORCH_CTX_New();

  GraphNode* x = G_Parameter(ctx, T_Build(ctx, 2, 3, 6, 1.0, 2.0, 3.0, 4.0, -5.0, -6.0));

  GraphNode* g = G_SumReduce0(ctx, x);

  forward(g);
  backward(g);

  assert(grad(x)->n == 2);
  assert(grad(x)->m == 3);

  int retval = 0;

  for (int i=0; i<6; i++)
    retval += check_almost_eq(grad(x)->data[i], 1.0);

  STORCH_CTX_Destroy(ctx);

  if (retval)
    printf("Error in by_row!");

  return retval;
}


static int
by_col() {
  STORCH_CTX ctx = STORCH_CTX_New();

  GraphNode* x = G_Parameter(ctx, T_Build(ctx, 2, 3, 6, 1.0, 2.0, 3.0, 4.0, -5.0, -6.0));

  GraphNode* g = G_SumReduce1(ctx, x);

  forward(g);
  backward(g);

  assert(grad(x)->n == 2);
  assert(grad(x)->m == 3);

  int retval = 0;

  for (int i=0; i<6; i++)
    retval += check_almost_eq(grad(x)->data[i], 1.0);

  STORCH_CTX_Destroy(ctx);

  if (retval)
    printf("Error in by_col!");

  return retval;
}

int
main(void)
{
  return by_row() + by_col();
}
