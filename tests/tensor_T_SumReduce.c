#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdlib.h>
#include <stdio.h>

int
main(void)
{

  T_eltype v[] = { 1, 2, 3, 4, -5, 6 };

  Tensor* t = T_Wrap(NULL, 2, 3, v);

  Tensor* s = T_SumReduce(NULL, t);

  T_eltype e[] = {-1, 12};

  int retval = check_almost_eq(s->data[0], e[0]) + check_almost_eq(s->data[1], e[1]);

  T_Destroy(t);
  T_Destroy(s);

  return retval ? EXIT_FAILURE : EXIT_SUCCESS;
}
