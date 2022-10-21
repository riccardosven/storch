#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdlib.h>

int
main(void)
{

  Tensor* t = T_Scalar(NULL, 1.234);

  int retval = almost_eq(t->data[0], 1.234) && t->n == 1 && t->m == 1;

  T_Destroy(t);
  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
