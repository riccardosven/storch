#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"
#include <stdlib.h>

int
main(void)
{

  Tensor* t = T_Scalar(1.234);

  int retval = almost_eq(t->data[0], 1.234) && t->n == 1 && t->m == 1;

  T_Destroy(t);
  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
