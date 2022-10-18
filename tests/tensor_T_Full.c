#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"
#include <stdlib.h>

int
main(void)
{

  Tensor* t = T_Full(NULL, 3, 2, 1.234);

  int retval = t->n == 3 && t->m == 2 &&
    almost_eq(t->data[0], 1.234) &&
    almost_eq(t->data[1], 1.234) &&
    almost_eq(t->data[2], 1.234) &&
    almost_eq(t->data[3], 1.234) &&
    almost_eq(t->data[4], 1.234) &&
    almost_eq(t->data[5], 1.234);

  T_Destroy(t);
  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
