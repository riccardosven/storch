#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"
#include <stdlib.h>

int
main(void)
{

  Tensor* t = T_Zeros(3, 2);

  int retval = almost_eq(T_GetItem(t, 1, 1), 0.0);

  T_SetItem(t, 2, 1, 1.3);
  retval = retval && almost_eq(T_GetItem(t, 2, 1), 1.3);

  T_SetItem(t, 2, 0, 1.1);
  T_SetItem(t, 2, 0, 0.9);

  retval = retval && almost_eq(t->data[2], 0.9) && almost_eq(t->data[3], 0.0) &&
           almost_eq(t->data[4], 1.1) && almost_eq(t->data[5], 1.3);

  T_Destroy(t);

  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
