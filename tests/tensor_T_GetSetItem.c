#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"
#include <stdlib.h>

#define all_almost_eq(a, b, c) almost_eq((a), (c)) && almost_eq((a), (b))

int
main(void)
{

  /* t = [0 , 1.9]
   *     [0.7, 0]
   *     [1.1, 0.9]
   */

  Tensor* t = T_Zeros(NULL, 3, 2);

  T_SetItem(t, 1, 0, 0.7);
  T_SetItem(t, 2, 0, 1.1);
  T_SetItem(t, 0, 1, 1.9);
  T_SetItem(t, 1, 1, 0.4);
  T_SetItem(t, 2, 1, 0.9);

  int retval =
    all_almost_eq(0.0, t->data[0], T_GetItem(t, 0, 0)) &&
    all_almost_eq(0.7, t->data[1], T_GetItem(t, 1, 0)) &&
    all_almost_eq(1.1, t->data[2], T_GetItem(t, 2, 0)) &&
    all_almost_eq(1.9, t->data[3], T_GetItem(t, 0, 1)) &&
    all_almost_eq(0.4, t->data[4], T_GetItem(t, 1, 1)) &&
    all_almost_eq(0.9, t->data[5], T_GetItem(t, 2, 1));

  T_Destroy(t);

  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
