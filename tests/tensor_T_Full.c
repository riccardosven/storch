#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdlib.h>

int
main(void)
{

  Tensor* t1 = T_Full(NULL, 3, 2, 1.234);

  int retval = t1->n == 3 && t1->m == 2 && almost_eq(t1->data[0], 1.234) &&
               almost_eq(t1->data[1], 1.234) && almost_eq(t1->data[2], 1.234) &&
               almost_eq(t1->data[3], 1.234) && almost_eq(t1->data[4], 1.234) &&
               almost_eq(t1->data[5], 1.234);

  Tensor* t2 = T_FullLike(NULL, t1, 3.0);

  retval = retval && t2->n == 3 && t2->m == 2 && almost_eq(t2->data[0], 3.0) &&
           almost_eq(t2->data[1], 3.0) && almost_eq(t2->data[2], 3.0) &&
           almost_eq(t2->data[3], 3.0) && almost_eq(t2->data[4], 3.0) &&
           almost_eq(t2->data[5], 3.0);

  T_Destroy(t1);
  T_Destroy(t2);

  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
