#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdlib.h>

int
main(void)
{

  Tensor* t1 = T_Zeros(NULL, 3, 2);
  Tensor* t2 = T_Ones(NULL, 3, 2);

  int retval = 1;
  for (size_t i = 0; i < T_nelems(t1); i++)
    retval = retval && t1->data[i] != t2->data[i];

  Tensor* t3 = T_Copy(NULL, t1);
  T_Copy_(t2, t1);

  retval = retval && t3->data != t1->data;
  for (size_t i = 0; i < T_nelems(t1); i++) {
    retval = retval && t1->data[i] == t2->data[i] && t1->data[i] == t3->data[i];
  }

  T_Destroy(t1);
  T_Destroy(t2);
  T_Destroy(t3);
  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
