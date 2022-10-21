#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdlib.h>

int
main(void)
{
  Tensor* t = T_New(NULL, 12, 14);

  int retval = t->n == 12 && t->m == 14 && t->data != NULL;

  T_Destroy(t);
  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
