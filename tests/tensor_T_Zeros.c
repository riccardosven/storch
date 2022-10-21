#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdlib.h>

int
main(void)
{
  Tensor* t = T_Zeros(NULL, 12, 14);

  int retval = t->n == 12 && t->m == 14 && t->data != NULL;

  for (size_t i = 0; i < T_nelems(t); i++) {
    retval = retval && (t->data[i] == 0);
  }

  T_Destroy(t);
  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
