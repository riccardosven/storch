#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"
#include <stdlib.h>

int
main(void)
{
  Tensor* t = T_Ones(12, 14);

  int retval = t->n == 12 && t->m == 14 && t->data != NULL;

  for (size_t i = 0; i < nelems(t); i++) {
    retval = retval && (t->data[i] == 1);
  }

  T_Destroy(t);
  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
