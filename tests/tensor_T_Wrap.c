#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"
#include <stdlib.h>

int
main(void)
{

  T_eltype d[] = { 1, 2, 3, 4, 5, 6 };

  Tensor* t = T_Wrap(3, 2, d);

  int retval = 1;

  for (size_t i = 0; i < nelems(t); i++)
    retval = retval && t->data[i] == d[i];

  T_Destroy(t);
  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
