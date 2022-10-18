#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"
#include <stdlib.h>

int
main(void)
{
  T_eltype va[] = { 1, 2, 3, 4, 5, 6 };
  Tensor* a = T_Wrap(3, 2, va);

  T_eltype vb[] = { 3, 2, 0, 7, -3, -2 };
  Tensor* b = T_Wrap(3, 2, vb);

  // s = log(a)
  Tensor* s = T_Log(a);
  int retval = 1;
  for (size_t i = 0; i < nelems(s); i++) {
    retval = retval && (s->data[i] == log(va[i]));
  }

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(s);

  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
