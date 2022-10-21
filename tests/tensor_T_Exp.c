#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdlib.h>

int
main(void)
{
  T_eltype va[] = { 1, 2, 3, 4, 5, 6 };
  Tensor* a = T_Wrap(NULL, 3, 2, va);

  T_eltype vb[] = { 3, 2, 0, 7, -3, -2 };
  Tensor* b = T_Wrap(NULL, 3, 2, vb);

  // s = log(a)
  Tensor* s = T_Exp(NULL, a);
  int retval = 1;
  for (size_t i = 0; i < T_nelems(s); i++) {
    retval = retval && (s->data[i] == exp(va[i]));
  }

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(s);

  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
