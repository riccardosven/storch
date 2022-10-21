#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdlib.h>

int
main(void)
{

  Tensor* t = T_Build(NULL, 3, 2, 6, 1.0, 2.0, -1.3, 2.2, 4.0, 5.1);

  T_eltype e[] = { 1.0, 2.0, -1.3, 2.2, 4.0, 5.1 };

  int retval = 0;

  for (int i = 0; i < 6; i++) {
    printf("%f == %f ?\n", t->data[i], e[i]);
    retval += check_almost_eq(t->data[i], e[i]);
  }

  T_Destroy(t);

  return retval ? EXIT_FAILURE : EXIT_SUCCESS;
}
