
#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdlib.h>

int
main(void)
{
  T_eltype va[] = { 1, 2, 3, 4, 5, 6 };
  Tensor* a = T_Wrap(NULL,3, 2, va);

  T_eltype vb[] = { 3, 2, 0, 7, -3, -2 };
  Tensor* b = T_Wrap(NULL,2, 3, vb);

  /* s = a@b
   * a = [1 4]  b = [3 0 -3]
   *     [2 5]      [2 7 -2]
   *     [3 6]
   */

  Tensor* s = T_MatMul(NULL, a, b);

  T_eltype vc[] = {
    1*3 + 4*2,
    2*3 + 5*2,
    3*3 + 6*2,
    1*0 + 4*7,
    2*0 + 5*7,
    3*0 + 6*7,
    -1*3 - 4*2,
    -2*3 - 5*2,
    -3*3 - 6*2};

  int retval = 1;
  for (size_t i = 0; i < T_nelems(s); i++) {
    retval = retval && (s->data[i] == vc[i]);
  }

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(s);

  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
