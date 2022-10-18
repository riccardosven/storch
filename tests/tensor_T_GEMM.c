#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

int
main(void)
{
  T_eltype va[] = { 1, 2, 3, 4, 5, 6 };
  Tensor* a = T_Wrap(2, 3, va);

  T_eltype vb[] = { 3, 2, 0, 7, -3, -2 };
  Tensor* b = T_Wrap(3, 2, vb);

  T_eltype vc[] = { 1, 2, 3, 4 };
  Tensor* c = T_Wrap(2, 2, vc);

  T_eltype alpha = 1.2;
  T_eltype beta = -0.3;

  /*
   * A = [1 3 5] B = [3  7] C = [1 3]
   *     [2 4 6]     [2 -3]     [2 4]
   *                 [0 -2]
   * a = 1.2  b = -0.3
   */

  // c = alpha*a@b + beta*c
  T_GEMM_(c, a, false, b, false, alpha, beta);

  T_eltype r[] = { alpha * (1 * 3 + 3 * 2 + 5 * 0) + beta * 1,
                   alpha * (2 * 3 + 4 * 2 + 6 * 0) + beta * 2,
                   alpha * (1 * 7 - 3 * 3 - 2 * 5) + beta * 3,
                   alpha * (2 * 7 - 4 * 3 - 2 * 6) + beta * 4 };

  int retval = 1;
  for (size_t i = 0; i < nelems(c); i++)
    retval = retval && r[i] == c->data[i];

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(c);

  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
