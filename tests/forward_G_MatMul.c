#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdlib.h>

int
main(void)
{

  /* a = [ 1, 4] b = [0.1, 0.3]
   *     [-2,-5]     [0.2, 0.4]
   *     [-3, 6]
   */

  STORCH_CTX ctx = STORCH_CTX_New();

  T_eltype v_a[] = { 1, -2, -3, 4, -5, 6 };
  T_eltype v_b[] = { 0.1, 0.2, 0.3, 0.4 };

  Tensor* t_a = T_Wrap(ctx, 3, 2, v_a);
  Tensor* t_b = T_Wrap(ctx, 2, 2, v_b);

  GraphNode* g = G_MatMul(ctx, G_Value(ctx, t_a), G_Value(ctx, t_b));

  forward(g);

  T_eltype v_ab[] = {
    1 * 0.1 + 4 * 0.2, -2 * 0.1 - 5 * 0.2, -3 * 0.1 + 6 * 0.2,
    1 * 0.3 + 4 * 0.4, -2 * 0.3 - 5 * 0.4, -3 * 0.3 + 6 * 0.4
  };

  int retval = 1;
  for (size_t i = 0; i < T_nelems(value(g)); i++) {
    retval = retval && almost_eq(value(g)->data[i], v_ab[i]);
  }

  STORCH_CTX_Destroy(ctx);

  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
