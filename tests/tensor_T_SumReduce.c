#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static int
by_row()
{

  Tensor* a = T_Build(NULL, 2, 3, 6, 1.0, 2.0, 3.0, 4.0, -5.0, -6.0);

  Tensor* t = T_SumReduce0(NULL, a);

  assert(t->n == 1);
  assert(t->m == 3);

  int retval = check_almost_eq(t->data[0], 1.0 + 2.0) +
               check_almost_eq(t->data[1], 3.0 + 4.0) +
               check_almost_eq(t->data[2], -5.0 - 6.0);

  printf("t: %f %f %f\n", t->data[0], t->data[1], t->data[2]);

  T_Destroy(t);
  T_Destroy(a);

  if (retval)
    printf("Failure in by_row.");

  return retval;
}

static int
by_col()
{

  Tensor* a = T_Build(NULL, 2, 3, 6, 1.0, 2.0, 3.0, 4.0, -5.0, -6.0);

  Tensor* t = T_SumReduce1(NULL, a);

  assert(t->n == 2);
  assert(t->m == 1);

  int retval = check_almost_eq(t->data[0], 1.0 + 3.0 - 5.0) +
               check_almost_eq(t->data[1], 2.0 + 4.0 - 6.0);

  T_Destroy(t);
  T_Destroy(a);

  if (retval)
    printf("Failure in by_col.");

  return retval;
}

int
main(void)
{
  return by_row() + by_col();
}
