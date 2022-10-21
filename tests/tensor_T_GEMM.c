#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdio.h>
#include <stdbool.h>
#include <cblas.h>
#include <stdlib.h>





int test_AB(void)
{
  T_eltype va[] = { 1, 2, 3, 4, 5, 6 };
  Tensor* a = T_Wrap(NULL,2, 3, va);

  T_eltype vb[] = { 3, 2, 0, 7, -3, -2 };
  Tensor* b = T_Wrap(NULL,3, 2, vb);

  T_eltype vc[] = { 1, 2, 3, 4 };
  Tensor* c = T_Wrap(NULL,2, 2, vc);

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

  T_eltype r1[] = { alpha * (1 * 3 + 3 * 2 + 5 * 0) + beta * 1,
                   alpha * (2 * 3 + 4 * 2 + 6 * 0) + beta * 2,
                   alpha * (1 * 7 - 3 * 3 - 2 * 5) + beta * 3,
                   alpha * (2 * 7 - 4 * 3 - 2 * 6) + beta * 4 };

  int retval = 1;
  for (size_t i = 0; i < T_nelems(c); i++)
    retval = retval && almost_eq(r1[i], c->data[i]);

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(c);

  return retval;
}

int
test_ATB(void)
{
  T_eltype va[] = {1, 3, 5};
  Tensor *a = T_Wrap(NULL,3, 1, va);
  T_eltype vb[] = {5,2,1, 1,-2, 5};
  Tensor *b = T_Wrap(NULL,3,2, vb);

  Tensor *c = T_Zeros(NULL, 1, 2);

  T_GEMM_(c, a, true, b, false, 1.0, 0.0);

  int retval = almost_eq(c->data[0], 16) && almost_eq(c->data[1],20);

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(c);

  return retval;
}


int
test_ABT(void)
{
  // A 2 x 3
  T_eltype va[] = {1., 2., 3., 4.,5., 6.};
  Tensor *a = T_Wrap(NULL,2, 3, va );

  // B 2 x 3
  T_eltype vb[] = {3., -3., .5, .7, .8, .9};
  Tensor *b = T_Wrap(NULL,2, 3 ,vb);

  // C 2 x 2
  T_eltype vc[] = {0., 0., 0., 0.};
  Tensor *c = T_Wrap(NULL,2, 2, vc);

  T_GEMM_(c, a, false, b, true, 1.0, 0.0);


  int retval = almost_eq(c->data[0], 8.5) &&
    almost_eq(c->data[1], 12.8) &&
  almost_eq(c->data[2], 3.6) &&
  almost_eq(c->data[3], 2.2);

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(c);

  return retval;
}


int
main(void)
{

  return test_AB() && test_ATB()  && test_ABT() ? EXIT_SUCCESS : EXIT_FAILURE;
}
