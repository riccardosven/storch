#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdlib.h>

int
test_matrix_matrix()
{

  Tensor* a = T_Build(NULL, 2, 3, 6, 1.0, 2.0, 3.0, 4.0, -5.0, 6.0);
  Tensor* b = T_Build(NULL, 2, 3, 6, 3.0, -2.0, 1.0, 2.0, 9.0, 1.0);

  Tensor* s = T_Div(NULL, a, b);

  T_eltype e[] = { 1.0 / 3.0, -1, 3, 2, -5.0 / 9.0, 6.0 };

  int retval = 0;
  for (size_t i = 0; i < 6; i++) {
    retval += check_almost_eq(s->data[i], e[i]);
  }

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(s);

  if (retval)
    printf("FAILED -> test_matrix_matrix\n");

  return retval;
}

int
test_matrix_row()
{
  /* 1 3 -5
   * 2 4  6
   */

  Tensor* a = T_Build(NULL, 2, 3, 6, 1., 2., 3., 4., -5., 6.);
  Tensor* b = T_Build(NULL, 1, 3, 3, 3., -2., -1.);

  T_eltype e1[] = { 1.0 / 3.0,  2.0 / 3.0, -3.0 / 2.0,
                    -4.0 / 2.0, 5.0 / 1.0, -6.0 / 1.0 };
  T_eltype e2[] = { 3.0 / 1.0,  3.0 / 2.0, -2.0 / 3.0,
                    -2.0 / 4.0, 1.0 / 5.0, -1.0 / 6.0 };

  Tensor* s1 = T_Div(NULL, a, b);
  Tensor* s2 = T_Div(NULL, b, a);

  int retval = 0;
  for (size_t i = 0; i < 6; i++) {
    retval += check_almost_eq(s1->data[i], e1[i]);
    retval += check_almost_eq(s2->data[i], e2[i]);
  }

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(s1);
  T_Destroy(s2);

  if (retval)
    printf("FAILED -> test_matrix_row\n");

  return retval;
}

int
test_matrix_col()
{
  /* 1 3 -5
   * 2 4  6
   */

  Tensor* a = T_Build(NULL, 2, 3, 6, 1., 2., 3., 4., -5., 6.);
  Tensor* b = T_Build(NULL, 2, 1, 2, -2., 3.);

  Tensor* s1 = T_Div(NULL, a, b);
  Tensor* s2 = T_Div(NULL, b, a);

  T_eltype e1[] = { -1.0 / 2.0, 2.0 / 3.0, -3.0 / 2.0,
                    4.0 / 3.0,  5.0 / 2.0, 6.0 / 3.0 };
  T_eltype e2[] = { -2.0 / 1.0, 3.0 / 2.0, -2.0 / 3.0,
                    3.0 / 4.0,  2.0 / 5.0, 3.0 / 6.0 };

  int retval = 0;
  for (size_t i = 0; i < 6; i++) {
    retval += check_almost_eq(s1->data[i], e1[i]);
    retval += check_almost_eq(s2->data[i], e2[i]);
  }

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(s1);
  T_Destroy(s2);

  if (retval)
    printf("FAILED -> test_matrix_col\n");

  return retval;
}

int
test_matrix_scalar()
{

  Tensor* a = T_Build(NULL, 2, 3, 6, 1., 2., 3., 4., -5., 6.);
  Tensor* b = T_Scalar(NULL, 1.1);

  Tensor* s1 = T_Div(NULL, a, b);
  Tensor* s2 = T_Div(NULL, b, a);

  T_eltype e1[] = { 1.0 / 1.1, 2.0 / 1.1,  3.0 / 1.1,
                    4.0 / 1.1, -5.0 / 1.1, 6.0 / 1.1 };
  T_eltype e2[] = { 1.1 / 1.0, 1.1 / 2.0,  1.1 / 3.0,
                    1.1 / 4.0, -1.1 / 5.0, 1.1 / 6.0 };

  int retval = 0;

  for (size_t i = 0; i < 6; i++) {
    retval += check_almost_eq(s1->data[i], e1[i]);
    retval += check_almost_eq(s2->data[i], e2[i]);
  }

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(s1);
  T_Destroy(s2);

  if (retval)
    printf("FAILED -> test_matrix_scalar\n");

  return retval;
}

int
main(void)
{

  return test_matrix_matrix() + test_matrix_row() + test_matrix_col() +
         test_matrix_scalar();
}
