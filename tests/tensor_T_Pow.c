#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdlib.h>

int
test_matrix_matrix() {
  T_eltype a_v[] = { 1, 2, 3, 4, -5, 6 };
  T_eltype b_v[] = { 3, -2, 1, 2, 9, 1 };

  Tensor* a = T_Wrap(NULL, 2, 3, a_v);
  Tensor* b = T_Wrap(NULL, 2, 3, b_v);

  Tensor* s = T_Pow(NULL, a, b);

  T_eltype e[] = {pow(1.0, 3.0), pow(2.0, -2.0), pow(3.0, 1.0), pow(4.0, 2.0), pow(-5.0,9.0), pow(6.0, 1.0)};

  int retval = 0;
  for (size_t i=0; i<6; i++) {
      retval += check_almost_eq(s->data[i], e[i]);
  }

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(s);

  return retval;
}

int
test_matrix_row() {
    /* 1 3 -5
     * 2 4  6
     */
  T_eltype a_v[] = { 1, 2, 3, 4, -5, 6 };
  T_eltype b_v[] = { 3, -2, 1};

  Tensor* a = T_Wrap(NULL, 2, 3, a_v);
  Tensor* b = T_Wrap(NULL, 1, 3, b_v);


  T_eltype e1[] = {pow(1.0, 3.0), pow(2.0,3.0), pow(3.0,-2.0), pow(4.0, -2.0), pow(-5.0, 1.0), pow(6.0, 1.0)};
  T_eltype e2[] = {pow(3.0, 1.0), pow(3.0,2.0), pow(-2.0,3.0), pow(-2.0, 4.0), pow(1.0, -5.0), pow(1.0, 6.0)};

  Tensor* s1 = T_Pow(NULL, a, b);
  Tensor* s2 = T_Pow(NULL, b, a);

  int retval = 0;
  for (size_t i=0; i<6; i++) {
      retval += check_almost_eq(s1->data[i], e1[i]);
      retval += check_almost_eq(s2->data[i], e2[i]);
  }

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(s1);
  T_Destroy(s2);

  return retval;
}

int
test_matrix_col() {
    /* 1 3 -5
     * 2 4  6
     */
  T_eltype a_v[] = { 1, 2, 3, 4, -5, 6 };
  T_eltype b_v[] = { -2, 3};

  Tensor* a = T_Wrap(NULL, 2, 3, a_v);
  Tensor* b = T_Wrap(NULL, 2, 1, b_v);

  Tensor* s1 = T_Pow(NULL, a, b);
  Tensor* s2 = T_Pow(NULL, b, a);

  T_eltype e1[] = {pow(1.0, -2.0), pow(2.0, 3.0), pow(3.0, -2.0), pow(4.0, 3.0), pow(-5.0, -2.0), pow(6.0, 3.0)};
  T_eltype e2[] = {pow(-2.0, 1.0), pow(3.0, 2.0), pow(-2.0, 3.0), pow(3.0, 4.0), pow(-2.0, -5.0), pow(3.0, 6.0)};

  int retval = 0;
  for (size_t i=0; i<6; i++) {
      retval += check_almost_eq(s1->data[i], e1[i]);
      retval += check_almost_eq(s2->data[i], e2[i]);
  }

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(s1);
  T_Destroy(s2);

  return retval;
}

int
test_matrix_scalar()
{
  T_eltype a_v[] = { 1, 2, 3, 4, -5, 6 };

  Tensor* a = T_Wrap(NULL, 2, 3, a_v);
  Tensor* b = T_Scalar(NULL, 1.1);

  Tensor* s1 = T_Pow(NULL, a, b);
  Tensor* s2 = T_Pow(NULL, b, a);

  T_eltype e1[] = {pow(1.0, 1.1), pow(2.0, 1.1), pow(3.0, 1.1), pow(4.0, 1.1), pow(-5.0, 1.1), pow(6.0, 1.1)};
  T_eltype e2[] = {pow(1.1, 1.0), pow(1.1, 2.0), pow(1.1, 3.0), pow(1.1, 4.0), pow(1.1, -5.0), pow(1.1, 6.0)};

  int retval = 0;

  for (size_t i=0; i<6; i++) {
      retval += check_almost_eq(s1->data[i], e1[i]);
      retval += check_almost_eq(s2->data[i], e2[i]);
  }

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(s1);
  T_Destroy(s2);

  return retval;

}


int
main(void)
{

  return test_matrix_matrix() + test_matrix_row() + test_matrix_col()+ test_matrix_scalar();
}
