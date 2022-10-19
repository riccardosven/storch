#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"
#include <stdlib.h>
#include <stdio.h>


int
test_matrix_matrix() {
  T_eltype a_v[] = { 1, 2, 3, 4, -5, 6 };
  T_eltype b_v[] = { 0, -2, 1, 2, 9, 1 };

  Tensor* a = T_Wrap(NULL, 2, 3, a_v);
  Tensor* b = T_Wrap(NULL, 2, 3, b_v);

  Tensor* s = T_BroadcastAdd(NULL, a, b);

  T_eltype e[] = {1, 0, 4, 6, 4, 7};

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
  T_eltype b_v[] = { 0, -2, 1};

  Tensor* a = T_Wrap(NULL, 2, 3, a_v);
  Tensor* b = T_Wrap(NULL, 1, 3, b_v);


  T_eltype e[] = {1, 2, 1, 2, -4, 7};

  Tensor* s1 = T_BroadcastAdd(NULL, a, b);
  Tensor* s2 = T_BroadcastAdd(NULL, b, a);

  int retval = 0;
  for (size_t i=0; i<6; i++) {
      retval += check_almost_eq(s1->data[i], e[i]);
      retval += check_almost_eq(s2->data[i], e[i]);
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
  T_eltype b_v[] = { -2, 2};

  Tensor* a = T_Wrap(NULL, 2, 3, a_v);
  Tensor* b = T_Wrap(NULL, 2, 1, b_v);

  Tensor* s1 = T_BroadcastAdd(NULL, a, b);
  Tensor* s2 = T_BroadcastAdd(NULL, b, a);

  T_eltype e[] = {-1, 4, 1, 6, -7, 8};

  int retval = 0;
  for (size_t i=0; i<6; i++) {
      retval += check_almost_eq(s1->data[i], e[i]);
      retval += check_almost_eq(s2->data[i], e[i]);
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

  Tensor* s1 = T_BroadcastAdd(NULL, a, b);
  Tensor* s2 = T_BroadcastAdd(NULL, b, a);

  T_eltype e[] = {2.1, 3.1, 4.1, 5.1, -3.9, 7.1};

  int retval = 0;

  for (size_t i=0; i<6; i++) {
      retval += check_almost_eq(s1->data[i], e[i]);
      retval += check_almost_eq(s2->data[i], e[i]);
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

  return test_matrix_matrix() + test_matrix_row()+ test_matrix_col() + test_matrix_scalar();
}
