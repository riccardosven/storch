#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"
#include <stdlib.h>

int
main(void)
{
  T_eltype v[] = { 1, 2, 3, 4, 5, 6 };

  Tensor* t = T_Wrap(NULL,3, 2, v);
  Tensor* o = T_OnesLike(NULL, t);

  // s = t + 1
  Tensor* s = T_Sum(NULL, t, o);
  int retval = 1;
  for (size_t i = 0; i < nelems(t); i++) {
    retval = retval && (s->data[i] == t->data[i] + 1);
  }

  // t = t + 1
  T_Sum_(t, t, o);
  for (size_t i = 0; i < nelems(t); i++) {
    retval = retval && (s->data[i] == t->data[i]);
  }

  // t += 1
  T_Add_(t, o);
  for (size_t i = 0; i < nelems(t); i++) {
    retval = retval && (t->data[i] == s->data[i] + 1);
  }

  T_Destroy(t);
  T_Destroy(o);
  T_Destroy(s);

  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
