#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"
#include <stdlib.h>

int
main(void)
{
  T_eltype v[] = { 1, 2, 3, 4, 5, 6 };

  Tensor* t = T_Wrap(3, 2, v);
  Tensor* o = T_OnesLike(t);

  // s = t - 1
  Tensor* s = T_Diff(t, o);

  int retval = s->data != t->data && s->data != o->data;

  for (size_t i = 0; i < nelems(t); i++) {
    retval = retval && (s->data[i] == t->data[i] - 1);
  }

  // t = s - 1
  T_Diff_(t, s, o);

  for (size_t i = 0; i < nelems(t); i++) {
    retval = retval && (t->data[i] == s->data[i] - 1);
  }

  // t -= 1
  T_Sub_(t, o);

  for (size_t i = 0; i < nelems(t); i++) {
    retval = retval && (t->data[i] == s->data[i] - 2);
  }

  // r = -t
  Tensor* r = T_Minus(t);
  for (size_t i = 0; i < nelems(t); i++) {
    retval = retval && (r->data[i] == -t->data[i]);
  }

  T_Destroy(t);
  T_Destroy(o);
  T_Destroy(s);
  T_Destroy(r);

  return retval ? EXIT_SUCCESS : EXIT_FAILURE;
}
