#include "scorch/tensor.h"
#include "arena.h"
#include "scorch/scorch.h"
#include <assert.h>
#include <cblas.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tgmath.h>

#define check_sizes(t, a, b)                                                   \
  assert(t->n == a->n && t->m == a->m && t->n == b->n && t->m == b->m)

#define max(a, b) ((a) > (b)) ? (a) : (b)

#define data(m, i, j) m->data[(i) + (m->n) * (j)]

static inline void
T_Broadcast_(T_eltype (*operation)(T_eltype, T_eltype),
             const Tensor* const t,
             const Tensor* const a,
             const Tensor* const b)
{
  if ((a->n == b->n) && (a->m == b->m)) {
    /* matrix-matrix */
    for (size_t i = 0; i < T_nelems(t); i++) {
      t->data[i] = operation(a->data[i], b->data[i]);
    }
  } else if ((b->n == 1) && (b->m == 1)) {
    /* matrix-scalar */
    for (size_t i = 0; i < T_nelems(t); i++) {
      t->data[i] = operation(a->data[i], b->data[0]);
    }
  } else if ((a->n == b->n) && (b->m == 1)) {
    /* matrix-column */
    for (size_t i = 0; i < T_nelems(a); i++) {
      t->data[i] = operation(a->data[i], b->data[i % a->n]);
    }
  } else if ((a->m == b->m) && (b->n == 1)) {
    /* matrix-row */
    for (size_t i = 0; i < T_nelems(a); i++) {
      t->data[i] = operation(a->data[i], b->data[i / a->n]);
    }
  } else if ((a->n == b->n) && (a->m == 1)) {
    /* column-matrix */
    for (size_t i = 0; i < T_nelems(b); i++) {
      t->data[i] = operation(a->data[i % b->n], b->data[i]);
    }
  } else if ((a->n == 1) && (a->m == b->m)) {
    /* row-matrix */
    for (size_t i = 0; i < T_nelems(b); i++) {
      t->data[i] = operation(a->data[i / b->n], b->data[i]);
    }
  } else if ((a->n == 1) && (a->m == 1)) {
    /* scalar-matrix */
    for (size_t i = 0; i < T_nelems(b); i++) {
      t->data[i] = operation(a->data[0], b->data[i]);
    }
  } else {
    assert(0);
  }
}

static inline T_eltype
op_sum(T_eltype a, T_eltype b)
{
  return a + b;
}

static inline T_eltype
op_mul(T_eltype a, T_eltype b)
{
  return a * b;
}

static inline T_eltype
op_diff(T_eltype a, T_eltype b)
{
  return a - b;
}

static inline T_eltype
op_div(T_eltype a, T_eltype b)
{
  return a / b;
}

size_t
T_nelems(const Tensor* const t)
{
  return t->n * t->m;
}

bool
isscalar(const Tensor* const t)
{
  return t->n == 1 && t->m == 1;
}

Tensor*
T_New(SCORCH_CTX ctx, size_t n, size_t m)
{
  Tensor* t = T_Malloc(ctx);
  t->n = n;
  t->m = m;
  t->data = malloc(sizeof(T_eltype) * n * m);

  return t;
}

Tensor*
T_Full(SCORCH_CTX ctx, size_t n, size_t m, T_eltype a)
{
  Tensor* t = T_New(ctx, n, m);
  for (size_t i = 0; i < T_nelems(t); i++)
    t->data[i] = a;
  return t;
}

Tensor*
T_FullLike(SCORCH_CTX ctx, const Tensor* const t, T_eltype a)
{
  return T_Full(ctx, t->n, t->m, a);
}

Tensor*
T_Ones(SCORCH_CTX ctx, size_t n, size_t m)
{
  return T_Full(ctx, n, m, 1.0);
}

Tensor*
T_Zeros(SCORCH_CTX ctx, size_t n, size_t m)
{
  return T_Full(ctx, n, m, 0.0);
}

Tensor*
T_ZerosLike(SCORCH_CTX ctx, const Tensor* const t)
{
  return T_Zeros(ctx, t->n, t->m);
}

Tensor*
T_OnesLike(SCORCH_CTX ctx, const Tensor* const t)
{
  return T_Ones(ctx, t->n, t->m);
}

Tensor*
T_Scalar(SCORCH_CTX ctx, T_eltype a)
{
  return T_Full(ctx, 1, 1, a);
}

void*
T_Destroy(Tensor* t)
{
  free(t->data);
  free(t);

  return NULL;
}

Tensor*
T_Wrap(SCORCH_CTX ctx, size_t n, size_t m, T_eltype s[static n * m])
{
  Tensor* t = T_New(ctx, n, m);
  for (size_t i = 0; i < n * m; i++)
    t->data[i] = s[i];

  return t;
}

void
T_Copy_(Tensor* const t, const Tensor* const a)
{
  assert(t->n == a->n && t->m == a->m);
  for (size_t i = 0; i < T_nelems(t); i++)
    t->data[i] = a->data[i];
}

Tensor*
T_Copy(SCORCH_CTX ctx, const Tensor* const a)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_Copy_(t, a);
  return t;
}

T_eltype
T_GetItem(const Tensor* const t, size_t i, size_t j)
{
  assert(i < t->n && j < t->m);

  return t->data[i + t->n * j];
}

void
T_SetItem(Tensor* const t, size_t i, size_t j, T_eltype d)
{
  assert(i < t->n && j < t->m);
  t->data[i + t->n * j] = d;
}

Tensor*
T_Sum(SCORCH_CTX ctx, const Tensor* const a, const Tensor* const b)
{
  size_t n = max(a->n, b->n);
  size_t m = max(a->m, b->m);
  Tensor* t = T_New(ctx, n, m);

  T_Sum_(t, a, b);

  return t;
}

void
T_Sum_(Tensor* const t, const Tensor* const a, const Tensor* const b)
{
  T_Broadcast_(op_sum, t, a, b);
}

void
T_Add_(Tensor* const t, const Tensor* const a)
{
  T_Sum_(t, t, a);
}

void
T_Diff_(Tensor* const t, const Tensor* const a, const Tensor* const b)
{
  T_Broadcast_(op_diff, t, a, b);
}

Tensor*
T_Diff(SCORCH_CTX ctx, const Tensor* const a, const Tensor* const b)
{
  size_t n = max(a->n, b->n);
  size_t m = max(a->m, b->m);
  Tensor* t = T_New(ctx, n, m);

  T_Diff_(t, a, b);
  return t;
}

void
T_Sub_(Tensor* const t, const Tensor* const a)
{
  T_Diff_(t, t, a);
}

void
T_Mul_(Tensor* const t, const Tensor* const a, const Tensor* const b)
{
  T_Broadcast_(op_mul, t, a, b);
}

Tensor*
T_Mul(SCORCH_CTX ctx, const Tensor* const a, const Tensor* const b)
{
  size_t n = max(a->n, b->n);
  size_t m = max(a->m, b->m);
  Tensor* t = T_New(ctx, n, m);

  T_Mul_(t, a, b);
  return t;
}

void
T_Div_(Tensor* const t, const Tensor* const a, const Tensor* const b)
{
  T_Broadcast_(op_div, t, a, b);
}

Tensor*
T_Div(SCORCH_CTX ctx, const Tensor* const a, const Tensor* const b)
{
  size_t n = max(a->n, b->n);
  size_t m = max(a->m, b->m);
  Tensor* t = T_New(ctx, n, m);

  T_Div_(t, a, b);
  return t;
}

void
T_Scale_(Tensor* const t, T_eltype a, const Tensor* const b)
{
  assert(t->n == b->n && t->m == b->m);
  for (size_t i = 0; i < T_nelems(t); i++)
    t->data[i] = a * b->data[i];
}

Tensor*
T_Scale(SCORCH_CTX ctx, T_eltype a, const Tensor* const b)
{
  Tensor* t = T_New(ctx, b->n, b->m);
  T_Scale_(t, a, b);
  return t;
}

void
T_SPow_(Tensor* const t, const Tensor* const a, T_eltype b)
{
  assert(t->n == a->n && t->m == a->m);
  for (size_t i = 0; i < T_nelems(t); i++)
    t->data[i] = pow(a->data[i], b);
}

Tensor*
T_SPow(SCORCH_CTX ctx, const Tensor* const a, T_eltype b)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_SPow_(t, a, b);

  return t;
}

void
T_Pow_(Tensor* const t, const Tensor* const a, const Tensor* const b)
{
  T_Broadcast_(pow, t, a, b);
}

Tensor*
T_Pow(SCORCH_CTX ctx, const Tensor* const a, const Tensor* const b)
{
  size_t n = max(a->n, b->n);
  size_t m = max(a->m, b->m);
  Tensor* t = T_New(ctx, n, m);

  T_Pow_(t, a, b);
  return t;
}

void
T_Minus_(Tensor* const t, const Tensor* const a)
{
  for (size_t i = 0; i < T_nelems(t); i++)
    t->data[i] = -a->data[i];
}

Tensor*
T_Minus(SCORCH_CTX ctx, const Tensor* const a)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_Minus_(t, a);
  return t;
}

void
T_Exp_(Tensor* const t, const Tensor* const a)
{
  assert(t->n == a->n && t->m == a->m);
  for (size_t i = 0; i < T_nelems(t); i++)
    t->data[i] = exp(a->data[i]);
}

Tensor*
T_Exp(SCORCH_CTX ctx, const Tensor* const a)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_Exp_(t, a);
  return t;
}

void
T_Log_(Tensor* const t, const Tensor* const a)
{
  assert(t->n == a->n && t->m == a->m);
  for (size_t i = 0; i < T_nelems(t); i++)
    t->data[i] = log(a->data[i]);
}

Tensor*
T_Log(SCORCH_CTX ctx, const Tensor* const a)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_Log_(t, a);
  return t;
}

void
T_GEMM_(Tensor* const t,
        const Tensor* const a,
        bool transpose_a,
        const Tensor* const b,
        bool transpose_b,
        T_eltype alpha,
        T_eltype beta)
{

  cblas_dgemm(CblasColMajor,
              transpose_a ? CblasTrans : CblasNoTrans,
              transpose_b ? CblasTrans : CblasNoTrans,
              t->n,
              t->m,
              transpose_a ? a->n : a->m,
              alpha,
              a->data,
              a->n,
              b->data,
              b->n,
              beta,
              t->data,
              t->n);
}

Tensor*
T_MatMul(SCORCH_CTX ctx, const Tensor* const a, const Tensor* const b)
{
  assert(a->m == b->n);

  Tensor* t = T_New(ctx, a->n, b->m);

  T_MatMul_(t, a, b);

  return t;
}

void
T_MatMul_(Tensor* const t, const Tensor* const a,const Tensor* const b)
{
  assert(t->n == a->n && t->m == b->m && a->m == b->n);

  /*
  for (size_t i=0; i<t->n; i++){
    for (size_t j=0; j<t->m; j++){
      t->data[i + j*t->n] = 0;
      for (size_t k=0; k< a->m; k++)
        t->data[i + j*t->n] += a->data[i + k*t->n] * b->data[k + j * b->n];
    }
  }
  */

  T_GEMM_(t, a, false, b, false, 1.0, 0.0);
}

Tensor*
T_SumReduce(SCORCH_CTX ctx, const Tensor* const a)
{
  Tensor* t = T_New(ctx, a->n, 1);
  T_SumReduce_(t, a);

  return t;
}

void
T_SumReduce_(Tensor* restrict const t, const Tensor* restrict const a)
{
  assert(t->n == a->n && t->m == 1);

  for (size_t i = 0; i < t->n; i++) {
    t->data[i] = 0;
    for (size_t j = 0; j < a->m; j++) {
      t->data[i] += a->data[i + a->n * j];
    }
  }
}
