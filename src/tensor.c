#include "scorch/tensor.h"
#include "scorch/scorch.h"
#include <assert.h>
#include "arena.h"
#include <cblas.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tgmath.h>

#define check_sizes(t, a, b)                                                   \
  assert(t->n == a->n && t->m == a->m && t->n == b->n && t->m == b->m)

size_t
nelems(Tensor* t)
{
  return t->n * t->m;
}

bool
isscalar(Tensor* t)
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
T_Zeros(SCORCH_CTX ctx, size_t n, size_t m)
{
  Tensor* t = T_New(ctx, n, m);
  for (size_t i = 0; i < nelems(t); i++)
    t->data[i] = 0.0;
  return t;
}

Tensor*
T_ZerosLike(SCORCH_CTX ctx, Tensor* t)
{
  return T_Zeros(ctx, t->n, t->m);
}

Tensor*
T_Ones(SCORCH_CTX ctx, size_t n, size_t m)
{
  Tensor* t = T_New(ctx, n, m);
  for (size_t i = 0; i < nelems(t); i++)
    t->data[i] = 1.0;
  return t;
}

Tensor*
T_OnesLike(SCORCH_CTX ctx, Tensor* t)
{
  return T_Ones(ctx, t->n, t->m);
}

Tensor*
T_Scalar(SCORCH_CTX ctx, T_eltype a)
{
  Tensor* t = T_New(ctx, 1, 1);
  t->data[0] = a;
  return t;
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
  for (size_t i = 0; i < n*m; i++)
    t->data[i] = s[i];

  return t;
}

void
T_Copy_(Tensor* t, Tensor* a)
{
  assert(t->n == a->n && t->m == a->m);
  for (size_t i = 0; i < nelems(t); i++)
    t->data[i] = a->data[i];
}

Tensor*
T_Copy(SCORCH_CTX ctx, Tensor* a)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_Copy_(t, a);
  return t;
}

T_eltype
T_GetItem(Tensor* t, size_t i, size_t j)
{
  assert(i < t->n && j < t->m);

  return t->data[i + t->n * j];
}

void
T_SetItem(Tensor* t, size_t i, size_t j, T_eltype d)
{
  assert(i < t->n && j < t->m);
  t->data[i + t->n * j] = d;
}

Tensor*
T_Sum(SCORCH_CTX ctx, Tensor* a, Tensor* b)
{
  Tensor* t = T_New(ctx, a->n, a->m);

  T_Sum_(t, a, b);

  return t;
}

void
T_Sum_(Tensor* t, Tensor* a, Tensor* b)
{
  check_sizes(t, a, b);

  for (size_t i = 0; i < (t->n * t->m); i++)
    t->data[i] = a->data[i] + b->data[i];
}

void
T_Add_(Tensor* t, Tensor* a)
{
  T_Sum_(t, t, a);
}

void
T_Diff_(Tensor* t, Tensor* a, Tensor* b)
{
  check_sizes(t, a, b);

  for (size_t i = 0; i < (t->n * t->m); i++)
    t->data[i] = a->data[i] - b->data[i];
}

Tensor*
T_Diff(SCORCH_CTX ctx, Tensor* a, Tensor* b)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_Diff_(t, a, b);
  return t;
}

void
T_Sub_(Tensor* t, Tensor* a)
{
  T_Diff_(t, t, a);
}

void
T_Mul_(Tensor* t, Tensor* a, Tensor* b)
{
  check_sizes(t, a, b);

  for (size_t i = 0; i < nelems(t); i++)
    t->data[i] = a->data[i] * b->data[i];
}

Tensor*
T_Mul(SCORCH_CTX ctx, Tensor* a, Tensor* b)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_Mul_(t, a, b);
  return t;
}

void
T_Div_(Tensor* t, Tensor* a, Tensor* b)
{
  check_sizes(t, a, b);
  for (size_t i = 0; i < nelems(t); i++)
    t->data[i] = a->data[i] / b->data[i];
}

Tensor*
T_Div(SCORCH_CTX ctx ,Tensor* a, Tensor* b)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_Div_(t, a, b);
  return t;
}

void
T_Scale_(Tensor* t, T_eltype a, Tensor* b)
{
  assert(t->n == b->n && t->m == b->m);
  for (size_t i = 0; i < nelems(t); i++)
    t->data[i] = a * b->data[i];
}

Tensor*
T_Scale(SCORCH_CTX ctx, T_eltype a, Tensor* b)
{
  Tensor* t = T_New(ctx, b->n, b->m);
  T_Scale_(t, a, b);
  return t;
}

void
T_SPow_(Tensor* t, Tensor* a, T_eltype b)
{
  assert(t->n == a->n && t->m == a->m);
  for (size_t i = 0; i < nelems(t); i++)
    t->data[i] = pow(a->data[i], b);
}

Tensor*
T_SPow(SCORCH_CTX ctx, Tensor* a, T_eltype b)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_SPow_(t, a, b);

  return t;
}

void
T_Pow_(Tensor* t, Tensor* a, Tensor* b)
{
  check_sizes(t, a, b);
  for (size_t i = 0; i < nelems(t); i++)
    t->data[i] = pow(a->data[i], b->data[i]);
}

Tensor*
T_Pow(SCORCH_CTX ctx, Tensor* a, Tensor* b)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_Pow_(t, a, b);
  return t;
}

void
T_Minus_(Tensor* t, Tensor* a)
{
  for (size_t i = 0; i < nelems(t); i++)
    t->data[i] = -a->data[i];
}

Tensor*
T_Minus(SCORCH_CTX ctx ,Tensor* a)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_Minus_(t, a);
  return t;
}

void
T_Exp_(Tensor* t, Tensor* a)
{
  assert(t->n == a->n && t->m == a->m);
  for (size_t i = 0; i < nelems(t); i++)
    t->data[i] = exp(a->data[i]);
}

Tensor*
T_Exp(SCORCH_CTX ctx, Tensor* a)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_Exp_(t, a);
  return t;
}

void
T_Log_(Tensor* t, Tensor* a)
{
  assert(t->n == a->n && t->m == a->m);
  for (size_t i = 0; i < nelems(t); i++)
    t->data[i] = log(a->data[i]);
}

Tensor*
T_Log(SCORCH_CTX ctx, Tensor* a)
{
  Tensor* t = T_New(ctx, a->n, a->m);
  T_Log_(t, a);
  return t;
}

void
T_GEMM_(Tensor* t, Tensor* a, bool Ta, Tensor* b, bool Tb, T_eltype alpha, T_eltype beta)
{

  cblas_dgemm(
      CblasColMajor,
      Ta ? CblasTrans : CblasNoTrans,
      Tb ? CblasTrans : CblasNoTrans,
      t->n,
      t->m,
      Ta ? a->n : a->m,
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
T_MatMul(SCORCH_CTX ctx, Tensor *a, Tensor*b)
{
  assert(a->m == b->n);

  Tensor* t = T_New(ctx, a->n, b->m);

  T_MatMul_(t, a, b);

  return t;
}

void
T_MatMul_(Tensor* t, Tensor* a,Tensor*  b)
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

