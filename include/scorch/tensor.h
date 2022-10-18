#ifndef SCORCH_TENSOR_H
#define SCORCH_TENSOR_H
#include <stdlib.h>
#include <stdbool.h>

typedef double T_eltype;

typedef struct
{
  size_t n; // rows
  size_t m; // columns
  T_eltype* data;
} Tensor;

size_t
nelems(Tensor* t);

Tensor* T_New(size_t, size_t);

Tensor* T_Zeros(size_t, size_t);

Tensor* T_Ones(size_t, size_t);

Tensor* T_ZerosLike(Tensor*);

Tensor* T_OnesLike(Tensor*);

void*
T_Destroy(Tensor* t);

Tensor*
T_Wrap(size_t n, size_t m, T_eltype[]);

Tensor* T_Scalar(T_eltype);

Tensor*
T_Copy(Tensor*);

void
T_Copy_(Tensor*, Tensor*);

T_eltype
T_GetItem(Tensor*, size_t, size_t);

void
T_SetItem(Tensor*, size_t, size_t, T_eltype);

Tensor*
T_Sum(Tensor*, Tensor*);

void
T_Sum_(Tensor*, Tensor*, Tensor*);

Tensor*
T_Diff(Tensor*, Tensor*);

void
T_Diff_(Tensor*, Tensor*, Tensor*);

void
T_Sub_(Tensor*, Tensor*);

void
T_Add_(Tensor*, Tensor*);

Tensor*
T_Mul(Tensor*, Tensor*);

void
T_Mul_(Tensor*, Tensor*, Tensor*);

Tensor*
T_Div(Tensor*, Tensor*);

void
T_Div_(Tensor*, Tensor*, Tensor*);

Tensor*
T_Scale(T_eltype, Tensor*);

void
T_Scale_(Tensor*, T_eltype, Tensor*);

Tensor*
T_SPow(Tensor*, T_eltype);

void
T_SPow_(Tensor*, Tensor*, T_eltype);

Tensor*
T_Pow(Tensor*, Tensor*);

void
T_Pow_(Tensor*, Tensor*, Tensor*);

Tensor*
T_Exp(Tensor*);

void
T_Exp_(Tensor*, Tensor*);

Tensor*
T_Log(Tensor*);

void
T_Log_(Tensor*, Tensor*);

Tensor*
T_Minus(Tensor*);

void
T_Minus_(Tensor*, Tensor*);

void
T_GEMM_(Tensor*, Tensor*, bool, Tensor*, bool, T_eltype, T_eltype);

Tensor*
T_MatMul(Tensor*, Tensor*);

void
T_MatMul_(Tensor*, Tensor*, Tensor*);

#endif // SCORCH_TENSOR_H
