#ifndef SCORCH_TENSOR_H
#define SCORCH_TENSOR_H

#include "scorch/scorch.h"
#include <stdbool.h>
#include <stddef.h>

typedef double T_eltype;

typedef struct scorch_tensor_s
{
  size_t n; // rows
  size_t m; // columns
  T_eltype* data;
} Tensor;

size_t
nelems(Tensor* t);

Tensor* T_New(SCORCH_CTX, size_t, size_t);

Tensor* T_Zeros(SCORCH_CTX, size_t, size_t);

Tensor* T_Ones(SCORCH_CTX, size_t, size_t);

Tensor*
T_ZerosLike(SCORCH_CTX, Tensor*);

Tensor*
T_OnesLike(SCORCH_CTX, Tensor*);

void*
T_Destroy(Tensor* t);

Tensor*
T_Wrap(SCORCH_CTX, size_t n, size_t m, T_eltype[static n*m]);

Tensor* T_Scalar(SCORCH_CTX, T_eltype);

Tensor*
T_Full(SCORCH_CTX, size_t n, size_t m, T_eltype);

Tensor*
T_FullLike(SCORCH_CTX, Tensor*, T_eltype);

Tensor*
T_Copy(SCORCH_CTX, Tensor*);

void
T_Copy_(Tensor*, Tensor*);

T_eltype
T_GetItem(Tensor*, size_t, size_t);

void
T_SetItem(Tensor*, size_t, size_t, T_eltype);

Tensor*
T_Sum(SCORCH_CTX, Tensor*, Tensor*);

void
T_Sum_(Tensor*, Tensor*, Tensor*);

Tensor*
T_Diff(SCORCH_CTX, Tensor*, Tensor*);

void
T_Diff_(Tensor*, Tensor*, Tensor*);

void
T_Sub_(Tensor*, Tensor*);

void
T_Add_(Tensor*, Tensor*);

Tensor*
T_Mul(SCORCH_CTX, Tensor*, Tensor*);

void
T_Mul_(Tensor*, Tensor*, Tensor*);

Tensor*
T_Div(SCORCH_CTX, Tensor*, Tensor*);

void
T_Div_(Tensor*, Tensor*, Tensor*);

Tensor*
T_Scale(SCORCH_CTX, T_eltype, Tensor*);

void
T_Scale_(Tensor*, T_eltype, Tensor*);

Tensor*
T_SPow(SCORCH_CTX, Tensor*, T_eltype);

void
T_SPow_(Tensor*, Tensor*, T_eltype);

Tensor*
T_Pow(SCORCH_CTX, Tensor*, Tensor*);

void
T_Pow_(Tensor*, Tensor*, Tensor*);

Tensor*
T_Exp(SCORCH_CTX, Tensor*);

void
T_Exp_(Tensor*, Tensor*);

Tensor*
T_Log(SCORCH_CTX, Tensor*);

void
T_Log_(Tensor*, Tensor*);

Tensor*
T_Minus(SCORCH_CTX, Tensor*);

void
T_Minus_(Tensor*, Tensor*);

void
T_GEMM_(Tensor*, Tensor*, bool, Tensor*, bool, T_eltype, T_eltype);

Tensor*
T_MatMul(SCORCH_CTX, Tensor*, Tensor*);

void
T_MatMul_(Tensor*, Tensor*, Tensor*);

#endif // SCORCH_TENSOR_H
