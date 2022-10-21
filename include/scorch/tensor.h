/** @file
 * Library of tensor operations.
 *
 * In Scorch, tensors are limited to two-dimensional data structures (think: matrices).
 * This library contains the basic mathematical operations on tensors.
 *
 * @note Tensors are dynamically allocated either on the stack or in a `SCORCH_CTX`.
 * @note Many operations use numpy-style broadcasting of unitary dimensions.
 *
 */
#ifndef SCORCH_TENSOR_H
#define SCORCH_TENSOR_H

#include "scorch/scorch.h"
#include <stdbool.h>
#include <stddef.h>

typedef double T_eltype; /**< Type of tensor elements. */

/** Tensor structure implementation.
 *
 * Wraps a contiguous segment of memory into a two-dimensional tensor in column-major order.
 */
typedef struct scorch_tensor_s
{
  size_t n; /**< Number of rows */
  size_t m; /**< Number of columns */
  T_eltype* data; /**< Pointer to the tensor data (column-major layout). */
} Tensor;

/** Find the number of elements of a tensor
 *
 * @param t[in] input tensor.
 *
 * @returns The number of elements (#rows times #columns);
 */
size_t
T_nelems(const Tensor* const t);

/**
 * Create a new tensor.
 *
 * @param ctx[in] Optional Scorch context.
 * @param n[in] Number of rows.
 * @param m[in] Number of columns.
 *
 * @returns A pointer to the allocated tensor.
 *
 * @note A `NULL` context requires explicit destruction!
 * @see T_New
 * @see T_Destroy
 */
Tensor*
T_New(SCORCH_CTX ctx, size_t n, size_t m);

/**
 * Create a new zero-initialized tensor.
 *
 * @param ctx[in] Optional Scorch context.
 * @param n[in] Number of rows.
 * @param m[in] Number of columns.
 *
 * @returns A pointer to the allocated tensor.
 *
 * @note A `NULL` context requires explicit destruction!
 *
 */
Tensor*
T_Zeros(SCORCH_CTX, size_t, size_t);

/**
 * Create a new tensor filled with ones.
 *
 * @param ctx[in] Optional Scorch context.
 * @param n[in] Number of rows.
 * @param m[in] Number of columns.
 *
 * @returns A pointer to the allocated tensor.
 *
 * @note A `NULL` context requires explicit destruction!
 *
 */
Tensor*
T_Ones(SCORCH_CTX, size_t, size_t);

/**
 * Create a new zero-initialized tensor with the same shape as the given tensor.
 *
 * @param ctx[in] Optional Scorch context.
 * @param t[in] Reference tensor.
 *
 * @returns A pointer to the allocated tensor.
 *
 * @note A `NULL` context requires explicit destruction!
 *
 */
Tensor*
T_ZerosLike(SCORCH_CTX, const Tensor* const t);

/**
 * Create a new one-initialized tensor with the same shape as the given tensor.
 *
 * @param ctx[in] Optional Scorch context.
 * @param t[in] Reference tensor.
 *
 * @returns A pointer to the allocated tensor.
 *
 * @note A `NULL` context requires explicit destruction!
 *
 */
Tensor*
T_OnesLike(SCORCH_CTX, const Tensor* const t);

/**
 * Destroy the given tensor and deallocate the memory; returns a `NULL` pointer.
 *
 * @param t[in] Pointer to the tensor to deallocate.
 *
 * @returns `NULL` pointer.
 *
 * @note Do not use on Tensors allocated in a `SCORCH_CTX`!
 *
 */
void*
T_Destroy(Tensor* t);

/**
 * Create a new `n` times `m` initialized with `data`.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[in] n Number of rows.
 * @param[in] m Number of columns.
 * @param[in] data Initialization data in column-major order.
 *
 * @returns A pointer to the allocated tesor.
 */
Tensor*
T_Wrap(SCORCH_CTX ctx, size_t n, size_t m, T_eltype data[static n*m]);

/**
 * Create a new 1 by 1 tensor.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[in] d Value of the scalar tensor.
 *
 * @returns A pointer to the allocated tesor.
 */
Tensor*
T_Scalar(SCORCH_CTX, T_eltype);

/**
 * Create a new `n` times `m` tensor where all the entries are equal to `d`.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[in] n Number of rows.
 * @param[in] m Number of columns.
 * @param[in] d Value to fill with.
 *
 * @returns A pointer to the allocated tesor.
 */
Tensor*
T_Full(SCORCH_CTX ctx, size_t n, size_t m, T_eltype d);

/**
 * Create a new tensor with the same shape as `t` where all the entries are equal to `d`.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[in] t Tensor whose shape is copied.
 * @param[in] d Value to fill with.
 *
 * @returns A pointer to the allocated tesor.
 */
Tensor*
T_FullLike(SCORCH_CTX ctx, const Tensor* const t, T_eltype d);

/**
 * Copy the given tensor `t`.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[in] t Tensor to copy
 *
 * @returns A pointer to the allocated tesor.
 *
 * @note Allocates a new tensor; appropriate destruction is required.
 */
Tensor*
T_Copy(SCORCH_CTX ctx, const Tensor* const t);

/**
 * In-place copy of a tensor: `t=a`.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[out] t Tensor to copy to.
 * @param[in] a Tensor to copy.
 *
 * @returns A pointer to the allocated tesor.
 *
 */
void
T_Copy_(Tensor* const t, const Tensor* const a);

/**
 * Get the `i,j`th element of the tensor.
 *
 * @param[in] t Tensor.
 * @param[in] i Row of the element.
 * @param[in] j Column of the element.
 *
 */
T_eltype
T_GetItem(const Tensor* const t, size_t i, size_t j);

/**
 * Set the value of the `i,j`th element of the tensor.
 *
 * @param[in,out] t Tensor; modified in place.
 * @param[in] i Row of the element.
 * @param[in] j Column of the element.
 * @param[in] d Value to set.
 *
 */
void
T_SetItem(Tensor* const t, size_t i , size_t j, T_eltype d);

/**
 * Element-wise tensor addition.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[in] a First tensor.
 * @param[in] b Second tensor.
 *
 * @returns A pointer to `a + b`.
 *
 */
Tensor*
T_Sum(SCORCH_CTX ctx , const Tensor* const a, const Tensor* const b);

/**
 * In-place the element-wise tensor addition: `t = a + b`.
 *
 * @param[out] t Output tensor.
 * @param[in] a First tensor.
 * @param[in] b Second tensor.
 *
 */
void
T_Sum_(Tensor* const t, const Tensor* const a, const Tensor* const b);

/**
 * Element-wise tensor difference.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[in] a First tensor.
 * @param[in] b Second tensor.
 *
 * @returns A pointer to `a - b`.
 *
 */
Tensor*
T_Diff(SCORCH_CTX, const Tensor* const a, const Tensor* const b);

/**
 * In-place the element-wise tensor difference: `t = a - b`.
 *
 * @param[out] t Output tensor.
 * @param[in] a First tensor.
 * @param[in] b Second tensor.
 *
 */
void
T_Diff_(Tensor* const t, const Tensor* const a, const Tensor* const b);

/**
 * In-place the element-wise tensor subtraction: `t -= a`.
 *
 * @param[in,out] t Tensor to operate upon.
 * @param[in] a Tensor to subtract.
 *
 */
void
T_Sub_(Tensor* const t, const Tensor* const a);

/**
 * In-place the element-wise tensor addition: `t += a`.
 *
 * @param[in,out] t Tensor to operate upon.
 * @param[in] a Tensor to add.
 *
 */
void
T_Add_(Tensor* const t, const Tensor* const a);

/**
 * Element-wise tensor multiplication.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[in] a First tensor.
 * @param[in] b Second tensor.
 *
 * @returns A pointer to `a * b`.
 *
 */
Tensor*
T_Mul(SCORCH_CTX, const Tensor* const a, const Tensor* const b);


/**
 * In-place the element-wise tensor multiplication: `t = a * b`.
 *
 * @param[out] t Output tensor.
 * @param[in] a First tensor.
 * @param[in] b Second tensor.
 *
 */
void
T_Mul_(Tensor* const t, const Tensor* const a, const Tensor* const b);

/**
 * Element-wise tensor division.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[in] a First tensor.
 * @param[in] b Second tensor.
 *
 * @returns A pointer to `a / b`.
 *
 */
Tensor*
T_Div(SCORCH_CTX ctx, const Tensor* const a, const Tensor* const b);

/**
 * In-place the element-wise tensor division: `t = a / b`.
 *
 * @param[out] t Output tensor.
 * @param[in] a First tensor.
 * @param[in] b Second tensor.
 *
 */
void
T_Div_(Tensor* const t, const Tensor* const a, const Tensor* const b);


/**
 * Element-wise tensor scaling.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[in] alpha The scaling factor.
 * @param[in] a The tensor to scale.
 *
 * @return A pointer to `alpha * a`.
 *
 */
Tensor*
T_Scale(SCORCH_CTX ctx, T_eltype alpha, const Tensor* const a);

/**
 * In-place element-wise tensor scaling: `t = alpha * a`.
 *
 * @param[out] t The scaled tensor.
 * @param[in] alpha The scaling factor.
 * @param[in] a The tensor to scale.
 *
 */
void
T_Scale_(Tensor* const t, T_eltype alpha , const Tensor* const a);

/**
 * Element-wise tensor powering.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[in] a The input tensor.
 * @param[in] e The exponent
 *
 * @return A pointer to `a ** e`.
 *
 */
Tensor*
T_SPow(SCORCH_CTX cxt, const Tensor* const a , T_eltype e);

/**
 * In-place element-wise tensor powering: `t = a ** e`.
 *
 * @param[out] t The output tensor.
 * @param[in] a The input tensor.
 * @param[in] e The exponent
 *
 */
void
T_SPow_(Tensor* const t, const Tensor* const a, T_eltype e);

/**
 * Element-wise broadcasting tensor powering.
 * @param[in] ctx Optional Scorch context.
 * @param[in] a The input tensor.
 * @param[in] e The exponent tensor.
 *
 * @return A pointer to `a ** b`.
 *
 */
Tensor*
T_Pow(SCORCH_CTX ctx, const Tensor* const a, const Tensor* const b);

/**
 * In-place element-wise broadcasting tensor powering: `t=a**e`.
 * @param[out] t The output tensor.
 * @param[in] a The input tensor.
 * @param[in] e The exponent tensor.
 *
 */
void
T_Pow_(Tensor* const t, const Tensor* const a, const Tensor* const b);

/**
 * Element-wise tensor exponential.
 * 
 * @param[in] ctx Optional Scorch context.
 * @param[in] a The input tensor.
 *
 * @returns A pointer to `exp(a)`.
 *
 */
Tensor*
T_Exp(SCORCH_CTX ctx, const Tensor* const a);

/**
 * In-place element-wise tensor exponential: `t=exp(a)`.
 * 
 * @param[out] t The output tensor.
 * @param[in] a The input tensor.
 *
 */
void
T_Exp_(Tensor* const t, const Tensor* const a);

/**
 * Element-wise tensor natural logarithm.
 * 
 * @param[in] ctx Optional Scorch context.
 * @param[in] a The input tensor.
 *
 * @returns A pointer to `log(a)`.
 *
 */
Tensor*
T_Log(SCORCH_CTX ctx, const Tensor* const a);

/**
 * In-place element-wise tensor natural logarith: `t=log(a)`.
 * 
 * @param[out] t The output tensor.
 * @param[in] a The input tensor.
 *
 */
void
T_Log_(Tensor* const t, const Tensor* const a);

/**
 * Tensor unitary negation.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[in] a The input tensor.
 *
 * @returns A pointer to `-a`.
 *
 */
Tensor*
T_Minus(SCORCH_CTX ctx, const Tensor* const a);

/**
 * In-place tensor unitary negation: `t=-a`.
 *
 * @param[out] t The output tensor.
 * @param[in] a The input tensor.
 *
 */
void
T_Minus_(Tensor* const t, const Tensor* const a);

/**
 * Wrapper for the `BLAS` generalized matrix-matrix multiplication:
 * `t = alpha * a(transpose?) @ b(transpose?) + beta * t`
 *
 *
 * @param[in, out] t The ouput tensor.
 * @param[in] a The first operand.
 * @param[in] transpose_a If `a` should be transposed before the multiplication.
 * @param[in] b The second operand.
 * @param[in] transpose_b If `b` should be transposed before the multiplication.
 * @param[in] alpha First scaling factor.
 * @param[in] beta Second scaling factor.
 */
void
T_GEMM_(Tensor* const t, const Tensor* const a, bool transpose_a, const Tensor* const b, bool transpose_b, T_eltype alpha, T_eltype beta);

/**
 * Matrix multiplication.
 *
 * @param[in] ctx: Optional  Scorch context.
 * @param[in] a First operand.
 * @param[in] b Second operand.
 *
 * @returns A pointer to `a@b`.
 *
 */
Tensor*
T_MatMul(SCORCH_CTX ctx, const Tensor* const a, const Tensor* const b);

/**
 * In-place matrix multiplication: `t=a@b`.
 *
 * @param[out] t Output tensor.
 * @param[in] a First operand.
 * @param[in] b Second operand.
 *
 */
void
T_MatMul_(Tensor* const t, const Tensor* const a, const Tensor* const b);

/**
 * Tensor column reduction operation.
 *
 * @param[in] ctx Optional Scorch context.
 * @param[in] a Operand.
 *
 * @returns a pointer to a column tensor `t` where `t[i] = sum(a[i, j] for all j)`.
 *
 */
Tensor*
T_SumReduce(SCORCH_CTX ctx, const Tensor* const a);

/**
 * In-place tensor column reduction operation: `t[i] = sum(a[i, j] for all j)`.
 *
 * @param[out] t Output tensor.
 * @param[in] a Operand tensor.
 *
 */
void
T_SumReduce_(Tensor* restrict const t, const Tensor* restrict const a);

#endif // SCORCH_TENSOR_H
