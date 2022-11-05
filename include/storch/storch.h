/** @file
 * Library of graph operations for autodifferentiation.
 *
 * Storch keeps a dynamic graph of operations that is defined via the functions
 * in this module. No actual operations are performed until the `G_forward`
 * function is called and the graph is realized. After graph realization, the
 * `G_backward` function can be used to compute the gradient graph and for al
 * variables in the graph.
 */
#ifndef STORCH_H
#define STORCH_H

#include <stdio.h>

typedef struct storch_ctx_s*
  STORCH_CTX; /**< Opaque data structure defining a graph contex */
typedef struct graphnode_s GraphNode; /**< A node in the storch graph. */
typedef struct storch_tensor_s Tensor;

/**
 * @name Graph Context functions
 *
 * Functions used to manipulate the `SCORCH_CTX` arena allocator.
 *
 * @{
 */

/**
 * Create a new graph context.
 *
 * A Graph context is essentially an arena allocator that keeps track of all
 * `GraphNode` and `Tensor` objects.
 * @note Objects allocated inside a `STORCH_CTX` should not be freed; the whole
 * context is destroyed with `STORCH_CTX_DESTROY`!
 *
 * @returns The created context.
 *
 */
STORCH_CTX
STORCH_CTX_New(void);

/**
 * Destroy a graph context.
 *
 * Iteratively destroys all `GraphNode` and `Tensor` objects allocated in the
 * context.
 *
 * @param[in] ctx Storch context to be destroyed.
 *
 */
void*
STORCH_CTX_Destroy(STORCH_CTX ctx);

/** @}*/

/** @name Graph transformation functions
 *
 * Functions used to define the computational graph.
 *
 * @{
 */

/**
 * Represents a fixed value.
 *
 * @param[in] ctx Storch context to manage the graph.
 * @param[in] x Tensor with the value.
 *
 * @note No gradient will be computed for this node.
 */
GraphNode*
G_Value(const STORCH_CTX ctx, Tensor* const x);

/**
 * Represents a variable parameter.
 *
 * @param[in] ctx Storch context to manage the graph.
 * @param[in] x Tensor with the value of the parameter.
 *
 * @note Gradient will be computed for this node.
 */
GraphNode*
G_Parameter(const STORCH_CTX ctx, Tensor* const x);

/**
 * Compute the product of two graphnodes.
 *
 * @param[in] ctx Storch context to manage the graph.
 * @param[in] x First operand.
 * @param[in] y Second operand.
 *
 * @returns A `GraphNode` representing `x * y`
 *
 */
GraphNode*
G_Product(const STORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

/**
 * Compute the sum of two graphnodes.
 *
 * @param[in] ctx Storch context to manage the graph.
 * @param[in] x First operand.
 * @param[in] y Second operand.
 *
 * @returns A `GraphNode` representing `x + y`
 *
 */
GraphNode*
G_Sum(const STORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

/**
 * Compute the difference of two graphnodes.
 *
 * @param[in] ctx Storch context to manage the graph.
 * @param[in] x First operand.
 * @param[in] y Second operand.
 *
 * @returns A `GraphNode` representing `x - y`.
 *
 */
GraphNode*
G_Diff(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

/**
 * Compute the element-wise division of two graphnodes.
 *
 * @param[in] ctx Storch context to manage the graph.
 * @param[in] x First operand.
 * @param[in] y Second operand.
 *
 * @returns A `GraphNode` representing `x / y`.
 *
 */
GraphNode*
G_Div(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

/**
 * Compute the element-wise exponential of a graphnode.
 *
 * @param[in] ctx Storch context to manage the graph.
 * @param[in] x Operand.
 *
 * @returns A `GraphNode` representing `exp(x)`.
 *
 */
GraphNode*
G_Exp(STORCH_CTX ctx, GraphNode* const x);

/**
 * Compute the element-wise natural logarithm of a graphnode.
 *
 * @param[in] ctx Storch context to manage the graph.
 * @param[in] x Operand.
 *
 * @returns A `GraphNode` representing `log(x)`.
 *
 */
GraphNode*
G_Log(STORCH_CTX ctx, GraphNode* const x);

/**
 * Compute the element-wise power of a graphnode.
 *
 * @param[in] ctx Storch context to manage the graph.
 * @param[in] x Base.
 * @param[in] y Exponent.
 *
 * @returns A `GraphNode` representing `x ** y`.
 *
 */
GraphNode*
G_Pow(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

/**
 * Compute the element-wise unary negation of a graphnode.
 *
 * @param[in] ctx Storch context to manage the graph.
 * @param[in] x Operand.
 *
 * @returns A `GraphNode` representing `-x`.
 *
 */
GraphNode*
G_Minus(STORCH_CTX ctx, GraphNode* const x);

/**
 * Compute the matrix multiplication of two graphnodes.
 *
 * @param[in] ctx Storch context to manage the graph.
 * @param[in] x First operand.
 * @param[in] y Second operand.
 *
 * @returns A `GraphNode` representing `x @ y`.
 *
 */
GraphNode*
G_MatMul(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

/**
 * Compute the row-wise unary reduction of a graphnode.
 *
 * @param[in] ctx Storch context to manage the graph.
 * @param[in] x Operand.
 *
 * @returns A `GraphNode` representing `sum(x[i, :] for all i)`.
 *
 */
GraphNode*
G_SumReduce0(STORCH_CTX ctx, GraphNode* const x);

/**
 * Compute the column-wise unary reduction of a graphnode.
 *
 * @param[in] ctx Storch context to manage the graph.
 * @param[in] x Operand.
 *
 * @returns A `GraphNode` representing `sum(x[:, i] for all i)`.
 *
 */
GraphNode*
G_SumReduce1(STORCH_CTX ctx, GraphNode* const x);

/** @} */

/** @name Graph and gradient evaluation
 *
 * Functions used to _realise_ (that is, evaluate) the forward value of the
 * graph starting from Values and Parameters and to compute the backward
 * computation of gradients in a realised graph.
 *
 * @{
 */

/**
 * Realize the forward graph.
 *
 * Running this function evaluates the graph iteratively starting from `x` until
 * it reaches `G_Value` or `G_Parameter` values. The value of the `Tensors` in
 * the `GraphNodes` can be read using the `value` function.
 *
 * @parameter[in] x GraphNode to evaluate.
 *
 */
void
forward(GraphNode* const x);

/**
 * Compute the backward graph.
 *
 * Running this function evaluates the graph backwar iteratively starting from
 * `x` until it reaches `G_Value` or `G_Parameter` values. During the
 * propagation, gradients are computed and deposited in all `GraphNode` objects.
 * These gradients can be read using the `grad` function.
 *
 * @parameter[in] x GraphNode to evaluate.
 *
 */
void
backward(GraphNode* const x);

/**
 * Read the value of a GraphNode.
 *
 * @parameter[in] x GraphNode to evaluate.
 *
 * @node Requires a forward pass!
 *
 */
Tensor*
value(const GraphNode* const x);

/**
 * Read the gradient of a GraphNode.
 *
 * @parameter[in] x GraphNode to evaluate.
 *
 * @node Requires both a forward and a backward pass!
 *
 */
Tensor*
grad(const GraphNode* const x);

/**
 * Print a tensor to stdout.
 *
 * @parameter[in] x Tensor to print.
 *
 */
void
print(Tensor* const x);

/** @} */
#endif // STORCH_H
