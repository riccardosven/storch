#ifndef STORCH_H
#define STORCH_H

#include <stdio.h>

typedef struct storch_ctx_s* STORCH_CTX;
typedef struct graphnode_s GraphNode;
typedef struct storch_tensor_s Tensor;

/* GRAPH CONTEXT AND ARENA ALLOCATOR */
STORCH_CTX
STORCH_CTX_New();

void*
STORCH_CTX_Destroy(STORCH_CTX ctx);

/* GRAPH TRANSFORMATION FUNCTIONS */
GraphNode*
G_Product(const STORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode*
G_Value(const STORCH_CTX ctx, Tensor* x);

GraphNode*
G_Parameter(const STORCH_CTX ctx, Tensor* x);

GraphNode*
G_Sum(const STORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode*
G_Diff(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode*
G_Div(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode*
G_Exp(STORCH_CTX ctx, GraphNode* const x);

GraphNode*
G_Log(STORCH_CTX ctx, GraphNode* const x);

GraphNode*
G_Pow(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode*
G_Minus(STORCH_CTX ctx, GraphNode* const x);

GraphNode*
G_MatMul(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode*
G_SumReduce(STORCH_CTX ctx, GraphNode* const x);

/* GRAPH AND GRADIENT EVALUATION */
void
forward(GraphNode* const x);
void
backward(GraphNode* const x);

Tensor*
value(const GraphNode* const x);

Tensor*
grad(const GraphNode* const x);

void
print(Tensor * const x);

#endif // STORCH_H
