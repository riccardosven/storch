#ifndef SCORCH_H
#define SCORCH_H

typedef struct scorch_ctx_s* SCORCH_CTX;
typedef struct graphnode_s GraphNode;
typedef struct scorch_tensor_s Tensor;

/* GRAPH CONTEXT AND ARENA ALLOCATOR */
SCORCH_CTX
SCORCH_CTX_New();

void*
SCORCH_CTX_Destroy(SCORCH_CTX ctx);

/* GRAPH TRANSFORMATION FUNCTIONS */
GraphNode*
G_Product(const SCORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode*
G_Value(const SCORCH_CTX ctx, Tensor* x);

GraphNode*
G_Parameter(const SCORCH_CTX ctx, Tensor* x);

GraphNode*
G_Sum(const SCORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode*
G_Diff(SCORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode*
G_Div(SCORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode*
G_Exp(SCORCH_CTX ctx, GraphNode* const x);

GraphNode*
G_Log(SCORCH_CTX ctx, GraphNode* const x);

GraphNode*
G_Pow(SCORCH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode*
G_Minus(SCORCH_CTX ctx, GraphNode* const x);

GraphNode*
G_MatMul(SCORCH_CTX ctx, GraphNode* const x ,GraphNode* const y);

/* GRAPH AND GRADIENT EVALUATION */
void
forward(GraphNode* const x);
void
backward(GraphNode* const x);

Tensor*
value(const GraphNode* const x);

Tensor*
grad(const GraphNode* const x);

#endif // SCORCH_H
