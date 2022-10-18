#ifndef SCORCH_H
#define SCORCH_H

#include "scorch/tensor.h"

typedef struct graph_ctx_s* GRAPH_CTX;
typedef struct graphnode_s GraphNode;


/* GRAPH CONTEXT AND ARENA ALLOCATOR */
GRAPH_CTX G_CTX_New();
void* G_CTX_Destroy(GRAPH_CTX ctx);


/* GRAPH TRANSFORMATION FUNCTIONS */
GraphNode* G_Product(const GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y);
GraphNode* G_Value(const GRAPH_CTX ctx, const Tensor x);
GraphNode* G_Parameter(const GRAPH_CTX ctx, const Tensor x);
GraphNode* G_Sum(const GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y);
GraphNode* G_Diff(GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y);
GraphNode* G_Div(GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y);
GraphNode* G_Exp(GRAPH_CTX ctx, GraphNode* const x);
GraphNode* G_Pow(GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y);
GraphNode* G_Minus(GRAPH_CTX ctx, GraphNode* const x);


/* GRAPH AND GRADIENT EVALUATION */
void forward(GraphNode* const x);
void backward(GraphNode* const x);

Tensor value(const GraphNode* const x);
Tensor grad(const GraphNode* const x);

#endif // SCORCH_H
