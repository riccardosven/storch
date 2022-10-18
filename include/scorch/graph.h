#ifndef SCORCH_GRAPH_H
#define SCORCH_GRAPH_H

#include <stdbool.h>
#include <stdlib.h>

#include "scorch/tensor.h"

typedef enum
{
  PARAMETER,
  NONE,
  VALUE,
  SUM,
  PRODUCT,
  DIFFERENCE,
  DIVISION,
  EXPONENTIAL,
  POWER,
  MINUS,
  N_OPS
} Op;

typedef struct node
{
  Tensor t;
  Tensor g;
  bool requires_grad;
  Op op;
  size_t arity;
  struct node** operands;
} GraphNode;

struct graph_ctx
{
  GraphNode** arena;
  size_t len;
  size_t cap;
} graph_ctx;

Tensor
value(const GraphNode*);

typedef struct graph_ctx* GRAPH_CTX;

GRAPH_CTX G_CTX_New();

void* G_Destroy(GraphNode* g);

void* G_CTX_Destroy(GRAPH_CTX ctx);

GraphNode* G_Product(const GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode* G_Value(const GRAPH_CTX ctx, const Tensor x);

GraphNode* G_Parameter(const GRAPH_CTX ctx, const Tensor x);

GraphNode* G_Sum(const GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode* G_Diff(GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode* G_Div(GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode* G_Exp(GRAPH_CTX ctx, GraphNode* const x);

GraphNode* G_Pow(GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y);

GraphNode* G_Minus(GRAPH_CTX ctx, GraphNode* const x);

#endif // GRAPH_H
