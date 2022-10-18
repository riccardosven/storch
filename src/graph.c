#include <stdbool.h>
#include <stdlib.h>

#include "arena.h"
#include "graph.h"
#include "ops.h"
#include "scorch/scorch.h"

static GraphNode*
G_New(SCORCH_CTX ctx, Op op, size_t arity)
{
  GraphNode* v = G_Malloc(ctx);

  v->t = NULL;
  v->g = NULL;
  v->requires_grad = false;
  v->op = op;
  v->arity = arity;
  v->operands = malloc(sizeof(GraphNode*) * arity);
  v->forward_f = NULL;
  v->backward_f = NULL;
  return v;
}

void*
G_Destroy(GraphNode* g)
{

  if (g->operands)
    free(g->operands);

  free(g);

  return NULL;
}

GraphNode*
G_Product(SCORCH_CTX ctx, GraphNode* const x, GraphNode* const y)
{

  GraphNode* v = G_New(ctx, PRODUCT, 2);

  v->operands[0] = x;
  v->operands[1] = y;
  v->forward_f = G_Product_Forward;
  v->backward_f = G_Product_Backward;

  return v;
}

GraphNode*
G_Value(SCORCH_CTX ctx, Tensor* x)
{

  GraphNode* v = G_New(ctx, VALUE, 0);

  v->t = x;
  return v;
}

GraphNode*
G_Parameter(SCORCH_CTX ctx, Tensor* x)
{
  GraphNode* v = G_New(ctx, PARAMETER, 0);
  v->t = x;
  v->requires_grad = true;

  return v;
}

GraphNode*
G_Sum(SCORCH_CTX ctx, GraphNode* const x, GraphNode* const y)
{

  GraphNode* v = G_New(ctx, SUM, 2);

  v->operands[0] = x;
  v->operands[1] = y;
  v->forward_f = G_Sum_Forward;
  v->backward_f = G_Sum_Backward;

  return v;
}

GraphNode*
G_Diff(SCORCH_CTX ctx, GraphNode* const x, GraphNode* const y)
{

  GraphNode* v = G_New(ctx, DIFFERENCE, 2);

  v->operands[0] = x;
  v->operands[1] = y;
  v->forward_f = G_Diff_Forward;
  v->backward_f = G_Diff_Backward;

  return v;
}

GraphNode*
G_Div(SCORCH_CTX ctx, GraphNode* const x, GraphNode* const y)
{
  GraphNode* v = G_New(ctx, DIVISION, 2);

  v->operands[0] = x;
  v->operands[1] = y;
  v->forward_f = G_Div_Forward;
  v->backward_f = G_Div_Backward;

  return v;
}

GraphNode*
G_Exp(SCORCH_CTX ctx, GraphNode* const x)
{
  GraphNode* v = G_New(ctx, EXPONENTIAL, 1);

  v->operands[0] = x;
  v->forward_f = G_Exp_Forward;
  v->backward_f = G_Exp_Backward;

  return v;
}

GraphNode*
G_Pow(SCORCH_CTX ctx, GraphNode* const x, GraphNode* const y)
{
  GraphNode* v = G_New(ctx, POWER, 2);

  v->operands[0] = x;
  v->operands[1] = y;

  v->forward_f = G_Pow_Forward;
  v->backward_f = G_Pow_Backward;

  return v;
}

GraphNode*
G_Minus(SCORCH_CTX ctx, GraphNode* const x)
{
  GraphNode* v = G_New(ctx, MINUS, 1);
  v->operands[0] = x;
  v->forward_f = G_Minus_Forward;
  v->backward_f = G_Minus_Backward;

  return v;
}

GraphNode*
G_MatMul(SCORCH_CTX ctx, GraphNode* const x, GraphNode* const y)
{
  GraphNode* v = G_New(ctx, MATMUL, 2);
  v->operands[0] = x;
  v->operands[1] = y;

  v->forward_f = G_MatMul_Forward;
  v->backward_f = G_MatMul_Backward;

  return v;
}
