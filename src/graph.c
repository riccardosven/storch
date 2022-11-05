#include <stdbool.h>
#include <stdlib.h>

#include "arena.h"
#include "graph.h"
#include "ops.h"
#include "storch/storch.h"

static GraphNode*
G_New(STORCH_CTX ctx, Op op, size_t arity)
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
G_Product(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y)
{

  GraphNode* v = G_New(ctx, PRODUCT, 2);

  v->operands[0] = x;
  v->operands[1] = y;
  v->forward_f = G_Product_Forward;
  v->backward_f = G_Product_Backward;

  return v;
}

GraphNode*
G_Value(STORCH_CTX ctx, Tensor* x)
{

  GraphNode* v = G_New(ctx, VALUE, 0);

  v->t = x;
  return v;
}

GraphNode*
G_Parameter(STORCH_CTX ctx, Tensor* x)
{
  GraphNode* v = G_New(ctx, PARAMETER, 0);
  v->t = x;
  v->requires_grad = true;

  return v;
}

GraphNode*
G_Sum(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y)
{

  GraphNode* v = G_New(ctx, SUM, 2);

  v->operands[0] = x;
  v->operands[1] = y;
  v->forward_f = G_Sum_Forward;
  v->backward_f = G_Sum_Backward;

  return v;
}

GraphNode*
G_Diff(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y)
{

  GraphNode* v = G_New(ctx, DIFFERENCE, 2);

  v->operands[0] = x;
  v->operands[1] = y;
  v->forward_f = G_Diff_Forward;
  v->backward_f = G_Diff_Backward;

  return v;
}

GraphNode*
G_Div(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y)
{
  GraphNode* v = G_New(ctx, DIVISION, 2);

  v->operands[0] = x;
  v->operands[1] = y;
  v->forward_f = G_Div_Forward;
  v->backward_f = G_Div_Backward;

  return v;
}

GraphNode*
G_Exp(STORCH_CTX ctx, GraphNode* const x)
{
  GraphNode* v = G_New(ctx, EXPONENTIAL, 1);

  v->operands[0] = x;
  v->forward_f = G_Exp_Forward;
  v->backward_f = G_Exp_Backward;

  return v;
}

GraphNode*
G_Log(STORCH_CTX ctx, GraphNode* const x)
{
  GraphNode* v = G_New(ctx, LOG, 1);

  v->operands[0] = x;
  v->forward_f = G_Log_Forward;
  v->backward_f = G_Log_Backward;

  return v;
}

GraphNode*
G_Pow(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y)
{
  GraphNode* v = G_New(ctx, POWER, 2);

  v->operands[0] = x;
  v->operands[1] = y;

  v->forward_f = G_Pow_Forward;
  v->backward_f = G_Pow_Backward;

  return v;
}

GraphNode*
G_Minus(STORCH_CTX ctx, GraphNode* const x)
{
  GraphNode* v = G_New(ctx, MINUS, 1);
  v->operands[0] = x;
  v->forward_f = G_Minus_Forward;
  v->backward_f = G_Minus_Backward;

  return v;
}

GraphNode*
G_MatMul(STORCH_CTX ctx, GraphNode* const x, GraphNode* const y)
{
  GraphNode* v = G_New(ctx, MATMUL, 2);
  v->operands[0] = x;
  v->operands[1] = y;

  v->forward_f = G_MatMul_Forward;
  v->backward_f = G_MatMul_Backward;

  return v;
}

GraphNode*
G_SumReduce0(STORCH_CTX ctx, GraphNode* const x)
{
  GraphNode* v = G_New(ctx, SUMREDUCE_ROW, 1);
  v->operands[0] = x;

  v->forward_f = G_SumReduce0_Forward;
  v->backward_f = G_SumReduce0_Backward;

  return v;
}

GraphNode*
G_SumReduce1(STORCH_CTX ctx, GraphNode* const x)
{
  GraphNode* v = G_New(ctx, SUMREDUCE_COL, 1);
  v->operands[0] = x;

  v->forward_f = G_SumReduce1_Forward;
  v->backward_f = G_SumReduce1_Backward;

  return v;
}

/*
GraphNode*
G_MeanReduce(STORCH_CTX, GraphNode* const x, )
{
  GraphNode *v = G_New(ctx, MEANREDUCE, 1);
  v->operands[0] = x;

  v->forward_f = G_SumReduceForward;
  v->backward_f = G_SumReduceForward;

  return v;
}
*/
