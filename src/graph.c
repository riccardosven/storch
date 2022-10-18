#include <stdbool.h>
#include <stdlib.h>

#include "graph.h"
#include "ops.h"
#include "scorch/scorch.h"

GRAPH_CTX
G_CTX_New()
{
  GRAPH_CTX ctx = malloc(sizeof(struct graph_ctx_s));
  ctx->arena = NULL;
  ctx->cap = 0;
  ctx->len = 0;

  return ctx;
}

static GraphNode*
G_Malloc(GRAPH_CTX ctx)
{
  if (ctx->cap == ctx->len) {
    ctx->cap = 2 * ctx->cap + 1;
    ctx->arena = realloc(ctx->arena, sizeof(GraphNode*) * ctx->cap);
  }

  GraphNode* g = malloc(sizeof *g);

  ctx->arena[ctx->len++] = g;

  return g;
}

static GraphNode*
G_New(GRAPH_CTX ctx, Op op, size_t arity)
{
  GraphNode* v = G_Malloc(ctx);

  v->t = 0.0;
  v->g = 0.0;
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

void*
G_CTX_Destroy(GRAPH_CTX ctx)
{
  while (ctx->len--) {
    G_Destroy(ctx->arena[ctx->len]);
  }

  free(ctx->arena);
  free(ctx);

  return NULL;
}

GraphNode*
G_Product(GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y)
{

  GraphNode* v = G_New(ctx, PRODUCT, 2);

  v->operands[0] = x;
  v->operands[1] = y;
  v->forward_f = G_Product_Forward;
  v->backward_f = G_Product_Backward;

  return v;
}

GraphNode*
G_Value(GRAPH_CTX ctx, const Tensor x)
{

  GraphNode* v = G_New(ctx, VALUE, 0);

  v->t = x;
  return v;
}

GraphNode*
G_Parameter(GRAPH_CTX ctx, const Tensor x)
{
  GraphNode* v = G_New(ctx, PARAMETER, 0);
  v->t = x;
  v->requires_grad = true;

  return v;
}

GraphNode*
G_Sum(GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y)
{

  GraphNode* v = G_New(ctx, SUM, 2);

  v->operands[0] = x;
  v->operands[1] = y;
  v->forward_f = G_Sum_Forward;
  v->backward_f = G_Sum_Backward;

  return v;
}

GraphNode*
G_Diff(GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y)
{

  GraphNode* v = G_New(ctx, DIFFERENCE, 2);

  v->operands[0] = x;
  v->operands[1] = y;
  v->forward_f = G_Diff_Forward;
  v->backward_f = G_Diff_Backward;

  return v;
}

GraphNode*
G_Div(GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y)
{
  GraphNode* v = G_New(ctx, DIVISION, 2);

  v->operands[0] = x;
  v->operands[1] = y;
  v->forward_f = G_Div_Forward;
  v->backward_f = G_Div_Backward;

  return v;
}

GraphNode*
G_Exp(GRAPH_CTX ctx, GraphNode* const x)
{
  GraphNode* v = G_New(ctx, EXPONENTIAL, 1);

  v->operands[0] = x;
  v->forward_f = G_Exp_Forward;
  v->backward_f = G_Exp_Backward;

  return v;
}

GraphNode*
G_Pow(GRAPH_CTX ctx, GraphNode* const x, GraphNode* const y)
{
  GraphNode* v = G_New(ctx, POWER, 2);

  v->operands[0] = x;
  v->operands[1] = y;

  v->forward_f = G_Pow_Forward;
  v->backward_f = G_Pow_Backward;

  return v;
}

GraphNode*
G_Minus(GRAPH_CTX ctx, GraphNode* const x)
{
  GraphNode* v = G_New(ctx, MINUS, 1);
  v->operands[0] = x;
  v->forward_f = G_Minus_Forward;
  v->backward_f = G_Minus_Backward;

  return v;
}
