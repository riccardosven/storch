#ifndef GRAPH_H
#define GRAPH_H

#include <stdlib.h>

typedef enum
{
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
  Op op;
  size_t arity;
  struct node** operands;
} GraphNode;

static struct graph_ctx
{
  GraphNode** arena;
  size_t len;
  size_t cap;
} graph_ctx;

Tensor
value(GraphNode*);

typedef struct graph_ctx* GRAPH_CTX;

GRAPH_CTX
G_CTX_New()
{
  GRAPH_CTX ctx = malloc(sizeof(struct graph_ctx));
  ctx->arena = NULL;
  ctx->cap = 0;
  ctx->len = 0;

  return ctx;
}

GraphNode*
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

GraphNode*
G_New(GRAPH_CTX ctx, Op op, size_t arity)
{
  GraphNode* v = G_Malloc(ctx);

  v->t = 0.0;
  v->g = 0.0;
  v->op = op;
  v->arity = arity;
  v->operands = malloc(sizeof(GraphNode*) * arity);
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
G_Product(GRAPH_CTX ctx, GraphNode* x, GraphNode* y)
{

  GraphNode* v = G_New(ctx, PRODUCT, 2);

  v->operands[0] = x;
  v->operands[1] = y;

  return v;
}

GraphNode*
G_Value(GRAPH_CTX ctx, Tensor x)
{

  GraphNode* v = G_New(ctx, VALUE, 0);

  v->t = x;

  return v;
}

GraphNode*
G_Sum(GRAPH_CTX ctx, GraphNode* x, GraphNode* y)
{

  GraphNode* v = G_New(ctx, SUM, 2);

  v->operands[0] = x;
  v->operands[1] = y;

  return v;
}

GraphNode*
G_Diff(GRAPH_CTX ctx, GraphNode* x, GraphNode* y)
{

  GraphNode* v = G_New(ctx, DIFFERENCE, 2);

  v->operands[0] = x;
  v->operands[1] = y;

  return v;
}

GraphNode*
G_Div(GRAPH_CTX ctx, GraphNode* x, GraphNode* y)
{
  GraphNode* v = G_New(ctx, DIVISION, 2);

  v->operands[0] = x;
  v->operands[1] = y;

  return v;
}

GraphNode*
G_Exp(GRAPH_CTX ctx, GraphNode* x)
{
  GraphNode* v = G_New(ctx, EXPONENTIAL, 1);

  v->operands[0] = x;

  return v;
}

GraphNode*
G_Pow(GRAPH_CTX ctx, GraphNode* x, GraphNode* y)
{
  GraphNode* v = G_New(ctx, POWER, 2);

  v->operands[0] = x;
  v->operands[1] = y;

  return v;
}

GraphNode*
G_Minus(GRAPH_CTX ctx, GraphNode* x)
{
  GraphNode* v = G_New(ctx, MINUS, 1);
  v->operands[0] = x;
  return v;
}

#endif // GRAPH_H
