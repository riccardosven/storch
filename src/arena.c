#include "arena.h"
#include "graph.h"
#include "storch/storch.h"
#include "storch/tensor.h"
#include <stdlib.h>

struct storch_ctx_s
{
  GraphNode** g_arena;
  size_t g_len;
  size_t g_cap;

  Tensor** t_arena;
  size_t t_len;
  size_t t_cap;
};

STORCH_CTX
STORCH_CTX_New()
{
  STORCH_CTX ctx = malloc(sizeof(struct storch_ctx_s));
  ctx->g_arena = NULL;
  ctx->g_cap = 0;
  ctx->g_len = 0;

  ctx->t_arena = NULL;
  ctx->t_cap = 0;
  ctx->t_len = 0;

  return ctx;
}
void*
STORCH_CTX_Destroy(STORCH_CTX ctx)
{
  while (ctx->g_len--)
    G_Destroy(ctx->g_arena[ctx->g_len]);

  while (ctx->t_len--)
    T_Destroy(ctx->t_arena[ctx->t_len]);

  free(ctx->g_arena);
  free(ctx->t_arena);

  free(ctx);

  return NULL;
}

GraphNode*
G_Malloc(STORCH_CTX ctx)
{
  GraphNode* g = malloc(sizeof *g);
  g->ctx = ctx;

  if (!ctx)
    return g;

  if (ctx->g_cap == ctx->g_len) {
    ctx->g_cap = 2 * ctx->g_cap + 1;
    ctx->g_arena = realloc(ctx->g_arena, sizeof(GraphNode*) * ctx->g_cap);
  }

  ctx->g_arena[ctx->g_len++] = g;

  return g;
}

Tensor*
T_Malloc(STORCH_CTX ctx)
{
  Tensor* t = malloc(sizeof *t);

  if (!ctx)
    return t;

  if (ctx->t_cap == ctx->t_len) {
    ctx->t_cap = 2 * ctx->t_cap + 1;
    ctx->t_arena = realloc(ctx->t_arena, sizeof(GraphNode*) * ctx->t_cap);
  }

  ctx->t_arena[ctx->t_len++] = t;

  return t;
}
