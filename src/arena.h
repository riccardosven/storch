#ifndef STORCH_ARENA_H
#define STORCH_ARENA_H

#include "storch/storch.h"
#include "storch/tensor.h"

typedef struct storch_ctx_s* STORCH_CTX;

STORCH_CTX
STORCH_CTX_New();

void*
STORCH_CTX_Destroy(STORCH_CTX ctx);

GraphNode*
G_Malloc(STORCH_CTX ctx);

Tensor*
T_Malloc(STORCH_CTX ctx);

#endif // STORCH_ARENA_H
