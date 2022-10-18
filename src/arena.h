#ifndef SCORCH_ARENA_H
#define SCORCH_ARENA_H

#include "scorch/scorch.h"
#include "scorch/tensor.h"

typedef struct scorch_ctx_s* SCORCH_CTX;

SCORCH_CTX
SCORCH_CTX_New();

void*
SCORCH_CTX_Destroy(SCORCH_CTX ctx);

GraphNode*
G_Malloc(SCORCH_CTX ctx);

Tensor*
T_Malloc(SCORCH_CTX ctx);


#endif // SCORCH_ARENA_H
