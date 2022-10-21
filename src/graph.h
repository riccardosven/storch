#ifndef STORCH_GRAPH_H
#define STORCH_GRAPH_H

#include <stdbool.h>

#include "ops.h"
#include "storch/storch.h"
#include "storch/tensor.h"

struct graphnode_s
{
  Tensor* t;
  Tensor* g;
  bool requires_grad;
  Op op;
  size_t arity;
  struct graphnode_s** operands;
  void (*forward_f)(struct graphnode_s*);
  void (*backward_f)(struct graphnode_s*);
  STORCH_CTX ctx;
};

void*
G_Destroy(GraphNode*);

#endif // STORCH_GRAPH_H
