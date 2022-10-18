#ifndef SCORCH_GRAPH_H
#define SCORCH_GRAPH_H

#include <stdbool.h>
#include <stdlib.h>

#include "scorch/tensor.h"
#include "scorch/scorch.h"
#include "ops.h"


struct graphnode_s
{
  Tensor t;
  Tensor g;
  bool requires_grad;
  Op op;
  size_t arity;
  struct graphnode_s** operands;
};


struct graph_ctx_s
{
  GraphNode** arena;
  size_t len;
  size_t cap;
};


#endif // SCORCH_GRAPH_H
