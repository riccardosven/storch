#ifndef SCORCH_GRAPH_H
#define SCORCH_GRAPH_H

#include <stdbool.h>

#include "ops.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

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
  SCORCH_CTX ctx;
};

void*
G_Destroy(GraphNode*);

#endif // SCORCH_GRAPH_H
