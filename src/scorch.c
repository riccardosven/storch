#include "scorch/scorch.h"
#include "graph.h"
#include "scorch/tensor.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

void
forward(GraphNode* const x)
{

  for (size_t i = 0; i < x->arity; i++) {
    forward(x->operands[i]);
    if (x->operands[i]->requires_grad)
      x->requires_grad = true;
  }

  if (x->forward_f)
    x->forward_f(x);
}

Tensor
value(const GraphNode* const x)
{
  return x->t;
}

static void
backward_impl(GraphNode* const x)
{
  assert(x->requires_grad);

  if (x->backward_f)
    x->backward_f(x);

  for (size_t i = 0; i < x->arity; i++)
    if (x->operands[i]->requires_grad) {
      backward_impl(x->operands[i]);
    }
}

void
backward(GraphNode* const x)
{
  x->g = 1;
  backward_impl(x);
}

Tensor
grad(const GraphNode* const x)
{
  return x->g;
}
