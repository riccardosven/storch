#include "scorch/scorch.h"
#include "graph.h"
#include "scorch/tensor.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
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

Tensor*
value(const GraphNode* const x)
{
  return x->t;
}

static void
backward_impl(GraphNode* const x)
{
  assert(x->requires_grad);

  if (!x->g)
    x->g = T_ZerosLike(x->t);

  for (size_t i = 0; i < x->arity; i++)
    if (!x->operands[i]->g) {
      size_t n = x->operands[i]->t->n;
      size_t m = x->operands[i]->t->m;

      printf("%zu %zu\n", n, m);

      x->operands[i]->g = T_Zeros(n, m);
    }

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
  x->g = T_OnesLike(x->t);
  backward_impl(x);
}

Tensor*
grad(const GraphNode* const x)
{
  return x->g;
}
