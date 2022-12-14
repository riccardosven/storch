#include "storch/storch.h"
#include "graph.h"
#include "storch/tensor.h"

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
    x->g = T_ZerosLike(x->ctx, x->t);

  for (size_t i = 0; i < x->arity; i++)
    if (!x->operands[i]->g) {

      x->operands[i]->g = T_ZerosLike(x->ctx, x->operands[i]->t);
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
  x->g = T_OnesLike(x->ctx, x->t);
  backward_impl(x);
}

Tensor*
grad(const GraphNode* const x)
{
  return x->g;
}

void
print(Tensor* const x)
{
  for (size_t i = 0; i < x->n; i++) {
    for (size_t j = 0; j < x->m; j++) {
      printf("%f ", x->data[i + x->n * j]);
    }
    printf("\n");
  }
}
