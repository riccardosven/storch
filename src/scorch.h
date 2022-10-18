#ifndef SCORCH_H
#define SCORCH_H

typedef double Tensor;

#include "graph.h"
#include "operations.h"
#include <assert.h>
#include <stdbool.h>

void
forward(GraphNode* x)
{

  for (size_t i = 0; i < x->arity; i++) {
    forward(x->operands[i]);
    if (x->operands[i]->requires_grad)
      x->requires_grad = true;
  }

  switch (x->op) {
    case PARAMETER:
    case VALUE:
      break;
    case PRODUCT:
      G_Product_Forward(x);
      break;
    case SUM:
      G_Sum_Forward(x);
      break;
    case DIFFERENCE:
      G_Diff_Forward(x);
      break;
    case DIVISION:
      G_Div_Forward(x);
      break;
    case EXPONENTIAL:
      G_Exp_Forward(x);
      break;
    case POWER:
      G_Pow_Forward(x);
      break;
    case MINUS:
      G_Minus_Forward(x);
      break;
    case NONE:
    default:
      assert(0);
  }
}

Tensor
value(GraphNode* x)
{
  return x->t;
}

static void
backward_impl(GraphNode* x)
{
  assert(x->requires_grad);

  switch (x->op) {
    case PARAMETER:
      break;
    case SUM:
      G_Sum_Backward(x);
      break;
    case PRODUCT:
      G_Product_Backward(x);
      break;
    case DIFFERENCE:
      G_Diff_Backward(x);
      break;
    case DIVISION:
      G_Div_Backward(x);
      break;
    case EXPONENTIAL:
      G_Exp_Backward(x);
      break;
    case POWER:
      G_Pow_Backward(x);
      break;
    case MINUS:
      G_Minus_Backward(x);
      break;
    case VALUE:
    case NONE:
    default:
      assert(0);
  }
  for (size_t i = 0; i < x->arity; i++)
    if (x->operands[i]->requires_grad) {
      backward_impl(x->operands[i]);
    }
}

void
backward(GraphNode* x)
{
  x->g = 1;
  backward_impl(x);
}

Tensor
grad(GraphNode* x)
{
  return x->g;
}

#endif // SCORCH_H
