#include <assert.h>
#include <tgmath.h>

#include "ops.h"
#include "graph.h"
#include "scorch/scorch.h"

void
G_Product_Forward(GraphNode* x)
{
  assert(x->op == PRODUCT);
  assert(x->arity == 2);

  x->t = value(x->operands[0]) * value(x->operands[1]);
}

void
G_Product_Backward(GraphNode* x)
{

  assert(x->arity == 2);

  x->operands[0]->g += x->g * value(x->operands[1]);
  x->operands[1]->g += x->g * value(x->operands[0]);
}

void
G_Sum_Forward(GraphNode* x)
{
  assert(x->op == SUM);
  assert(x->arity == 2);

  x->t = value(x->operands[0]) + value(x->operands[1]);
}

void
G_Sum_Backward(GraphNode* x)
{
  assert(x->op == SUM);
  assert(x->arity == 2);

  x->operands[0]->g += x->g;
  x->operands[1]->g += x->g;
}

void
G_Diff_Forward(GraphNode* x)
{
  assert(x->op == DIFFERENCE);
  assert(x->arity == 2);

  x->t = value(x->operands[0]) - value(x->operands[1]);
}

void
G_Diff_Backward(GraphNode* x)
{
  assert(x->op == DIFFERENCE);
  assert(x->arity == 2);

  x->operands[0]->g += x->g;
  x->operands[1]->g -= x->g;
}

void
G_Div_Forward(GraphNode* x)
{
  assert(x->op == DIVISION);
  assert(x->arity == 2);

  x->t = value(x->operands[0]) / value(x->operands[1]);
}

void
G_Div_Backward(GraphNode* x)
{
  assert(x->op == DIVISION);
  assert(x->arity == 2);

  x->operands[0]->g += x->g / value(x->operands[1]);
  x->operands[1]->g -=
    x->g * value(x->operands[0]) / pow(value(x->operands[1]), 2);
}

void
G_Exp_Forward(GraphNode* x)
{
  assert(x->op == EXPONENTIAL);
  assert(x->arity == 1);

  x->t = exp(value(x->operands[0]));
}

void
G_Exp_Backward(GraphNode* x)
{
  assert(x->op == EXPONENTIAL);
  assert(x->arity == 1);

  x->operands[0]->g += x->g * value(x);
}

void
G_Pow_Forward(GraphNode* x)
{
  assert(x->op == POWER);
  assert(x->arity == 2);

  x->t = pow(value(x->operands[0]), value(x->operands[1]));
}

void
G_Pow_Backward(GraphNode* x)
{
  assert(x->op == POWER);
  assert(x->arity == 2);

  Tensor t0 = value(x->operands[0]);
  Tensor t1 = value(x->operands[1]);

  x->operands[0]->g += x->g * t1 * pow(t0, t1 - 1);
  x->operands[1]->g += x->g * value(x) * log(t0);
}

void
G_Minus_Forward(GraphNode* x)
{
  assert(x->op == MINUS);
  assert(x->arity == 1);

  x->t = -value(x->operands[0]);
}

void
G_Minus_Backward(GraphNode* x)
{
  assert(x->op == MINUS);
  assert(x->arity == 1);

  x->operands[0]->g -= x->g;
}
