#include <assert.h>
#include <tgmath.h>

#include "graph.h"
#include "ops.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

void
G_Product_Forward(GraphNode* x)
{
  assert(x->op == PRODUCT);
  assert(x->arity == 2);

  x->t = T_Mul(value(x->operands[0]), value(x->operands[1]));
}

void
G_Product_Backward(GraphNode* x)
{

  assert(x->op == PRODUCT);
  assert(x->arity == 2);

  T_GEMM_(x->operands[0]->g, x->g, value(x->operands[1]), 1.0, 1.0);
  T_GEMM_(x->operands[1]->g, x->g, value(x->operands[0]), 1.0, 1.0);
}

void
G_Sum_Forward(GraphNode* x)
{
  assert(x->op == SUM);
  assert(x->arity == 2);

  x->t = T_Sum(value(x->operands[0]), value(x->operands[1]));
}

void
G_Sum_Backward(GraphNode* x)
{
  assert(x->op == SUM);
  assert(x->arity == 2);

  T_Add_(grad(x->operands[0]), grad(x));
  T_Add_(grad(x->operands[1]), grad(x));
}

void
G_Diff_Forward(GraphNode* x)
{
  assert(x->op == DIFFERENCE);
  assert(x->arity == 2);

  x->t = T_Diff(value(x->operands[0]), value(x->operands[1]));
}

void
G_Diff_Backward(GraphNode* x)
{
  assert(x->op == DIFFERENCE);
  assert(x->arity == 2);

  T_Add_(x->operands[0]->g, x->g);
  T_Sub_(x->operands[1]->g, x->g);
}

void
G_Div_Forward(GraphNode* x)
{
  assert(x->op == DIVISION);
  assert(x->arity == 2);

  x->t = T_Div(value(x->operands[0]), value(x->operands[1]));
}

void
G_Div_Backward(GraphNode* x)
{
  assert(x->op == DIVISION);
  assert(x->arity == 2);

  // x->operands[0]->g += x->g / value(x->operands[1]);
  Tensor* a = T_Div(x->g, value(x->operands[1]));
  T_Add_(x->operands[0]->g, a);

  // x->operands[1]->g -= x->g * value(x->operands[0]) /
  // pow(value(x->operands[1]), 2);
  T_Copy_(a, value(x->operands[0]));
  Tensor* b = T_SPow(value(x->operands[1]), 2);
  T_Div_(a, a, b);
  T_Mul_(a, x->g, a);
  T_Sub_(x->operands[1]->g, a);

  T_Destroy(a);
  T_Destroy(b);
}

void
G_Exp_Forward(GraphNode* x)
{
  assert(x->op == EXPONENTIAL);
  assert(x->arity == 1);

  x->t = T_Exp(value(x->operands[0]));
}

void
G_Exp_Backward(GraphNode* x)
{
  assert(x->op == EXPONENTIAL);
  assert(x->arity == 1);

  T_GEMM_(x->operands[0]->g, x->g, value(x), 1.0, 1.0);
}

void
G_Pow_Forward(GraphNode* x)
{
  assert(x->op == POWER);
  assert(x->arity == 2);

  Tensor* b = value(x->operands[1]);

  assert(b->n == 1 && b->m == 1);

  x->t = T_SPow(value(x->operands[0]), b->data[0]);
}

void
G_Pow_Backward(GraphNode* x)
{
  assert(x->op == POWER);
  assert(x->arity == 2);

  Tensor* a = T_Copy(value(x->operands[0]));
  Tensor* t = value(x->operands[1]);
  assert(t->n == 1 && t->m == 1);

  T_eltype b = *(value(x->operands[1])->data);

  Tensor* c = T_Log(a);

  // x->operands[0]->g += x->g * a * pow(a, b-1);
  T_SPow_(a, a, b - 1);
  T_Scale_(a, b, a);
  T_Mul_(a, x->g, a);
  T_Add_(x->operands[0]->g, a);

  // x->operands[1]->g += x->g * value(x) * log(t0);
  T_Mul_(c, value(x), c);
  T_Mul_(c, x->g, c);
  T_Add_(x->operands[1]->g, c);

  T_Destroy(a);
  T_Destroy(c);
}

void
G_Minus_Forward(GraphNode* x)
{
  assert(x->op == MINUS);
  assert(x->arity == 1);

  x->t = T_Minus(value(x->operands[0]));
}

void
G_Minus_Backward(GraphNode* x)
{
  assert(x->op == MINUS);
  assert(x->arity == 1);

  // x->operands[0]->g -= x->g;
  T_Sub_(x->operands[0]->g, x->g);
}
