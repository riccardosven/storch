#include <assert.h>
#include <tgmath.h>

#include "graph.h"
#include "ops.h"
#include "storch/storch.h"
#include "storch/tensor.h"

void
G_Product_Forward(GraphNode* x)
{
  assert(x->op == PRODUCT);
  assert(x->arity == 2);

  x->t = T_Mul(x->ctx, value(x->operands[0]), value(x->operands[1]));
}

void
G_Product_Backward(GraphNode* x)
{

  assert(x->op == PRODUCT);
  assert(x->arity == 2);

  Tensor* t = T_Mul(NULL, x->g, value(x->operands[1]));
  T_Add_(x->operands[0]->g, t);

  T_Mul_(t, x->g, value(x->operands[0]));
  T_Add_(x->operands[1]->g, t);

  T_Destroy(t);
}

void
G_Sum_Forward(GraphNode* x)
{
  assert(x->op == SUM);
  assert(x->arity == 2);

  x->t = T_Sum(x->ctx, value(x->operands[0]), value(x->operands[1]));
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

  x->t = T_Diff(x->ctx, value(x->operands[0]), value(x->operands[1]));
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

  x->t = T_Div(x->ctx, value(x->operands[0]), value(x->operands[1]));
}

void
G_Div_Backward(GraphNode* x)
{
  assert(x->op == DIVISION);
  assert(x->arity == 2);

  // x->operands[0]->g += x->g / value(x->operands[1]);
  Tensor* a = T_Div(NULL, x->g, value(x->operands[1]));
  T_Add_(x->operands[0]->g, a);

  // x->operands[1]->g -= x->g * value(x->operands[0]) /
  // pow(value(x->operands[1]), 2);
  T_Copy_(a, value(x->operands[0]));
  Tensor* b = T_SPow(NULL, value(x->operands[1]), 2);
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

  x->t = T_Exp(x->ctx, value(x->operands[0]));
}

void
G_Exp_Backward(GraphNode* x)
{
  assert(x->op == EXPONENTIAL);
  assert(x->arity == 1);

  Tensor* t = T_Mul(NULL, x->g, value(x));

  T_Add_(grad(x->operands[0]), t);

  T_Destroy(t);
}

void
G_Pow_Forward(GraphNode* x)
{
  assert(x->op == POWER);
  assert(x->arity == 2);

  x->t = T_Pow(x->ctx, value(x->operands[0]), value(x->operands[1]));
}

void
G_Pow_Backward(GraphNode* x)
{
  assert(x->op == POWER);
  assert(x->arity == 2);

  Tensor* b = T_Copy(NULL, value(x->operands[1]));
  Tensor* o = T_OnesLike(NULL, b);

  T_Diff_(o, b, o);

  // x->operands[0]->g += x->g * b * pow(a, b-1);
  Tensor* a = T_Pow(NULL, value(x->operands[0]), o);
  T_Mul_(a, b, a);
  T_Mul_(a, grad(x), a);
  T_Add_(grad(x->operands[0]), a);

  // x->operands[1]->g += x->g * value(x) * log(t0);
  Tensor* c = T_Log(NULL, value(x->operands[0]));
  T_Mul_(c, value(x), c);
  T_Mul_(c, grad(x), c);
  T_Add_(grad(x->operands[1]), c);

  T_Destroy(a);
  T_Destroy(b);
  T_Destroy(c);
  T_Destroy(o);
}

void
G_Log_Forward(GraphNode* x)
{
  assert(x->op == LOG);
  assert(x->arity == 1);
  x->t = T_Log(x->ctx, value(x->operands[0]));
}

void
G_Log_Backward(GraphNode* x)
{
  assert(x->op == LOG);
  assert(x->arity == 1);
  Tensor* t = T_Div(NULL, grad(x), value(x->operands[0]));
  T_Add_(grad(x->operands[0]), t);
  T_Destroy(t);
}

void
G_Minus_Forward(GraphNode* x)
{
  assert(x->op == MINUS);
  assert(x->arity == 1);

  x->t = T_Minus(x->ctx, value(x->operands[0]));
}

void
G_Minus_Backward(GraphNode* x)
{
  assert(x->op == MINUS);
  assert(x->arity == 1);

  // x->operands[0]->g -= x->g;
  T_Sub_(grad(x->operands[0]), grad(x));
}

void
G_MatMul_Forward(GraphNode* x)
{
  assert(x->op == MATMUL);
  assert(x->arity == 2);

  x->t = T_MatMul(x->ctx, value(x->operands[0]), value(x->operands[1]));
}

void
G_MatMul_Backward(GraphNode* x)
{
  assert(x->op == MATMUL);
  assert(x->arity == 2);

  Tensor* W = x->operands[0]->t;
  Tensor* X = x->operands[1]->t;
  Tensor* dY = x->g;

  Tensor* dW = x->operands[0]->g;
  Tensor* dX = x->operands[1]->g;

  T_GEMM_(dW, dY, false, X, true, 1.0, 1.0);
  T_GEMM_(dX, W, true, dY, false, 1.0, 1.0);

  /*
  y = WX
  dw = dyX'
  dx = W'dy
  */
}

void
G_SumReduce_Forward(GraphNode* x)
{
  assert(x->op == SUMREDUCE);
  assert(x->arity == 1);

  x->t = T_SumReduce(x->ctx, value(x->operands[0]));
}

void
G_SumReduce_Backward(GraphNode* x)
{

  /*
   * y = ones'x
   * dy = ones'dx
   */
  assert(x->op == SUMREDUCE);
  assert(x->arity == 1);

  T_Sum_(grad(x->operands[0]), grad(x->operands[0]), grad(x));
}
