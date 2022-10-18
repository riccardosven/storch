#include <tgmath.h>

#include "common.h"
#include "scorch/scorch.h"
#include "scorch/tensor.h"

int
integration_1(void)
{
  /* g = x^2 * y + 15*x*y + 3/x */

  SCORCH_CTX ctx = SCORCH_CTX_New();

  T_eltype x_v = 1.51;
  T_eltype y_v = 2.192;

  Tensor* t_x = T_Scalar(ctx, x_v);
  Tensor* t_y = T_Scalar(ctx, y_v);
  Tensor* t_2 = T_Scalar(ctx, 2);
  Tensor* t_3 = T_Scalar(ctx, 3);
  Tensor* t_15 = T_Scalar(ctx,15);

  GraphNode* x = G_Parameter(ctx, t_x);
  GraphNode* y = G_Parameter(ctx, t_y);

  GraphNode* g =
    G_Sum(ctx,
          G_Sum(ctx,
                G_Product(ctx, G_Pow(ctx, x, G_Parameter(ctx, t_2)), y),
                G_Product(ctx, G_Product(ctx, x, y), G_Parameter(ctx, t_15))),
          G_Div(ctx, G_Parameter(ctx, t_3), x));

  forward(g);
  backward(g);

  int retval = check_almost_eq(value(g)->data[0],
                               pow(x_v, 2) * y_v + 15 * x_v * y_v + 3 / x_v);
  retval += check_almost_eq(grad(x)->data[0],
                            2 * x_v * y_v + 15 * y_v - 3 / pow(x_v, 2));
  SCORCH_CTX_Destroy(ctx);

  return retval;
}

int
integration_2(void)
{
  /* g = 1/(1 + exp(x)) */
  SCORCH_CTX ctx = SCORCH_CTX_New();

  T_eltype x_v = 0.84;
  Tensor* t_x = T_Scalar(ctx, x_v);
  Tensor* t_1 = T_Scalar(ctx, 1);

  GraphNode* x = G_Parameter(ctx, t_x);

  GraphNode* g =
    G_Div(ctx,
          G_Parameter(ctx, t_1),
          G_Sum(ctx, G_Parameter(ctx, t_1), G_Exp(ctx, G_Minus(ctx, x))));

  forward(g);
  backward(g);

  T_eltype sigma = 1 / (1 + exp(-x_v));

  int retval = check_almost_eq(value(g)->data[0], sigma);
  retval += check_almost_eq(grad(x)->data[0], sigma * (1 - sigma));

  SCORCH_CTX_Destroy(ctx);

  return retval;
}

int
integration_3(void)
{

  /* g = a * (x + b * c) */
  SCORCH_CTX ctx = SCORCH_CTX_New();

  T_eltype x_v = 0.88;

  T_eltype a_v = 4;
  T_eltype b_v = 5;
  T_eltype c_v = 3;

  GraphNode* x = G_Parameter(ctx, T_Scalar(ctx, x_v));
  GraphNode* a = G_Value(ctx, T_Scalar(ctx, a_v));
  GraphNode* b = G_Value(ctx, T_Scalar(ctx, b_v));
  GraphNode* c = G_Value(ctx, T_Scalar(ctx, c_v));

  GraphNode* g = G_Product(ctx, a, G_Sum(ctx, x, G_Product(ctx, b, c)));

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(x)->data[0], a_v);

  SCORCH_CTX_Destroy(ctx);

  return retval;
}

int
main(void)
{
  return integration_1() + integration_2() + integration_3();
}
