#include <tgmath.h>

#include "common.h"
#include "scorch/scorch.h"

int
integration_1(void)
{
  /* g = x^2 * y + 15*x*y + 3/x */

  GRAPH_CTX ctx = G_CTX_New();

  T_eltype x_v = 1.51;
  T_eltype y_v = 2.192;

  Tensor* t_x = T_Scalar(x_v);
  Tensor* t_y = T_Scalar(y_v);
  Tensor* t_2 = T_Scalar(2);
  Tensor* t_3 = T_Scalar(3);
  Tensor* t_15 = T_Scalar(15);

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

  T_Destroy(t_x);
  T_Destroy(t_y);
  T_Destroy(t_2);
  T_Destroy(t_3);
  T_Destroy(t_15);

  G_CTX_Destroy(ctx);

  return retval;
}

int
integration_2(void)
{
  /* g = 1/(1 + exp(x)) */
  GRAPH_CTX ctx = G_CTX_New();

  T_eltype x_v = 0.84;
  Tensor* t_x = T_Scalar(x_v);
  Tensor* t_1 = T_Scalar(1);

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

  G_CTX_Destroy(ctx);

  return retval;
}

int
integration_3(void)
{

  /* g = a * (x + b * c) */
  GRAPH_CTX ctx = G_CTX_New();

  T_eltype x_v = 0.88;

  T_eltype a_v = 4;
  T_eltype b_v = 5;
  T_eltype c_v = 3;

  Tensor* t_x = T_Scalar(x_v);

  Tensor* t_a = T_Scalar(a_v);
  Tensor* t_b = T_Scalar(b_v);
  Tensor* t_c = T_Scalar(c_v);

  GraphNode* x = G_Parameter(ctx, t_x);
  GraphNode* a = G_Value(ctx, t_a);
  GraphNode* b = G_Value(ctx, t_b);
  GraphNode* c = G_Value(ctx, t_c);

  GraphNode* g = G_Product(ctx, a, G_Sum(ctx, x, G_Product(ctx, b, c)));

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(x)->data[0], a_v);

  G_CTX_Destroy(ctx);
  T_Destroy(t_x);
  T_Destroy(t_a);
  T_Destroy(t_b);
  T_Destroy(t_c);

  return retval;
}

int
main(void)
{
  return integration_1() + integration_2() + integration_3();
}
