#include <tgmath.h>

#include "common.h"
#include "storch/storch.h"
#include "storch/tensor.h"

int
integration_1(void)
{
  /* g = x^2 * y + 15*x*y + 3/x */

  STORCH_CTX ctx = STORCH_CTX_New();

  T_eltype x_v = 1.51;
  T_eltype y_v = 2.192;

  Tensor* t_x = T_Scalar(ctx, x_v);
  Tensor* t_y = T_Scalar(ctx, y_v);
  Tensor* t_2 = T_Scalar(ctx, 2);
  Tensor* t_3 = T_Scalar(ctx, 3);
  Tensor* t_15 = T_Scalar(ctx, 15);

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
  STORCH_CTX_Destroy(ctx);

  if (retval)
    printf("Error in integration_1");

  return retval;
}

int
integration_2(void)
{
  /* g = 1/(1 + exp(x)) */
  STORCH_CTX ctx = STORCH_CTX_New();

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

  STORCH_CTX_Destroy(ctx);

  if (retval)
    printf("Error in integration_2");

  return retval;
}

int
integration_3(void)
{

  /* g = a * (x + b * c) */
  STORCH_CTX ctx = STORCH_CTX_New();

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

  STORCH_CTX_Destroy(ctx);

  if (retval)
    printf("Error in integration_3");

  return retval;
}

int
integration_4(void)
{
  /* g = sum_i ( (x_i + i)**2) */
  STORCH_CTX ctx = STORCH_CTX_New();
  /* 0.5  1.1 -0.7
   * 0.7 -0.2  4.4
   */

  /*
  T_eltype b_v[] = { 1, 1, 2, 2, 3, 3 };

  T_eltype x_v[] = { 0.5, 0.7, 1.1, -0.2, -0.7, 4.4 };

  GraphNode* x = G_Parameter(ctx, T_Wrap(ctx, 2, 3, x_v));
  GraphNode* bias = G_Value(ctx, T_Wrap(ctx, 2, 3, b_v));
  */
  GraphNode* x =
    G_Parameter(ctx, T_Build(ctx, 2, 3, 6, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0));
  GraphNode* bias =
    G_Value(ctx, T_Build(ctx, 2, 3, 6, 0.5, 0.7, 1.1, -0.2, -0.7, 4.4));

  GraphNode* g = G_SumReduce1(
    ctx, G_Pow(ctx, G_Sum(ctx, x, bias), G_Value(ctx, T_Full(ctx, 2, 3, 2.0))));

  forward(g);
  backward(g);

  int retval = check_almost_eq(value(g)->data[0], 17.15) +
               check_almost_eq(value(g)->data[1], 60.89);

  retval += check_almost_eq(grad(x)->data[0], 3.0);
  retval += check_almost_eq(grad(x)->data[1], 3.4);
  retval += check_almost_eq(grad(x)->data[2], 6.2);
  retval += check_almost_eq(grad(x)->data[3], 3.6);
  retval += check_almost_eq(grad(x)->data[4], 4.6);
  retval += check_almost_eq(grad(x)->data[5], 14.8);

  STORCH_CTX_Destroy(ctx);

  if (retval)
    printf("Error in integration_4");

  return retval;
}

int
integration_5(void)
{
  /* g = [x 0] ( [x y] ** [a ; b] ) [y ; y] */
  /* g = x**(a + 1) y + x  y **(a+1) */

  STORCH_CTX ctx = STORCH_CTX_New();

  T_eltype x = 0.13;
  T_eltype y = 0.52;
  T_eltype a = 1.2;
  T_eltype b = 0.99;

  GraphNode* g1 = G_Parameter(ctx, T_Build(ctx, 1, 2, 2, x, 0.0));
  GraphNode* g2 = G_Parameter(ctx, T_Build(ctx, 1, 2, 2, x, y));
  GraphNode* g3 = G_Parameter(ctx, T_Build(ctx, 2, 1, 2, a, b));
  GraphNode* g4 = G_Parameter(ctx, T_Build(ctx, 2, 1, 2, y, y));

  GraphNode* f = G_MatMul(ctx, G_MatMul(ctx, g1, G_Pow(ctx, g2, g3)), g4);

  forward(f);
  backward(f);

  int retval = check_almost_eq(value(f)->data[0], -0.03668616);

  STORCH_CTX_Destroy(ctx);

  if (retval)
    printf("Error in integration_5");

  return retval;
}

int
main(void)
{
  return integration_1() + integration_2() + integration_3() + integration_4() +
         integration_5();
}
