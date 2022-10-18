#include <tgmath.h>

#include "scorch/scorch.h"
#include "common.h"

int
integration_1(void)
{
  /* g = x^2 * y + 15*x*y + 3/x */

  GRAPH_CTX ctx = G_CTX_New();

  Tensor x_v = 1.51;
  Tensor y_v = 2.192;

  GraphNode* x = G_Parameter(ctx, x_v);
  GraphNode* y = G_Parameter(ctx, y_v);

  GraphNode* g =
    G_Sum(ctx,
          G_Sum(ctx,
                G_Product(ctx, G_Pow(ctx, x, G_Parameter(ctx, 2)), y),
                G_Product(ctx, G_Product(ctx, x, y), G_Parameter(ctx, 15))),
          G_Div(ctx, G_Parameter(ctx, 3), x));

  forward(g);
  backward(g);

  int retval = check_almost_eq(value(g), pow(x_v, 2) * y_v + 15 * x_v * y_v + 3 / x_v);
  retval += check_almost_eq(grad(x), 2 * x_v * y_v + 15 * y_v - 3 / pow(x_v, 2));

  G_CTX_Destroy(ctx);

  return retval;
}

int
integration_2(void)
{
  /* g = 1/(1 + exp(x)) */
  GRAPH_CTX ctx = G_CTX_New();

  Tensor x_v = 0.84;

  GraphNode* x = G_Parameter(ctx, x_v);

  GraphNode* g =
    G_Div(ctx,
          G_Parameter(ctx, 1),
          G_Sum(ctx, G_Parameter(ctx, 1), G_Exp(ctx, G_Minus(ctx, x))));

  forward(g);
  backward(g);

  Tensor sigma = 1 / (1 + exp(-x_v));

  int retval = check_almost_eq(value(g), sigma);
  retval += check_almost_eq(grad(x), sigma * (1 - sigma));

  G_CTX_Destroy(ctx);

  return retval;
}

int
integration_3(void)
{

  /* g = a * (x + b * c) */
  GRAPH_CTX ctx = G_CTX_New();

  Tensor x_v = 0.88;
  Tensor a_v = 4;
  Tensor b_v = 5;
  Tensor c_v = 3;

  GraphNode* x = G_Parameter(ctx, x_v);
  GraphNode* a = G_Value(ctx, a_v);
  GraphNode* b = G_Value(ctx, b_v);
  GraphNode* c = G_Value(ctx, c_v);
  GraphNode* y1 = G_Product(ctx, b, c);
  GraphNode* y2 = G_Sum(ctx, x, y1);
  GraphNode* g = G_Product(ctx, a, y2);

  forward(g);
  backward(g);

  int retval = check_almost_eq(grad(x), a_v);
  retval += check_almost_eq(grad(b), 0.0);

  G_CTX_Destroy(ctx);

  return retval;
}


int
main(void)
{
  return integration_1() + integration_2() + integration_3();
}
