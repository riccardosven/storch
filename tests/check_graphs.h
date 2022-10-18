#include <check.h>
#include <tgmath.h>

#include "../src/scorch.h"
#include "common.h"

START_TEST(integration_1)
{
  /* g = x^2 * y + 15*x*y + 3/x */

  GRAPH_CTX ctx = G_CTX_New();

  Tensor x_v = 1.51;
  Tensor y_v = 2.192;

  GraphNode* x = G_Value(ctx, x_v);
  GraphNode* y = G_Value(ctx, y_v);

  GraphNode* g =
    G_Sum(ctx,
          G_Sum(ctx,
                G_Product(ctx, G_Pow(ctx, x, G_Value(ctx, 2)), y),
                G_Product(ctx, G_Product(ctx, x, y), G_Value(ctx, 15))),
          G_Div(ctx, G_Value(ctx, 3), x));

  forward(g);
  backward(g);

  assert_almost_eq(value(g), pow(x_v, 2) * y_v + 15 * x_v * y_v + 3 / x_v);
  assert_almost_eq(grad(x), 2 * x_v * y_v + 15 * y_v - 3 / pow(x_v, 2));

  G_CTX_Destroy(ctx);
}
END_TEST

START_TEST(integration_2)
{
  /* g = 1/(1 + exp(x)) */
  GRAPH_CTX ctx = G_CTX_New();

  Tensor x_v = 0.84;

  GraphNode* x = G_Value(ctx, x_v);

  GraphNode* g =
    G_Div(ctx,
          G_Value(ctx, 1),
          G_Sum(ctx, G_Value(ctx, 1), G_Exp(ctx, G_Minus(ctx, x))));

  forward(g);
  backward(g);

  Tensor sigma = 1 / (1 + exp(-x_v));

  assert_almost_eq(value(g), sigma);
  assert_almost_eq(grad(x), sigma * (1 - sigma));

  G_CTX_Destroy(ctx);
}
END_TEST

Suite*
graph_suite(void)
{
  Suite* s;
  TCase* tc_core;
  s = suite_create("Graphs");

  tc_core = tcase_create("Core");

  tcase_add_test(tc_core, integration_1);
  tcase_add_test(tc_core, integration_2);
  suite_add_tcase(s, tc_core);

  return s;
}
