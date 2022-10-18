#include <check.h>
#include <tgmath.h>

#include "../src/scorch.h"
#include "common.h"

START_TEST(backward_G_Product)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* a = G_Value(ctx, 5);
  GraphNode* b = G_Value(ctx, 3);

  GraphNode* g = G_Product(ctx, a, b);

  forward(g);
  backward(g);

  assert_almost_eq(grad(a), 3);
  assert_almost_eq(grad(b), 5);

  G_CTX_Destroy(ctx);
}
END_TEST

START_TEST(backward_G_Value)
{
  GRAPH_CTX ctx = G_CTX_New();
  GraphNode* g = G_Value(ctx, 2);

  forward(g);
  backward(g);

  assert_almost_eq(grad(g), 1);

  G_CTX_Destroy(ctx);
}
END_TEST

START_TEST(backward_G_Sum)
{
  GRAPH_CTX ctx = G_CTX_New();
  GraphNode* a = G_Value(ctx, 13);
  GraphNode* b = G_Value(ctx, 24);
  GraphNode* g = G_Sum(ctx, a, b);

  forward(g);
  backward(g);

  assert_almost_eq(grad(a), 1);
  assert_almost_eq(grad(b), 1);

  G_CTX_Destroy(ctx);
}
END_TEST

START_TEST(backward_G_Diff)
{
  GRAPH_CTX ctx = G_CTX_New();
  GraphNode* a = G_Value(ctx, 13);
  GraphNode* b = G_Value(ctx, 24);
  GraphNode* g = G_Diff(ctx, a, b);

  forward(g);
  backward(g);

  assert_almost_eq(grad(a), 1);
  assert_almost_eq(grad(b), -1);

  G_CTX_Destroy(ctx);
}
END_TEST

START_TEST(backward_G_Div)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* a = G_Value(ctx, 14);
  GraphNode* b = G_Value(ctx, 7);

  GraphNode* g = G_Div(ctx, a, b);

  forward(g);
  backward(g);

  assert_almost_eq(grad(a), 1.0 / 7.0);
  assert_almost_eq(grad(b), -14.0 / (7.0 * 7.0));

  G_CTX_Destroy(ctx);
}
END_TEST

START_TEST(backward_G_Exp)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* a = G_Value(ctx, 7.3);
  GraphNode* g = G_Exp(ctx, a);

  forward(g);
  backward(g);

  assert_almost_eq(grad(a), exp(7.3));

  G_CTX_Destroy(ctx);
}
END_TEST

START_TEST(backward_G_Pow)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* a = G_Value(ctx, 4.4);
  GraphNode* b = G_Value(ctx, 3.2);

  GraphNode* g = G_Pow(ctx, a, b);

  forward(g);
  backward(g);

  assert_almost_eq(grad(a), 3.2 * pow(4.4, 2.2));
  assert_almost_eq(grad(b), pow(4.4, 3.2) * log(4.4));

  G_CTX_Destroy(ctx);
}
END_TEST

START_TEST(backward_G_Minus)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* x = G_Value(ctx, 9.91);

  GraphNode* g = G_Minus(ctx, x);

  forward(g);
  backward(g);

  assert_almost_eq(grad(x), -1);

  G_CTX_Destroy(ctx);
}
END_TEST

Suite*
backward_suite(void)
{
  Suite* s;
  TCase* tc_core;
  s = suite_create("Backward");

  tc_core = tcase_create("Core");

  tcase_add_test(tc_core, backward_G_Value);
  tcase_add_test(tc_core, backward_G_Sum);
  tcase_add_test(tc_core, backward_G_Diff);
  tcase_add_test(tc_core, backward_G_Div);
  tcase_add_test(tc_core, backward_G_Product);
  tcase_add_test(tc_core, backward_G_Exp);
  tcase_add_test(tc_core, backward_G_Pow);
  suite_add_tcase(s, tc_core);

  return s;
}
