#include <check.h>
#include <tgmath.h>

#include "../src/scorch.h"
#include "common.h"



START_TEST(forward_G_Pow)
{
  GRAPH_CTX ctx = G_CTX_New();

  GraphNode* g = G_Pow(ctx, G_Value(ctx, 13.2), G_Value(ctx, 2.2));

  forward(g);

  assert_almost_eq(value(g), pow(13.2, 2.2));

  G_CTX_Destroy(ctx);
}
END_TEST

START_TEST(forward_G_Minus)
{
  GRAPH_CTX ctx = G_CTX_New();
  GraphNode* g = G_Minus(ctx, G_Value(ctx, 8.88));

  forward(g);
  assert_almost_eq(value(g), -8.88);

  G_CTX_Destroy(ctx);
}
END_TEST

Suite*
forward_suite(void)
{
  Suite* s;
  TCase* tc_core;
  s = suite_create("Graph");

  tc_core = tcase_create("Core");

  tcase_add_test(tc_core, forward_G_Value);
  tcase_add_test(tc_core, forward_G_Sum);
  tcase_add_test(tc_core, forward_G_Diff);
  tcase_add_test(tc_core, forward_G_Div);
  tcase_add_test(tc_core, forward_G_Product);
  tcase_add_test(tc_core, forward_G_Exp);
  tcase_add_test(tc_core, forward_G_Pow);
  tcase_add_test(tc_core, forward_G_Minus);
  suite_add_tcase(s, tc_core);

  return s;
}
