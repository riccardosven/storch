#include <check.h>
#include <tgmath.h>

#include "../src/scorch.h"
#include <stdio.h>

#include "check_backward.h"
#include "check_forward.h"
#include "check_graphs.h"

#include "common.h"

int
main(void)
{
  SRunner* sr = srunner_create(forward_suite());
  srunner_add_suite(sr, backward_suite());
  srunner_add_suite(sr, graph_suite());

  srunner_run_all(sr, CK_NORMAL);
  int number_failed = srunner_ntests_failed(sr);

  srunner_free(sr);
  return number_failed;
}
