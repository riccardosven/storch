#include <check.h>
#include <tgmath.h>
#include <stdio.h>

#define EPS 1e-6
#define check_almost_eq(A, B) (fabs((A) - (B)) >= EPS)
#define almost_eq(A, B) (fabs((A) - (B)) <= EPS)
#define ck_assert_almost_eq(A, B) ck_assert(fabs((A) - (B)) <= EPS)
