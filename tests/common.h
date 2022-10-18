#include <assert.h>
#include <tgmath.h>

#define EPS 1e-6
#define assert_almost_eq(A, B) assert(fabs((A) - (B)) <= EPS)
