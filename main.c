#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "scorch.h"


int
main()
{

  GRAPH_CTX ctx = G_CTX_New();

  GraphNode *x = G_Parameter(ctx, 14);

  GraphNode *t = G_Product(ctx, G_Product(ctx, G_Product(ctx, G_Product(ctx, x,x), x), x), x);

  forward(t);

  printf("x: %f\n", value(x));

  backward(t);

  printf("x^5: %f\n", value(t));
  printf("dt/dx: %f\n", x->g);


  ctx = G_CTX_Destroy(ctx);

  return 0;
}
