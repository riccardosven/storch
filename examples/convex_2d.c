#include <stdio.h>
#include <stdlib.h>

#include "scorch/scorch.h"
#include "scorch/tensor.h"


GraphNode *
objectiveFunction(SCORCH_CTX ctx, GraphNode * x)
{
  T_eltype tgt[] = {0.3, -1.2};
  GraphNode* x_t = G_Value(ctx, T_Wrap(ctx, 2,1, tgt));

  return G_MatMul(ctx, 
      G_Value(ctx, T_Ones(ctx, 1,2)),
      G_Pow(ctx,
        G_Diff(ctx, x, x_t),
        G_Value(ctx, T_Full(ctx, 2, 1, 2.0))
        )
      );
}


void
optimize(Tensor *x_v, size_t n_epochs)
{
  while (--n_epochs)
  {
    SCORCH_CTX ctx = SCORCH_CTX_New();
    GraphNode *x = G_Parameter(ctx, x_v);
    GraphNode *loss = objectiveFunction(ctx, x);

    forward(loss);
    backward(loss);

    Tensor* delta = T_Scale(ctx, 0.1, grad(x));

    T_Sub_(x_v, delta);

    printf("x:[% 2.2f % 2.2f] f(x):% 3.4f\n", x_v->data[0], x_v->data[1], value(loss)->data[0]);

    SCORCH_CTX_Destroy(ctx);
  }

}


int
main(void)
{
  T_eltype x0[] = {3.0, 2.0};  // Initial position

  Tensor *x_v = T_Wrap(NULL, 2, 1, x0);

  optimize(x_v, 50);

  T_Destroy(x_v);

  return EXIT_SUCCESS;


}
