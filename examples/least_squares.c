
#include  "storch/storch.h"
#include "storch/tensor.h"
#include <stdio.h>
#include <stdlib.h>


void
getMean(size_t N, T_eltype data[static N])
{
  Tensor *m_t = T_Scalar(NULL, 15.0);

  for (size_t epoch=0; epoch<20; epoch++) {
    STORCH_CTX ctx = STORCH_CTX_New();

    GraphNode *m = G_Parameter(ctx, m_t);
    GraphNode* t = G_Value(ctx, T_Zeros(ctx, 1,1));

    for (size_t i=0; i<N; i++) {
      t = G_Sum(ctx,
          t,
          G_Pow(ctx,
            G_Diff(ctx,
              G_Value(ctx,
                T_Scalar(ctx, data[i])),
              m),
            G_Value(ctx, T_Scalar(ctx, 2))
            )
          );
    }

    forward(t);
    backward(t);

    printf("f(m) = %3.3f  df(m) = %3.3f m = %3.3f\n", value(t)->data[0], grad(m)->data[0], m_t->data[0]);
    Tensor* d = T_Scale(ctx, 0.05, grad(m));
    T_Sub_(m_t, d);

    STORCH_CTX_Destroy(ctx);
  }
  printf("MEAN: %f\n",  m_t->data[0]);
  T_Destroy(m_t);

}


int
main(void)
{

  size_t N = 5;

  T_eltype data[] = {
    1.0, 2.0, 3.0, 4.0, 5.0
  };

  getMean(N, data);

  return EXIT_SUCCESS;

}
