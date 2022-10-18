#include "scorch/scorch.h"
#include "scorch/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


GraphNode*
G_LogSumExp(SCORCH_CTX ctx, GraphNode * x) {
  GraphNode* ones = G_Value(ctx, T_Ones(ctx, 3, 3));

  return G_Log(ctx, G_MatMul(ctx, ones, G_Exp(ctx, x)));
}


GraphNode*
modelForward(SCORCH_CTX ctx, GraphNode *x, GraphNode* w, GraphNode* b) {

  GraphNode *s = G_Sum(ctx,
        G_MatMul(ctx, w, x),
        b);

  return G_Diff(ctx, s, G_LogSumExp(ctx, s));
}

typedef struct  {
  size_t n_samples;
  size_t n_features;
  size_t n_labels;
  T_eltype *features;
  T_eltype *labels;
} Dataset;

void
trainLoop(Dataset d, Tensor *w_v, Tensor *b_v)
{

  for (size_t epoch = 0; epoch < 500; epoch++) {
    SCORCH_CTX ctx = SCORCH_CTX_New();
    GraphNode *w = G_Parameter(ctx, w_v);
    GraphNode *b = G_Parameter(ctx, b_v);

    GraphNode *loss = G_Value(ctx, T_Zeros(ctx, 1,1));

    for (size_t i =0; i<d.n_samples; i++) {
      GraphNode* g_x = G_Value(ctx, T_Wrap(ctx, d.n_features, 1, d.features + (d.n_features * i)));
      GraphNode* g_y = G_Value(ctx, T_Wrap(ctx, 1, d.n_labels, d.labels + (d.n_labels * i)));
      GraphNode * score  = modelForward(ctx, g_x, w, b);
      loss = G_Diff(ctx, loss, G_MatMul(ctx, g_y, score));
    }
    forward(loss);
    backward(loss);

    for (size_t i=0;  i< nelems(w_v); i++)
      printf("%f ", grad(w)->data[i]);
    for (size_t i=0; i<nelems(b_v); i++)
      printf("%f ", grad(b)->data[i]);
    printf("\n");
 //printf("%f %f %f\n", value(loss)->data[0], grad(b)->data[0], value(b)->data[0]);
    T_Scale_(grad(w), 0.005, grad(w));
    T_Add_(value(w), grad(w));

    T_Scale_(grad(b), 0.005, grad(b));
    T_Add_(value(b), grad(b));

    SCORCH_CTX_Destroy(ctx);
  }

}




int
main(void)
{

  char* fname = "iris.csv";

  FILE* fp = fopen(fname, "r");
  if (!fp) {
    fprintf(stderr, "Could not find %s dataset.\n", fname);
    return EXIT_FAILURE;
  }

  size_t n_samples = 150;
  size_t n_features = 4;
  size_t n_classes = 3;

  T_eltype features[n_samples * n_features];
  T_eltype targets[n_classes * n_samples];
  char variety[24];
  char buffer[128];

  fgets(buffer, 128, fp);  // Skip header

  size_t i = 0;
  while (fgets(buffer, 128, fp)) {
    sscanf(buffer,
           "%le,%le,%le,%le,%s",
           &features[i * n_features + 0],
           &features[i * n_features + 1],
           &features[i * n_features + 2],
           &features[i * n_features + 3],
           variety);

    printf("%s\n", variety);

    if (strcmp(variety, "\"Setosa\"") == 0) {
      targets[i * n_classes + 0] = 1.0;
      targets[i * n_classes + 1] = 0.0;
      targets[i * n_classes + 2] = 0.0;
    } else if (strcmp(variety, "\"Virginica\"") == 0) {
      targets[i * n_classes + 0] = 0.0;
      targets[i * n_classes + 1] = 1.0;
      targets[i * n_classes + 2] = 0.0;
    } else if (strcmp(variety, "\"Versicolor\"") == 0) {
      targets[i * n_classes + 0] = 0.0;
      targets[i * n_classes + 1] = 0.0;
      targets[i * n_classes + 2] = 1.0;
    }

    printf("%0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f\n", 
           features[i * n_features + 0],
           features[i * n_features + 1],
           features[i * n_features + 2],
           features[i * n_features + 3],
      targets[i * n_classes + 0],
      targets[i * n_classes + 1],
      targets[i * n_classes + 2]);
    i++;
  }

  Dataset d = {
    .n_features = n_features,
    .n_labels = n_classes,
    .n_samples = n_samples,
    .features = features,
    .labels = targets};

  Tensor *weight = T_Ones(NULL, n_classes, n_features);
  Tensor *bias = T_Ones(NULL, n_classes, 1);


  trainLoop(d, weight, bias);

  T_Destroy(weight);
  T_Destroy(bias);

  fclose(fp);

  return EXIT_SUCCESS;
}
