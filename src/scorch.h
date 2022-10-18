#ifndef SCORCH_H
#define SCORCH_H

typedef float Tensor;

#include <assert.h>
#include "graph.h"
#include "operations.h"


void forward(GraphNode* x) {

  for (size_t i=0; i<x->arity; i++)
    forward(x->operands[i]);

  switch (x->op) {
    case VALUE: break;
    case PRODUCT: G_Product_Forward(x); break;
    case SUM: G_Sum_Forward(x); break;
    case DIFFERENCE: G_Diff_Forward(x); break;
    case DIVISION:  G_Div_Forward(x); break;
    case EXPONENTIAL:  G_Exp_Forward(x); break;
    case NONE:
    default: assert(0);
  }
}


Tensor value(GraphNode*x) {
  return x->t;
}


void backward(GraphNode *x) {
  switch (x->op) {
   case VALUE: break;
   case SUM: G_Sum_Backward(x); break;
   case PRODUCT: G_Product_Backward(x); break;
   case DIFFERENCE: G_Diff_Backward(x); break;
   case DIVISION: G_Div_Backward(x); break;
   case EXPONENTIAL: G_Exp_Backward(x); break;
   case NONE:
   default: assert(0);
  }
  for (size_t i=0; i < x->arity; i++)
    backward(x->operands[i]);
}


#endif // SCORCH_H
