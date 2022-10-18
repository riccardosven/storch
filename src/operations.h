#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <assert.h>
#include "graph.h"
#include <tgmath.h>



void G_Product_Forward(GraphNode* x) {
  assert(x->op == PRODUCT);
  assert(x->arity == 2);

  x->t = value(x->operands[0]) * value(x->operands[1]);
}


void G_Product_Backward(GraphNode *x){

  assert(x->arity == 2);

  x->operands[0]->g += x->g * value(x->operands[1]);
  x->operands[1]->g += x->g * value(x->operands[0]);
}



void G_Sum_Forward(GraphNode *x) {
  assert(x->op == SUM);
  assert(x->arity == 2);

  x->t = value(x->operands[0]) + value(x->operands[1]);
}


void G_Sum_Backward(GraphNode *x) {
  assert(x->op == SUM);
  assert(x->arity == 2);

  x->operands[0]->g += x->g;
  x->operands[1]->g += x->g;

}



void G_Diff_Forward(GraphNode *x) {
  assert(x->op == DIFFERENCE);
  assert(x->arity == 2);

  x->t = value(x->operands[0]) - value(x->operands[1]);
}


void G_Diff_Backward(GraphNode *x) {
  assert(x->op == DIFFERENCE);
  assert(x->arity == 2);

  x->operands[0]->g += x->g;
  x->operands[1]->g -= x->g;

}




void G_Div_Forward(GraphNode *x) {
  assert(x->op == DIVISION);
  assert(x->arity == 2);

  x->t = value(x->operands[0])/value(x->operands[1]);
}


void G_Div_Backward(GraphNode *x) {
  assert(x->op == DIVISION);
  assert(x->arity == 2);

  x->operands[0]->g += x->g / value(x->operands[1]);
  x->operands[1]->g -= x->g / pow(value(x->operands[1]), 2);
}



void G_Exp_Forward(GraphNode *x) {
  assert(x->op == EXPONENTIAL);
  assert(x->arity == 1);

  x->t = exp(value(x));
}


void G_Exp_Backward(GraphNode *x) {
  assert(x->op == EXPONENTIAL);
  assert(x->arity == 1);

  x->operands[0]->g += value(x);
}

#endif // OPERATIONS_H
