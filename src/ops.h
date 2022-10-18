#ifndef SCORCH_OPS_H
#define SCORCH_OPS_H

#include "scorch/scorch.h"

typedef enum
{
  DIFFERENCE,
  DIVISION,
  EXPONENTIAL,
  LOG,
  MATMUL,
  MINUS,
  NONE,
  PARAMETER,
  POWER,
  PRODUCT,
  SUM,
  VALUE,
  N_OPS
} Op;

void
G_Diff_Forward(GraphNode* x);

void
G_Diff_Backward(GraphNode* x);

void
G_Div_Forward(GraphNode* x);

void
G_Div_Backward(GraphNode* x);

void
G_Exp_Forward(GraphNode* x);

void
G_Exp_Backward(GraphNode* x);

void
G_Log_Forward(GraphNode* x);

void
G_Log_Backward(GraphNode* x);

void
G_MatMul_Forward(GraphNode* x);

void
G_MatMul_Backward(GraphNode* x);

void
G_Minus_Forward(GraphNode* x);

void
G_Minus_Backward(GraphNode* x);

void
G_Pow_Forward(GraphNode* x);

void
G_Pow_Backward(GraphNode* x);

void
G_Product_Forward(GraphNode* x);

void
G_Product_Backward(GraphNode* x);

void
G_Sum_Forward(GraphNode* x);

void
G_Sum_Backward(GraphNode* x);

#endif // SCORCH_OPS_H
