#ifndef STORCH_OPS_H
#define STORCH_OPS_H

#include "storch/storch.h"

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
  SUMREDUCE_COL,
  SUMREDUCE_ROW,
  MEANREDUCE,
  N_OPS,
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

void
G_SumReduce0_Forward(GraphNode* x);

void
G_SumReduce0_Backward(GraphNode* x);

void
G_SumReduce1_Forward(GraphNode* x);

void
G_SumReduce1_Backward(GraphNode* x);

#endif // STORCH_OPS_H
