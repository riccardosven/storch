#ifndef SCORCH_H
#define SCORCH_H

#include "scorch/graph.h"

void forward(GraphNode* const x);

Tensor value(const GraphNode* const x);

void backward(GraphNode* const x);

Tensor grad(const GraphNode* const x);

#endif // SCORCH_H
