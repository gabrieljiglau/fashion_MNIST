#pragma once 

#include <torch/torch.h>


torch::Tensor lossWeights(torch::Tensor lossNext, torch::Tensor weightsNext, torch::Tensor activationDerivative);

torch::Tensor lossBiases(torch::Tensor loss);

torch::Tensor oneHotEncode(torch::Tensor tensor, int length);

int checkPredictions(torch::Tensor softmaxOutput, torch::Tensor groundTruth);