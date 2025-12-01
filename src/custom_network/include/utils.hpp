#pragma once 

#include <torch/torch.h>
#include "activations.hpp"
#include "losses.hpp"


torch::Tensor lossLastLayer(torch::Tensor activation, torch::Tensor activationPrev, torch::Tensor target, 
                activationType activationName, lossType lossName);

static torch::Tensor lossHidenWeights(torch::Tensor lossNext, torch::Tensor weightsNext, torch::Tensor activationDerivative, bool forBiases);

static torch::Tensor lossHiddenBiases(torch::Tensor lossNext, torch::Tensor weightsNext, torch::Tensor activationDerivative, bool forBiases);

torch::Tensor lossHidden(torch::Tensor lossNext, torch::Tensor weightsNext, torch::Tensor activationDerivative, bool forBiases);

torch::Tensor oneHotEncode(torch::Tensor tensor, int length);