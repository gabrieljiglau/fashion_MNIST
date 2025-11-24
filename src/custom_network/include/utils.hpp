#pragma once 

#include <torch/torch.h>
#include "activations.hpp"
#include "losses.hpp"


torch::Tensor lossLastLayer(torch::Tensor activation, torch::Tensor activationPrev, torch::Tensor target, 
                activationType activationName, lossType lossName);

torch::Tensor lossHidden(torch::Tensor lossNext, torch::Tensor weightsNext, torch::Tensor activationDerivative);

torch::Tensor oneHotEncode(torch::Tensor tensor, int length);