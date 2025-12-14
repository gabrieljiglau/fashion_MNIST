#pragma once 

#include "losses.hpp"
#include "activations.hpp"
#include <torch/torch.h>


torch::Tensor lossWeights(torch::Tensor lossNext, torch::Tensor weightsNext, torch::Tensor activationDerivative);

torch::Tensor lossBiases(torch::Tensor loss);

torch::Tensor oneHotEncode(torch::Tensor tensor, int length);

int checkPredictions(torch::Tensor softmaxOutput, torch::Tensor groundTruth);

class FeedForwardNetwork;
std::vector<std::unique_ptr<FeedForwardNetwork>> networkSweep(int noInputs, int noOutputs, std::array<float, 4> learningRate, 
                                        std::array<float, 3> weightDecay, std::array<int, 4> batchSize, std::array<int, 3> numHidden, 
                                        Loss lossFunction, std::array<activationType, 3> activations, float percentage);