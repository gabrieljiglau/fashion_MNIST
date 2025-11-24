#pragma once

#include <torch/torch.h>

enum lossType{

    MSE,
    CROSS_ENTROPY
};


class Loss{

    lossType lossFunction;

    static float mse(torch::Tensor activation, torch::Tensor target);

    static float crossEntropy(torch::Tensor activation, torch::Tensor target);

    public:
    Loss(lossType lossFunction): lossFunction(lossFunction) {};

    float totalLoss(torch::Tensor activation, torch::Tensor target);

    lossType getLossType();
};