#pragma once

#include <torch/torch.h>

enum activationType{
    SOFTMAX,
    RELU,
    SIGMOID,
    NONE
};

class ActivationFunction{

    activationType actName;

    static torch::Tensor softmax(torch::Tensor z);

    static torch::Tensor relu(torch::Tensor z);

    static torch::Tensor sigmoid(torch::Tensor z);

    static torch::Tensor reluDerivative(torch::Tensor z);

    static torch::Tensor sigmoidDerivative(torch::Tensor z);

    public:

    ActivationFunction(activationType actName): actName(actName) {};

    torch::Tensor activateHidden(torch::Tensor z);

    torch::Tensor derivative(torch::Tensor z);

    activationType getName();
};