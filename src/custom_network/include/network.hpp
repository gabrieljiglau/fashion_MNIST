#pragma once

#include<torch/torch.h>
#include <optional>
#include <vector>
#include "losses.hpp"
#include "activations.hpp"


class FeedForwardNetwork{

    int numLayers = 0;
    int miniBatchSize = 1; 
    float learningRate;
    float weightDecay;

    std::vector<ActivationFunction> activationFunctions;
    torch::Tensor weights;
    torch::Tensor biases;

    Loss lossFunction;

    void checkModel();

    std::tuple<torch::Tensor, torch::Tensor> heInitialization(const int numNeurons1, const int numNeurons2, bool isHidden);

    public:

    FeedForwardNetwork(float learningRate, float weightDecay, Loss lossFunction, int miniBatchSize): 
                       learningRate(learningRate), weightDecay(weightDecay), lossFunction(lossFunction), miniBatchSize(miniBatchSize) {};

    void addLayer(const int numNeurons1, const int numNeurons2, std::optional<activationType> actName=std::nullopt);
    

    // functia asta ar trebui sa fie 'private'
    torch::Tensor forward(torch::Tensor xBatch);

    void backward(torch::Tensor xBatch, torch::Tensor yOneHot, torch::Tensor activations, int batchSize);

    template<typename LoaderType>
    void train(LoaderType trainSet, int epochs=10);

    void predict(torch::Tensor xTest);
};