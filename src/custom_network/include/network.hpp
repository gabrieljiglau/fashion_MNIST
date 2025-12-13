#pragma once

#include<torch/torch.h>
#include <optional>
#include <tuple>
#include <array>
#include <vector>
#include "utils.hpp"
#include "losses.hpp"
#include "activations.hpp"


class FeedForwardNetwork{

    int miniBatchSize = 1; 
    float learningRate;
    float weightDecay;

    int numLayers = 0;
    std::vector<int> layerSizes;

    std::vector<ActivationFunction> activationFunctions;
    std::vector<torch::Tensor> weights;
    std::vector<torch::Tensor> biases;

    Loss lossFunction;

    bool checkModel();

    std::tuple<torch::Tensor, torch::Tensor> heInitialization(const int numNeurons1, const int numNeurons2, bool isHidden);

    public:

    FeedForwardNetwork(float learningRate, float weightDecay, Loss lossFunction, int miniBatchSize): 
                       learningRate(learningRate), weightDecay(weightDecay), lossFunction(lossFunction), miniBatchSize(miniBatchSize) {};


    void addLayer(const int numNeurons1, const int numNeurons2, std::optional<activationType> actName=std::nullopt);
    

    // functia asta ar trebui sa fie 'private'
    std::vector<torch::Tensor> forward(torch::Tensor xBatch);

    void backward(torch::Tensor xBatch, torch::Tensor yOneHot, std::vector<torch::Tensor> activations, int batchSize);

    void predict(torch::Tensor xTest);

    static FeedForwardNetwork buildStandardNetwork(int noInputs, std::array<int, 2> numHidden, int noOutputs,
                                            float learningRate, float batchSize, float weightDecay,
                                            std::array<activationType, 3> activationFunctions, const Loss lossFun);


    template<typename LoaderType>
    std::tuple<float, std::vector<torch::Tensor>, std::vector<torch::Tensor>> train(LoaderType &trainSet, int epochs);
};

template<typename LoaderType>
std::tuple<float, std::vector<torch::Tensor>, std::vector<torch::Tensor>> FeedForwardNetwork::train(LoaderType &trainSet, int epochs){
        
    assert(checkModel() == true);

    // initialize the layers
    for (int i = 0; i < this->weights.size(); i++){

        int numNeurons1 = this->weights[i].size(0);
        int numNeurons2 = this->weights[i].size(1);

        bool isHidden = false;
        if (i != 0 || i != this->weights.size() - 1){
            isHidden = true;
        }
            
        auto [newWeights, newBiases] = heInitialization(numNeurons1, numNeurons2, isHidden);
        this->weights[i] = newWeights;
        this->biases[i] = newBiases;

    }

    float loss = 0;
    int batchNumber = 0;

    int totalInstances = 0;
    int correctLabels = 0;
    float accuracy = 0;

    for (int epoch = 0; epoch < epochs; epoch++){

        std::cout << "Epoch " << epoch + 1 << " ====> ";

        float epochLoss = 0;
        torch::Tensor yOneHot;
        for (auto &batch: trainSet){

            batchNumber += 1;

            torch::Tensor xTrain = batch.data; // [batch_size, no_RGB_channels, img_height, img_width]
            xTrain = xTrain.to(torch::kFloat64).flatten(1); // [batch_size, img_height X img_width
            totalInstances += xTrain.size(1);
        
            torch::Tensor yTrain = batch.target;
            yTrain = yTrain.to(torch::kFloat64);
            yOneHot = oneHotEncode(yTrain, 10); // of shape [batch_size, num_classes=10]
        
            std::vector<torch::Tensor> activations = forward(xTrain);

            torch::Tensor lastActivation = activations[activations.size() - 1];
            correctLabels += checkPredictions(lastActivation, yTrain);
    
            backward(xTrain, yOneHot, activations, xTrain.size(0));

            epochLoss += this->lossFunction.totalLoss(lastActivation, yOneHot);
        }

        std::cout << "Total loss " << epochLoss / batchNumber << std::endl;
        accuracy = (float(correctLabels) / totalInstances) * 100;
        std::cout << "Prediction accuracy " << accuracy << "%" << std::endl;
    }

    return std::make_tuple(accuracy, this->weights, this->biases);
}
