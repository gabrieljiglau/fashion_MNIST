#pragma once

#include <Eigen/Dense>
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
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;

    Loss lossFunction;

    void checkModel();

    Eigen::MatrixXd heInitialization(const int numNeurons1, const int numNeurons2);

    public:

    FeedForwardNetwork(float learningRate, float weightDecay, Loss lossFunction, int miniBatchSize): 
                       learningRate(learningRate), weightDecay(weightDecay), lossFunction(lossFunction), miniBatchSize(miniBatchSize) {};

    void addLayer(const int numNeurons1, const int numNeurons2, std::optional<activationType> actName=std::nullopt);
    
    Eigen::MatrixXd forward(Eigen::MatrixXd xBatch);

    void backward(Eigen::MatrixXd xBatch, Eigen::MatrixXd yOneHot, Eigen::MatrixXd activations, int batchSize);

    void train(std::vector<Eigen::VectorXd> xTrain, std::vector<Eigen::VectorXd> yTrain, int epochs=10);

    void predict(Eigen::VectorXd xTest);
};