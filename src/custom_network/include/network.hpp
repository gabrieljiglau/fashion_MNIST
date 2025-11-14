#pragma once

#include <Eigen/Dense>
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
    std::vector<Eigen::MatrixXd> activations;

    lossType lossFunction;

    void checkModel();

    Eigen::MatrixXd heInitialization(const int numNeurons1, const int numNeurons2);

    public:

    FeedForwardNetwork(float learningRate, float weightDecay, int miniBatchSize): 
                       learningRate(learningRate), weightDecay(weightDecay), miniBatchSize(miniBatchSize) {};
    ~FeedForwardNetwork(); // destructor cannot have any parameters

    void addLayer(const int numNeurons1, const int numNeurons2);
    
    void addActivation(activationType actName);
    
    void forward(std::vector<Eigen::VectorXd>);

    std::vector<Eigen::MatrixXd> backward(Eigen::MatrixXd xBatch, Eigen::MatrixXd yOneHot, int batchSize);

    void train(std::vector<Eigen::VectorXd> xTrain, std::vector<Eigen::VectorXd> yTrain, int epochs=10);

    void predict(Eigen::VectorXd xTest);
};