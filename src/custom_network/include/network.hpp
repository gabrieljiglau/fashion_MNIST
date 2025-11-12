#pragma once

#include <Eigen/Dense>
#include <vector>
#include "losses.hpp"
#include "activations.hpp"

class FeedForwardNetwork{

    int numLayers = 0;
    float learningRate;
    float weightDecay;

    std::vector<ActivationFunction> activationFunctions;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;

    lossType lossFunction;

    void checkModel();

    Eigen::MatrixXd heInitialization(const int numNeurons1, const int numNeurons2);

    public:

    FeedForwardNetwork(float learningRate, float weightDecay): learningRate(learningRate), weightDecay(weightDecay) {};
    ~FeedForwardNetwork(); // destructor cannot have any parameters

    void addLayer(const int numNeurons1, const int numNeurons2);
    
    void addActivation(activationType actName);
    
    Eigen::MatrixXd forward(std::vector<Eigen::VectorXd>);

    Eigen::MatrixXd backward();

    void train(std::vector<Eigen::VectorXd> xTrain, std::vector<Eigen::VectorXd> yTrain);

    void predict(Eigen::VectorXd xTest);
};