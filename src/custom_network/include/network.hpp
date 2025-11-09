#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include "activations.hpp"

class FeedForwardNetwork{

    int numLayers = 0;
    std::vector<ActivationFunction> activationFunctions;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;
    std::string lossFunction;
    float learningRate;
    float weightDecay;

    public:
    FeedForwardNetwork(float learningRate, float weightDecay): learningRate(learningRate), weightDecay(weightDecay) {};
    ~FeedForwardNetwork(); // destructor cannot have any parameters

    int addLayer(const int numNeurons, std::string activationFunction);

    Eigen::MatrixXd forward(Eigen::VectorXd xIn);

    Eigen::MatrixXd backward();

    void train(Eigen::MatrixXd xTrain, Eigen::VectorXd yTrain);

    void predict(Eigen::VectorXd xTest);
};