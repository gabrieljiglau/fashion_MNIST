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

    public:
    FeedForwardNetwork();
    ~FeedForwardNetwork();

    int addLayer(const int numNeurons, std::string activationFunction);
};