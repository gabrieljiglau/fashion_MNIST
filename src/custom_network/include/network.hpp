#pragma once

#include <string>
#include <vector>

class FeedForwardNetwork{

    int numLayers;
    int numNeurons;
    std::vector<std::string> activationFunctions;
    std::string lossFunction;
    float learningRate;

    public:
    FeedForwardNetwork();
    ~FeedForwardNetwork();
};