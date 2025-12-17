#pragma once

#include "losses.hpp"
#include "activations.hpp"
#include "network.hpp"
#include <memory>

class NetworkBuilder{

    int noInputs = 0;
    int noOutputs = 0;
    
    float lr = 0.1;
    float wd = 0.1;
    int bs = 32;

    std::array<int, 2> numHidden = {32, 32};
    std::array<activationType, 3> activations;

    lossType crossEntropy = lossType::CROSS_ENTROPY;
    Loss lossFunction{crossEntropy};

    FeedForwardNetwork buildStandardNetwork(int noInputs, std::array<int, 2> numHidden, int noOutputs, std::array<activationType, 3> activations,
                                            float lr, float wd, int bs, const Loss lossFunction){
                                                  
        /*
        creates a FeedForwardNetwork with 2 hidden layers, with the hyperparameters passed to the function
        */
    
        FeedForwardNetwork network(lr, wd, lossFunction, bs);
    
        network.addLayer(noInputs, numHidden[0], activations[0]); // input -> hidden 1
        network.addLayer(numHidden[0], numHidden[1], activations[1]); // hidden 1 -> hidden 2
        network.addLayer(numHidden[1], noOutputs, activations[2]); // hidden 2 -> output layer 
    
        return network;
    }

    public:
    
    NetworkBuilder& setInputs(int noInputs) {this->noInputs = noInputs; return *this;}
    NetworkBuilder& setOutputs(int noOutputs) {this->noOutputs = noOutputs; return *this;}
    NetworkBuilder& setLearningRate(float lr) {this->lr = lr; return *this;}
    NetworkBuilder& setWeightDecay(float wd) {this->wd = wd; return *this;}
    NetworkBuilder& setBatchSize(int bs) {this->bs = bs; return *this;}
    NetworkBuilder& setNumHidden(std::array<int, 2> numHidden) {this->numHidden = numHidden; return *this;}
    NetworkBuilder& setActivations(std::array<activationType, 3> activations) {this->activations = activations; return *this;}
    NetworkBuilder& setLossFunction(Loss lossFunction) {this->lossFunction = lossFunction; return *this;}

    FeedForwardNetwork build() {return buildStandardNetwork(noInputs, numHidden, noOutputs, activations, lr, wd, bs, lossFunction);}
};
