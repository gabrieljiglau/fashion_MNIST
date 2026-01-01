#pragma once

#include "losses.hpp"
#include "activations.hpp"
#include "network.hpp"
#include "../../libtorch/torch_network.cpp"
#include <torch/nn/modules/container/any.h>

template<typename Derived>
class NetworkBuilder{

    protected:

        /*
        initialized variables for the implicit constructor
        */

        int noInputs = 0;
        int noOutputs = 0;
        
        float lr = 0.1;
        float wd = 0.1;
        int bs = 32;

        template<typename... Args>
        auto build(Args&&... args){ // &&: forward reference
            return static_cast<Derived*>(this)->buildNetwork(args ...);
        }

        NetworkBuilder& setInputs(int noInputs) {this->noInputs = noInputs; return *this;}
        NetworkBuilder& setOutputs(int noOutputs) {this->noOutputs = noOutputs; return *this;}
        NetworkBuilder& setLearningRate(float lr) {this->lr = lr; return *this;}
        NetworkBuilder& setBatchSize(int bs) {this->bs = bs; return *this;}
};


class CustomBuilder: NetworkBuilder<CustomBuilder>{


    std::array<int, 2> numHidden = {32, 32};
    std::array<activationType, 3> activations;

    lossType crossEntropy = lossType::CROSS_ENTROPY;
    Loss lossFunction{crossEntropy};


    FeedForwardNetwork buildNetwork(int noInputs, std::array<int, 2> numHidden, int noOutputs, std::array<activationType, 3> activations,
                                            float lr, float wd, int bs){
                
        /*
        creates a FeedForwardNetwork with 2 hidden layers, with the hyperparameters passed to the function
        */

        FeedForwardNetwork network(lr, wd, this->lossFunction, bs);

        network.addLayer(noInputs, numHidden[0], activations[0]); // input -> hidden 1
        network.addLayer(numHidden[0], numHidden[1], activations[1]); // hidden 1 -> hidden 2
        network.addLayer(numHidden[1], noOutputs, activations[2]); // hidden 2 -> output layer 

        return network;
    }
    
    NetworkBuilder& setWeightDecay(float wd) {this->wd = wd; return *this;}
    NetworkBuilder& setNumHidden(std::array<int, 2> numHidden) {this->numHidden = numHidden; return *this;}
    NetworkBuilder& setActivations(std::array<activationType, 3> activations) {this->activations = activations; return *this;}
    NetworkBuilder& setLossFunction(Loss lossFunction) {this->lossFunction = lossFunction; return *this;}

};

class TorchBuilder: NetworkBuilder<CustomBuilder>{

    /*
    initialized variables for the implicit constructor
    */

    torch::nn::AnyModule loss;
    std::unique_ptr<torch::optim::Optimizer> optimizer; // make it explicit to be SGD

    std::shared_ptr<TorchNetwork> buildNetwork(int noInputs, std::array<int, 2> numHidden, int noOutputs, float lr, int bs,
                                              torch::nn::AnyModule loss){
        std::shared_ptr<TorchNetwork> network = std::make_shared<TorchNetwork>(noInputs, numHidden, noOutputs, bs);
        network->setOptimizer(std::make_unique<torch::optim::SGD>(
                            network->parameters(), 
                            torch::optim::SGDOptions(lr))
                        ); // therefore, this line seems redundant

        network->setLossFunction(loss);
        
        return network;
    }

    NetworkBuilder& setLossFunction(torch::nn::AnyModule loss) {this->loss = loss; return *this;}
};
