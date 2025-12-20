#include "include/network.hpp"
#include "include/utils.hpp"
#include <ATen/TensorIndexing.h>
#include <random>
#include <assert.h>
#include <iostream>
#include <torch/types.h>
#include <tuple>


bool FeedForwardNetwork::checkModel(){

    if (this->numLayers != this->weights.size() or this->numLayers != this->biases.size()){
        std::cout << "Mismatch between the number of layers: " << this->numLayers << " and weights: " << this->weights.size()
                  << " or biases: " << this->biases.size() << std::endl;
        return false;
    }



    // check if c1 == r2 as in:  current(r1, c1) x next(r2, c2)
    for (int i = 0; i < this->weights.size() - 1; i++){

        assert(this->weights[i].sizes().size() == 2);

        torch::Tensor current = this->weights[i];
        torch::Tensor next = this->weights[i + 1];

        if (current.size(1) != next.size(0)){
            std::cout << "Mismatch between layer " << i + 1 << ", shape: " << current.sizes() 
                      <<" and layer " << i + 2 << ", shape: " << next.sizes();

            return false;
        }
    }

    std::cout << "Network OK 👍 " << std::endl;
    return true;
}


std::tuple<torch::Tensor, torch::Tensor> FeedForwardNetwork::heInitialization(const int numNeurons1, const int numNeurons2, bool isHidden){
    
    torch::Tensor weights = torch::ones({numNeurons1, numNeurons2});
    torch::Tensor biases = torch::zeros({numNeurons2});

    std::random_device rd;
    std::mt19937 seed(rd());
    std::normal_distribution<float> normalDistribution(0.0f, std::sqrt(2.0f / numNeurons1));

    for (int i = 0; i < numNeurons1; i++){
        if (isHidden){
            for (int j = 0; j < numNeurons2; j++){
                weights[i][j] = normalDistribution(seed);
            }
        }
    }

    for (int i = 0; i < numNeurons2; i++){
        if (isHidden){
            biases[i] = normalDistribution(seed);
        }
    }

    return std::make_tuple(weights, biases);
}

void FeedForwardNetwork::addLayer(const int numNeurons1, const int numNeurons2, std::optional<activationType> actName){

    this->weights.push_back(torch::zeros({numNeurons1, numNeurons2}));
    this->biases.push_back(torch::zeros({numNeurons1}));    
    
    if (actName.has_value()){
        this->activationFunctions.push_back(ActivationFunction(actName.value()));
    }

    // a vector of ints, representing the number of neurons in each layer
    this->layerSizes.push_back(numNeurons1);
    this->numLayers += 1;
}


std::vector<torch::Tensor> FeedForwardNetwork::forward(torch::Tensor xBatch){

    // contains both the preactivation, as well as the activation itself, that's why the size is numLayers + 1
    std::vector<torch::Tensor> activations(this->numLayers + 1);

    // layer 0: do nothing  
    activations[0] = xBatch;

    // z_l = W_l * a_l-1 + b_l
    // a_l = activation(z_l), for all layers, with the exception of the last one
    for (int layerIdx = 0; layerIdx < activations.size() - 1; layerIdx++){

        torch::Tensor z = torch::matmul(activations[layerIdx].to(torch::kFloat64), this->weights[layerIdx].to(torch::kFloat64));
        z += this->biases[layerIdx].to(torch::kFloat64);
        activations[layerIdx + 1] = this->activationFunctions[layerIdx].activateHidden(z);
      

        //std::cout << activations[layerIdx].sizes() << std::endl;
    }

    return activations;
}


void FeedForwardNetwork::backward(torch::Tensor xBatch, torch::Tensor yOneHot, std::vector<torch::Tensor> activations, int batchSize){
    
    assert(this->lossFunction.getLossType() == CROSS_ENTROPY);
    
    // initialize them with the weights/biases, to have the same shape
    std::vector<torch::Tensor> gradientWeights = this->weights;
    std::vector<torch::Tensor> gradientBiases = this->biases;

    // dl/dz output
    torch::Tensor delta = activations[this->numLayers] - yOneHot;

    for (int layerIdx = this->numLayers - 1; layerIdx >= 0; layerIdx--){

        if (layerIdx == this->numLayers - 1){ // loss from the output layer, computable directly
            
            torch::Tensor gradW = torch::matmul(activations[layerIdx].transpose(0, 1), delta);
            gradientWeights[layerIdx] = gradW + this->weightDecay * this->weights[layerIdx]; // add the L2 penalty
            
            gradientBiases[layerIdx] = lossBiases(delta);

        } else {

            // (w_l+1 @ delta_l+1) * a_l (the next activation)
            torch::Tensor activationDerivative = this->activationFunctions[layerIdx].derivative(activations[layerIdx + 1]);
            delta = lossWeights(delta, this->weights[layerIdx + 1], activationDerivative);
            
            // std::cout << "shape(dL) = " << delta.sizes() << std::endl;
            // std::cout << "activations[layerIdx].sizes() " << activations[layerIdx].sizes() << std::endl;


            //delta * a_l - 1
            gradientWeights[layerIdx] = torch::matmul(activations[layerIdx].transpose(0, 1).to(torch::kFloat64), delta);
            gradientWeights[layerIdx] += this->weightDecay * this->weights[layerIdx];
            
            gradientBiases[layerIdx] = lossBiases(delta);
        }
    }

    // weights and biases update
    for (int i = 1; i < this->numLayers; i++){
        this->weights[i] -= this->learningRate * (gradientWeights[i] / batchSize);
        this->biases[i] -= this->learningRate * (gradientBiases[i] / batchSize);   
    }

}


