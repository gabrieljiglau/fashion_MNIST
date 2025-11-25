#include "include/network.hpp"
#include <random>
#include <assert.h>
#include <iostream>
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
    torch::Tensor biases = torch::ones({numNeurons1});

    std::random_device rd;
    std::mt19937 seed(rd());
    std::normal_distribution<float> normalDistribution(0.0f, std::sqrt(2.0f / numNeurons1));

    for (int i = 0; i < numNeurons1; i++){

        if (!isHidden){
            biases[i] = normalDistribution(seed);
        }

        for (int j = 0; j < numNeurons2; j++){
            weights[i][j] = normalDistribution(seed);
        }
    }

    return std::make_tuple(weights, biases);
}


void FeedForwardNetwork::addLayer(const int numNeurons1, const int numNeurons2, std::optional<activationType> actName){

    this->numLayers += 1;

    if (this->numLayers == 1){ // input layer, do nothing
        this->weights.push_back(torch::zeros({numNeurons1, numNeurons2}));
        this->biases.push_back(torch::zeros({numNeurons1}));
    } else {
        this->weights.push_back(torch::ones({numNeurons1, numNeurons2}));
        this->biases.push_back(torch::ones({numNeurons1}));    
    }

    if (actName.has_value()){
        this->activationFunctions.push_back(ActivationFunction(actName.value()));
    }

    this->layerSizes.push_back(std::make_tuple(numNeurons1, numNeurons2));
}


std::vector<torch::Tensor> FeedForwardNetwork::forward(torch::Tensor xBatch){


    /// TODO: sau la modul cum merg activarile pe batch-uri ????
    // cred ca ar trebuie sa mai fie inca o dimensiune pentru batch -> tensori cu 3 coordonate


    // layer 0: do nothing
    std::vector<torch::Tensor> activations(this->numLayers);
    for (int i = 0; i < activations.size(); i++){
        activations[i] = torch::zeros({this->layerSizes[i][0], this->layerSizes[i][1]});
    }

    // z_l = W_l * a_l-1 + b_l
    // a_l = activation(z_l)
    for (int layerIdx = 1; layerIdx < this->numLayers; layerIdx++){
        std::cout << "activations[layerIdx - 1].sizes(): " << activations[layerIdx - 1].sizes() << std::endl;
        std::cout << "this->weights[layerIdx]: " << this->weights[layerIdx].sizes() << std::endl;

        torch::Tensor z = this->weights[layerIdx] * activations[layerIdx - 1] + this->biases[layerIdx];
        activations[layerIdx] = this->activationFunctions[layerIdx - 1].activateHidden(z);
    }

    return activations;
}


void FeedForwardNetwork::backward(torch::Tensor xBatch, torch::Tensor yOneHot, std::vector<torch::Tensor> activations, int batchSize){
    
    assert(this->lossFunction.getLossType() == CROSS_ENTROPY);
    
    // initialize them with the weights/biases, to have the same shape
    std::vector<torch::Tensor> gradientWeights = this->weights;
    std::vector<torch::Tensor> gradientBiases = this->biases;

    // dl/dz output
    torch::Tensor dL = activations[this->numLayers - 1] - yOneHot[batchSize - 1]; // off by 1 errors ??

    // the previous activations; and the L2 penalty
    // does the matrix need to be transposed ??
    gradientWeights[0] = dL * activations[this->numLayers - 2] + this->weightDecay * this->weights[numLayers - 1]; 
    for (int layerIdx = this->numLayers - 2; layerIdx > 1; layerIdx--){
        torch::Tensor activationDerivative = this->activationFunctions[layerIdx].derivative(activations[layerIdx]);
        gradientWeights[layerIdx] = lossHidden(gradientWeights[layerIdx - 1], this->weights[layerIdx + 1], activationDerivative);
        gradientBiases[layerIdx] = lossHidden(gradientBiases[layerIdx - 1], this->biases[layerIdx + 1], torch::ones(this->biases[layerIdx].sizes()));
    }

    // weights and biases update
    for (int i = 0; i < batchSize; i++){
        this->weights[i] -= this->learningRate * (gradientWeights[i] / batchSize);
        this->biases[i] -= this->learningRate * (gradientBiases[i] / batchSize);   
    }

}


