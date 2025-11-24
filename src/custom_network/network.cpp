#include "include/network.hpp"
#include "include/utils.hpp"
#include <random>
#include <assert.h>
#include <iostream>
#include <tuple>


/// TODO: modify the loops to support torch::Tensor only

/// now the weights and biases are multi dimensional tensors


void FeedForwardNetwork::checkModel(){

    if (this->numLayers != this->weights.sizes().size() or this->numLayers != this->biases.sizes().size()){
        std::cout << "Mismatch between the number of layers: " << this->numLayers << " and weights: " << this->weights.sizes().size()
                  << " or biases: " << this->biases.sizes().size();
        return;
    }



    // check if c1 == r2 as in:  current(r1, c1) x next(r2, c2)
    for (int i = 0; i < this->weights.sizes().size() - 1; i++){

        assert(this->weights[i].sizes().size() == 2);

        torch::Tensor current = this->weights[i];
        torch::Tensor next = this->weights[i + 1];

        if (current.size(1) != next.size(0)){
            std::cout << "Mismatch between layer " << i + 1 << ", shape: " << current.sizes() 
                      <<" and layer " << i + 2 << ", shape: " << next.sizes();

            return;
        }
    }

    std::cout << "Network OK 👍 " << std::endl;
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
    this->weights = torch::cat({this->weights, torch::ones({numNeurons1, numNeurons2})}, 0);
    this->biases = torch::cat({this->biases, torch::ones({numNeurons1})}, 0);

    if (actName.has_value()){
        this->activationFunctions.push_back(ActivationFunction(actName.value()));
    }
}


torch::Tensor FeedForwardNetwork::forward(torch::Tensor xBatch){

    // layer 0: do nothing
    torch::Tensor activations = torch::ones({xBatch.size(0), xBatch.size(1)});
    activations[0] = weights[0]; 

    // z_l = W_l * a_l-1 + b_l
    // a_l = activation(z_l)
    for (int layerIdx = 1; layerIdx < this->numLayers; layerIdx++){
        torch::Tensor z = this->weights[layerIdx] * activations[layerIdx - 1] + this->biases[layerIdx];
        activations[layerIdx] = this->activationFunctions[layerIdx - 1].activateHidden(z);
    }

    return activations;
}


void FeedForwardNetwork::backward(torch::Tensor xBatch, torch::Tensor yOneHot, torch::Tensor activations, int batchSize){
    
    assert(this->lossFunction.getLossType() == CROSS_ENTROPY);
    
    // initialize them with the weights/biases, to have the same shape
    torch::Tensor gradientWeights = this->weights;
    torch::Tensor gradientBiases = this->biases;

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

template<typename LoaderType>
void FeedForwardNetwork::train(LoaderType trainSet, int epochs){
    
    checkModel();

    // initialize each layer
    for (int i = 0; i < this->weights.sizes().size(); i++){

        bool isHidden = (i> 0) ? true : false;
        int numNeurons1 = this->weights[i].size(0);
        int numNeurons2 = this->weights[i].size(1);
        
        auto [newWeights, newBiases] = heInitialization(numNeurons1, numNeurons2, isHidden);
        this->weights[i] = newWeights;
        this->biases[i] = newBiases;
    }


    float loss = 0;

    int batchNumber = 0;

    for (int epoch = 0; epoch < epochs; epoch++){

        std::cout << "Epoch ====> " << epoch + 1 << std::endl;

        // deja ai xTrain si yTrain pregatite

        for (auto &batch: *trainSet){
            
            std::cout << "Now processing instances from batch " << batchNumber + 1 << std::endl;
            torch::Tensor xTrain = batch.data; // [batch_size, no_RGB_channels, img_height, img_width]
            xTrain = xTrain.to(torch::kFloat64).flatten(1); // [batch_size, img_height X img_width]
    
            torch::Tensor yTrain = batch.target;
            yTrain = yTrain.to(torch::kFloat64);
            yTrain = oneHotEncode(yTrain, 10); // of shape [batch_size, target_dim=10]
    
            torch::Tensor activations = forward(xTrain);
            backward(xTrain, yTrain, activations, xTrain.size(0));

            loss += this->lossFunction.totalLoss(activations, yTrain);
            std::cout << "Total loss " << loss / xTrain.size(1) << std::endl;
        }
    }
}
