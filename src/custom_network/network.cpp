#include "include/network.hpp"
#include "include/activations.hpp"
#include "include/losses.hpp"
#include "include/utils.hpp"
#include <Eigen/Core>
#include <random>
#include <assert.h>
#include <iostream>


void FeedForwardNetwork::checkModel(){

    if (this->numLayers != this->weights.size() or this->numLayers != this->biases.size()){
        std::cout << "Mismatch between the number of layers and weights/biases";
        return;
    }

    // check if c1 = r2 as in:  current(r1, c1) x next(r2, c2)
    for (int i = 0; i < this->weights.size() - 1; i++){
        Eigen::MatrixXd current = this->weights[i];
        Eigen::MatrixXd next = this->weights[i + 1];

        if (current.cols() != next.rows()){
            std::cout << "Mismatch between layer " << i + 1 << ", shape: " << current.rows() << " x " << current.cols() 
                      <<" and layer " << i + 2 << ", shape: " << next.rows() << " x " << next.cols();

            return;
        }

    }
}


Eigen::MatrixXd FeedForwardNetwork::heInitialization(const int numNeurons1, const int numNeurons2){
    
    Eigen::MatrixXd weights = Eigen::MatrixXd(numNeurons1, numNeurons2);

    std::random_device rd;
    std::mt19937 seed(rd());
    std::normal_distribution<float> normalDistribution(0, std::sqrt(2 / numNeurons1));

    for (int i = 0; i < numNeurons1; i++){
        for (int j = 0; j < numNeurons2; j++){
            weights(i, j) = normalDistribution(seed);
        }
    }

    return weights;
}


void FeedForwardNetwork::addLayer(const int numNeurons1, const int numNeurons2){


    this->numLayers += 1;
    Eigen::MatrixXd weights = heInitialization(numNeurons1, numNeurons2);
    this->weights.push_back(weights);
    this->biases.push_back(Eigen::VectorXd::Zero(numNeurons1));
}

void FeedForwardNetwork::addActivation(activationType actName){

    this->activationFunctions.push_back(ActivationFunction(actName));
}

Eigen::MatrixXd FeedForwardNetwork::forward(Eigen::MatrixXd xBatch){

    // layer 0: do nothing
    Eigen::MatrixXd activations(xBatch.rows(), xBatch.cols());
    activations.row(0) = weights[0];

    // z_l = W_l * a_l-1 + b_l
    // a_l = activation(z_l)
    for (int layerIdx = 1; layerIdx < this->numLayers; layerIdx++){
        Eigen::MatrixXd z = this->weights[layerIdx] * activations.row(layerIdx - 1) + this->biases[layerIdx];
        activations.row(layerIdx) = this->activationFunctions[layerIdx].activateHidden(z);
    }

    return activations;
}


void FeedForwardNetwork::backward(Eigen::MatrixXd xBatch, Eigen::MatrixXd yOneHot, Eigen::MatrixXd activations, int batchSize){
    
    assert(this->lossFunction.getLossType() == CROSS_ENTROPY);
    
    Eigen::MatrixXd gradients(xBatch.rows(), xBatch.cols());
    // off by 1 errors ??
    Eigen::VectorXd dL = activations.row(this->numLayers - 1) - yOneHot.row(batchSize - 1);

    // the previous activations; and the L2 penalty
    // does the matrix need to be transposed ??
    gradients.row(0) = dL * activations.row(this->numLayers - 2) + this->weightDecay * this->weights[numLayers - 1]; 
    for (int layerIdx = this->numLayers - 2; layerIdx > 1; layerIdx--){
        Eigen::VectorXd activationDerivative = this->activationFunctions[layerIdx].derivative(activations.row(layerIdx));
        gradients.row(layerIdx) = lossHidden(gradients.row(layerIdx - 1), this->weights[layerIdx + 1], activationDerivative);
    }

    // + trebuie sa calculezi si derivata pierderii la bias: dL/db = exact delta_l


    // aici nu stiu daca matricea 'gradients' are forma (shape) corecta

    // weights update
    for (int i = 0; i < batchSize; i++){
        this->weights[i] -= this->learningRate * gradients.row(i);
        
    }

}

void FeedForwardNetwork::train(std::vector<Eigen::VectorXd> xTrain, std::vector<Eigen::VectorXd> yTrain, int epochs){
    
    checkModel();

    // split the data into miniBatches; one of the splits will have less than this->miniBatchSize items
    int evenBatches = xTrain.size() / this->miniBatchSize;
    int unevenBatchSize = xTrain.size() % this->miniBatchSize;
    

    float loss = 0;
    auto start = xTrain.begin();
    for (int epoch = 0; epoch < epochs; epoch++){

        std::cout << "Epoch ====> " << epoch + 1 << std::endl;

        for (int instanceIdx = 0; instanceIdx < xTrain.size(); ){

            int batchSize = (instanceIdx < evenBatches) ? this->miniBatchSize : unevenBatchSize;
            std::vector<Eigen::VectorXd> xSlice(start, start + batchSize);
            std::vector<Eigen::VectorXd> ySlice(start, start + batchSize);

            std::cout << "Now processing instances " << instanceIdx << " : " << instanceIdx + batchSize << std::endl;

            Eigen::MatrixXd xBatch = stackVectors(xSlice);
            Eigen::MatrixXd yBatch = stackVectors(ySlice);
            
            Eigen::MatrixXd activations = forward(xBatch);
            Eigen::MatrixXd gradients = backward(xBatch, yBatch, activations, batchSize);

            loss += this->lossFunction.totalLoss(activations, yBatch);
            
            start += batchSize;
            instanceIdx += batchSize;
        }

        std::cout << "Total loss " << loss / xTrain.size() << std::endl;
    }

}
