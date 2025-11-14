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

void FeedForwardNetwork::forward(std::vector<Eigen::VectorXd> xIn){

    /* xIn has potentially a number of this->miniBatchSize instances of type Eigen::VectorXd;
       , but I need them inside a MatrixXd
    */

    Eigen::MatrixXd xBatch = Eigen::MatrixXd(xIn[0].rows(), xIn.size());
    for (int i = 0; i < xBatch.rows(); i++){
        xBatch.col(i) = xIn[i]; // by default VectorXd is a column  vector
    }

    // layer 0: do nothing
    this->activations.push_back(this->weights[0]);
    for (int layerIdx = 1; layerIdx < this->numLayers; layerIdx++){
        Eigen::MatrixXd z = this->weights[layerIdx] * this->activations[layerIdx - 1] + this->biases[layerIdx];
        this->activations.push_back(this->activationFunctions[layerIdx].activateHidden(z));
    }
}


std::vector<Eigen::MatrixXd> FeedForwardNetwork::backward(Eigen::MatrixXd xBatch, Eigen::MatrixXd yOneHot, int batchSize){
    
    assert(this->lossFunction == CROSS_ENTROPY);
    
    std::vector<Eigen::MatrixXd> gradients;

    // when using minibatches activations are matrices ??

    // direct formula for dL/dz in the last layer, when using cross entropy as the loss function
    Eigen::VectorXd dL = this->activations[this->numLayers - 1] - yOneHot.row(batchSize - 1);

    // off by 1 errors ??

    // the previous activations; and the L2 penalty
    
    // aici 
    // de modificat actualizarile sa utilizeze functiile din 'utils.hpp' (lossLastLayer)

    gradients.push_back(dL * this->activations[this->numLayers - 2] + this->weightDecay * this->weights[numLayers - 1]); 
    for (int layerIdx = this->numLayers - 2; layerIdx > 1; layerIdx--){
        Eigen::MatrixXd activationDerivative = this->activationFunctions[layerIdx].derivative(this->activations[layerIdx]);
        gradients.push_back(lossHidden(gradients[gradients.size() - 1], this->weights[layerIdx + 1], activationDerivative));
    }

    return gradients;
}

void FeedForwardNetwork::train(std::vector<Eigen::VectorXd> xTrain, std::vector<Eigen::VectorXd> yTrain, int epochs=10){
    
    checkModel();

    // divide the data into miniBatches, one of the splits will have less than this->miniBatchSize items

    // turn yTrain into Eigen::MatrixXd
    for (int epoch = 0; epoch < epochs; epoch++){
        // forward
        // calculezi loss
        // backward
    }

}
