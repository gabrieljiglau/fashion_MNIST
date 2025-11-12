#include "include/network.hpp"
#include "include/activations.hpp"
#include <Eigen/Core>
#include <random>
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

Eigen::MatrixXd FeedForwardNetwork::forward(std::vector<Eigen::VectorXd> xIn){

    // prepare the mini batch
    Eigen::MatrixXd xBatch = Eigen::MatrixXd(xIn[0].rows(), xIn.size());
    for (int i = 0; i < xBatch.rows(); i++){
        xBatch.col(i) = xIn[i]; // by default VectorXd is a column  vector
    }

    // layer 0: do nothing
    Eigen::MatrixXd prevActivations = this->weights[0];
    for (int layerIdx = 1; layerIdx < this->numLayers; layerIdx++){
        Eigen::MatrixXd z = this->weights[layerIdx] * prevActivations + this->biases[layerIdx];
        ActivationFunction actFunction = this->activationFunctions[layerIdx];
        prevActivations = actFunction.activateHidden(z);
    }

    return prevActivations;
}


Eigen::MatrixXd FeedForwardNetwork::backward(){

    // last layer: total loss, computable directly
    for (int layerIdx = this->numLayers - 1; layerIdx > 1; layerIdx--){

    }

}

void FeedForwardNetwork::train(std::vector<Eigen::VectorXd> xTrain, std::vector<Eigen::VectorXd> yTrain){
    checkModel();
}
