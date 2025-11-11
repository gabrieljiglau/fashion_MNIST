#include "include/network.hpp"
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

void FeedForwardNetwork::addActivation(ActivationFunction activationType){
    this->activationFunctions.push_back(activationType);

}

Eigen::MatrixXd FeedForwardNetwork::forward(Eigen::VectorXd xIn){

    
}
