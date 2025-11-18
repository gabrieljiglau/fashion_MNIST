#include <Eigen/Dense>
#include <cmath>
#include "include/losses.hpp"


// !! posibil aici sa fie nevoie sa modifici functiile deoarece nu mai primesc ca parametru VectorXd, ci MatrixXd 

float Loss::crossEntropy(Eigen::MatrixXd activation, Eigen::MatrixXd target){
    
    /*
    activation  -> the output from the activation funtion (i.e. softmax, relu)
    targetIndex -> target in one-hot encoding form
    */

    int targetIndex = 0;
    for (int i = 0; i < target.size(); i++){
        if (target(i) == 1){
            targetIndex = i;
            break;
        }
    }

    return -std::log10(activation(targetIndex));
}

float Loss::mse(Eigen::MatrixXd activation, Eigen::MatrixXd target){

    float error = 0;
    for (int i = 0; i < activation.size(); i++){
        error += std::pow(activation(i) - target(i), 2) / 2;
    }
    return error;
}

float Loss::totalLoss(Eigen::MatrixXd activation, Eigen::MatrixXd target){

    if (this->lossFunction == MSE){
        return mse(activation, target);
    }

    if (this->lossFunction == CROSS_ENTROPY){
        return crossEntropy(activation, target);
    }

    // fallback; to check when calling
    return 0;
}

lossType Loss::getLossType(){
    return this->lossFunction;
}
