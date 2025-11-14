#include <Eigen/Dense>
#include <cmath>
#include "include/losses.hpp"


float Loss::crossEntropy(Eigen::VectorXd activation, Eigen::VectorXd target){
    
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

float Loss::mse(Eigen::VectorXd activation, Eigen::VectorXd target){

    float error = 0;
    for (int i = 0; i < activation.size(); i++){
        error += std::pow(activation(i) - target(i), 2) / 2;
    }
    return error;
}

float Loss::totalLoss(Eigen::VectorXd activation, Eigen::VectorXd target){

    if (this->lossFunction == MSE){
        return mse(activation, target);
    }

    if (this->lossFunction == CROSS_ENTROPY){
        return crossEntropy(activation, target);
    }

    // fallback; to check when calling
    return 0;
}
