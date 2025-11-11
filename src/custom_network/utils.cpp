#include <Eigen/Dense>
#include "include/utils.hpp"


Eigen::MatrixXd lossLastLayer(Eigen::MatrixXd activation, Eigen::MatrixXd activationPrev, Eigen::VectorXd target, 
    activationType activationName, lossType lossName){


    // when using mini bacthes, the activations are matrices

    if (lossName == MSE){
        Eigen::MatrixXd dL = activation - target;// d_L / d_a
        return dL * ActivationFunction::derivative(activation, activationName);
    }

    if (lossName == CROSS_ENTROPY){
        if (activationName == SOFTMAX){
            return activation - target;
        }
    }

    return Eigen::MatrixXd::Zero(activation.rows(), activation.cols());
}


Eigen::MatrixXd lossHidden(Eigen::MatrixXd lossNext, Eigen::MatrixXd weightsNext, Eigen::MatrixXd activationDerivative){

    return weightsNext * lossNext * activationDerivative;
}
