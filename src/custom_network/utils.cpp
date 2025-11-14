#include <Eigen/Dense>
#include "include/utils.hpp"
#include "include/activations.hpp"


Eigen::MatrixXd lossLastLayer(Eigen::MatrixXd activation, Eigen::MatrixXd activationPrev, Eigen::VectorXd target, 
    activationType activationName, lossType lossName){

    // !! these need to be changed, the networks works on mini batches now !
    if (lossName == MSE){
        Eigen::MatrixXd dL = activation - target;// d_L / d_a
        ActivationFunction actFunction = ActivationFunction(activationName);
        return dL * actFunction.derivative(activation);
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
