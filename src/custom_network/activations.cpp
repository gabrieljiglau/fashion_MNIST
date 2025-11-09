#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include "include/activations.hpp"


Eigen::MatrixXd ActivationFunction::softmax(Eigen::MatrixXd &z){

    // by default, the operations performed by Eigen are on matrices
    Eigen::MatrixXd zExp = (z.array().colwise() - z.rowwise().maxCoeff().array()).exp().matrix();
    Eigen::VectorXd zExpSum = zExp.rowwise().sum();
    
    for (int i = 0; i < zExp.rows(); i++){
        for (int j = 0; j < zExp.cols(); j++){
            zExp(i, j) /= zExpSum(i);
        }
    }

    return zExp;
}

Eigen::MatrixXd ActivationFunction::relu(Eigen::MatrixXd &z){

    for (int i = 0; i < z.rows(); i++){
        for (int j = 0; j < z.cols(); j++){
            z(i, j) = std::max(0.0, z(i, j));
        }
    }

    return z;
}

Eigen::MatrixXd ActivationFunction::sigmoid(Eigen::MatrixXd &z){

    /*
    sigma(x) = 1 / (1 + e^-x); equivalent to e^x / (1 + e^x)
    */

    Eigen::MatrixXd zExp = (z.array().colwise() - z.rowwise().maxCoeff().array()).exp().matrix();

    for (int i = 0; i < zExp.rows(); i++){
        for (int j = 0; j < zExp.cols(); j++){
            zExp(i, j) /= zExp(i, j) + 1;
        }
    }

    return zExp;
}

Eigen::MatrixXd ActivationFunction::reluDerivative(Eigen::MatrixXd &z){

    /*
    d_relu/d_z = 1 if z > 0 else 0
    */

    for (int i = 0; i < z.rows(); i++){
        for (int j = 0; j < z.cols(); j++){
            z(i, j) = (z(i, j) > 0) ? 0 : 1 ;
        }
    }

    return z;
}


Eigen::MatrixXd ActivationFunction::sigmoidDerivative(Eigen::MatrixXd &z){
    
    /*
    d_sigma/d_z = sigma(x)(1 - sigma(x))
    */

    Eigen::MatrixXd sigma = sigmoid(z);
    Eigen::MatrixXd oneMinusSigma = sigma;

    for (int i = 0; i < z.rows(); i++){
        for (int j = 0; j < z.cols(); j++){
            oneMinusSigma(i, j) = 1 - sigma(i, j);
        }
    }

    return sigma * oneMinusSigma;
}  

Eigen::MatrixXd ActivationFunction::activateHidden(Eigen::MatrixXd &z){

    if (this->activationName == RELU){
        return relu(z);
    }

    if (this->activationName == SIGMOID){
        return sigmoid(z);
    }

    // fallback; to check when calling
    return Eigen::MatrixXd::Ones(z.rows(), z.cols());
}

Eigen::MatrixXd ActivationFunction::derivative(Eigen::MatrixXd &z){

    if (this->activationName == RELU){
        return reluDerivative(z);
    }

    if (this->activationName == SIGMOID){
        return sigmoid(z);
    }

    // fallback; to check when calling
    return Eigen::MatrixXd::Ones(z.rows(), z.cols());
}
