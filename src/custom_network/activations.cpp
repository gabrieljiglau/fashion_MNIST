#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include "include/activations.hpp"


// TO-DO: modify the loops to support Eigen::Vectors only

Eigen::VectorXd ActivationFunction::softmax(Eigen::VectorXd z){

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

Eigen::VectorXd ActivationFunction::relu(Eigen::VectorXd z){

    for (int i = 0; i < z.rows(); i++){
        for (int j = 0; j < z.cols(); j++){
            z(i, j) = std::max(0.0, z(i, j));
        }
    }

    return z;
}

Eigen::VectorXd ActivationFunction::sigmoid(Eigen::VectorXd z){

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

Eigen::VectorXd ActivationFunction::reluDerivative(Eigen::VectorXd z){

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


Eigen::VectorXd ActivationFunction::sigmoidDerivative(Eigen::VectorXd z){
    
    /*
    d_sigma/d_z = sigma(x)(1 - sigma(x))
    */

    Eigen::MatrixXd sigma = z;
    Eigen::MatrixXd oneMinusSigma = sigma;

    for (int i = 0; i < z.rows(); i++){
        for (int j = 0; j < z.cols(); j++){
            oneMinusSigma(i, j) = 1 - sigma(i, j);
        }
    }

    return sigma * oneMinusSigma;
}  

Eigen::VectorXd ActivationFunction::activateHidden(Eigen::VectorXd z){

    if (this->actName == RELU){
        return relu(z);
    }

    if (this->actName == SIGMOID){
        return sigmoid(z);
    }

    // fallback; to check when calling
    return Eigen::MatrixXd::Ones(z.rows(), z.cols());
}

Eigen::VectorXd ActivationFunction::derivative(Eigen::VectorXd z){

    if (this->actName == RELU){
        return reluDerivative(z);
    }

    if (this->actName == SIGMOID){
        return sigmoid(z);
    }

    // fallback; to check when calling
    return Eigen::VectorXd::Ones(z.size());
}