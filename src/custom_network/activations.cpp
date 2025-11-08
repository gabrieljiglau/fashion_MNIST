#include <Eigen/Dense>
#include <numbers>
#include <cmath>
#include "include/activations.hpp"
#include <iostream>



Eigen::MatrixXd ActivationFunction::softmax(Eigen::MatrixXd &z){

    // by default, the operations performed by Eigen are on matrices
    Eigen::MatrixXd zExp = (z.array().rowwise() - z.rowwise().maxCoeff().array()).exp();
    Eigen::VectorXd zExpSum = zExp.rowwise().sum();

    return zExp.array().rowwise() / zExpSum.array();
}


/*
Eigen::MatrixXd ActivationFunction::activateHidden(Eigen::MatrixXd z){

    if (this->actName == SOFTMAX) {
        return 
    } else if (this->actName == RELU){
        return 
    } else if (this->actName == SIGMOID){

    }
}

Eigen::MatrixXd ActivationFunction::acivateOutput(Eigen::MatrixXd z, Loss l){


}
*/