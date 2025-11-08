#pragma once

#include "losses.hpp"
#include <Eigen/Dense>

enum activationType{
    SOFTMAX,
    RELU,
    SIGMOID
};

class ActivationFunction{

    activationType actName;

    public:
    ActivationFunction(const activationType type) : actName(type) {}

    Eigen::MatrixXd activateHidden(Eigen::MatrixXd z);

    Eigen::MatrixXd acivateOutput(Eigen::MatrixXd z, Loss l);

    static Eigen::MatrixXd softmax(Eigen::MatrixXd &z);

    static Eigen::MatrixXd relu(Eigen::MatrixXd &z);

    static Eigen::MatrixXd sigmoid(Eigen::MatrixXd &z);

};