#pragma once
#include <Eigen/Dense>

enum activationType{
    SOFTMAX,
    RELU,
    SIGMOID
};

class ActivationFunction{

    static Eigen::MatrixXd softmax(Eigen::MatrixXd &z);

    static Eigen::MatrixXd relu(Eigen::MatrixXd &z);

    static Eigen::MatrixXd sigmoid(Eigen::MatrixXd &z);

    static Eigen::MatrixXd reluDerivative(Eigen::MatrixXd &z);

    static Eigen::MatrixXd sigmoidDerivative(Eigen::MatrixXd &z);

    public:

    static Eigen::MatrixXd activateHidden(Eigen::MatrixXd &z, activationType aType);

    static Eigen::MatrixXd derivative(Eigen::MatrixXd &z, activationType aType);
};