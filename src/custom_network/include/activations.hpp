#pragma once
#include <Eigen/Dense>

enum activationType{
    SOFTMAX,
    RELU,
    SIGMOID
};

class ActivationFunction{

    activationType actName;

    static Eigen::MatrixXd softmax(Eigen::MatrixXd &z);

    static Eigen::MatrixXd relu(Eigen::MatrixXd &z);

    static Eigen::MatrixXd sigmoid(Eigen::MatrixXd &z);

    static Eigen::MatrixXd reluDerivative(Eigen::MatrixXd &z);

    static Eigen::MatrixXd sigmoidDerivative(Eigen::MatrixXd &z);

    public:

    ActivationFunction(activationType actName): actName(actName) {};

    Eigen::MatrixXd activateHidden(Eigen::MatrixXd &z);

    Eigen::MatrixXd derivative(Eigen::MatrixXd &z);
};