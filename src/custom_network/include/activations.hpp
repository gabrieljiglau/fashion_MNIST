#pragma once
#include <Eigen/Dense>

enum activationType{
    SOFTMAX,
    RELU,
    SIGMOID
};

class ActivationFunction{

    activationType actName;

    static Eigen::VectorXd softmax(Eigen::VectorXd z);

    static Eigen::VectorXd relu(Eigen::VectorXd z);

    static Eigen::VectorXd sigmoid(Eigen::VectorXd z);

    static Eigen::VectorXd reluDerivative(Eigen::VectorXd z);

    static Eigen::VectorXd sigmoidDerivative(Eigen::VectorXd z);

    public:

    ActivationFunction(activationType actName): actName(actName) {};

    Eigen::VectorXd activateHidden(Eigen::VectorXd z);

    Eigen::VectorXd derivative(Eigen::VectorXd z);
};