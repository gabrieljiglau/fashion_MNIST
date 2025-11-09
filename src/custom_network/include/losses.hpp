#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

enum lossType{

    MSE,
    CROSS_ENTROPY
};


class Loss{

    lossType lossFunction;

    static float mse(Eigen::VectorXd activation, Eigen::VectorXd target);

    static float crossEntropy(Eigen::VectorXd activation, Eigen::VectorXd target);

    public:
    Loss(lossType lossFunction): lossFunction(lossFunction) {};

    Eigen::MatrixXd lossLastLayer(Eigen::MatrixXd activation, lossType l);

    Eigen::MatrixXd lossHidden(Eigen::MatrixXd lossNext, Eigen::MatrixXd weightsNext, Eigen::MatrixXd activationDerivative);

    float totalLoss(Eigen::VectorXd activation, Eigen::VectorXd target);
};