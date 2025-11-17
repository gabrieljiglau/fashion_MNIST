#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

enum lossType{

    MSE,
    CROSS_ENTROPY
};


class Loss{

    lossType lossFunction;

    static float mse(Eigen::MatrixXd activation, Eigen::MatrixXd target);

    static float crossEntropy(Eigen::MatrixXd activation, Eigen::MatrixXd target);

    public:
    Loss(lossType lossFunction): lossFunction(lossFunction) {};

    float totalLoss(Eigen::MatrixXd activation, Eigen::MatrixXd target);

    lossType getLossType();
};