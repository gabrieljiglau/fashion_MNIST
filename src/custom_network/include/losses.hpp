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

    float totalLoss(Eigen::VectorXd activation, Eigen::VectorXd target);
};