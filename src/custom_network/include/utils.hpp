#pragma once 

#include <Eigen/Dense>
#include "activations.hpp"
#include "losses.hpp"


Eigen::MatrixXd lossLastLayer(Eigen::MatrixXd activation, Eigen::MatrixXd activationPrev, Eigen::VectorXd target, 
                activationType activationName, lossType lossName);

Eigen::MatrixXd lossHidden(Eigen::MatrixXd lossNext, Eigen::MatrixXd weightsNext, Eigen::MatrixXd activationDerivative);