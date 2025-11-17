#pragma once 

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "activations.hpp"
#include "losses.hpp"


Eigen::MatrixXd stackVectors(std::vector<Eigen::VectorXd> vectors);

Eigen::MatrixXd lossLastLayer(Eigen::MatrixXd activation, Eigen::MatrixXd activationPrev, Eigen::MatrixXd target, 
                activationType activationName, lossType lossName);

Eigen::MatrixXd lossHidden(Eigen::MatrixXd lossNext, Eigen::MatrixXd weightsNext, Eigen::MatrixXd activationDerivative);