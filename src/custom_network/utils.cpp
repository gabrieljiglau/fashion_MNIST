#include <Eigen/Dense>
#include <torch/types.h>
#include "include/utils.hpp"
#include "include/activations.hpp"


Eigen::MatrixXd stackVectors(std::vector<Eigen::VectorXd> vectors){

    /*
    returns a matrix, whose rows are vectors from the input 
    */

    int rows = vectors.size();
    int cols = vectors[0].size();
    Eigen::MatrixXd matrix(rows, cols);

    for (int i = 0; i < rows; i++){
        matrix.row(i) = vectors[i].transpose();
    }

    return matrix;
}


// keep target as a VectorXd, since the operations are done on batches ??
Eigen::MatrixXd lossLastLayer(Eigen::MatrixXd activation, Eigen::MatrixXd activationPrev, Eigen::VectorXd target, 
    activationType activationName, lossType lossName){

 
    if (lossName == CROSS_ENTROPY){
        if (activationName == SOFTMAX){
            return activation - target;
        }
    }

    return Eigen::MatrixXd::Zero(activation.rows(), activation.cols());
}


Eigen::MatrixXd lossHidden(Eigen::MatrixXd lossNext, Eigen::MatrixXd weightsNext, Eigen::MatrixXd activationDerivative){

    return weightsNext * lossNext * activationDerivative;
}

torch::Tensor oneHotEncode(torch::Tensor tensor, int length){

    torch::Tensor oneHot = torch::zeros({tensor.size(0), length});

    for (int i = 0; i < tensor.size(0); i++){
        int target = tensor[i].item<int>();
        oneHot.index_put_({i, target}, 1); 
    }
    
    return oneHot;
}