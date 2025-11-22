#include <algorithm>
#include <tuple>
#include "include/activations.hpp"


// TO-DO: modify the loops to support Eigen::Vectors only

torch::Tensor ActivationFunction::softmax(torch::Tensor z){

    // max(1: reduce over columns, true: keep dimensions)
    auto [zMax, idx] = z.max(1, true);
    torch::Tensor zExp = (z - zMax).exp();
    torch::Tensor zExpSum = zExp.sum(1, true);

    return zExp / zExpSum;
}

torch::Tensor ActivationFunction::relu(torch::Tensor z){


    /*
    works, but it is slow
    */

    /*
    for (int i = 0; i < z.size(0); i++){
        for (int j = 0; j < z.size(1); j++){
            z[i][j] = std::max(0.0, z[i][j].item<double>());
        }
    }
    */

    return torch::clamp_min(z, 0);
}

torch::Tensor ActivationFunction::sigmoid(torch::Tensor z){

    /*
    sigma(x) = 1 / (1 + e^-x); equivalent to e^x / (1 + e^x)
    */

    auto [zMax, idx] = z.max(1, true);
    torch::Tensor zExp = (z - zMax).exp();
    
    return zExp / zExp.add_(1.0); // add_ modifies the tensor in-place
}

torch::Tensor ActivationFunction::reluDerivative(torch::Tensor z){

    /*
    d_relu/d_z = 1 if z > 0 else 0
    */

    for (int i = 0; i < z.rows(); i++){
        for (int j = 0; j < z.cols(); j++){
            z(i, j) = (z(i, j) > 0) ? 0 : 1 ;
        }
    }

    return z;
}


torch::Tensor ActivationFunction::sigmoidDerivative(torch::Tensor z){
    
    /*
    d_sigma/d_z = sigma(x)(1 - sigma(x))
    */

    Eigen::MatrixXd sigma = z;
    Eigen::MatrixXd oneMinusSigma = sigma;

    for (int i = 0; i < z.rows(); i++){
        for (int j = 0; j < z.cols(); j++){
            oneMinusSigma(i, j) = 1 - sigma(i, j);
        }
    }

    return sigma * oneMinusSigma;
}  

torch::Tensor ActivationFunction::activateHidden(torch::Tensor z){

    if (this->actName == RELU){
        return relu(z);
    }

    if (this->actName == SIGMOID){
        return sigmoid(z);
    }

    // fallback; to check when calling
    return Eigen::MatrixXd::Ones(z.rows(), z.cols());
}

torch::Tensor ActivationFunction::derivative(torch::Tensor z){

    if (this->actName == RELU){
        return reluDerivative(z);
    }

    if (this->actName == SIGMOID){
        return sigmoid(z);
    }

    // fallback; to check when calling
    return Eigen::VectorXd::Ones(z.size());
}