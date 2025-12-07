#include <ATen/ops/clamp_min.h>
#include <cassert>
#include "include/activations.hpp"


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
    
    /*
    std::cout << z << std::endl;
    std::cout << "separator " << std::endl;
    std::cout << torch::clamp_min(z, 0) << std::endl;
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

    return (z > 0).to(torch::kFloat64);
}


torch::Tensor ActivationFunction::sigmoidDerivative(torch::Tensor sigma){
    
    /*
    d_sigma/d_z = sigma(x)(1 - sigma(x))
    */

    torch::Tensor oneMinusSigma = 1 - sigma;

    return sigma * oneMinusSigma;
}  

torch::Tensor ActivationFunction::activateHidden(torch::Tensor z){

    assert(z.sizes().size() == 2);

    if (this->actName == RELU){
        return relu(z);
    }

    if (this->actName == SIGMOID){
        return sigmoid(z);
    }

    if (this->actName == SOFTMAX){
        return softmax(z);
    }

    if (this->actName == NONE){
        return z;
    }

    // fallback; to check when calling
    return torch::ones({z.size(0), z.size(1)});
}

torch::Tensor ActivationFunction::derivative(torch::Tensor z){

    assert(z.sizes().size() == 2);

    if (this->actName == RELU){
        return reluDerivative(z);
    }

    if (this->actName == SIGMOID){
        return sigmoid(z);
    }

    // fallback; to check when calling
    return torch::ones({z.size(0), z.size(1)});
}

activationType ActivationFunction::getName(){
    return this->actName;
}