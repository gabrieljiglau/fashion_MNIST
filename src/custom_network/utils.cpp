#include "include/utils.hpp"


torch::Tensor lossWeights(torch::Tensor lossNext, torch::Tensor weightsNext, torch::Tensor activationDerivative){

    return torch::matmul(lossNext.to(torch::kFloat64), weightsNext.to(torch::kFloat64).transpose(0, 1)) * activationDerivative.to(torch::kFloat64);
}

torch::Tensor lossBiases(torch::Tensor losses){

    // before: shape [batch, num_biases], after [num_biases]
    return losses.sum(0);
}


// aici de fapt erau bune alea 2 functii, pentru ca fac lucruri diferite !!!

torch::Tensor oneHotEncode(torch::Tensor tensor, int length){

    torch::Tensor oneHot = torch::zeros({tensor.size(0), length});

    for (int i = 0; i < tensor.size(0); i++){
        int target = tensor[i].item<int>();
        oneHot.index_put_({i, target}, 1); 
    }
    
    return oneHot;
}