#include "include/utils.hpp"
#include <ATen/core/interned_strings.h>


torch::Tensor lossWeights(torch::Tensor lossNext, torch::Tensor weightsNext, torch::Tensor activationDerivative){

    return torch::matmul(lossNext.to(torch::kFloat64), weightsNext.to(torch::kFloat64).transpose(0, 1)) * activationDerivative.to(torch::kFloat64);
}

torch::Tensor lossBiases(torch::Tensor losses){

    // before: shape [batch, num_biases], after [num_biases]
    return losses.sum(0);
}


torch::Tensor oneHotEncode(torch::Tensor tensor, int length){

    torch::Tensor oneHot = torch::zeros({tensor.size(0), length});

    for (int i = 0; i < tensor.size(0); i++){
        int target = tensor[i].item<int>();
        oneHot.index_put_({i, target}, 1); 
    }
    
    return oneHot;
}

int checkPredictions(torch::Tensor softmaxOutput, torch::Tensor groundTruth){

    /*
    return the number of correctly labeled examples 
    */

    torch::Tensor predictions = torch::argmax(softmaxOutput, 1);
    assert(predictions.sizes() == groundTruth.sizes()); // and they should be [batch_size]
    
    int correctPredictions = 0;
    for (int i = 0; i < predictions.size(0); i++){
        if (predictions[i].item<int>() == groundTruth[i].item<int>()){
            correctPredictions += 1;
        }
    }

    return correctPredictions;
}

void hyperparameterSweep(){

    /// TODO: randomized search

}