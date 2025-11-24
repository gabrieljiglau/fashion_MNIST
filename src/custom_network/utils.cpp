#include "include/utils.hpp"
#include "include/activations.hpp"



// keep target as a VectorXd, since the operations are done on batches ??
torch::Tensor lossLastLayer(torch::Tensor activation, torch::Tensor activationPrev, torch::Tensor target, activationType activationName, 
                            lossType lossName){

    assert(activation.sizes() == 2);

    if (lossName == CROSS_ENTROPY){
        if (activationName == SOFTMAX){
            return activation - target;
        }
    }

    return torch::ones({activation.size(0), activation.size(1)});
}


torch::Tensor lossHidden(torch::Tensor lossNext, torch::Tensor weightsNext, torch::Tensor activationDerivative){

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