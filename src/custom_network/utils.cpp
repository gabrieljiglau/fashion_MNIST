#include "include/utils.hpp"
#include "include/activations.hpp"



/// TODO: ce mai este si cu functia asta ??
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

/// TODO: reparat functiile pe aici
static torch::Tensor lossHiddenWeights(torch::Tensor lossNext, torch::Tensor weightsNext, torch::Tensor activationDerivative){

    std::cout << "lossNext.sizes(): " << lossNext.sizes() << std::endl;
    std::cout << "weightsNext.sizes(): " << weightsNext.sizes() << std::endl;
    std::cout << "activationDerivative.sizes(): " << activationDerivative.sizes() << std::endl;


    return torch::matmul(torch::matmul(lossNext.to(torch::kFloat64), weightsNext.to(torch::kFloat64)), activationDerivative.transpose(0, 1).to(torch::kFloat64));
}

static torch::Tensor lossHiddenBiases(torch::Tensor lossNext, torch::Tensor weightsNext, torch::Tensor activationDerivative, bool forBiases){

    // (vezi si cearta-te cu GepeTo)...
}

torch::Tensor lossHidden(torch::Tensor lossNext, torch::Tensor weightsNext, torch::Tensor activationDerivative, bool forBiases){

    if (forBiases){
        return lossHiddenBiases(lossNext, weightsNext, activationDerivative); // 
    } else {
        return lossHiddenWeights(lossNext, weightsNext, activationDerivative);
    }

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