#include <cmath>
#include "include/losses.hpp"


float Loss::crossEntropy(torch::Tensor activation, torch::Tensor target){
    
    /*
    activation  -> the output from the activation funtion (i.e. softmax, relu)
    targetIndex -> target in one-hot encoding form
    */

    int targetIndex = 0;
    for (int i = 0; i < target.size(0); i++){
        if (target[i].item<int>() == 1){
            targetIndex = i;
            break;
        }
    }

    // aici s-ar putea sa fie probleme, fiindca activarea este o matrice/un tensor 2D
    return -std::log10(activation[targetIndex].item<float>());
}

float Loss::mse(torch::Tensor activation, torch::Tensor target){

    torch::Tensor loss = activation - target;
    loss *= loss;
    loss /= 2;


    return loss.sum().item<float>();
}

float Loss::totalLoss(torch::Tensor activation, torch::Tensor target){

    if (this->lossFunction == MSE){
        return mse(activation, target);
    }

    /// TODO: here you should add the L2 penalty (add the squared sum of coefficients to the loss function) 
    if (this->lossFunction == CROSS_ENTROPY){
        return crossEntropy(activation, target);
    }

    // fallback; to check when calling
    return 0;
}

lossType Loss::getLossType(){
    return this->lossFunction;
}
