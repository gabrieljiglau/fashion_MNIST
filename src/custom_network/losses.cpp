#include "include/losses.hpp"


float Loss::crossEntropy(torch::Tensor activation, torch::Tensor target){
    
    /*
    activation  -> the output from the activation funtion (i.e. softmax)
    targetIndex -> target in one-hot encoding form
    */

    double epsilon = 1e-9;


    // the activations are off ??
    //std::cout << "activation" << activation << std::endl;
    //std::cout << "target" << target << std::endl;
    
    //std::cout << "logits : " << activation << std::endl;
    activation += epsilon;
    torch::Tensor logits = torch::log(activation);
    //std::cout << "logits : " << logits << std::endl;
    torch::Tensor loss = -(target * logits).sum(1).mean();

    return loss.item<float>();
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

    /// the L2 penalty is added in the gradients computation
    if (this->lossFunction == CROSS_ENTROPY){
        return crossEntropy(activation, target);
    }

    // fallback; to check when calling
    return 0;
}

lossType Loss::getLossType(){
    return this->lossFunction;
}
