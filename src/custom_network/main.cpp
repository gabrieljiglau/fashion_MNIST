#include <torch/data/transforms/tensor.h>
#include <torch/torch.h>
#include <iostream>


int main(){

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    
    std::string dataPath = "~/gabriel/data/";
    auto trainSet = torch::data::datasets::MNIST(
                        dataPath,
                        torch::data::datasets::MNIST::Mode::kTrain)
                        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                        .map(torch::data::transforms::Stack<>()); 
    
}