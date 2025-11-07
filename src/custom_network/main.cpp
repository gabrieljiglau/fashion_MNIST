#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/transforms/stack.h>
#include <torch/data/transforms/tensor.h>
#include "include/data_loaders.hpp"
#include <torch/torch.h>
#include <iostream>


int main(){

    //std::string dataPath = "/home/gabriel/Documents/HolyC/fashion_MNIST/data/";
    std::string dataPath = std::getenv("DATA_PATH"); // full path to the dataset
    int batchSize = 64;
    int numWorkers = 3; 
    auto [trainSet, testSet] = loadMnist(dataPath, batchSize, numWorkers);
    
}