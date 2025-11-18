#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/transforms/stack.h>
#include <torch/data/transforms/tensor.h>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <iostream>
#include "include/data_loaders.hpp"
#include "include/activations.hpp"
#include "include/losses.hpp"
#include "include/utils.hpp"
#include "include/network.hpp"

int main(){

    //std::string dataPath = "/home/gabriel/Documents/HolyC/fashion_MNIST/data/";
    std::string dataPath = std::getenv("DATA_PATH"); // full path to the dataset
    int batchSize = 64;
    int numWorkers = 3; 
    auto [trainSet, testSet] = loadMnist(dataPath, batchSize, numWorkers);

    float learningRate = 1e-5;
    float weightDecay = 1e-3;
    lossType loss = CROSS_ENTROPY;
    Loss lossFunction(loss);
    FeedForwardNetwork network(learningRate, weightDecay, lossFunction, batchSize);

    // de adaugat aici straturile

    // itetaring through the data loaders
    for (auto &batch: *trainSet){
        torch::Tensor x = batch.data;
        torch::Tensor y = batch.target;

        Eigen::MatrixXd X = torchToEigen(x);
        Eigen::MatrixXd Y = torchToEigen(y);
    }

    
}