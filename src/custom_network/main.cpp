#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/transforms/stack.h>
#include <torch/data/transforms/tensor.h>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <iostream>
#include <optional>
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
    
    lossType crossEntropy = CROSS_ENTROPY;
    activationType relu = RELU;

    Loss lossFunction(crossEntropy);
    FeedForwardNetwork network(learningRate, weightDecay, lossFunction, batchSize);


    network.addLayer(784, 128); // input layer: no activation, it just passes the input
    network.addLayer(128, 128, relu); // hidden 1
    network.addLayer(128, 128, relu); // hidden 2
    network.addLayer(128, 10); // output layer

    // de folosit doar torch::Tensor, fara Eigen::Matrix, fiindca e idiot sa faci asta

    for (auto &batch: *trainSet){
        
        torch::Tensor x = batch.data; // [batch_size, no_RGB_channels, img_height, img_width]
        x = x.to(torch::kFloat64).flatten(1); // [batch_size, img_height X img_width]

        torch::Tensor y = batch.target;
        y = y.to(torch::kFloat64);
        y = oneHotEncode(y, 10); // of shape [batch_size, target_dim=10]
    }

    
}