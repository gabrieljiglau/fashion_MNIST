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
    int epochs = 2;
    
    lossType crossEntropy = CROSS_ENTROPY;
    activationType relu = RELU;

    Loss lossFunction(crossEntropy);
    FeedForwardNetwork network(learningRate, weightDecay, lossFunction, batchSize);


    /// TODO: modul cum construiesc straturile si adaug functiile de activare este suspect (trebuie modificat)
    network.addLayer(784, 128); // input -> hidden 1
    network.addLayer(128, 128, relu); // hidden 1 -> hidden 2
    network.addLayer(128, 10, relu); // hidden 2 -> output layer  

    network.train(*trainSet, epochs);

    
}