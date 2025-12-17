#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/transforms/stack.h>
#include <torch/data/transforms/tensor.h>
#include <torch/torch.h>
#include <iostream>
#include "include/data_loaders.hpp"
#include "include/activations.hpp"
#include "include/losses.hpp"
#include "sweep.cpp"
#include "include/network.hpp"

int main(){

    //std::string dataPath = "/home/gabriel/Documents/HolyC/fashion_MNIST/data/";
    std::string dataPath = std::getenv("DATA_PATH"); // full path to the dataset
    int batchSize = 64;
    int numWorkers = 3; 
    auto [trainSet, testSet] = loadMnist(dataPath, batchSize, numWorkers);  

    float learningRate = 1e-5;
    float weightDecay = 1e-3;
    int epochs = 3;
    
    lossType crossEntropy = CROSS_ENTROPY;

    activationType none = NONE;
    activationType relu = RELU;
    activationType softmax = SOFTMAX;

    Loss lossFunction(crossEntropy);
    
    /*
    FeedForwardNetwork network(learningRate, weightDecay, lossFunction, batchSize);

    network.addLayer(784, 128, none); // input -> hidden 1
    network.addLayer(128, 128, relu); // hidden 1 -> hidden 2
    network.addLayer(128, 10, softmax); // hidden 2 -> output layer  

    network.train(*trainSet, epochs);
    */

    /// TODO: mutat load mnist (functia) ca parametru in sweep()
    hyperparameterSweep()
}