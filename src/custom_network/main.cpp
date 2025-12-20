#include <fstream>
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

    activationType none = NONE;
    activationType relu = RELU;
    activationType softmax = SOFTMAX;

    lossType crossEntropy = CROSS_ENTROPY;
    Loss lossFunction(crossEntropy);
    int numWorkers = 3;
    
    // run sweep
    /*
    std::array<float, 4> learningRate{0.1, 0.01, 0.001, 0.0001};
    std::array<float, 3> weightDecay{0.1, 0.01, 0.001};
    std::array<int, 4> batchSize{16, 32, 64, 128};
    std::array<int, 4> numHidden{32, 64, 128, 256};
    std::array<activationType, 3> activations{none, relu, softmax};

    int epochs = 10;
    float searchPercentage = 0.2;

    int noInputs = 784;
    int noOutputs = 10;

    std::string path="/home/gabriel/Documents/HolyC/fashion_MNIST/models/hyperparams_custom.csv";
    
    hyperparameterSweep(loadMnistTestSet, dataPath, numWorkers, learningRate, weightDecay, batchSize, numHidden, activations,
                        lossFunction, searchPercentage, noInputs, noOutputs, epochs, path);
    */
    
    // the parameters found from the sweep
    FeedForwardNetwork network(0.1, 0.001, lossFunction, 16);

    network.addLayer(784, 128, none); // input -> hidden 1
    network.addLayer(128, 128, relu); // hidden 1 -> hidden 2
    network.addLayer(128, 10, softmax); // hidden 2 -> output layer  

    auto [trainSet, testSet] = loadMnist(dataPath, 128, numWorkers); 

    // training
    /*
    auto [accuracy, weights, biases] = network.train(*trainSet, "test", 50);

    torch::save(weights, "/home/gabriel/Documents/HolyC/fashion_MNIST/models/weights.pt");
    torch::save(biases, "/home/gabriel/Documents/HolyC/fashion_MNIST/models/biases.pt");
    */

    // testing

    std::vector<torch::Tensor> weights;
    std::vector<torch::Tensor> biases;

    torch::load(weights, "/home/gabriel/Documents/HolyC/fashion_MNIST/models/weights.pt");
    torch::load(biases, "/home/gabriel/Documents/HolyC/fashion_MNIST/models/biases.pt");

    network.setWeights(weights);
    network.setBiases(biases);
    network.fit(*testSet, "validate");
    
}