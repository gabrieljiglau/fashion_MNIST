#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/transforms/stack.h>
#include <torch/data/transforms/tensor.h>
#include <torch/torch.h>
#include "../data_loaders.cpp"
#include "../custom_network/include/activations.hpp"
#include "../custom_network/include/losses.hpp"
#include "../custom_network/sweep.cpp"
#include "../custom_network/include/network.hpp"


int main(){

    // "/home/gabriel/Documents/HolyC/fashion_MNIST/data/";
    std::string dataPath = std::getenv("DATA_PATH"); // full path to the dataset

    // "/home/gabriel/Documents/HolyC/fashion_MNIST/models/"
    std::string dir = std::getenv("MODEL_DIR");
    std::string weightsPath = dir + "vanilla_network/weights.pt";
    std::string biasesPath = dir + "vanilla_network/biases.pt";

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
    FeedForwardNetwork network(0.01, 0.001, lossFunction, 16);

    network.addLayer(784, 128, relu); // input -> hidden 1
    network.addLayer(128, 128, relu); // hidden 1 -> hidden 2
    network.addLayer(128, 10, softmax); // hidden 2 -> output layer  

    auto [trainSet, testSet] = loadMnist(dataPath, 16, numWorkers); 

    /*
    // training
    auto [accuracy, weights, biases] = network.fit(*trainSet, "train", 50);
    torch::save(weights, weightsPath);
    torch::save(biases, biasesPath);
    */
    // testing

    std::vector<torch::Tensor> weights;
    std::vector<torch::Tensor> biases;

    torch::load(weights, weightsPath);
    torch::load(biases, biasesPath);

    network.setWeights(weights);
    network.setBiases(biases);
    network.fit(*testSet, "validate");
}