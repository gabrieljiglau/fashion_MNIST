#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/transforms/stack.h>
#include <torch/data/transforms/tensor.h>
#include <torch/torch.h>
#include <iostream>
#include <optional>
#include "include/data_loaders.hpp"
#include "include/activations.hpp"
#include "include/losses.hpp"
#include "include/utils.hpp"
#include "include/network.hpp"

std::array<float, 4> hyperparameterSweep(std::array<float, 4> learningRate, std::array<float, 3> weightDecay,
                                         std::array<int, 4> batchSize, std::array<int, 3> numHidden, 
                                         std::array<activationType, 3> activationFunctions, Loss lossFunction,
                                         float percentage, int noInputs, int noOutputs, int epochs){
    
    std::vector<std::array<int, 5>> permutations = assignPermutations(learningRate, weightDecay, batchSize, numHidden, percentage);
    
    /// TODO: reparat aici cum dai ca argument 'permutation' si ce anume pastrezi in ele .. (lr bs wd numNeurons1  numNeurons2 )

    for (auto permutation : permutations){
        FeedForwardNetwork network = FeedForwardNetwork::buildStandardNetwork(noInputs, noOutputs, activationFunctions, lossFunction, permutation);
    }
    // nu inteleg de ce nu am voie sa fac aceasta declaratie aici
    FeedForwardNetwork network = buildStandardNetwork(noInputs, std::array<int, 2> numHidden, int noOutputs,
        float learningRate, float batchSize, float weightDecay,
        std::array<activationType, 3> activationFunctions, const Loss lossFun);
}

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
    FeedForwardNetwork network(learningRate, weightDecay, lossFunction, batchSize);

    network.addLayer(784, 128, none); // input -> hidden 1
    network.addLayer(128, 128, relu); // hidden 1 -> hidden 2
    network.addLayer(128, 10, softmax); // hidden 2 -> output layer  

    network.train(*trainSet, epochs);

    /// TODO: add hyperparameter sweep using random search (si de pus in utils)
    // trebuie pentru asta sa creeze retele neuronale in mod dinamic, sa rulezi fiecare configuratie, si sa salvezi parametrii 'buni'

    
}