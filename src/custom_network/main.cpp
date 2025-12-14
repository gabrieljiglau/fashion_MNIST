#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/transforms/stack.h>
#include <torch/data/transforms/tensor.h>
#include <torch/torch.h>
#include <iostream>
#include <optional>
#include <fstream>
#include "include/data_loaders.hpp"
#include "include/activations.hpp"
#include "include/losses.hpp"
#include "include/utils.hpp"
#include "include/network.hpp"

template<typename LoaderType>
std::array<float, 4> hyperparameterSweep(LoaderType &trainSet, std::array<float, 4> learningRate, std::array<float, 3> weightDecay,
                                         std::array<int, 4> batchSize, std::array<int, 3> numHidden, 
                                         std::array<activationType, 3> activationFunctions, Loss lossFunction,
                                         float percentage, int noInputs, int noOutputs, int epochs=10, std::string path="../models/hyperparams_custom.txt"){
    
    std::vector<std::unique_ptr<FeedForwardNetwork>> networkConfigs = networkSweep(noInputs, noOutputs, learningRate, weightDecay, batchSize,
                                                                  numHidden, lossFunction, activationFunctions, percentage);
    
    int bestIdx = 0;
    float bestPrecision = 0;
    
    for (int i = 0; i < networkConfigs.size(); i++){
        auto& network = networkConfigs[i];
        auto [precision, weights, biases] = network->train(*trainSet, epochs);
        if (precision > bestPrecision){
            bestPrecision = precision;
            bestIdx = i;
        }
    }
    

    std::ofstream bestConfig(path);
    if (!bestConfig) {
        std::cout << "Couldn't open " << path;
    }

    auto& bestNetwork = networkConfigs[bestIdx];

    // uita-te in network.hpp
    bestConfig << "learningRate: " << bestNetwork.


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
    
    /*
    FeedForwardNetwork network(learningRate, weightDecay, lossFunction, batchSize);

    network.addLayer(784, 128, none); // input -> hidden 1
    network.addLayer(128, 128, relu); // hidden 1 -> hidden 2
    network.addLayer(128, 10, softmax); // hidden 2 -> output layer  

    network.train(*trainSet, epochs);
    */

    /// TODO: add hyperparameter sweep using random search (si de pus in utils)
    // trebuie pentru asta sa creeze retele neuronale in mod dinamic, sa rulezi fiecare configuratie, si sa salvezi parametrii 'buni'

    
}