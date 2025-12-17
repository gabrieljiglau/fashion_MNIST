#include "include/network_builder.hpp"
#include "include/losses.hpp"
#include "include/activations.hpp"
#include <random>
#include <fstream>


template<typename T>
bool contains(std::vector<T> &vector, T &toFind){
    return std::find(vector.begin(), vector.end(), toFind) != vector.end();
}

static std::vector<FeedForwardNetwork> networkPermutations(int noInputs, int noOutputs, std::array<float, 4> learningRate, 
                                                             std::array<float, 3> weightDecay, std::array<int, 4> batchSize, std::array<int, 3> numHidden, 
                                                             Loss lossFunction, std::array<activationType, 3> activations, float percentage){
    
    // use randomized search (percentage * the complete search-space)
    int searchSpace = learningRate.size() * weightDecay.size() * batchSize.size() * numHidden.size() * numHidden.size();
    int usedSearchSpace = int(percentage * searchSpace);

    // avoid reevaluations
    std::vector<std::array<int, 5>> usedPermutations;

    // choose uniformly for each hyperparameter
    std::random_device rd;
    std::mt19937_64 seed(rd());
    std::uniform_int_distribution<> distribution1(0, 3); // for the hyperparameters that hold 4 values
    std::uniform_int_distribution<> distribution2(0, 2); // for those that hold 3 values

    std::vector<FeedForwardNetwork> networksConfigurations;
    NetworkBuilder builder;
    while (usedPermutations.size() < usedSearchSpace){

        std::array<int, 5> currentPermutation;
        for (int i = 0; i < 2; i++){
            currentPermutation[i] = distribution1(seed);
        }

        for (int i = 0; i < 3; i++){
            currentPermutation[i + 2] = distribution2(seed);
        }

        if (!contains(usedPermutations, currentPermutation)){
            float lr = learningRate[currentPermutation[0]];
            float bs = batchSize[currentPermutation[1]];
            float wd = weightDecay[currentPermutation[2]];
            int numNeurons1 = numHidden[currentPermutation[3]];
            int numNeurons2 = numHidden[currentPermutation[4]];
            
            FeedForwardNetwork network = builder
                                        .setInputs(noInputs)
                                        .setNumHidden(std::array<int, 2> {numNeurons1, numNeurons2})
                                        .setOutputs(noOutputs)
                                        .setLearningRate(lr)
                                        .setWeightDecay(wd)
                                        .setBatchSize(bs)
                                        .setActivations(activations)
                                        .setLossFunction(lossFunction)
                                        .build();

            networksConfigurations.push_back(network);
            usedPermutations.push_back(currentPermutation);
        }
    }

    return networksConfigurations;
}


/// TODO: utilizat aici function templates, pt ca altfel nu poti sa dai ca parametru 'batch size'
template<typename LoaderType>
void hyperparameterSweep(LoaderType &trainSet, std::array<float, 4> learningRate, std::array<float, 3> weightDecay,
                                         std::array<int, 4> batchSize, std::array<int, 3> numHidden, 
                                         std::array<activationType, 3> activationFunctions, Loss lossFunction,
                                         float percentage, int noInputs, int noOutputs, int epochs=10, std::string path="../models/hyperparams_custom.csv"){
    
    std::vector<FeedForwardNetwork> networkConfigs = networkPermutations(noInputs, noOutputs, learningRate, weightDecay, batchSize,
                                                                  numHidden, lossFunction, activationFunctions, percentage);
    
    int bestIdx = 0;
    float bestPrecision = 0;
    
    for (int i = 0; i < networkConfigs.size(); i++){
        auto& network = networkConfigs[i];
        auto [precision, weights, biases] = network.train(*trainSet, epochs);
        if (precision > bestPrecision){
            bestPrecision = precision;
            bestIdx = i;
        }
    }
    

    std::ofstream bestPath(path);
    if (!bestPath) {
        std::cout << "Couldn't open " << path << std::endl;
        return;
    }

    auto& bestNetwork = networkConfigs[bestIdx];

    bestPath << bestNetwork.getLearningRate() << "," << bestNetwork.getHiddenSizes() << bestNetwork.getMiniBatchSize() << ",";
    bestPath << bestNetwork.getWeightDecay();
}