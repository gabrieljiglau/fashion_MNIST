#include "include/utils.hpp"
#include "include/network_builder.hpp"
#include <memory>
#include <random>

torch::Tensor lossWeights(torch::Tensor lossNext, torch::Tensor weightsNext, torch::Tensor activationDerivative){

    return torch::matmul(lossNext.to(torch::kFloat64), weightsNext.to(torch::kFloat64).transpose(0, 1)) * activationDerivative.to(torch::kFloat64);
}

torch::Tensor lossBiases(torch::Tensor losses){

    // before: shape [batch, num_biases], after [num_biases]
    return losses.sum(0);
}


torch::Tensor oneHotEncode(torch::Tensor tensor, int length){

    torch::Tensor oneHot = torch::zeros({tensor.size(0), length});

    for (int i = 0; i < tensor.size(0); i++){
        int target = tensor[i].item<int>();
        oneHot.index_put_({i, target}, 1); 
    }
    
    return oneHot;
}

int checkPredictions(torch::Tensor softmaxOutput, torch::Tensor groundTruth){

    /*
    return the number of correctly labeled examples 
    */

    torch::Tensor predictions = torch::argmax(softmaxOutput, 1);
    assert(predictions.sizes() == groundTruth.sizes()); // and they should be [batch_size]
    
    int correctPredictions = 0;
    for (int i = 0; i < predictions.size(0); i++){
        if (predictions[i].item<int>() == groundTruth[i].item<int>()){
            correctPredictions += 1;
        }
    }

    return correctPredictions;
}

template<typename T>
bool contains(std::vector<T> &vector, T &toFind){
    return std::find(vector.begin(), vector.end(), toFind) != vector.end();
}

std::vector<std::unique_ptr<FeedForwardNetwork>> networkSweep(int noInputs, int noOutputs, std::array<float, 4> learningRate, 
                                                             std::array<float, 3> weightDecay, std::array<int, 4> batchSize, std::array<int, 3> numHidden, 
                                                             Loss lossFunction, std::array<activationType, 3> activations, float percentage){
    // use randomized search (percentage * the complete search-space)
    int searchSpace = learningRate.size() * weightDecay.size() * batchSize.size() * numHidden.size() * numHidden.size();
    int usedSearchSpace = int(percentage * searchSpace);

    // avoid reevaluations
    std::vector<std::array<int, 5>> usedPermutations;

    // choose uniformly from each hyperparameter
    std::random_device rd;
    std::mt19937_64 seed(rd());
    std::uniform_int_distribution<> distribution1(0, 3); // for the hyperparameters that hold 4 values
    std::uniform_int_distribution<> distribution2(0, 2); // for those that hold 3 values

    std::vector<std::unique_ptr<FeedForwardNetwork>> networksConfigurations;
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
            
            std::unique_ptr<FeedForwardNetwork> network = std::make_unique<FeedForwardNetwork>(
                                         builder
                                        .setInputs(noInputs)
                                        .setNumHidden(std::array<int, 2> {numNeurons1, numNeurons2})
                                        .setOutputs(noOutputs)
                                        .setLearningRate(lr)
                                        .setWeightDecay(wd)
                                        .setBatchSize(bs)
                                        .setActivations(activations)
                                        .setLossFunction(lossFunction)
                                        .build());

            networksConfigurations.push_back(network);
            usedPermutations.push_back(currentPermutation);
        }
    }

    return networksConfigurations;
}