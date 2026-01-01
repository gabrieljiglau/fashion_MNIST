#include <../libtorch/torch_network.cpp>
#include <ATen/ops/cross_entropy_loss.h>
#include <memory>
#include <torch/nn/modules/loss.h>
#include <torch/serialize/output-archive.h>
#include <torch/utils.h>
#include <../data_loaders.cpp>

int main(){

    // "/home/gabriel/Documents/HolyC/fashion_MNIST/data/";
    std::string dataPath = std::getenv("DATA_PATH"); // full path to the dataset

    float learningRate = 0.01;
    int batchSize = 16;
    int numWorkers = 3;

    int numInputs = 784;
    std::array<int, 2> numHidden{128, 128};
    int numOutputs = 10;

    auto [trainSet, testSet] = loadMnist(dataPath, batchSize, numWorkers); 


    std::shared_ptr<TorchNetwork> network = std::make_shared<TorchNetwork>(numInputs, numHidden, numOutputs, batchSize);
    network->setOptimizer(std::make_unique<torch::optim::SGD>(
                        network->parameters(), 
                        torch::optim::SGDOptions(learningRate))
                    );

    network->setLossFunction(torch::nn::AnyModule(torch::nn::NLLLoss()));

    int numEpochs = 50;
    std::string savingPath = "/home/gabriel/Documents/HolyC/fashion_MNIST/models/libtorch.pt";

    // train
    //fit(*trainSet, numEpochs, network, savingPath, "train");
    

    // test
    fit(*testSet, numEpochs, network, savingPath, "validate");

}
