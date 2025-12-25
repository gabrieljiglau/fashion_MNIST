#include <../libtorch/torch_network.cpp>
#include <ATen/ops/cross_entropy_loss.h>
#include <memory>
#include <torch/nn/modules/loss.h>
#include <../custom_network/include/data_loaders.hpp>
#include <torch/serialize/output-archive.h>
#include <torch/utils.h>

// &data: the address of data
template<typename LoaderType>
void fit(LoaderType &data, int numEpochs, std::shared_ptr<TorchNetwork> network, std::string path, std::string mode){

    if (mode == "validate"){
        torch::NoGradGuard noGrad;
    }

    for(int epoch = 1; epoch <= numEpochs; epoch++){
        std::cout << "Epoch : " << epoch << std::endl;
        
        float loss = 0;
        int batchNumber = 0;

        int correctLabels = 0;
        int totalInstances = 0;

        for(auto &batch: data){

            batchNumber += 1;

            network->optimizer->zero_grad(); //reset gradients

            torch::Tensor xTrain = batch.data;
            xTrain = xTrain.flatten(1);
            totalInstances += xTrain.size(0);

            torch::Tensor prediction = network->forward(xTrain);
            torch::Tensor target = batch.target;

            std::cout << "  prediction.argmax(1).item<int>(); = " <<  prediction.argmax(1).item<int>();
            int label = prediction.argmax(1).item<int>();

            if (label == batch.target.template item<int>()){
                correctLabels += 1;
            }

            torch::Tensor currentLoss = network->loss.forward(prediction, batch.target);
            loss += currentLoss.item<float>();

            if (mode == "train"){
                currentLoss.backward();
                network->optimizer->step();
            }
        }

        std::cout << "Loss = " << float(loss / batchNumber) << std::endl;
        std::cout << "Prediction accuracy " << float(correctLabels / totalInstances) << std::endl;
    }
    
    torch::serialize::OutputArchive archive;

    network->layers[0]->save(archive);
    network->layers[1]->save(archive);
    network->layers[2]->save(archive);
    
    archive.save_to(path);
}

int main(){

    std::string dataPath = std::getenv("DATA_PATH"); // full path to the dataset

    float learningRate = 0.001;
    int batchSize = 64;
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

    network->setLossFunction(torch::nn::AnyModule(torch::nn::CrossEntropyLoss()));

    int numEpochs = 50;
    std::string savingPath = "/home/gabriel/Documents/HolyC/fashion_MNIST/models/libtorch.pt";

    fit(*trainSet, numEpochs, network, savingPath, "train");

}
