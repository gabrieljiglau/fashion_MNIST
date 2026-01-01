#include <array>
#include <memory>
#include <torch/nn/modules/linear.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/sgd.h>
#include <torch/torch.h>
#include <../custom_network/include/utils.hpp>

class TorchNetwork: public torch::nn::Module{

    public:
    std::array<torch::nn::Linear, 3> layers{nullptr, nullptr, nullptr};
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    torch::nn::AnyModule loss;
    int batchSize;

    TorchNetwork(int noInputs, std::array<int, 2> numHidden, int noOutputs, int batchSize){
        this->layers[0] = register_module("fc1", torch::nn::Linear(noInputs, numHidden[0]));
        this->layers[1] = register_module("fc2", torch::nn::Linear(numHidden[0], numHidden[1]));
        this->layers[2] = register_module("fc3", torch::nn::Linear(numHidden[1], noOutputs));
        this->batchSize = batchSize;
    }

    torch::Tensor forward(torch::Tensor xBatch){

        // shape(xBatch) = [batch_size, noFeatures/numOutputNeurons];
        xBatch = torch::relu(this->layers[0]->forward(xBatch));
        xBatch = torch::relu(this->layers[1]->forward(xBatch));
        xBatch = torch::log_softmax(this->layers[2]->forward(xBatch), 1);

        return xBatch;
    }

    void setOptimizer(std::unique_ptr<torch::optim::Optimizer> opt){
        this->optimizer = std::move(opt);
    }

    void setLossFunction(torch::nn::AnyModule loss){
        this->loss = std::move(loss);
    }

    // &data: the address of data
    template<typename LoaderType>
    void fit(LoaderType &data, int numEpochs, std::shared_ptr<TorchNetwork> network, std::string path, std::string mode){

        /// TODO: modifica functia asta sa fie parte din TorchNetwork, 
        //        fara sa mai fie nevoie sa dai ca parametru std::shared_ptr<TorchNetwork> network

        if (mode == "validate"){
            torch::NoGradGuard noGrad;
            numEpochs = 1;

            // the same manner in which they were saved
            torch::load(network, path);
        }

        for(int epoch = 1; epoch <= numEpochs; epoch++){
            std::cout << "Epoch : " << epoch << std::endl;
            
            float loss = 0;
            int batchNumber = 0;

            int correctLabels = 0;
            int totalInstances = 0;

            for(auto &batch: data){

                batchNumber += 1;

                if (mode == "train"){
                    network->optimizer->zero_grad(); //reset gradients   
                }

                torch::Tensor xTrain = batch.data;
                xTrain = xTrain.flatten(1);
                totalInstances += xTrain.size(0);

                torch::Tensor prediction = network->forward(xTrain);
                torch::Tensor target = batch.target;

                correctLabels += checkPredictions(prediction, target);
                /* equivalent of
                prediction = torch::argmax(prediction, 1);
                correctLabels += prediction.eq(batch.target).sum().item<int>();
                */

                torch::Tensor currentLoss = network->loss.forward(prediction, target);
                loss += currentLoss.item<float>();

                if (mode == "train"){
                    currentLoss.backward();
                    network->optimizer->step();
                }
            }

            std::cout << "Loss = " << float(loss / batchNumber) << std::endl;
            std::cout << "Prediction accuracy " << (float(correctLabels) / totalInstances) * 100 << "%" << std::endl;
        }
        
        if (mode == "train"){
            torch::save(network, path);
        }
    }

};
