#include <array>
#include <memory>
#include <torch/nn/modules/linear.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/sgd.h>
#include <torch/torch.h>

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
        xBatch = torch::softmax(this->layers[2]->forward(xBatch), 1);

        return xBatch;
    }

    void setOptimizer(std::unique_ptr<torch::optim::Optimizer> opt){
        this->optimizer = std::move(opt);
    }

    void setLossFunction(torch::nn::AnyModule loss){
        this->loss = std::move(loss);
    }

};