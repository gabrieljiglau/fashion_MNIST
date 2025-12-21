#include <array>
#include <memory>
#include <torch/nn/modules/linear.h>
#include <torch/optim/sgd.h>
#include <torch/torch.h>

class TorchNetwork: torch::nn::Module{

    public:
    std::array<std::shared_ptr<torch::nn::LinearImpl> ,3> layers;
    torch::optim::SGD optimizer; // ma latra

    TorchNetwork(int noInputs, std::array<int, 2> numHidden, int noOutputs){
        this->layers[0] = register_module("fc1", torch::nn::Linear(noInputs, numHidden[0]));
        this->layers[1] = register_module("fc2", torch::nn::Linear(numHidden[0], numHidden[1]));
        this->layers[2] = register_module("fc3", torch::nn::Linear(numHidden[1], noOutputs));
    }

    torch::Tensor forward(torch::Tensor xBatch){

        // shape(xBatch) = [batch_size, noFeatures/numOutputNeurons];

        xBatch = torch::relu(this->layers[0]->forward(xBatch));
        xBatch = torch::relu(this->layers[1]->forward(xBatch));
        xBatch = torch::softmax(this->layers[2]->forward(xBatch), 1);

        return xBatch;

    }
};