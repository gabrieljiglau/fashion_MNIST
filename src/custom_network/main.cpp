#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/transforms/stack.h>
#include <torch/data/transforms/tensor.h>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <iostream>
#include "include/activations.hpp"
#include "include/data_loaders.hpp"


int main(){

    //std::string dataPath = "/home/gabriel/Documents/HolyC/fashion_MNIST/data/";
    std::string dataPath = std::getenv("DATA_PATH"); // full path to the dataset
    int batchSize = 64;
    int numWorkers = 3; 
    //auto [trainSet, testSet] = loadMnist(dataPath, batchSize, numWorkers);
    Eigen::MatrixXd A(3, 3);
    A << 1, 2, 3,
          2, 4, 6,
          0, 0, 1;

    
    std::cout << ActivationFunction::softmax(A) << std::endl;
    //std::cout << A.rowwise() / A.rowwise().maxCoeff().transpose().array << std::endl;
}