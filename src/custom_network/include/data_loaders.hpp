#pragma once
#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/torch.h>


static inline auto loadMnistTrainSet(const std::string &dataPath, const int batchSize, const int numWorkers){

    auto trainSet = torch::data::datasets::MNIST(
                    dataPath,
                    torch::data::datasets::MNIST::Mode::kTrain)
                    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                    .map(torch::data::transforms::Stack<>());

    return torch::data::make_data_loader(
        std::move(trainSet),
        torch::data::DataLoaderOptions()
        .batch_size(batchSize)
        .workers(numWorkers)); 
}


static inline auto loadMnistTestSet(const std::string &dataPath, const int batchSize, const int numWorkers){
    
    auto testSet = torch::data::datasets::MNIST(
                    dataPath,
                    torch::data::datasets::MNIST::Mode::kTest)
                    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                    .map(torch::data::transforms::Stack<>());

    return torch::data::make_data_loader(
        std::move(testSet),
        torch::data::DataLoaderOptions()
        .batch_size(batchSize)
        .workers(numWorkers));
}


inline auto loadMnist(const std::string &dataPath, const int batchSize, const int numWorkers){

    auto trainSet = loadMnistTrainSet(dataPath, batchSize, numWorkers);
    auto testSet = loadMnistTestSet(dataPath, batchSize, numWorkers);

    return std::make_tuple(std::move(trainSet), std::move(testSet));
}

