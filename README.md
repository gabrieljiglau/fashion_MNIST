# Classification task on the fashion MNIST dataset, using feedforward neural networks

## Custom network

Supports creation of feedforward neural networks with arbitray number of hidden layers.
The backward and forward passes were built around torch::Tensor. 
It was tested, so far, on classification problems (cross-entropy as the loss metric).

Defining the activation and loss types:

```
    activationType none = NONE;
    activationType relu = RELU;
    activationType softmax = SOFTMAX;

    lossType crossEntropy = CROSS_ENTROPY;
```

Defining the network:

```
    // learningRate, weightDecay, lossType, batchSize
    FeedForwardNetwork network(0.1, 0.001, lossFunction, 16);

    network.addLayer(784, 128, none); // input -> hidden 1
    network.addLayer(128, 128, relu); // hidden 1 -> hidden 2
    network.addLayer(128, 10, softmax); // hidden 2 -> output layer  
```

Training and validating
```
    auto [accuracy, weights, biases] = network.fit(*trainSet, "train", epochs);
    
    network.fit(*testSet, "validate");
```


It also includes a function for selecting the best hyperparameters, using random search: ```hyperparameterSweep(..)```.


Results:
```
training: Prediction accuracy 88.6661%
testing: Prediction accuracy 86.58%
```

In order to run the project, from root:
```
export DATA_PATH=fullPath/To/fashion_MNIST/data/
cmake --build fashion_MNIST
./build/fashion_MNIST
```

## Libtorch

The same task, but this time is uses predefined functions (i.e. autodifferentiation) from the libtorch API.

Results: 
```
training: to add
testing: to add
```
