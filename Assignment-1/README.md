# Problem Statement:
The goal of this Assignment is to implement our own Feed-forward neural network, back-propagation code, and use the gradient descents and its variants with back-propagation for classification task and keep track of our experiments using [wandb.ai](https://wandb.ai/).<br /> 
The Assignment can be found [here](https://wandb.ai/miteshk/assignments/reports/Assignment-1--VmlldzozNjk4NDE?accessToken=r7ndsh8lf4wlxyjln7phvvfb8ftvc0n4lyn4tiowdg06hhzpzfzki4jrm28wqh44).

# Prerequisites:

```
Python 3.7.10
Numpy 1.19.5
```
# Dataset:
We have used Fashion-MNIST dataset.

# Installing:
+ Clone/download this repository.
+ For running in google colab, install wandb using following command - ```!pip install wandb```.
+ For running locally, install wandb using following command.
```
pip install wandb
pip install numpy
pip install keras
```

## Question-1:
There are ten classes in the Fashion-MNIST data set and here is a dictionary relating the model's numerical labels and corresponding class names.\
Class_labels_names = 
{       "0": "T-shirt/Top",         "1": "Trouser",
        "2": "Pullover",            "3": "Dress",
        "4": "Coat",                "5": "Sandal",
        "6": "Shirt",               "7": "Sneaker",
        "8": "Bag",                 "9": "Ankle Boot",     }
 
#### Solution Approach:
+ Create an array of available class as ig[].
+ check each image of our input data belongs to which class.
+ Then store that image and remove its class from the available class array.
+ To get the first image which is 0'th from each class, Iterate through all the images.
+ Then plot the images.
+ Integrate wandb to log the images and keep track of the experiment using wandb.ai.

## Question-2:
#### Solution Approach:
+ Feed-forward neural network (Feed_Frwd_Nw1()) has been implemented which takes in the training dataset(xtrain, ytrain), testing dataset(xtest, ytest), weights, biases,           activation function and loss function.
+ Initialize the randomized weights, biases as per the number of inputs, hidden & output layer specification.
+ Implement loss functions such as:
        1.cross entropy
        2. Mean squared error
+ Implement Activation functions such as:
     - sigmoid, tanh, relu...etc
+ our code provides the flexibility in choosing the above mentioned parameters.
+ and provides flexibility in choosing the number of hidden layers and neurons in each hidden layer.


## Question-3:
* Back propagation algorithm implemented with the support of the following optimization function and the code works for any batch size:
    * SGD 
    * Momentum based gradient descent 
    * Nesterov accelerated gradient descent
    * RMS Prop
    * ADAM
    * NADAM

#### Solution Approach:
+ Make use of uotput of the feed-forward neural network in the previous question.
+ Initialize one hot function to encode the labels of images.
* Implement the activation functions and their gradients.
    * sgd
    * softmax
    * Rel
    * tanh
+ Initialize the randomized parameters using the 'random' in python.
+ Initialize predictions, accuracy and loss functions.
+ loss functions are:
    + Mean squared Error
    + Cross entropy
+ Initialize the gradient descent functions.
+ and Initialize the training function to use the above functions.


## Question-4:
#### Solution Approach:
+ Split the training data in the ratio of 9:1.
+ The standard training & test split of fashion_mnist has been used with 60000 training images and 20000 test images & labels.
+ 10% shuffled training data was kept aside as validation data for the hyperparameter search i.e, 2000 images.
+ wandb.sweeps() provides an functionality to comapre the different combinations of the hyperparameters for the training purpose.
+ we are avail with 3 types of search strategies which are:
    + grid
    + random
    + Bayes
+ By considering the number of parameters given, there are totally 11664 combinations are possible.
+ grid : It checks through all the possible combinations of hyperparameters. If there are n hyperparameters and m options of each hyperparameter. 
  There will be m^n number of runs to see the final picture, hence grid search strategy wont work beacause it would be a computationally intensive.
+ random:
+ Bayesian: 
+ There are 2 options left to choose.
+ we chose random search. and we obtained a maximum validation accuracy of 88%. #need to update
+ after picking the sweep function, set the sweep function of wandb by setting up the different parameters in sweep configuration i.e, s_config().
+ By using the code below we can see the results in our wandb project.
+ wandb.agent(sweep_id,train).

## Question-5:
#### Solution Approach:
* The best accuracy across all the models is a validation accuracy of 89% and this is obtained for the following setting of hyperparameters:
         * Learning rate: 0.0055
         * Mode of initialization: Xavier
         * Optimizer: Adam
         * Number of hidden layers:  1
         * Number of neurons in every hidden layer: 32
         * lambda for L2 regularization (weight decay): 0.00005
         * Number of epochs: 15
         * Batch size: 16
         * Activation function: Relu
         * Loss function: ce
 + The graph containing a summary of validation accuracies for all the models is shown in the wandb report.


## Question-6:
#### Solution Approach:
+ Adam optimizer has provided the best val_accuracy in the above experiments, that has also reflected positively in the correlation table. Since the images that we are using       are black and white the input is definitely sparse, all the black pixels are represented with 0. So the idea used in rmsprop and adam to manipulate the learning rate             according to past updates, so that b does not undergo many updates in comparison to w checks out.
+  Small batch size might help to generalize well but may not be able to converge to the global minima. Similarly large batch size may cost us in terms of cost also in terms        generalization.  So we need a batch size that is neither too small or too large this idea also checks out in correlation table with a small positive correlation value for        batch size.
+ 

###### Recommendations to attain 95% accuracy:
+ In our assignment, we are using Fashion-MNIST dataset which contains images and we know that convolutional neural networks are good for datasets containing images in comparison to neural networks. 
+ Using a CNN architecture that involves parameter sharing and local connectivity is bound to give improvements on this image classification task. Thus, using convolutional neural networks we can attain accuracy up to 95%. 

## Question-7:
#### Solution Approach:
+ Get the best model.
* Report the best accuracy.
        * The best model configuration is:
        * learning_rate:
        * epochs: 
         * no_hidden_layer: 
         * size_hidden_layers
         * optimizer: 
         * batch_size:
         * activation: 
         * weight_initializations: 
         * weight_decay: 
         * loss_function:
         
+ Implement a function to calculate confusion matrix.
+ Plot and integrate wandb to keep track using wandb.






## Question-8:
#### Solution Approach:
+ First implement the squared error loss function.
+ then get the ouput of the squared error loss and cross entropy.
+ Integrate the outputs of squared error loss and cross entropy loss to see automatically generated plot on wandb.
+ It can be seen that MSE loss function based run for the best model configuration clearly under performs compared to cross entropy loss function based configuration. 
+ This is attributed to the inherent probabilistic nature of the problem statement which essentially tries to fit a probability distribution corresponding to the image              classification. Hence a probability based loss function such as cross entropy is more suitable than a distance function such as squared error.
+ the accuracy would also depend on hyperparameter configurations as well. So it is believed that a hyper parameter search is again necessitated for a better and informed          comparison to be drawn. 

## Question-10:
#### Solution Approach:
+ Since MNIST is a much simpler dataset, and a very similar image classification task with the same number of classes, the configurations of hyperparameters that worked well        for Fashion-MNIST is expected to work well for MNIST too.
+ Although transfer learning from the pre trained Fashion MNIST dataset's best model configuration for the digits MNIST dataset is an extremely viable option for faster           training and better initialization of the network, in the current implementation of the code, transfer learning has not been used. 
+ 





















