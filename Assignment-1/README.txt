#question-1:
There are ten classes in the Fashion-MNIST data set and here is a dictionary relating the model's numerical labels and corresponding class names. 
Class_labels_names = 
{       "0": "T-shirt/Top",          "1": "Trouser",
         "2": "Pullover",            "3": "Dress",
        "4": "Coat",                 "5": "Sandal",
        "6": "Shirt",                "7": "Sneaker",
        "8": "Bag",                  "9": "Ankle Boot",     }
Solution Approach:
1. Create an array of available class as ig[].
2. check each image of our input data belongs to which class.
3. Then store that image and remove its class from the available class array.
4. To get the first image which is 0'th from each class, Iterate through all the images.
5. Then plot the images.
6. Integrate wandb to log the images and keep track of the experiment using wandb.ai.
.................................................................................................................................................................................
Question-2:
Solution Approach:
1.Feed-forward neural network (Feed_Frwd_Nw1()) has been implemented which takes in the training dataset(xtrain, ytrain), testing dataset(xtest, ytest), weights, biases, activation function and loss function.
2. Initialize the randomized weights, biases as per the number of inputs, hidden & output layer specification.
3. Implement loss functions such as:
      i. cross entropy
      ii. Mean squared error
4. Implement Activation functions such as:
     i. sigmoid, tanh, relu...etc
5. our code provides the flexibility in choosing the above mentioned parameters.
6. and provides flexibility in choosing the number of hidden layers and neurons i each hidden layer.

.................................................................................................................................................................................


Question-3:

Back propagation algorithm implemented with the support of the following optimization function and the code works for any batch size:
    i.  SGD
    ii. Momentum based gradient descent
    iii.Nesterov accelerated gradient descent
    iv. RMS Prop
    v.  ADAM
    vi. NADAM

Solution Approach:
1. Make use of uotput of the feed-forward neural network in the previous question.
2. Initialize one hot function to encode the labels of images.
3. Implement the activation functions and their gradients.
    i. sgd
    ii. softmax
   iii. Relu
    iv. tanh
4. Initialize the randomized parameters using the 'random' in python.
5. Initialize predictions, accuracy and loss functions.
6. loss functions are:
    i. Mean squared Error
    ii. Cross entropy
7. Initialize the gradient descent functions.
8. and Initialize the training function to use the above functions.

.................................................................................................................................................................................

Question-4:
Solution Approach:
1. Split the training data in the ratio of 9:1.
2. The standard training & test split of fashion_mnist has been used with 60000 training images and 20000 test images & labels.
3. 10% shuffled training data was kept aside as validation data for the hyperparameter search i.e, 2000 images.
4. wandb.sweeps() provides an functionality to comapre the different combinations of the hyperparameters for the training purpose.
5. we are avail with 3 types of search strategies which are:
    i. grid
    ii. random
    iii. Bayes
6. By considering the number of parameters given, there are totally 11664 combinations are possible.
7. grid : It checks through all the possible combinations of hyperparameters. If there are n hyperparameters and m options of each hyperparameter. 
  There will be m^n number of runs to see the final picture, hence grid search strategy wont work beacause it would be a computationally intensive.
8. random:
9. Bayesian: 
10. There are 2 options left to choose.
11. we chose random search. and we obtained a maximum validation accuracy of 88%. #need to update
12. after picking the sweep function, set the sweep function of wandb by setting up the different parameters in sweep configuration i.e, s_config().
13. By using the code below we can see the results in our wandb project.
14. wandb.agent(sweep_id,train).
.................................................................................................................................................................................

Question-5:
Solution Approach:
The best accuracy across all the models is a validation accuracy of 89% and this is obtained for the following setting of hyperparameters:
         1. Learning rate: 0.0055
         2. Mode of initialization: Xavier
         3. Optimizer: Adam
         4. Number of hidden layers:  1
         5. Number of neurons in every hidden layer: 32
         6. lambda for L2 regularization (weight decay): 0.00005
         7. Number of epochs: 15
         8. Batch size: 16
         9. Activation function: Relu
         10. Loss function: ce
 The graph containing a summary of validation accuracies for all the models is shown in the wandb report.

.................................................................................................................................................................................

Question-6:
Solution Approach:
1. Adam optimizer has provided the best val_accuracy in the above experiments, that has also reflected positively in the correlation table. Since the images that we are using      are black and white the input is definitely sparse, all the black pixels are represented with 0. So the idea used in rmsprop and adam to manipulate the learning rate            according to past updates, so that b does not undergo many updates in comparison to w checks out.
2. Small batch size might help to generalize well but may not be able to converge to the global minima. Similarly large batch size may cost us in terms of cost also in terms       generalization.  So we need a batch size that is neither too small or too large this idea also checks out in correlation table with a small positive correlation value for       batch size.
3. 

Recommendations to attain 95% accuracy:
In our assignment, we are using Fashion-MNIST dataset which contains images and we know that convolutional neural networks are good for datasets containing images in comparison to neural networks. Using a CNN architecture that involves parameter sharing and local connectivity is bound to give improvements on this image classification task. Thus, using convolutional neural networks we can attain accuracy up to 95%. 
.................................................................................................................................................................................

Question-7:
Solution Approach:
1. Get the best model.
2. Report the best accuracy.
3. The best model configuration is:
         learning_rate: 
         epochs: 
         no_hidden_layer: 
         size_hidden_layers
         optimizer: 
         batch_size:
         activation: 
         weight_initializations: 
         weight_decay: 
         loss_function:
         
4. Implement a function to calculate confusion matrix.
5. Plot and integrate wandb to keep track using wandb.





.................................................................................................................................................................................

Question-8:
Solution Approach:
1. First implement the squared error loss function.
2. then get the ouput of the squared error loss and cross entropy.
3. Integrate the outputs of squared error loss and cross entropy loss to see automatically generated plot on wandb.
4. It can be seen that MSE loss function based run for the best model configuration clearly under performs compared to cross entropy loss function based configuration. 
5. This is attributed to the inherent probabilistic nature of the problem statement which essentially tries to fit a probability distribution corresponding to the image            classification. Hence a probability based loss function such as cross entropy is more suitable than a distance function such as squared error.
6. the accuracy would also depend on hyperparameter configurations as well. So it is believed that a hyper parameter search is again necessitated for a better and informed          comparison to be drawn. 
.................................................................................................................................................................................

Question-10:
Solution Approach:
1. Since MNIST is a much simpler dataset, and a very similar image classification task with the same number of classes, the configurations of hyperparameters that worked well      for Fashion-MNIST is expected to work well for MNIST too.
2. Although transfer learning from the pre trained Fashion MNIST dataset's best model configuration for the digits MNIST dataset is an extremely viable option for faster           training and better initialization of the network, in the current implementation of the code, transfer learning has not been used. 
3. 





















