question-1:
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


Question-5:
Solution Approach:


Question-6:
Solution Approach:


















