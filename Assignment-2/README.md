## Splitting folders and saving to drive:
+ Intially, As we were given inature_12k.zip which contains train and test data. we unzipped the file and saved to drive at location /content/drive/MyDrive/inaturalist_12K (folder contains two subfolders i.e. train and val where we used val folders data as test data)
+ we further have had split the train folder into trainable data and validation data. we have kept validation and training data in /content/drive/MyDrive/outputfin.
+ we further have used these locations to load the data.
+ we have provided splitting_folders_and_saving_to_drive.ipynb which performs the above task. 

## Part-A

#Question-1:
Initially we prepared a data i.e. Loading the images as arrays.
+ we load the training data contained in /content/drive/MyDrive/outputfin/train into the two arrays i.e. (x_train, y_train)/(x_train_un, y_train_un).
+ x_train contains all the training images but they are forcefully aguemented and y_train contains labels corresponding to the images.
+ x_train_un contains all the training images but they are not aguemented and y_train_un contains labels corresponding to the images.


+ we load the validation data contained in /content/drive/MyDrive/outputfin/val into the two arrays i.e. (x_val, y_val)/(x_val_un, y_val_un).
+ x_val contains all the validation images but they are forcefully aguemented and y_val contains labels corresponding to the images.
+ x_val_un contains all the validation images but they are not aguemented and y_val_un contains labels corresponding to the images.


+ we load the test data contained in /content/drive/MyDrive/inaturalist_12K/val into the two arrays i.e. (x_test, y_test)/(x_test_un, y_test_un).
+ x_test contains all the testing images but they are forcefully aguemented and y_train contains labels corresponding to the images.
+ x_test_un contains all the testing images but they are not aguemented and y_test_un contains labels corresponding to the images.


## The loading of data is done by the following function:
# function for prepating data takes two parameters
+ 1.drive_path - path to drive i.e training or validataion or test
+ 2.want_aug - Takes 'YES' or 'NO' values to gather info wheter to augment data or not
+ returns images array and labels array
## def prepare_data(drive_path,want_aug):

### To create the covolutional neural network we use the following function:

#this function creates a sequential  model and returns the model
+ The following are the parameters taken
        + 1.fil=filter sizes for each layer(all the 5 convolution layers)
        + 2.ker=kernel dimensions for each layer
        + 3.activ=activation function for the dense layer 
        + 4.batchnormalization=takes 'YES' or 'NO' which tells weather to normalize the data or not
        + 5.dropout=the percentage of dropout scaled from 0 to 1
        + 6.neurons=number of neurons for the dense layer
+ Returns the newely created CNN which can be used to compile and fit the data.
# def create_CNN(fil,ker,activ,batchnormalisation,dropout,neurons):
The data for the parameters of this function is gathered by command line arguments(optional)

## Sweeps:
def conlay(): This is the sweep function which takes in the sweep configuration parameter and creates CNN by calling create_CNN(fil,ker,activ,batchnormalisation,dropout,neurons) and the compiles and fit the model for each of the hyperparamter configuration.

## Question-4:
By taking the best hyperparameter configuration, we create, compile, fit and save the model in google drive for future usage 
 # 4-b:
 Here we have used function def getImages(): 
+ This function is used to accumulate 30 test pics from x_test where for each Image we gather 2 images i.e augmented and deaugmented Image.
+ Function returns a list of lists where each inner list contains augmented and deaugmented Image for a particular image def getImages():


+ later we have used the function to generate the samples and plot them.

## Question-5:
In this question we have initially loaded the saved model from the drive and also prepared the test data. 
later for the guided back-propagation we have used the following function:

## def guidedprop(model,input):
      + this takes input as
      + 1.model- the saved model (best)
      + 2.input-the data on which guided back propagation has to be done
      + Returns input image and gradient image

## Part-B:
After loading teh required data we used the following functions:
# def finetune(type,freez,epoch,optim,batch):
+ this function take the following parameters
       + 1.type - the type of the pretrained model to be used
       +2.freez- which layers to freez (-1 indicates freeizing all layers, 
       + negative number 'n' indicates freezing n-1 layers from the back, and positive number 'p' freezing starting p-1 layers)
       + 3.epochs - number of epochs to fit the model
       + 4.batch - batchsize for training
this creates , trains and returns the model 

# sweep function for hyperparamater tuning:
def hypertune():


## Dataset:
We have used iNaturalist[] dataset.

## Report:
Report for this Assignment can be found [here](https://wandb.ai/pratap49/CS6910-assignment-1/reports/Assignment-1--VmlldzoxNjA0NjQ2).

## Authors:
[pratap](https://github.com/pratap977)\
[ganesh](https://github.com/27-ganesh-07)























