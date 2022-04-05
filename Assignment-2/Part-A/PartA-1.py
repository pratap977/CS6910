import numpy as np
import tensorflow as tf
from tensorflow import keras
import math as mh
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Dense,Flatten
from keras.preprocessing.image import load_img as im
from keras.preprocessing.image import save_img as sim
import os
from random import shuffle
import numpy as np
from PIL import Image
from torchvision import transforms
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers,models
import wandb
import pathlib
import sys
from google.colab import drive
drive.mount('/content/drive')




#gettig labels for different classes and assigning values
class_labels = os.listdir("/content/drive/MyDrive/outputfin/train")
lab={l:i for l,i in zip(class_labels,range(10))}




##############################################################
#function for prepating data takes two parameters
#1.drive_path - path to drive i.e training or validataion or test
#2.want_aug - Takes 'YES' or 'NO' values to gather info wheter to augment data or not
def prepare_data(drive_path,want_aug):
  imgs_tr = []
  truth_tr = []
  #data augmentation rules here
  data_augmentation  = transforms.Compose([
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
  for l in class_labels:
    #### for each folder in the val dir we iterate though all the files 
    if l.startswith("."):
      continue
    imgs_per_class = os.listdir(os.path.join(drive_path, l)) #getting list of file names
    for img in imgs_per_class:
      #iterating to each file in the folder 
      img_path = os.path.join(drive_path, l, img)
      img=Image.open(img_path)
      img = img.resize((224,224))
      if img.mode == 'L':
                  continue
      if(want_aug=='YES'):
        normalized_image = data_augmentation(img)
        imgs_tr.append(np.asarray(normalized_image).transpose(1,2,0))
      else:
        normalized_image = img
        imgs_tr.append(np.asarray(normalized_image))
      truth_tr.append(lab[l])
      
  #contain augmented images(imgs_tr) and labels(truth_tr)
  return np.asarray(imgs_tr) , np.asarray(truth_tr) 



#####################################################################
# Preparing training set (with augmentation), validation set and test test
####################################################################
# Preparing training data with augmentation
# Loading data from directory


############augmented Training data############ 
x_train,y_train=prepare_data("/content/drive/MyDrive/outputfin/train",'YES')


#############augmented validation data############ 
x_val,y_val=prepare_data("/content/drive/MyDrive/outputfin/val",'YES')

# #############augmented test data############
x_test,y_test=prepare_data("/content/drive/MyDrive/inaturalist_12K/val",'YES')





#####################################################################
# Preparing training set (without augmentation), validation set and test test
####################################################################
# Preparing training data withou augmentation
# Loading data from directory


#############augmented Training data############ 
x_train_un,y_train_un=prepare_data("/content/drive/MyDrive/outputfin/train",'NO')


#############augmented validation data############ 
x_val_un,y_val_un=prepare_data("/content/drive/MyDrive/outputfin/val",'NO')

#############augmented test data############
x_test_un,y_test_un=prepare_data("/content/drive/MyDrive/inaturalist_12K/val",'NO')






#this function creates a sequential  model and returns the model
#The following are the parameters taken
#1.fil=filter sizes for each layer(all the 5 convolution layers)
#2.ker=kernel dimensions for each layer
#3.activ=activation function for the dense layer 
#4.batchnormalization=takes 'YES' or 'NO' which tells weather to normalize the data or not
#5.dropout=the percentage of dropout scaled from 0 to 1
#6.neurons=number of neurons for the dense layer


##example to call this function:::
##create_CNN([64,128,256,512,1024],[3,3,3,3,3],'sigmoid','NO',0.5,32)   
def create_CNN(fil,ker,activ,batchnormalisation,dropout,neurons):
  cnn=models.Sequential()
  cnn.add(layers.Conv2D(filters=fil[0], kernel_size=(ker[0], ker[0]), activation='relu',input_shape=(224, 224, 3)))
  cnn.add(layers.MaxPooling2D((2, 2)))
  if(batchnormalisation=='YES'):
    cnn.add(layers.BatchNormalization())
        
        
  cnn.add(layers.Conv2D(filters=fil[1], kernel_size=(ker[1], ker[1]), activation='relu'))
  cnn.add(layers.MaxPooling2D((2, 2)))
  if(batchnormalisation=='YES'):
    cnn.add(layers.BatchNormalization())
                                                  
  cnn.add(layers.Conv2D(filters=fil[2], kernel_size=(ker[2], ker[2]), activation='relu'))
  cnn.add(layers.MaxPooling2D((2, 2)))
  if(batchnormalisation=='YES'):
    cnn.add(layers.BatchNormalization())
         
  cnn.add(layers.Conv2D(filters=fil[3], kernel_size=(ker[3], ker[3]), activation='relu'))
  cnn.add(layers.MaxPooling2D((2, 2)))
  if(batchnormalisation=='YES'):
    cnn.add(layers.BatchNormalization())

  cnn.add(layers.Conv2D(filters=fil[4], kernel_size=(ker[4], ker[4]), activation='relu'))
  cnn.add(layers.MaxPooling2D((2, 2)))
  
  if(batchnormalisation=='YES'):
    cnn.add(layers.BatchNormalization())


  cnn.add(layers.Flatten())
  if(dropout!=0):
    cnn.add(layers.Dropout(dropout))
  cnn.add(layers.Dense(neurons, activation=activ))
  
  cnn.add(layers.Dense(10, activation='softmax'))
  return cnn


if(len(sys.argv)!=16):
    print("Please give vorrect arguments")
    exit(0)
else:
    fil=[]
    for i in range(1,6):
        fil.append(int(sys.argv[i]))
    ker=[]
    for i in range(6,11):
        ker.append(int(sys.argv[i]))
    activ=sys.argv[11]
    batchnormalisation=sys.argv[12]
    dropout=sys.argv[13]
    neurons=sys.argv[14]
    batch_size=sys.argv[15]
    epochs=ephocs[16]

    #example:::::    
    #cnn1=create_CNN([64,128,256,512,1024],[3,3,3,3,3],'sigmoid','NO',0.5,32)
        
    cnn1=create_CNN(fil,ker,activ,batchnormalisation,dropout,neurons)
    cnn1.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    cnn1.fit(x_train,y_train, batch_size=batch_size,epochs=epochs,validation_data=(x_val,y_val))
    tloss,tacc=cnn1.evaluate(x_test,y_test)
    print("Test loss:",tloss,"Test Accuracy:",tacc)





