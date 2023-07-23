#Importing necessary libraries
import numpy as np
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import tensorflow
import FeatureExtraction
import model


#Mounting google Drive
# drive.mount('/content/drive', force_remount=False) 

#Hyperparameter for feature extraction
hop_length = 256 #the default spacing between frames
n_fft = 128 #number of samples 
mfcc= 20    #Change it to 128 for more number of features


"""--------------------------------------------------------------------------------------------------------------------------------------------------------------"""
#Origial Dataset

FeatureExtraction.explore('genres_original')  #Exploring the dataset

# Feature Extraction of Augmented Audio
features_main,lable_main = FeatureExtraction.get_features('3sec',mfcc,n_fft = 128,hop_length = 256)
lable_main = FeatureExtraction.factorized_lable(lable_main)
features_main = FeatureExtraction.normalisation(features_main)

# Split twice to get the validation set
X_train, X_test, y_train, y_test = train_test_split(features_main, lable_main, test_size=0.1, random_state=123, stratify=lable_main)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

#Print the shapes of the dataset
print(X_train.shape, X_test.shape, X_val.shape, len(y_train), len(y_test), len(y_val))

model = model.lstmmodel() #Defining Model
model,history_model = model.training(X_train, y_train, X_val, y_val,model)  #Training the model

#Visualising the model results
FeatureExtraction.graph(history_model)  
TrainLoss, Trainacc = model.evaluate(X_train,y_train)   #Calculating loss and accuracy on training set
TestLoss, Testacc = model.evaluate(X_test, y_test)  #Calculating loss and accuracy on testing set
y_pred=model.predict(X_test)
print('Confusion_matrix: ',tensorflow.math.confusion_matrix(y_test, np.argmax(y_pred,axis=1)))



"""--------------------------------------------------------------------------------------------------------------------------------------------------------------"""
#Augmented Dataset (Generated from GANS)

FeatureExtraction.explore('Augmented')  #Exploring the dataset

# Feature Extraction of Augmented Audio
feature_augmented,lable_augmented = FeatureExtraction.get_features('Aug3sec')
lable_augmented = FeatureExtraction.factorized_lable(lable_augmented)
feature_augmented = FeatureExtraction.normalisation(feature_augmented)

# Train-Val Split for augmented Data. The train set is taken from the previous set.
X_train_aug, X_val_aug, y_train_aug, y_val_aug = train_test_split(feature_augmented,lable_augmented, test_size=0.2, random_state=123, stratify=lable_augmented)
X_train_aug.shape, X_val_aug.shape, len(y_train_aug), len(y_val_aug)

model = model.lstmmodel() #Defining Model
model_aug,history_model_aug = model.training(X_train_aug,y_train_aug, X_val_aug, y_val_aug,model)   #Training the model

#Visualising the model results
FeatureExtraction.graph(history_model_aug)
TrainLoss, Trainacc = model_aug.evaluate(X_train_aug,y_train_aug)   #Calculating loss and accuracy on training set
TestLoss, Testacc = model_aug.evaluate(X_test, y_test)  #Calculating loss and accuracy on testing set
y_pred=model.predict(X_test)
print('Confusion_matrix: ',tensorflow.math.confusion_matrix(y_test, np.argmax(y_pred,axis=1)))