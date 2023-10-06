#Importing necessary packages
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision as vision
from torchvision import transforms
import random
%matplotlib inline
from collections import Counter
import torch.nn as nn
from torch.optim import SGD,Adam
from google.colab import drive
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter
import utils

# Mounting google drive #
# drive.mount('/content/drive', force_remount=False)

#Defining path for image dataset
data_path = "/data/images_original"

#Defining transformation for the images
transformation = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),  #Converting images to tensor
    transforms.Resize((180,180))  #Resizing the tensor shape to (180,180)
])
data =vision.datasets.ImageFolder(root = data_path, transform= transformation)


#Performing train-test-validation split
train_data, val_data, test_data = torch.utils.data.random_split(data,[0.7,0.2,0.1])
BATCH_SIZE = 32 #Setting batch size

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)  #Loading train set
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)  #Loading val set
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)  #Loading test set

#Setting up GPU
device=torch.device('cuda')


model = utils.classifier3(100,train_data_loader,val_data_loader) # Runnning Classifier3 for 100 Epochs

# Save the entire model
torch.save(model, 'model/model.pth')