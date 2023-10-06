#Importing necessary packages
import torch.nn as nn
import torch.optim as optim
import torch

input_size=3*180*180
hidden_size1=128
hidden_size2=256
output_size=10

device=torch.device('cuda')

class Net(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()

        self.relu=nn.ReLU()
        self.layer1=nn.Linear(in_features,hidden_size1)
        self.layer2=nn.Linear(hidden_size1,hidden_size2)
        self.layer3=nn.Linear(hidden_size2,out_features)

    def forward(self,x):
        x=x.flatten(start_dim=1)
        x=self.layer1(x)
        x=self.relu(x)
        x=self.layer2(x)
        x=self.relu(x)
        x=self.layer3(x)

        return x
    

#Defining the network
class Net2(nn.Module):
  def __init__(self):
    super().__init__()
    self.relu=nn.ReLU()
    self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3)# input  ---> (*,3,180,180) Output  ---> (*,32,178,178)
    self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5)# input  ---> (*,32,178,178) Output ---> (*,64,174,174)
    self.pool1=nn.MaxPool2d(kernel_size=(2,2)) # input  ---> (*,64,174,174) Output  ---> (*,64,87,87)

    self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3) # input ---> (*,64,87,87) Output ---> (*,128,85,85)
    self.conv4=nn.Conv2d(in_channels=128,out_channels=32,kernel_size=5) # input ---> (*,128,85,85) Output ---> (*,32,81,81)
    self.pool2=nn.MaxPool2d(kernel_size=(2,2)) # input ---> (*,32,81,81) output ---> (*,32,40,40)

    self.flatten=nn.Flatten() #input --->(*,32,40,40) output---> (*,51200)
    self.fc1=nn.Linear(in_features=32*40*40,out_features=256) # input ---> (*,51200) output ---> (*,256)
    self.fc2=nn.Linear(in_features=256,out_features=10) #input ---> (*,256) output ---> (*,10)

  def forward(self,x):
    x=self.conv1(x)
    x=self.relu(x)
    x=self.conv2(x)
    x=self.relu(x)
    x=self.pool1(x)

    x=self.conv3(x)
    x=self.relu(x)
    x=self.conv4(x)
    x=self.relu(x)
    x=self.pool2(x)

    x=self.flatten(x)
    x=self.fc1(x)
    x=self.relu(x)
    x=self.fc2(x)
    return x
  

#Defining the network
class Net3(nn.Module):
  def __init__(self):
    super().__init__()
    self.relu=nn.ReLU()
    self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3)# input  ---> (*,3,180,180) Output  ---> (*,32,178,178)
    self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5)# input  ---> (*,32,178,178) Output ---> (*,64,174,174)
    self.pool1 = nn.MaxPool2d(kernel_size=(2,2)) # input  ---> (*,64,174,174) Output  ---> (*,64,87,87)

    self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3) # input ---> (*,64,87,87) Output ---> (*,128,85,85)
    self.conv4 = nn.Conv2d(in_channels=128,out_channels=32,kernel_size=5) # input ---> (*,128,85,85) Output ---> (*,32,81,81)
    self.pool2 = nn.MaxPool2d(kernel_size=(2,2)) # input ---> (*,32,81,81) output ---> (*,32,40,40)

    self.flatten = nn.Flatten() #input --->(*,32,40,40) output---> (*,51200)
    self.fc1 = nn.Linear(in_features=32*40*40,out_features=256) # input ---> (*,51200) output ---> (*,256)
    self.fc2 = nn.Linear(in_features=256,out_features=10) #input ---> (*,256) output ---> (*,10)

    self.batchnorm1 = nn.BatchNorm2d(32)
    self.batchnorm2 = nn.BatchNorm2d(64)
    self.batchnorm3 = nn.BatchNorm2d(128)
    self.batchnorm4 = nn.BatchNorm2d(32)

  def forward(self,x):
    x=self.conv1(x)
    x=self.relu(x)
    x= self.batchnorm1(x)
    x=self.conv2(x)
    x=self.relu(x)
    x= self.batchnorm2(x)
    x=self.pool1(x)

    x=self.conv3(x)
    x=self.relu(x)
    x= self.batchnorm3(x)
    x=self.conv4(x)
    x=self.relu(x)
    x= self.batchnorm4(x)
    x=self.pool2(x)

    x=self.flatten(x)
    x=self.fc1(x)
    x=self.relu(x)
    x=self.fc2(x)
    return x
