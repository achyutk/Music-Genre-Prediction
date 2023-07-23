#Importing necessary packages
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import torch
from CommonFunctions import visualise_training
from CommonFunctions import get_accuracy

input_size=3*180*180
hidden_size1=128
hidden_size2=256
output_size=10

device=torch.device('cuda')


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
  

#Training 3rd network i.e Convolution Neural Netork of above arhitecture with Batch Normalistion layers
def classifier3(epochs,train_data_loader,val_data_loader):

  model=Net3().to(device) #Creating an object of the model
  optimizer=Adam(model.parameters())  # Setting the optimiser
  loss_fn=nn.CrossEntropyLoss() # Setting the loss function

  #List to store accuracy of model after each epoch
  train_acc=[]
  val_acc = []

  #Training the model
  for epoch in range(epochs):
      loop=tqdm(train_data_loader)
      loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
      epoch_loss=0.
      for (imgs,labels) in loop:
          optimizer.zero_grad()
          imgs=imgs.to(device)
          labels=labels.to(device)
          outputs=model(imgs)
          loss=loss_fn(outputs,labels)
          loss.backward()
          optimizer.step()

          #Calculating Running Loss
          epoch_loss=0.9*epoch_loss+0.1*loss.item()
          loop.set_postfix(loss=epoch_loss)

      model.eval()

      #Getting model accuracy for train and validation
      t_acc=get_accuracy(train_data_loader,model,device)
      v_acc=get_accuracy(val_data_loader,model,device)

      #Appending the accuracies to the list
      train_acc.append(t_acc)
      val_acc.append(v_acc)
      model.train()

      print("Epoch {},loss {:.4f}, Accuracy {}, ValAccuracy {}".format(epoch+1,epoch_loss,t_acc,v_acc))


  visualise_training(epochs,train_acc,val_acc)

  return model