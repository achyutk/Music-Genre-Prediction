#Importing necessary packages
import torch.nn as nn
import torch.optim as optim
import torch
from CommonFunctions import visualise_training
from CommonFunctions import get_accuracy

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
    


#Training 1st netwrok FeedForward Neural Netork
def classifier1(epochs,train_data_loader,val_data_loader):
  model=Net(input_size,output_size)

  model=model.cuda()

  # Setting the optimiser
  optimizer=optim.Adam(model.parameters(),lr=3e-4)

  #Setting loss function
  loss_fn=nn.CrossEntropyLoss()
  running_loss=0.0

  #List to store accuracy of model after each epoch
  train_acc=[]
  val_acc = []

  #Training the model
  for epoch in range(epochs):
      correct = 0
      for imgs,labels in train_data_loader:
          imgs=imgs.cuda()
          labels=labels.cuda()
          output=model(imgs)
          loss=loss_fn(output,labels)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          #Calculating Running Loss
          running_loss=0.99*running_loss+0.01*loss.item()
          # _,predicted=torch.max(output.data,1)
          # correct+=(predicted==labels).sum()

      model.eval()
      #Getting model accuracy for train and validation
      train_accuracy = get_accuracy(train_data_loader,model,device)
      val_accuracy = get_accuracy(val_data_loader,model,device)
      model.train()

      #Appending the accuracies to the list
      train_acc.append(train_accuracy)
      val_acc.append(val_accuracy)
      print("Epoch {},loss {:.4f}, Accuracy {}, ValAccuracy {}".format(epoch+1,running_loss,train_accuracy,val_accuracy))

  print()
  #Visualising the accuracy
  visualise_training(epochs,train_acc,val_acc)

  return model
