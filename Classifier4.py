#Importing necessary packages
import torch.nn as nn
import torch
from CommonFunctions import visualise_training
from CommonFunctions import get_accuracy
import Classifier1

input_size=3*180*180
hidden_size1=128
hidden_size2=256
output_size=10

device=torch.device('cuda')

#Training 1st FeedForward Neural Netork with RMSOptimiser
def classifier4(epochs,train_data_loader,val_data_loader):
  model=Classifier1.Net(input_size,output_size)

  model=model.cuda()   #Creating an object of the model

  # Setting the optimiser
  optimizer=torch.optim.RMSprop(model.parameters(),lr=0.001)

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


