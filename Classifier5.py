#Importing necessary packages
import torch.nn as nn
import torch
from CommonFunctions import visualise_training
from CommonFunctions import get_accuracy
import Classifier2

input_size=3*180*180
hidden_size1=128
hidden_size2=256
output_size=10

device=torch.device('cuda')

#Training 1st Convolution Neural Netork with RMSOptimiser
def classifier5(epochs,train_data_loader,val_data_loader):

  model=Classifier2.Net2().to(device) #Creating an object of the model
  optimizer=torch.optim.RMSprop(model.parameters(),lr=0.001)  # Setting the optimiser
  loss_fn=nn.CrossEntropyLoss() #Setting loss function

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