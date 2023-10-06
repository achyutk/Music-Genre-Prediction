#Importing necessary packages
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import models
import torch
import torch.nn as nn
import torch.optim as optim,Adam
from tqdm import tqdm
%matplotlib inline



input_size=3*180*180
hidden_size1=128
hidden_size2=256
output_size=10

device=torch.device('cuda')


#Function to plot training and validatioin accuracy for each epoch
def visualise_training(epoch,train_acc,val_acc):
  x = [x+1 for x in range(epoch)]

  try:
    ta = [x.detach().cpu().numpy() for x in train_acc]
    plt.plot(x, ta, label ='Training accuracy')
  except:
    plt.plot(x, train_acc, label ='Training accuracy')

  plt.plot(x, val_acc, '-.', label ='Validation accuracy')

  plt.xlabel("X-axis data")
  plt.ylabel("Y-axis data")
  plt.legend()
  plt.title('Accuracy in each epochs')
  plt.show()


#Function to calculate accuracy of the model
def get_accuracy(dataloader,model,device):

  total=len(dataloader.dataset)
  correct=0

  for data in dataloader:
    imgs,labels=data
    imgs=imgs.to(device)
    labels=labels.to(device)
    outputs=model(imgs)
  # the second return value is the index of the max i.e. argmax

    _,predicted=torch.max(outputs.data,1)
    correct+=(predicted==labels).sum()


  return (correct/total).item()


#Training 1st netwrok FeedForward Neural Netork
def classifier1(epochs,train_data_loader,val_data_loader):
  model=models.Net(input_size,output_size)

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


#Training 2nd network i.e Convolution Neural Netork
def classifier2(epochs,train_data_loader,val_data_loader):

  model=models.Net2().to(device) #Creating an object of the model
  optimizer=Adam(model.parameters())   # Setting the optimiser
  loss_fn=nn.CrossEntropyLoss()  # Setting the loss function

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


#Training 3rd network i.e Convolution Neural Netork of above arhitecture with Batch Normalistion layers
def classifier3(epochs,train_data_loader,val_data_loader):

  model=models.Net3().to(device) #Creating an object of the model
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


#Training 1st FeedForward Neural Netork with RMSOptimiser
def classifier4(epochs,train_data_loader,val_data_loader):
  model=models.Net(input_size,output_size)

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


#Training 1st Convolution Neural Netork with RMSOptimiser
def classifier5(epochs,train_data_loader,val_data_loader):

  model=models.Net2().to(device) #Creating an object of the model
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


#Training 2nd Convolution Neural Netork with RMSOptimiser
def classifier6(epochs,train_data_loader,val_data_loader):

  model=models.Net3().to(device) #Creating an object of the model
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