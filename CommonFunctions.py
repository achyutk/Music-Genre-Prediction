#Importing necessary packages
import matplotlib.pyplot as plt
import torch
%matplotlib inline
import matplotlib.pyplot as plt


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
