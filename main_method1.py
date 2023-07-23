#Importing necessary packages
import matplotlib.pyplot as plt
import torch
import torchvision as vision
from torchvision import transforms
%matplotlib inline
import matplotlib.pyplot as plt
import Classifier1
import Classifier2
import Classifier3
import Classifier4
import Classifier5
import Classifier6
from CommonFunctions import get_accuracy

#Defining path for image dataset
data_path = "/content/drive/MyDrive/archive/Data/images_original"

#Defining transformation for the images
transformation = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),  #Converting images to tensor
    transforms.Resize((180,180))  #Resizing the tensor shape to (180,180)
])
data =vision.datasets.ImageFolder(root = data_path, transform= transformation)


#Visualising data
data_visu =vision.datasets.ImageFolder(root = data_path, transform= transforms.Resize((180,180))) #Loading dataset

itr=iter(data_visu)
fig=plt.figure(figsize=(10, 10))
fig.tight_layout()
plt.subplots_adjust( wspace=1, hspace=1)
#Iterating over images
for i in range(4):
            img,label=next(itr)
            t=fig.add_subplot(2,2,i+1)
            # set the title of the image equal to its label
            t.set_title(str(label))
            t.axes.get_xaxis().set_visible(False)
            t.axes.get_yaxis().set_visible(False)
            plt.imshow(img,cmap='gray_r')


#Performing train-test-validation split
train_data, val_data, test_data = torch.utils.data.random_split(data,[0.7,0.2,0.1])


BATCH_SIZE = 32 #Setting batch size

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)  #Loading train set
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)  #Loading val set
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)  #Loading test set

#Setting up GPU
device=torch.device('cuda')


model1 = Classifier1.classifier1(50,train_data_loader,val_data_loader)  # Runnning Classifier1 for 50 Epochs
model2 = Classifier1.classifier1(100,train_data_loader,val_data_loader) #Running Classifier 1 for 100 epochs
#Model Description
for c in model1.children():
    print(c)


model3 = Classifier2.classifier2(50,train_data_loader,val_data_loader) # Runnning Classifier2 for 50 Epochs
model4 = Classifier2.classifier2(100,train_data_loader,val_data_loader) # Runnning Classifier2 for 100 Epochs
#Model Description
for c in model3.children():
    print(c)

model5 = Classifier3.classifier3(50,train_data_loader,val_data_loader)  # Runnning Classifier3 for 50 Epochs
model6 = Classifier3.classifier3(100,train_data_loader,val_data_loader) # Runnning Classifier3 for 100 Epochs
#Model Description
for c in model5.children():
    print(c)

model7 = Classifier4.classifier4(50,train_data_loader,val_data_loader)  # Runnning Classifier4 for 50 Epochs
model8 = Classifier4.classifier4(100,train_data_loader,val_data_loader) # Runnning Classifier4 for 100 Epochs

model9 = Classifier5.classifier5(50,train_data_loader,val_data_loader)  # Runnning Classifier5 for 50 Epochs
model10 = Classifier5.classifier5(100,train_data_loader,val_data_loader)    # Runnning Classifier5 for 100 Epochs

model11 = Classifier6.classifier6(50,train_data_loader,val_data_loader) # Runnning Classifier6 for 50 Epochs
model12 = Classifier6.classifier6(100,train_data_loader,val_data_loader)    # Runnning Classifier6 for 100 Epochs

#Getting accuracy of the models on the test dataset
test_results_1 = get_accuracy(test_data_loader,model1,device)
test_results_2 = get_accuracy(test_data_loader,model2,device)
test_results_3 = get_accuracy(test_data_loader,model3,device)
test_results_4 = get_accuracy(test_data_loader,model4,device)
test_results_5 = get_accuracy(test_data_loader,model5,device)
test_results_6 = get_accuracy(test_data_loader,model6,device)
test_results_7 = get_accuracy(test_data_loader,model7,device)
test_results_8 = get_accuracy(test_data_loader,model8,device)
test_results_9 = get_accuracy(test_data_loader,model9,device)
test_results_10 = get_accuracy(test_data_loader,model10,device)
test_results_11 = get_accuracy(test_data_loader,model11,device)
test_results_12 = get_accuracy(test_data_loader,model12,device)

#Dictionary to store results on the test set
results = {
    "Model1": test_results_1,
    "Model2": test_results_2,
    "Model3": test_results_3,
    "Model4": test_results_4,
    "Model5": test_results_5,
    "Model6": test_results_6,
    "Model7": test_results_7,
    "Model8": test_results_8,
    "Model9": test_results_9,
    "Model10": test_results_10,
    "Model11": test_results_11,
    "Model12": test_results_12
}

#Visulaising the results
print(results)