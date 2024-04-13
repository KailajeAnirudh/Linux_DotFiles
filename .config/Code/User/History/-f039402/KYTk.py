import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

transform = transforms.Compose(
    [transforms.RandAugment(), transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Defining the model
class View(nn.Module):
    def __init__(self,o):
        super().__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)
    
class allcnn_t(nn.Module):
    def __init__(self, c1=96, c2= 192):
        super().__init__()
        d = 0.5

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                nn.ReLU(True),
                nn.BatchNorm2d(co))

        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(3,c1,3,1,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            nn.Dropout(d),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            nn.Dropout(d),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,10,1,1),
            nn.AvgPool2d(8),
            View(10))

        print('Num parameters: ', sum([p.numel() for p in self.m.parameters()]))

    def forward(self, x):
        return self.m(x)
    
# The training loop
train_loss_values = [] 
train_error_values = [] 
test_loss_values = [] 
test_error_values = [] 
learning_rates = [] 
epochs = 10
t = 0
steps_per_epoch = len(trainloader)
T_max = steps_per_epoch*epochs
T_0 = T_max/5 
lr_max = 0.0088 
model = allcnn_t()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr_max, momentum=0.9, weight_decay=0.001)
for epoch in range(epochs): 
  for i, data in enumerate(trainloader, 0): 
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      if t <= T_0:
        lr = 10**(-4) + (t/T_0)*lr_max  
      else: 
        lr = lr_max*np.cos((np.pi/2)*((t-T_0)/(T_max-T_0))) + 10**(-6) 

      for g in optimizer.param_groups:
        g['lr'] = lr 
      learning_rates.append(lr)
      optimizer.step()
      t+=1
      loss_t = loss.item()
      correct = 0
      total = 0.  
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      acc_tr_t =  100 * correct / total     
      train_loss_values.append(loss_t) 
      train_error_values.append(100-acc_tr_t)
      if i%(steps_per_epoch//10) == 0:
         print(f"Epoch: {(epoch+1):2d}/{epochs}, Step: {i+1}/{steps_per_epoch}, Train Loss: {loss_t:.3f}, Train Error: {acc_tr_t:.3f}")

  average_test_loss = 0
  average_test_accuracy = 0
  for i, data in enumerate(testloader, 0):
    correct = 0
    total = 0. 
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    average_test_accuracy +=  100 * correct / total   
    loss = criterion(outputs, labels)
    average_test_loss += loss.item()
  average_test_accuracy /= len(testloader)
  average_test_loss /= len(testloader)
  test_error_values.append(100-average_test_accuracy)    
  test_loss_values.append(average_test_loss)
  print(f"_______________________________________\nEpoch: {(epoch+1):2d}/{epochs}, Step: {i+1}/{steps_per_epoch}, Test Loss: {average_test_loss:.3f}, Test Error: {average_test_accuracy:.3f}\n_______________________________________\n")

  fig = plt.figure()
  plt.plot(np.arange(len(train_loss_values))+1, train_loss_values)
  plt.xlabel("Weight Updates")
  plt.ylabel("Training loss for the batch")
  plt.savefig(f'/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/02_Homework/04_Homework4/src/report/Training_Loss_{epoch}.png')

  fig = plt.figure()
  plt.plot(np.arange(len(train_error_values))+1, train_error_values)
  plt.xlabel("Weight Updates")
  plt.ylabel("Training error for the batch")
  plt.savefig(f'/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/02_Homework/04_Homework4/src/report/Training_Error_{epoch}.png')

  fig = plt.figure()
  plt.plot(np.arange(len(test_loss_values))+1, test_loss_values)
  plt.xlabel("Weight Updates")
  plt.ylabel("Test loss for the batch")
  plt.savefig(f'/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/02_Homework/04_Homework4/src/report/Testing_Loss_{epoch}.png')

  fig = plt.figure()
  plt.plot(np.arange(len(test_error_values))+1, test_error_values)
  plt.xlabel("Weight Updates")
  plt.ylabel("Testing error for the batch")
  plt.savefig(f'/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/02_Homework/04_Homework4/src/report/Training_Loss_{epoch}.png')
  
  
  