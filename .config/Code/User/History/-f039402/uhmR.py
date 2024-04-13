# torch and torchvision imports
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

model = allcnn_t().to(device).to(torch.float16)
epochs = 100
criterion = nn.CrossEntropyLoss()
lr = 1e-5
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

# The training loop

# model = net.to(device)
total_step = len(trainloader)
plot = False
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [40, 80, epochs-20], gamma = 0.1, verbose=True)
overall_step = 0
train_loss_values = []
train_error = []
val_loss_values = []
val_error = []
lrs = []
lr_finder_iter_lim = 125

for i, (images, labels) in enumerate(trainloader):
    # Move tensors to configured device
    images = images.to(device).to(torch.float16)
    labels = labels.to(device)
    #Forward Pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_loss_values.append(loss.item())
    lrs.append(lr)
    lr *= 1.1
    for op_params in optimizer.param_groups:
            op_params['lr'] = lr
    print(f"{i+1}/100")
    if i == lr_finder_iter_lim:
        break

lrs = np.array(lrs); train_loss_values = np.array(train_loss_values)
fig = plt.figure()
plt.plot(lrs, train_loss_values)
# plt.plot(lrs[1:-1], np.convolve(train_loss_values, np.ones(5)/5, mode='same')[1:-1])
plt.plot(lrs[np.argmin(train_loss_values)], train_loss_values.min(), '.r')
plt.xlabel('Learning Rate')
plt.ylabel('Training loss on the batch')
plt.xlim(1e-5/1.1, 1e-5*1.1**(lr_finder_iter_lim+1))
plt.savefig('Learning rate finder curve.png')
plt.show()


lr_max = lrs[np.argmin(np.array(train_loss_values))]/10
del model, optimizer

model = allcnn_t().to(device).to(torch.float16)
epochs = 100
criterion = nn.CrossEntropyLoss()
lr = 1e-4
t = 0

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)
model_name = 'cnn_curve'
total_step = len(trainloader)
t0 = total_step//20
T = total_step*epochs
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [40, 80, epochs-20], gamma = 0.1, verbose=True)
overall_step = 0
train_loss_values = []
train_error = []
val_loss_values = []
val_error = []
train_lrs=[]
for epoch in range(epochs):
    correct = 0
    total = 0
    flag = 0
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):
        # Move tensors to configured device
        images = images.to(device).to(torch.float16)
        labels = labels.to(device)
        #Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        t += 1
        if t<=t0:
            lr = 1e-4 + (t/t0)*lr_max
        else:
            lr = lr_max*np.cos((np.pi/2)*(t-t0)/(T-t0))
        
        for op_params in optimizer.param_groups:
            op_params['lr'] = lr

        optimizer.step()
        train_loss_values.append(running_loss)
        train_lrs.append(lr)
        train_error.append(100-100*correct/total)
        
        if (i+1) % 30 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))
        if plot:
            info = { ('loss_' + model_name): loss.item() }

            # for tag, value in info.items():
            #   logger.scalar_summary(tag, value, overall_step+1)
    

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(testloader):
            images = images.to(device).to(torch.float16)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
    val_error.append(100-100*correct/total)
    val_loss_values.append(running_loss)


print("Done")



