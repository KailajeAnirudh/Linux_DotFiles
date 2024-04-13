import torch, torch.nn as nn, torch.optim as optim
from torchmetrics.classification import MultilabelAUROC
import numpy as np,  matplotlib.pyplot as plt, pandas as pd, pickle
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from ResnetModel import *
writer = SummaryWriter()
# from google.colab import drive
# drive.mount('/gdrive')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

X_train = torch.from_numpy(np.transpose(np.load('src/X_train.npy'), axes = (0,2,1))).float()
X_test = torch.from_numpy(np.transpose(np.load('src/X_test.npy'), axes = (0,2,1))).float()
y_train = pd.read_pickle('src/y_train.pickle').to_numpy()
y_test = pd.read_pickle('src/y_test.pickle').to_numpy()

class_list = []
for classes in y_train:
    class_list += classes 
class_list = set(class_list)
class_list
diagSupclassDict = {val:i for i, val in enumerate(class_list)}
diagSupclassDict['Nodiag'] = 5
print(diagSupclassDict)

train_label_mapping = torch.zeros((X_train.shape[0], len(diagSupclassDict)))
print(f"-"*(1+30+5+35))
for i, classes in enumerate(y_train):
    for diagclass in classes:
        train_label_mapping[i, diagSupclassDict[diagclass]] = 1
    if len(classes) == 0:
        train_label_mapping[i, diagSupclassDict['Nodiag']] = 1
    
    print(f"|  {str(y_train[i]):>30}  |  {str(train_label_mapping[i]):<30}   |")

test_label_mapping = torch.zeros((X_test.shape[0], len(diagSupclassDict)))
print(f"-"*(1+30+5+35))
for i, classes in enumerate(y_test):
    for diagclass in classes:
        test_label_mapping[i, diagSupclassDict[diagclass]] = 1
    if len(classes) == 0:
        test_label_mapping[i, diagSupclassDict['Nodiag']] = 1
    
    print(f"|  {str(y_train[i]):>30}  |  {str(test_label_mapping[i]):<30}   |")

train_dataset = torch.utils.data.TensorDataset(X_train, train_label_mapping)
test_dataset = torch.utils.data.TensorDataset(X_test, test_label_mapping)
x = X_train[0:1]
print(x.shape)

class PositionalEncoding(nn.Module):
    def __init__(self, max_len=1000, emb_size=12):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-np.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Transformer(nn.Transformer):
    def __init__(self, emb_size=12, nhead=6, depth=6, hidden_size=128, seq_length=1000, num_classes=5):
        super(Transformer, self).__init__(d_model=emb_size, nhead=nhead, num_encoder_layers=depth, num_decoder_layers=depth, dim_feedforward=hidden_size)
    
        self.pos_encoder = PositionalEncoding(seq_length, emb_size)
        self.decoder = nn.Linear(emb_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear1 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, num_classes)
        
    def __forward_impl(self, x):
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.decoder(x)
        """The output is (seq_len, batch_size)"""
        x = torch.relu(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x
    
    def forward(self, x):
        return self.__forward_impl(x)

"""Transformer needs X input as (seq_len, batch_size, channels)"""
model = Transformer(nhead=6, hidden_size=512, depth=3).to(device)
# resnetModel = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=6).to(device)

# resnetModel(X_train[0:1].to(device))
print("Forward pass")
X_train = X_train.transpose(0,1).transpose(0,2)
model(X_train[:, :17, :].to(device))

