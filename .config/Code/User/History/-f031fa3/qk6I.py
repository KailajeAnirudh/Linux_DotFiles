import torch
import torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import pickle

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.m = nn.Sequential(
            nn.Linear(1,200), 
            nn.ReLU(),
            nn.BatchNorm1d(200),
            # nn.Dropout(0.001),
            nn.Linear(200, 200), 
            nn.ReLU(),
            nn.BatchNorm1d(200),
            # nn.Dropout(0.001),
            nn.Linear(200,1)
            )

        print('Num parameters: ', sum([p.numel() for p in self.m.parameters()]))

    def forward(self, x):
        return self.m(x)
    
def f_star(x):
    return torch.sin(10*torch.pi*x**4)

def del_fin(n, model):
    model.eval()
    with torch.no_grad():
        x = torch.linspace(0,1, n).reshape(-1,1)
        pred_y = model(x)
        y = f_star(x)
    return torch.abs(y-pred_y).max()

def del_fout(n, model):
    model.eval()
    with torch.no_grad():
        x = torch.linspace(0,1.5,n).reshape(-1,1)
        pred_y = model(x)
        y = f_star(x)
    return torch.abs(y-pred_y).max()

def train(n):
    x_train = torch.rand((n,1))
    # x_train = torch.linspace(0,1, 1000).reshape(-1,1)
    y_train = f_star(x_train).reshape(-1,1)
    criterion = nn.MSELoss()
    lr_max = 3e-3
    lr = 1e-4
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr = lr, weight_decay=1e-3, nesterov=True, momentum=0.9)
    lrs = []
    train_loss = []
    T_max = 2500
    T_0  = T_max/5
    for t in range(T_max):
        lrs.append(lr)
        if t <= T_0:
            lr = 10**(-4) + (t/T_0)*lr_max  
        else: 
            lr = lr_max*np.cos((np.pi/2)*((t-T_0)/(T_max-T_0))) + 10**(-6)
        ypred = model(x_train)
        optimizer.zero_grad()
        loss = criterion(ypred, y_train)
        loss.backward()
        optimizer.step()
        # lr *= 1.05
        train_loss.append(loss.item())
        for g in optimizer.param_groups:
            g['lr'] = lr

    return train_loss, del_fin(1000, model), del_fout(1000, model)

ns = np.logspace(1, 3, 20).astype(int)
trials = 5
del_ins = np.zeros((len(ns), trials))
del_outs = np.zeros_like(del_ins)
# del_in_summary = np.zeros((len(ns), 2))
# del_out_summary = np.zeros((len(ns), 2))
train_losses = np.zeros_like(del_ins)

for i, n in tqdm(enumerate(ns)):
    for j in range(trials):
        train_loss, fin, fout = train(n)
        train_losses[i,j] = train_loss[-1]
        del_ins[i,j] = fin
        del_outs[i, j] = fout

with open('dels.pickle', 'wb') as f:
    pickle.dump((train_losses, del_ins, del_outs), f)

