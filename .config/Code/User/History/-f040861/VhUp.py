import numpy as np
import os
import torch, torchvision as thv
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import datetime
import logging
from tqdm import tqdm

runstart = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if not 'src' in os.getcwd():
    for (root, dir, files) in os.walk(os.getcwd()):
        if 'src' in dir:
            os.chdir(os.path.join(root, 'src'))
            break

folders = ['./logs', './report', './logs/TorchNeuralNetwork']
for folder in folders:
    if not os.path.exists(folder):
        os.mkdir(folder)

logging.basicConfig(filename=f'./logs/TorchNeuralNetwork/{runstart}_TorchNeuralLog.log', level=logging.DEBUG, format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s')
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
logging.getLogger("svm").setLevel(logging.DEBUG)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.debug(f'Using {device} device.')

logging.debug('Loading MNIST dataset...')
train = thv.datasets.MNIST(root='./', train=True, download=True)
test = thv.datasets.MNIST(root='./', train=False, download=True)

# Filter data and targets for digits 0 and 1
mask_train = (train.targets == 0) | (train.targets == 1)
mask_test = (test.targets == 0) | (test.targets == 1)

data_train = train.data[mask_train].numpy().reshape(-1, 28 * 28)
target_train = train.targets[mask_train].numpy()

data_test = test.data[mask_test].numpy().reshape(-1, 28 * 28)
target_test = test.targets[mask_test].numpy()

train = torch.utils.data.TensorDataset(torch.from_numpy(data_train).float(), torch.from_numpy(target_train).long())
test = torch.utils.data.TensorDataset(torch.from_numpy(data_test).float(), torch.from_numpy(target_test).long())

loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
valloader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)