import numpy as np, os
from itertools import product
import torch, torchvision as thv, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt, imageio.v2 as imageio, datetime, logging
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

data_train = train.data[mask_train].numpy().reshape(-1, 28 * 28)/255
target_train = train.targets[mask_train].numpy()
target_train[target_train == 1] = -1; target_train[target_train == 0] = 1

data_test = test.data[mask_test].numpy().reshape(-1, 28 * 28)/255
target_test = test.targets[mask_test].numpy()
target_test[target_test == 1] = -1; target_test[target_test == 0] = 1

regularization_lambdas = np.array([0.1, 1, 10, 100, 1000])
nesterov_momentum_factors = np.array([0.75, 0.8, 0.85, 0.9, 0.95])

w0 = np.random.standard_normal((28*28)); wt = w0
u0 = np.zeros_like(w0); ut = u0
lr  = 0.01

def get_grad(w, lambda_i, X=data_train, Y = target_train, w0 = np.zeros()):
    grad = -(1/X.shape[0])*(X.T @((Y*(np.exp(-Y*((w@X.T)+w0)))).T) /(np.exp(-Y*((w@X.T)+w0))+1).T) + lambda_i*w.T
    return grad

def get_loss(w, lambda_i, X=data_train, Y = target_train, w0 = np.zeros()):
    loss = (1/X.shape[0])*(np.log(1+np.exp(-Y*(X.T@w+w0)))) + (lambda_i/2)*(np.linalg.norm(w)**2 + np.linalg.norm(w0))
for lambda_i, rho in product(regularization_lambdas, nesterov_momentum_factors):
    for t in range(niters):
        u_t = rho*ut - get_grad(wt+lr*rho*ut, lambda_i=lambda_i)
        wt = wt + lr*ut


