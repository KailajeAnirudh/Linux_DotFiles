import numpy as np, os
from itertools import product
import torch, torchvision as thv, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt, imageio.v2 as imageio, datetime, logging
from tqdm import tqdm
import pickle

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

regularization_lambdas = np.array([1, 10, 20, 40, 80, 100])
nesterov_momentum_factors = np.array([0.75, 0.8, 0.85, 0.9, 0.95])
batch_sizes = np.array([128, 64, 8], dtype=np.uint8)

wt = np.random.standard_normal((28*28))
w0 = 0
u0 = np.zeros_like(wt); ut = u0
uw0 = 0
lr  = 0.01; niters = 100
gradient_loss = np.zeros((len(regularization_lambdas), niters))
nesterov_loss = np.zeros((len(regularization_lambdas)*len(nesterov_momentum_factors),niters))
SGD_loss = np.zeros((len(batch_sizes)*len(regularization_lambdas), niters))
SGD_nesterov_loss = np.zeros((len(batch_sizes)*len(regularization_lambdas)*len(nesterov_momentum_factors), niters))

def get_grad(w, lambda_i, X=data_train, Y=target_train, w0 = 0):
    sigmoid_arg = np.exp(-Y*(X@w + w0))
    sigmoid = sigmoid_arg/(1+sigmoid_arg)
    grad = -(1/X.shape[0])* sigmoid@(Y[:, np.newaxis]*X)+ lambda_i*w
    gradw0 = -(Y*sigmoid*(1/X.shape[0])).sum() + lambda_i*w0
    return grad, gradw0

def get_loss(w, lambda_i, X=data_train, Y=target_train, w0 = 0):
    loss = (1/X.shape[0])*(np.log(1+np.exp(-Y*(X@w + w0)))).sum(axis = 0) + (lambda_i/2)*(np.linalg.norm(w)**2 + w0**2)
    return loss 


"""Standard Gradient Descent"""
for i, lambda_i in enumerate(regularization_lambdas):
    wt = np.random.standard_normal((28*28))
    w0 = 0
    for t in tqdm(range(niters)):
        gradw, gradw0 = get_grad(w=wt, lambda_i=lambda_i, w0 = w0)
        wt = wt - lr*gradw
        w0 = w0 - lr*gradw0
        gradient_loss[i, t] = get_loss(w=wt, lambda_i=lambda_i, w0=w0)


"""Nesterov's Method"""
for i, params in enumerate(product(nesterov_momentum_factors, regularization_lambdas)):
    rho, lambda_i = params
    wt = np.random.standard_normal((28*28))
    w0 = 0
    u = wt; u0 = w0
    for t in tqdm(range(niters)):
        gradw, gradw0 = get_grad(w=wt, lambda_i=lambda_i, w0=w0)
        u1 = wt - lr*gradw
        u0_1 = w0 - lr*gradw0
        wt = (1+rho)*u1 - rho*u
        w0 = (1+rho)*u0_1 - rho*u0
        u = u1
        u0 = u0_1
        nesterov_loss[i,t] = get_loss(w=wt, lambda_i=lambda_i, w0=w0)

"""Stochastic Gradient Descent"""
batch_size = 128
for i, params in enumerate(product(batch_sizes,regularization_lambdas)):
    batch_size, lambda_i = params
    wt = np.random.standard_normal((28*28))
    w0 = 0
    for t in tqdm(range(niters)):
        idx = np.random.randint(data_train.shape[0], size=batch_size)
        x = data_train[idx]
        y = target_train[idx]
        gradw, gradw0 = get_grad(w=wt, X= x, Y=y,lambda_i=lambda_i, w0 = w0)
        wt = wt - lr*gradw
        w0 = w0 - lr*gradw0
        SGD_loss[i, t] = get_loss(w=wt, lambda_i=lambda_i, w0=w0)


"""SGD with Nesterov's Method"""
for i, params in enumerate(product(batch_sizes, nesterov_momentum_factors, regularization_lambdas)):
    batch_sizes, rho, lambda_i = params
    wt = np.random.standard_normal((28*28))
    w0 = 0
    u = wt; u0 = w0
    for t in tqdm(range(niters)):
        idx = np.random.randint(data_train.shape[0], size=batch_size)
        x = data_train[idx]
        y = target_train[idx]
        gradw, gradw0 = get_grad(w=wt, X= x, Y= y, lambda_i=lambda_i, w0=w0)
        u1 = wt - lr*gradw
        u0_1 = w0 - lr*gradw0
        wt = (1+rho)*u1 - rho*u
        w0 = (1+rho)*u0_1 - rho*u0
        u = u1
        u0 = u0_1
        SGD_nesterov_loss[i,t] = get_loss(w=wt, lambda_i=lambda_i, w0=w0)

with open('Losses.pickle', 'wb') as f:
    pickle.dump((gradient_loss, nesterov_loss, SGD_loss, SGD_nesterov_loss), f)

"""Plotting Gradient Descent Results"""
fig = plt.figure()
for i in range(gradient_loss.shape[0]):
    plt.semilogy(np.array(range(gradient_loss.shape[1])), gradient_loss[i], label = regularization_lambdas[i])
plt.ylabel('Training Loss (log-scale)')
plt.xlabel('Weight updates')
plt.legend([f'$\lambda = {regularization_lambdas[i]}$' for i in range(len(regularization_lambdas))])
plt.savefig('/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/02_Homework/04_Homework4/src/report/GD/Gradient Descent.png')
plt.show()

"""Plotting Nesterov Gradient Descent Results"""
for i in range(len(nesterov_momentum_factors)):
    fig = plt.figure()
    for j in range(len(regularization_lambdas)):
        plt.semilogy(np.array(range(nesterov_loss.shape[1])), nesterov_loss[i*len(nesterov_momentum_factors)+j])
    plt.ylabel('Training Loss (log-scale)')
    plt.xlabel('Weight updates')
    plt.legend([f'$\lambda = {regularization_lambdas[k]}$, ' for k in range(len(regularization_lambdas))])
    plt.title(f'Nesterov Gradient Descent with Momentum Factor{nesterov_momentum_factors[i]}')
    plt.savefig(f'/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/02_Homework/04_Homework4/src/report/Nesterov/Nesterov Gradient Descent with Momentum Factor{nesterov_momentum_factors[i]}.png')
    plt.close()

"""Plotting Stochastic gradient Descent REsults"""
for i in range(len(batch_sizes)):
    fig = plt.figure()
    for j in range(len(regularization_lambdas)):
        plt.semilogy(np.arange(len(SGD_loss.shape[1])), SGD_loss[i*len(regularization_lambdas) + j])
    plt.ylabel('Training Loss (log-scale)')
    plt.xlabel('Weight Updates')
    plt.legend([f'$\lambda = {regularization_lambdas[k]}$, ' for k in range(len(regularization_lambdas))])
    plt.title(f'Stochastic Gradient Descent with Batch Size{batch_sizes[i]}')
    plt.savefig(f'/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/02_Homework/04_Homework4/src/report/SGD/Stochastic Gradient Descent with Batch Size{batch_sizes[i]}.png')
    plt.close()

"""Plotting Nesterov SGD"""
for i in range(len(batch_sizes)):
    for j in range(len(nesterov_momentum_factors)):
        fig = plt.figure()
        for k in range(len(regularization_lambdas)):
            plt.semilogy(np.arange(len(SGD_nesterov_loss.shape[1])), SGD_nesterov_loss[i*len(batch_sizes)+j*len(nesterov_momentum_factors)+j])
        plt.ylabel('Training Loss (log-scale)')
        plt.xlabel('Weight Updates')
        plt.legend([f'$\lambda = {regularization_lambdas[k]}$, ' for k in range(len(regularization_lambdas))])
        plt.title(f'Nesterov Stochastic Gradient Descent with Batch Size{batch_sizes[i]}')
        plt.savefig(f'/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/02_Homework/04_Homework4/src/report/NesterovSGD/Nestereov Stochastic Gradient Descent with Batch Size{batch_sizes[i]} and Momentum{nesterov_momentum_factors[j]}.png')
        plt.close()
