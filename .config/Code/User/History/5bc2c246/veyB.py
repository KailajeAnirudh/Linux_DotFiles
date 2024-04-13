import os, numpy as np, matplotlib.pyplot as plt, pickle, pandas as pd
from tqdm import tqdm
import torch
from utils import *

data_folder = '/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/03_Project/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
data, raw_labels = load_dataset(data_folder, 100)

tasks = ['diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm']
task_labels = []
for task in tasks:
    task_labels.append(compute_label_aggregations(raw_labels, folder= data_folder, ctype=task))

