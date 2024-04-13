# %%
import numpy as np, pandas as pd, os, torch, torchaudio, torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import wfdb
import ast
from IPython.display import Audio
from torchaudio.utils import download_asset
import seaborn as sns

# %%
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = '/home/anirudhkailaje/Documents/01_UPenn/01_ESE5460/03_Project/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sampling_rate=100
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
scp_codes = pd.read_csv(path+'scp_statements.csv', index_col=0)
scp_codes.loc[:, ['diagnostic', 'form', 'rhythm']].fillna(0, inplace=True)
scp_codes[['diagnostic', 'form', 'rhythm']] = scp_codes[['diagnostic', 'form', 'rhythm']].apply(pd.to_numeric)
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
Y[['Diag', 'Form', 'Rhythm']] = 0
for idx in Y.index.values:
    labels = Y.loc[idx].scp_codes
    for key in labels.keys():
        if labels[key] > 0:
            Y.loc[idx, ['Diag', 'Form', 'Rhythm']] = scp_codes.loc[key][['diagnostic', 'form', 'rhythm']].values
Y.loc[:,['Diag', 'Form', 'Rhythm']].fillna(0, inplace=True)
X = load_raw_data(Y, sampling_rate, path)

scp_list = Y.scp_codes.values.tolist()
unique_scp = []
for scps in scp_list:
    unique_scp += list(scps.keys())
len(set(unique_scp))
Y_train = Y[(Y.strat_fold != 10)&(Y.strat_fold!= 9)]


# %%
Y.loc[:, ['Diag','Form', 'Rhythm']].fillna(0, inplace=True)

# %%
Y.fillna(0, inplace=True)

# %%
scp_codes[['diagnostic', 'form', 'rhythm']] = scp_codes[['diagnostic', 'form', 'rhythm']].fillna(0)
scp_codes['super_class'] = scp_codes.diagnostic*1+scp_codes.form*2+scp_codes.rhythm*4
scp_codes.sort_values(by = ['super_class'], inplace= True)
scp_codes['superclass_desc'] = ""
scp_codes.loc[scp_codes.super_class == 1, 'superclass_desc'] = "Diagnostic" 
scp_codes.loc[scp_codes.super_class == 2, 'superclass_desc'] = "Form" 
scp_codes.loc[scp_codes.super_class == 4, 'superclass_desc'] = "Rhythm" 
scp_codes.loc[scp_codes.super_class == 3, 'superclass_desc'] = "Diagnostic and Form"
class2num = {key: i for i, key in enumerate(scp_codes.index.values)}

# %%
counts = np.zeros(len(class2num), dtype=np.longlong)
for i in range(len(Y)):
    for key in Y.scp_codes.iloc[i].keys():
        counts[class2num[key]] += 1



# %%
Y_train = Y[(Y.strat_fold != 10)&(Y.strat_fold!= 9)].reset_index()
X_train = X[np.where((Y.strat_fold != 10)&(Y.strat_fold!= 9))]

Y_val = Y[Y.strat_fold == 9].reset_index()
X_val = X[np.where(Y.strat_fold ==9)]

Y_test = Y[Y.strat_fold == 10].reset_index()
X_test = X[np.where(Y.strat_fold == 10)]

# %%
np.savez_compressed('X_train.npz', X_train)
np.savez_compressed('X_val.npz', X_val)
np.savez_compressed('X_test.npz', X_test)
Y_train.to_csv('Y_train.csv')
Y_val.to_csv('Y_val.csv')
Y_test.to_csv('Y_test.csv')



