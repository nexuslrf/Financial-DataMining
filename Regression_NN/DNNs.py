import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import tqdm
import torch.cuda as cuda
import argparse
import os
import shutil
from Dataset import Seq2SeqDataLoader

class DNN(nn.Module):
    def __init__(self, inDim, hidDim, layers, num_classes=1, batchnorm=True, activation=nn.LeakyReLU, dropout=0.2):
        super(DNN, self).__init__()
        if type(hidDim) is not list:
            hidDim = [hidDim] * layers
        self.inDim = inDim
        Dims = [inDim] + hidDim
        if batchnorm:
            self.layers = nn.Sequential(
                *[nn.Sequential(nn.Linear(Dims[i], Dims[i + 1]),
                                nn.BatchNorm1d(Dims[i + 1]), activation()) for i in range(layers)]
            )
        else:
            self.layers = nn.Sequential(
                *[nn.Sequential(nn.Linear(Dims[i], Dims[i + 1]),
                                activation()) for i in range(layers)]
            )
        self.dropout = nn.Dropout(dropout)
        self.pred = nn.Linear(Dims[-1], num_classes)
    def forward(self, x):
        if len(x.shape) != 2:
            x = x.reshape(-1, self.inDim)
        x = self.layers(x)
        x = self.dropout(x)
        x = self.pred(x)
        return x

class CNN(nn.Module):
    def __init__(self, inDim, hidDim, prev_steps, layers=1, pool_size=4, kernelsize=[3], num_classes=1, batchnorm=True, activation=nn.LeakyReLU,  dropout=0.2):
        super(CNN, self).__init__()
        # a CNN
        if type(hidDim) is not list:
            hidDim = [hidDim] * layers
        if type(kernelsize) is not list:
            kernelsize = [kernelsize]

        Dims = [inDim] + hidDim

        if batchnorm:
            self.convs = nn.ModuleList(
                [nn.Sequential(
                    *[
                        nn.Sequential(
                        nn.Conv1d(Dims[j], Dims[j+1], i, padding=i//2,),
                        nn.BatchNorm1d(Dims[j+1]),
                        activation(),
                        nn.MaxPool1d(pool_size)
                        )
                        for j in range(layers)
                    ]
                    )
                    for i in kernelsize
                ]
            )
        else:
            self.convs = nn.ModuleList(
                [nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(Dims[j], Dims[j + 1], i, padding=i//2),
                            activation(),
                            nn.MaxPool1d(pool_size)
                        )
                        for j in range(layers)
                    ]
                )
                    for i in kernelsize
                ]
            )
        self.dropout = nn.Dropout(dropout)
        self.pred = nn.Linear(prev_steps//(pool_size**layers)*len(kernelsize)*Dims[-1], num_classes)
    def forward(self, x, len_first=True):
        if len_first:
            x = x.permute(1,2,0)
        conv_out = []
        for i in range(len(self.convs)):
            conv_out.append(self.convs[i](x))
        x = torch.cat(conv_out,2)
        x = x.view(x.size(0),-1)
        x = self.dropout(x)
        x = self.pred(x)
        return x


if __name__ == '__main__':
    batch_size = 64
    df_val = pd.read_csv('../DataSet/ValSet.csv')
    indicators = df_val.columns.values[:108].tolist()
    market_stat = ['midPrice', 'LastPrice', 'Volume', 'LastVolume', 'Turnover', 'LastTurnover',
                   'OpenInterest', 'UpperLimitPrice', 'LowerLimitPrice', 'UpdateMinute']
    features = indicators + market_stat

    val_data = df_val[features].values
    val_label = df_val['label'].values
    val_date = df_val[['Day', 'am_pm']]
    val_date['Day'] -= 31
    val_loader = Seq2SeqDataLoader(val_data, val_label, 32, 1,
                                   batch_size, 10, val_date, False)
    # M = DNN(118, 128, 3)
    M = CNN(118, 64, 32, 2)
    inputs, tar = val_loader.get_batch()
    M(inputs)
    print(M)
    torch.utils.data.DataLoader(
                val_loader, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True, sampler=None)
