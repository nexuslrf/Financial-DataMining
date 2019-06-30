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
from RNN import RNNSeq2Seq
from Transfermer_stock import Transformer_stock
from wavelet_trans import wavelet_transform
import json

cuda.empty_cache()

class AE(nn.Module):
    def __init__(self, inDim, hidDim, layers, batchnorm=True, activation=nn.LeakyReLU):
        super(AE, self).__init__()
        if type(hidDim) is not list:
            hidDim = [hidDim] * layers

        Dims = [inDim] + hidDim
        if batchnorm:
            self.encoder = nn.Sequential(
                *[nn.Sequential(nn.Linear(Dims[i], Dims[i + 1]),
                                nn.BatchNorm1d(Dims[i + 1]), activation()) for i in range(layers)]
            )
            self.decoder = nn.Sequential(
                *[nn.Sequential(nn.Linear(Dims[i], Dims[i - 1]),
                                nn.BatchNorm1d(Dims[i - 1]), activation()) for i in range(layers, 1, -1)],
                nn.Linear(Dims[1], Dims[0]),
            )
        else:
            self.encoder = nn.Sequential(
                *[nn.Sequential(nn.Linear(Dims[i], Dims[i + 1]),
                                activation()) for i in range(layers)]
            )
            self.decoder = nn.Sequential(
                *[nn.Sequential(nn.Linear(Dims[i], Dims[i - 1]),
                                activation()) for i in range(layers - 1, 0, -1)],
                nn.Linear(Dims[1], Dims[0]),
            )

    def forward(self, x):
        embed = self.encoder(x)
        reconstruct = self.decoder(embed)
        return embed, reconstruct

    def get_embed(self, x):
        self.eval()
        with torch.no_grad():
            return self.encoder(x)

class SAE(nn.Module):
    def __init__(self, inDim, hidDim, layers, batchnorm=True, activation=nn.LeakyReLU):
        super(SAE, self).__init__()
        if type(hidDim) is not list:
            hidDim = [hidDim] * layers

        Dims = [inDim] + hidDim

        if batchnorm:
            self.encoder = nn.ModuleList(
                [nn.Sequential(nn.Linear(Dims[i], Dims[i + 1]), nn.BatchNorm1d(Dims[i + 1]), activation()) \
                 for i in range(layers)]
            )
        else:
            self.encoder = nn.ModuleList(
                [nn.Sequential(nn.Linear(Dims[i], Dims[i + 1]),  activation()) \
                 for i in range(layers)]
            )
        self.decoder = nn.ModuleList(
            [nn.Linear(Dims[i + 1], Dims[i]) for i in range(layers)]
        )

    def forward(self, x, level=0):
        with torch.no_grad():
            for i in range(level):
                x = self.encoder[i].eval()(x)
        embed = self.encoder[level](x)
        reconstruct = self.decoder[level](embed)
        return embed, reconstruct, x

    def get_embed(self, x, level):
        self.eval()
        with torch.no_grad():
            for i in range(level):
                x = self.encoder[i](x)
            embed = self.encoder[level](x)
            return embed

if __name__ == '__main__':
    M = AE(236, [256,192,160,128], 4)
    print(M)