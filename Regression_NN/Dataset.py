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
import torch.utils.data
class Seq2SeqDataLoader(torch.utils.data.Dataset):
    def __init__(self, data, label, prev_step, future_step, batch_size, days, date, padding=True):
        super(type(self), self).__init__()
        self.data = data
        self.label = label
        self.data_size = data.shape[1]
        self.prev_step = prev_step
        self.future_step = future_step
        self.days = days
        self.batch_size = batch_size
        self.date = date
        self.am_per_day = [date[(date['Day'] == i) & (date['am_pm'] == True)].__len__() for i in range(days)]
        self.pm_per_day = [date[(date['Day'] == i) & (date['am_pm'] == False)].__len__() for i in range(days)]

        self.candidates = []
        self.session_idx = np.zeros(len(label), dtype=int)

        if padding:
            prev_step = 1
        offset = 0
        for i in range(days):
            # AM
            self.candidates += list(range(offset + prev_step - 1, offset + self.am_per_day[i] - future_step + 1))
            self.session_idx[offset + prev_step - 1:offset + self.am_per_day[i] - future_step + 1] = \
                np.arange(self.am_per_day[i] - future_step - prev_step + 2)
            offset += self.am_per_day[i]
            # PM
            self.candidates += list(range(offset + prev_step - 1, offset + self.pm_per_day[i] - future_step + 1))
            self.session_idx[offset + prev_step - 1:offset + self.pm_per_day[i] - future_step + 1] = \
                np.arange(self.pm_per_day[i] - future_step - prev_step + 2)
            offset += self.pm_per_day[i]

        self.candidates = np.array(self.candidates)

        self.num_batch = len(self.candidates) // batch_size
        self.left = len(self.candidates) % batch_size
        self.num_batch = self.num_batch if self.left == 0 else self.num_batch + 1
        self.left = self.left if self.left != 0 else self.batch_size

        self.cnt = 0
        self.reset()

    def get_batch(self):
        batch_size = self.batch_size if self.cnt != self.num_batch - 1 else self.left

        out = np.zeros([self.prev_step, batch_size, self.data_size])
        tar = np.zeros([batch_size, self.future_step])
        for i in range(batch_size):
            cur_idx = self.randlist[self.cnt * self.batch_size + i]
            forward_step = min(self.session_idx[cur_idx] + 1, self.prev_step)
            out[-forward_step:, i, :] = self.data[cur_idx - forward_step + 1:cur_idx + 1, :]
            tar[i, :] = self.label[cur_idx:cur_idx + self.future_step]

        self.cnt += 1
        if self.cnt % self.num_batch == 0:
            self.reset()
        return torch.Tensor(out), torch.Tensor(tar)

    def reset(self):
        self.cnt = 0
        self.randlist = np.random.permutation(self.candidates)

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, index):
        out = np.zeros([self.prev_step, self.data_size])
        cur_idx = self.candidates[index]
        forward_step = min(self.session_idx[cur_idx] + 1, self.prev_step)
        out[-forward_step:, :] = self.data[cur_idx - forward_step + 1:cur_idx + 1, :]
        tar = self.label[cur_idx:cur_idx + self.future_step]
        return out, tar
