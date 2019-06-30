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
from SAEs import SAE

cuda.empty_cache()

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--layers', default=2, type=int)
parser.add_argument('--feature_size', default='128', type=str)
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--optim', default='Adam', type=str)
parser.add_argument('--sched', type=str, default='none')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--save_epoch', type=int, default=1)
parser.add_argument('--decay_factor', default=0.98, type=float)
parser.add_argument('--wavelet_trans', action='store_true')
parser.add_argument('--merge_WT', action='store_true')
parser.add_argument('--activation', default='LeakyReLU', type=str)
parser.add_argument('--no_BN', action='store_false')
args = parser.parse_args()
args.feature_size = json.loads(args.feature_size)
print('parsed options:', vars(args))

sched_lambda = {
    'none': lambda epoch: 1,
    'decay': lambda epoch: max(args.decay_factor ** epoch, 1e-4),
}


def train(train_loader, net, criterion, optimizer, level):
    net.train()
    log = []
    t = tqdm.trange(train_loader.num_batch)
    for b in t:
        inputs, tar = train_loader.get_batch()
        if inputs is None:
            print('Skip it!')
            continue
        if args.gpu != None:
            inputs = inputs.cuda(args.gpu).squeeze(0)

        embed, outputs, x = net(inputs, level)
        loss = criterion(outputs, x)
        optimizer.zero_grad()
        loss.backward()
        # for p in net.parameters():
        #     p.grad.data.clamp_(-5, 5)
        optimizer.step()

        log.append(loss.data.item())
        t.set_description('Train ML (loss=%.6f)' % loss)
        #             print("Test: [{}][{}]\tLoss:{:.04f}".format(b,val_loader.num_batch,loss))
    loss_avg = np.array(log).mean()
    return loss_avg

def validate(val_loader, net, criterion, level):
    log = []
    net.eval()
    with torch.no_grad():
        t = tqdm.trange(val_loader.num_batch)
        for b in t:
            inputs, tar = val_loader.get_batch()
            if inputs is None:
                print('Skip it!')
                continue
            if args.gpu != None:
                inputs = inputs.cuda(args.gpu).squeeze(0)

            embed, outputs, x = net(inputs, level)
            loss = criterion(outputs, x)
            log.append(loss.data.item())

            t.set_description('Eval ML (loss=%.6f)' % loss)
        #             print("Test: [{}][{}]\tLoss:{:.04f}".format(b,val_loader.num_batch,loss))
        loss_avg = np.array(log).mean()
        return loss_avg

def main():
    df_train = pd.read_csv('../DataSet/TrainSet.csv')
    df_val = pd.read_csv('../DataSet/ValSet.csv')
    indicators = df_train.columns.values[:108].tolist()
    market_stat = ['midPrice', 'LastPrice', 'Volume', 'LastVolume', 'Turnover', 'LastTurnover',
                   'OpenInterest', 'UpperLimitPrice', 'LowerLimitPrice', 'UpdateMinute']
    features = indicators + market_stat
    train_data = df_train[features].values
    train_label = df_train['label'].values
    train_date = df_train[['Day', 'am_pm']]

    val_data = df_val[features].values
    val_label = df_val['label'].values
    val_date = df_val[['Day', 'am_pm']]
    val_date['Day'] -= 31

    if args.wavelet_trans:
        # train_data_WT = np.empty((2, *train_data.shape))
        if args.merge_WT:
            train_WT = np.empty((train_data.shape[0],2*train_data.shape[1]))
            val_WT = np.empty((val_data.shape[0], 2*val_data.shape[1]))
            for j in range(len(features)):
                A, D = wavelet_transform(train_data[:, j])
                train_WT[:, j] = A[:]
                train_WT[:, j+len(features)] = D[:]
                A, D = wavelet_transform(val_data[:, j])
                val_WT[:, j] = A[:]
                val_WT[:, j + len(features)] = D[:]
            train_data = train_WT
            val_data = val_WT
        else:
            for j in range(len(features)):
                A, D = wavelet_transform(train_data[:, j])
                train_data[:, j] = A[:]
                A, D = wavelet_transform(val_data[:, j])
                val_data[:, j] = A[:]

    criterion = nn.MSELoss().cuda(args.gpu)

    # Normalization
    if args.normalize:
        mean = train_data.mean(0).reshape(1, -1)
        std = train_data.std(0).reshape(1, -1)
        train_data = (train_data - mean) / std
        val_data = (val_data - mean) / std

    net = SAE(train_data.shape[1], args.feature_size, args.layers, args.no_BN, nn.__dict__[args.activation])
    net = net.cuda(args.gpu)

    train_loader = Seq2SeqDataLoader(train_data, train_label, 1, 1,
                                     args.batch_size, 30, train_date, False)
    val_loader = Seq2SeqDataLoader(val_data, val_label, 1, 1,
                                   args.batch_size, 10, val_date, False)

    train_log = open(f'train_log_SAE_{args.suffix}.log', 'w')
    val_log = open(f'val_log_SAE_{args.suffix}.log', 'w')

    best_loss = 10
    is_best = False

    for L in range(args.layers):

        if args.optim == 'SGD':
            optimizer = torch.optim.SGD(list(net.encoder[L].parameters()) + \
                                        list(net.decoder[L].parameters()),
                                        args.lr,
                                        momentum=0.9,
                                        weight_decay=1e-4)
        elif args.optim == 'Adam':
            optimizer = torch.optim.Adam(list(net.encoder[L].parameters()) + \
                                         list(net.decoder[L].parameters()),
                                         args.lr, weight_decay=1e-4)

        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, sched_lambda[args.sched])

        for epoch in range(args.epochs):
            # training
            loss_train = train(train_loader, net, criterion, optimizer, L)
            train_log.write('{:.6f}\n'.format(loss_train))
            loss_val = validate(val_loader, net, criterion, L)
            val_log.write('{:.6f}\n'.format(loss_val))
            print(f"Level {L} Epoch {epoch} Train Loss Avg {loss_train} Eval Loss Avg {loss_val}")

            sched.step()

            save_dir = 'checkpoint_SAE_{}.pth.tar'.format(args.suffix)

            if (epoch + 1) % args.save_epoch == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'loss_train': loss_train,
                    'state_dict': net.state_dict(),
                    'inDim': train_data.shape[1],
                    'hidDim': args.feature_size,
                    'layers': args.layers,
                    'BN': args.no_BN,
                    'activation': args.activation
                }, is_best, save_dir)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_SAE_{}.pth.tar'.format(args.suffix))

if __name__ == '__main__':
    main()
    # M = SAE(118, [128,96,64], 3)
    # print(M)