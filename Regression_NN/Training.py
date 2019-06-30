import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import itertools
import tqdm
import torch.cuda as cuda
import argparse
import json
import os
import shutil
from Dataset import Seq2SeqDataLoader
from RNN import RNNSeq2Seq
from Transfermer_stock import Transformer_stock
from wavelet_trans import wavelet_transform
from SAEs import SAE
from DNNs import DNN, CNN
cuda.empty_cache()

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='Transformer', type=str, help='Transformer | GRU')
parser.add_argument('--epochs', default=60, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--prev_step', default=30, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--future_step', default=1, type=int)
parser.add_argument('--layers', default=2, type=int)
parser.add_argument('--feature_size', default='128', type=str)
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--padding', action='store_true')
parser.add_argument('--optim', default='Adam', type=str)
parser.add_argument('--sched', type=str, default='none')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--save_epoch', type=int, default=1)
parser.add_argument('--decay_factor', default=0.98, type=float)
parser.add_argument('--decoder', action='store_true')
parser.add_argument('--num_classes', default=1, type=int)
parser.add_argument('--wavelet_trans', action='store_true')
parser.add_argument('--merge_WT', action='store_true')
parser.add_argument('--SAE', default='', type=str)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--multitask', action='store_true')
parser.add_argument('--lambd', default=0.1, type=float)
parser.add_argument('--kernel_size', default='3', type=str)
parser.add_argument('--pool_size', default=2, type=int)
parser.add_argument('--no_BN', action='store_false')
parser.add_argument('--no_indicator', action='store_true')
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--resume', default='', type=str)
args = parser.parse_args()
print('parsed options:', vars(args))

args.feature_size = json.loads(args.feature_size)
args.kernel_size = json.loads(args.kernel_size)

sched_lambda = {
    'none': lambda epoch: 1,
    'decay': lambda epoch: max(args.decay_factor ** epoch, 1e-4),
}

if args.num_classes == 3:
    classes = [0, 1, 2]
    def class_map(x):
        if x < 0:
            return 0
        elif x > 0:
            return 2
        else:
            return 1

else:
    classes = []
    classes_range = []
    z = 0
    for i in range(-10, 11):
        classes.append(z)
        classes_range.append(i + 0.5)
        z += 1
    classes_range.pop()


    def class_map(x):
        global classes_range
        p = 0
        for i in classes_range:
            if x < i:
                return p
            p += 1
        return p

def get_up_down(x):
    z = torch.zeros_like(x, dtype=torch.long)
    z[x==0] = 1
    z[x>0] = 2
    return z.squeeze(1)

def train(train_loader, net, criterion, optimizer):
    net.train()
    log = []
    acc = []
    t = tqdm.trange(train_loader.num_batch)
    for b in t:
        inputs, tar = train_loader.get_batch()
        if inputs is None:
            print('Skip it!')
            continue
        if args.gpu != None:
            inputs = inputs.cuda(args.gpu)
            tar = tar.cuda(args.gpu)

        if args.num_classes > 1:
            tar = tar.long().squeeze(1)
            outputs = net(inputs)
            acc_t = accuracy(outputs, tar)[0].item()
            acc.append(acc_t)
            log.append(loss.data.item())
            t.set_description('Train ML (loss=%.3f, acc=%.3f)' % (loss, acc_t))

        if args.multitask:
            up_down_label = get_up_down(tar)
            outputs, h_outs = net(inputs)
            mse = criterion[0](outputs, tar)
            loss = mse + args.lambd * criterion[1](h_outs, up_down_label)
            log.append(mse.data.item())
            acc_t = accuracy(outputs, up_down_label)[0].item()
            acc.append(acc_t)
            t.set_description('Train ML (loss=%.3f, acc=%.3f)' % (loss, acc_t))

        else:
            outputs = net(inputs)
            loss = criterion(outputs, tar)
            t.set_description('Train ML (loss=%.6f)' % loss)
            log.append(loss.data.item())

        optimizer.zero_grad()
        loss.backward()
        # for p in net.parameters():
        #     p.grad.data.clamp_(-5, 5)
        optimizer.step()

    loss_avg = np.array(log).mean()
    acc_avg = np.array(acc).mean()
    return loss_avg, acc_avg


def validate(val_loader, net, criterion):
    log = []
    acc = []
    net.eval()
    with torch.no_grad():
        t = tqdm.trange(val_loader.num_batch)
        for b in t:
            inputs, tar = val_loader.get_batch()
            if inputs is None:
                print('Skip it!')
                continue
            if args.gpu != None:
                inputs = inputs.cuda(args.gpu)
                tar = tar.cuda(args.gpu)

            if args.num_classes > 1:
                tar = tar.long().squeeze(1)
                outputs = net(inputs)
                acc_t = accuracy(outputs, tar)[0].item()
                acc.append(acc_t)
                log.append(loss.data.item())
                t.set_description('Eval ML (loss=%.3f, acc=%.3f)' % (loss,acc_t))

            if args.multitask:
                up_down_label = get_up_down(tar)
                outputs, h_outs = net(inputs)
                mse = criterion[0](outputs, tar)
                loss = mse + args.lambd * criterion[1](h_outs, up_down_label)
                log.append(mse.data.item())
                acc_t = accuracy(outputs, up_down_label)[0].item()
                acc.append(acc_t)
                t.set_description('Eval ML (loss=%.3f, acc=%.3f)' % (loss, acc_t))

            else:
                outputs = net(inputs)
                loss = criterion(outputs, tar)
                t.set_description('Eval ML (loss=%.6f)' % loss)
                log.append(loss.data.item())

        loss_avg = np.array(log).mean()
        acc_avg = np.array(acc).mean()
        return loss_avg, acc_avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_{}_{}.pth.tar'.format(args.model, args.suffix))

#### Main
def main():
    global classes
    df_train = pd.read_csv('../DataSet/TrainSet.csv')
    df_val = pd.read_csv('../DataSet/ValSet.csv')
    indicators = df_train.columns.values[:108].tolist()
    market_stat = ['midPrice', 'LastPrice', 'Volume', 'LastVolume', 'Turnover', 'LastTurnover',
                   'OpenInterest', 'UpperLimitPrice', 'LowerLimitPrice', 'UpdateMinute']
    if args.no_indicator:
        features = market_stat
    else:
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


    if args.num_classes > 1:
        train_class = df_train['label'].map(class_map)
        val_class = df_val['label'].map(class_map)
        class_cnt = train_class.value_counts(sort=False)[classes]
        max_cnt = class_cnt.max()
        class_weight = ((class_cnt / max_cnt) ** -0.5).values
        class_weight = class_weight / class_weight.mean()
        train_label = train_class.values
        val_label = val_class.values

        criterion = nn.CrossEntropyLoss(torch.Tensor(class_weight)).cuda(args.gpu)
    else:
        criterion = nn.MSELoss().cuda(args.gpu)

    if args.multitask:
        # up 2 down 0 unchange 1
        up_down = lambda x: 0 if x<0 else 2 if x>0 else 1
        train_class = df_train['label'].map(up_down)
        class_cnt = train_class.value_counts(sort=False)[[0,1,2]]
        max_cnt = class_cnt.max()
        up_down_weight = ((class_cnt / max_cnt) ** -1).values
        up_down_weight = up_down_weight / up_down_weight.mean()
        up_down_criterion = nn.CrossEntropyLoss(
            torch.Tensor(up_down_weight)).cuda(args.gpu)
        criterion = (criterion, up_down_criterion)

    # Normalization
    if args.normalize:
        mean = train_data.mean(0).reshape(1, -1)
        std = train_data.std(0).reshape(1, -1)
        train_data = (train_data - mean) / std
        val_data = (val_data - mean) / std

    if args.SAE!='':
        ckpt = torch.load(args.SAE)
        train_ae = np.zeros((train_data.shape[0], ckpt['hidDim'][-1]))
        val_ae = np.zeros((val_data.shape[0], ckpt['hidDim'][-1]))
        ae = SAE(ckpt['inDim'], ckpt['hidDim'], ckpt['layers'],
                 ckpt['BN'], nn.__dict__[ckpt['activation']])
        ae.load_state_dict(ckpt['state_dict'])
        ae.cuda(args.gpu)
        bs = 1024
        offset = 0
        while offset + bs < train_data.shape[0]:
            train_ae[offset:offset + bs, :] = \
                ae.get_embed(torch.Tensor(train_data[offset:offset + bs, :]).cuda(args.gpu),
                             ckpt['layers']-1).data.cpu().numpy()
            offset += bs
        train_ae[offset:, :] = \
            ae.get_embed(torch.Tensor(train_data[offset:, :]).cuda(args.gpu),
                         ckpt['layers'] - 1).data.cpu().numpy()

        offset = 0
        while offset + bs < val_data.shape[0]:
            val_ae[offset:offset + bs, :] = \
                ae.get_embed(torch.Tensor(val_data[offset:offset + bs, :]).cuda(args.gpu),
                             ckpt['layers'] - 1).data.cpu().numpy()
            offset += bs
        val_ae[offset:, :] = \
            ae.get_embed(torch.Tensor(val_data[offset:, :]).cuda(args.gpu),
                         ckpt['layers'] - 1).data.cpu().numpy()

        train_data = train_ae
        val_data = val_ae
        del ckpt

    if args.model == 'Transformer':
        net = Transformer_stock(args.prev_step, args.future_step, args.num_classes, num_layers=args.layers,
                                feature_dim=train_data.shape[1], model_dim=args.feature_size, ffn_dim=512, 
                                gpu=args.gpu, num_heads=args.num_heads, multitask=args.multitask)
    elif args.model == 'DNN':
        net = DNN(train_data.shape[1], args.feature_size, args.layers, args.num_classes, batchnorm=args.no_BN)
    elif args.model == 'CNN':
        net = CNN(train_data.shape[1], args.feature_size, args.prev_step, args.layers, args.pool_size,
                  args.kernel_size, batchnorm=args.no_BN, num_classes=args.num_classes)
    else:
        net = RNNSeq2Seq(train_data.shape[1], args.feature_size, args.layers, args.num_classes,
                         args.gpu, args.decoder, backend=args.model, multitask=args.multitask)

    net = net.cuda(args.gpu)

    train_loader = Seq2SeqDataLoader(train_data, train_label, args.prev_step, args.future_step,
                                     args.batch_size, 30, train_date, args.padding)
    val_loader = Seq2SeqDataLoader(val_data, val_label, args.prev_step, args.future_step,
                                   args.batch_size, 10, val_date, args.padding)

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device("cuda:{}".format(args.gpu)))
        # args.start_epoch = checkpoint['epoch']
        # best_acc1 = checkpoint['best_acc1']
        net.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (loss_val: {}) !"
              .format(args.resume, checkpoint['loss_val']))
        del checkpoint

    if args.evaluate:
        net.eval()
        train_pred = np.zeros((train_data.shape[0], 1))
        val_pred = np.zeros((val_data.shape[0], 1))
        if not args.padding:
            print('Padding Error')
            return

        train_loader_seq = DataLoader(
            train_loader, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, sampler=None)
        val_loader_seq = DataLoader(
            val_loader, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, sampler=None)
        with torch.no_grad():
            offset = 0
            for i, (input, target) in tqdm.tqdm(enumerate(train_loader_seq)):
                bs = input.shape[0]
                input = input.transpose(0,1).float().cuda(args.gpu)
                pred = net(input).data.cpu().numpy()
                train_pred[offset:offset+bs, :] = pred
                offset += bs

            offset = 0
            for i, (input, target) in tqdm.tqdm(enumerate(val_loader_seq)):
                bs = input.shape[0]
                input = input.transpose(0, 1).float().cuda(args.gpu)
                pred = net(input).data.cpu().numpy()
                val_pred[offset:offset + bs, :] = pred
                offset += bs

        file_name = 'pred_res'+ args.resume[11:-8]+'.pt'
        torch.save({'train_pred': train_pred, 'val_pred': val_pred}, file_name)

        return




    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                     args.lr, weight_decay=args.weight_decay)

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, sched_lambda[args.sched])

    train_log = open(f'train_log_{args.model}_{args.suffix}.log', 'w')
    val_log = open(f'val_log_{args.model}_{args.suffix}.log', 'w')

    best_loss = 10
    is_best = False

    for epoch in range(args.epochs):
        # training
        loss_train, acc_train = train(train_loader, net, criterion, optimizer,)
        if args.num_classes > 1 or args.multitask:
            print(f"Epoch {epoch} Train, Loss Avg {loss_train} Acc Avg {acc_train}")
            train_log.write('{:.6f} {:.6}\n'.format(loss_train, acc_train))
        else:
            print(f"Epoch {epoch} Train, Loss Avg {loss_train}")
            train_log.write('{:.6f}\n'.format(loss_train))

        loss_val, acc_val = validate(val_loader, net, criterion,)
        if args.num_classes > 1 or args.multitask:
            print(f"Epoch {epoch} Eval, Loss Avg {loss_val} Acc Avg {acc_val}")
            train_log.write('{:.6f} {:.6}\n'.format(loss_val, acc_val))
        else:
            val_log.write('{:.6f}\n'.format(loss_val))
            print(f"Epoch {epoch} Eval, Loss Avg {loss_val}")
        sched.step()

        save_dir = 'checkpoint_{}_{}.pth.tar'.format(args.model,args.suffix)

        if (epoch + 1) % args.save_epoch == 0:
            is_best = best_loss > loss_val
            best_loss = min(best_loss, loss_val)
            save_checkpoint({
                'epoch': epoch + 1,
                'loss_train': loss_train,
                'loss_val': loss_val,
                'state_dict': net.state_dict(),
            }, is_best, save_dir)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
