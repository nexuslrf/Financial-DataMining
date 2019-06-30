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

cuda.empty_cache()



# parser = argparse.ArgumentParser()
#
# parser.add_argument('--epochs', default=60, type=int)
# parser.add_argument('--lr', default=0.001, type=float)
# parser.add_argument('--gpu', default=0, type=int)
# parser.add_argument('--prev_step', default=30, type=int)
# parser.add_argument('--batch_size', default=256, type=int)
# parser.add_argument('--future_step', default=1, type=int)
# parser.add_argument('--layers', default=1, type=int)
# parser.add_argument('--feature_size', default=64, type=int)
# parser.add_argument('--normalize', action='store_true')
# parser.add_argument('--padding', action='store_true')
# parser.add_argument('--optim', default='Adam',type=str)
# parser.add_argument('--sched',  type=str, default='none')
# parser.add_argument('--suffix', type=str, default='')
# parser.add_argument('--save_epoch', type=int, default=1)
# parser.add_argument('--decay_factor', default=0.98, type=float)
# parser.add_argument('--decoder', action='store_true')
# args = parser.parse_args()
# print('parsed options:', vars(args))
#
# sched_lambda = {
#         'none': lambda epoch: 1,
#         'decay': lambda epoch: max(args.decay_factor ** epoch, 1e-4),
#         }

class RNNSeq2Seq(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer=1, n_out=1, gpu=None, decoder=True,
                 backend='GRU', dropout=0.2, multitask=False):
        super().__init__()

        self.gpu = gpu
        self.backend = backend
        self.multitask = multitask
        self.input_dim, self.n_hidden, self.n_layer = input_dim, n_hidden, n_layer
        self.encoder = nn.__dict__[backend](self.input_dim, self.n_hidden, self.n_layer, dropout=dropout)
        self.decoder = decoder
        if decoder:
            self.decoder = nn.__dict__[backend](self.input_dim, self.n_hidden, self.n_layer, dropout=dropout)


        # self.reg = nn.Sequential(
        #     nn.Linear(self.n_hidden, self.n_hidden // 2),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(self.n_hidden // 2, n_out)
        # )
        self.reg = nn.Linear(self.n_hidden, n_out)
        if multitask:
            self.up_down_pred = nn.Linear(self.n_hidden, 3)

    def forward(self, x, h=None, future_step=1):
        lens, batch_size = x.size(0), x.size(1)
        h = h if h else self.init_hidden(batch_size)
        if self.multitask:
            h_outs = []
        outs = []
        gru_out, h = self.encoder(x, h)
        indata = x[-1:,:,:]
        for i in range(future_step):
            if self.decoder:
                gru_out, h = self.decoder(indata, h)
            #             gru_out = gru_out.transpose(0,1).reshape(batch_size, -1)
            outs.append(self.reg(gru_out[-1, :, :]))
            if self.multitask:
                h_outs.append(self.up_down_pred(gru_out[-1, :, :]))
        outs = torch.stack(outs)
        if outs.shape[0]==1:
            outs = outs.squeeze(0)
        if self.multitask:
            h_outs = torch.stack(h_outs)
            if h_outs.shape[0] == 1:
                h_outs = h_outs.squeeze(0)
            return outs, h_outs
        return outs

    def init_hidden(self, batch_size):
        init_h = torch.zeros((self.n_layer, batch_size, self.n_hidden))
        if self.backend == 'LSTM':
            init_c = torch.zeros((self.n_layer, batch_size, self.n_hidden))
            if self.gpu is not None:
                return (init_h.cuda(self.gpu), init_c.cuda(self.gpu))
            else:
                return (init_h, init_c)
        else:
            if self.gpu is not None:
                return init_h.cuda(self.gpu)
            else:
                return init_h

class Seq2SeqDataLoader():
    def __init__(self, data, label, prev_step, future_step, batch_size, days, date, padding=True):
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
            self.session_idx[offset+prev_step-1:offset+self.am_per_day[i]-future_step+1] = \
                np.arange(self.am_per_day[i]-future_step-prev_step+2)
            offset += self.am_per_day[i]
            # PM
            self.candidates += list(range(offset + prev_step - 1, offset + self.pm_per_day[i] - future_step + 1))
            self.session_idx[offset+prev_step-1:offset+self.pm_per_day[i]-future_step+1] = \
                np.arange(self.pm_per_day[i]-future_step-prev_step+2)
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
            forward_step = min(self.session_idx[cur_idx]+1, self.prev_step)
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
        forward_step = min(self.session_idx[cur_idx]+1, self.prev_step)
        out[-forward_step:,:] = self.data[cur_idx - forward_step + 1:cur_idx + 1, :]
        tar = self.label[cur_idx:cur_idx + self.future_step]
        return out,tar

def train(train_loader, net, criterion, optimizer, epoch):
    net.train()
    log = []
    t = tqdm.trange(train_loader.num_batch)
    for b in t:
        inputs, tar = train_loader.get_batch()
        if inputs is None:
            print('Skip it!')
            continue
        if args.gpu != None:
            inputs = inputs.cuda(0)
            tar = tar.cuda(0)
        outputs = net(inputs)
        loss = criterion(outputs, tar)
        optimizer.zero_grad()
        loss.backward()
        for p in net.parameters():
            p.grad.data.clamp_(-5, 5)
        optimizer.step()
        log.append(loss.data.item())
        t.set_description('Train ML (loss=%.6f)' % loss)
    #         print("Epoch:[{}/{}]\t[{}/{}]\tLoss:{:.4f}".format(epoch,epochs,b,train_loader.num_batch,loss))
    loss_avg = np.array(log).mean()
    return loss_avg

def validate(val_loader, net, criterion, epoch):
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
                inputs = inputs.cuda(0)
                tar = tar.cuda(0)
            outputs = net(inputs)
            loss = criterion(outputs, tar)
            log.append(loss.data.item())
            t.set_description('Eval ML (loss=%.6f)' % loss)
        #             print("Test: [{}][{}]\tLoss:{:.04f}".format(b,val_loader.num_batch,loss))
        loss_avg =np.array(log).mean()
        return loss_avg
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_GRU_{}.pth.tar'.format(args.suffix))
#### Main
def main():
    df_train = pd.read_csv('../DataSet/TrainSet.csv')
    df_val = pd.read_csv('../DataSet/ValSet.csv')
    indicators = df_train.columns.values[:108].tolist()
    market_stat = ['midPrice',  'LastPrice', 'Volume', 'LastVolume', 'Turnover', 'LastTurnover',
        'OpenInterest', 'UpperLimitPrice', 'LowerLimitPrice', 'UpdateMinute']
    features = indicators + market_stat
    train_data = df_train[features].values
    train_label = df_train['label'].values
    train_date = df_train[['Day', 'am_pm']]

    val_data = df_val[features].values
    val_label = df_val['label'].values
    val_date = df_val[['Day', 'am_pm']]
    val_date['Day'] -= 31
    # Normalization
    if args.normalize:
        mean = train_data.mean(0).reshape(1,-1)
        std = train_data.std(0).reshape(1,-1)
        train_data = (train_data - mean) / std
        val_data = (val_data - mean) / std

    net = RNNSeq2Seq(118, args.feature_size, args.layers, args.gpu, args.decoder)
    net = net.cuda(args.gpu)

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), args.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), 
                                args.lr, weight_decay=1e-4)
    
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, sched_lambda[args.sched])

    criterion = nn.MSELoss().cuda(args.gpu)

    train_loader = Seq2SeqDataLoader(train_data, train_label, args.prev_step, args.future_step, 
            args.batch_size, 30, train_date, args.padding)
    val_loader = Seq2SeqDataLoader(val_data, val_label, args.prev_step, args.future_step, 
            args.batch_size, 10, val_date, args.padding)

    train_log = open(f'train_log_{args.suffix}.log', 'w')
    val_log = open(f'val_log_{args.suffix}.log', 'w')

    best_loss = 10
    is_best = False

    for epoch in range(args.epochs):
        # training
        loss_train = train(train_loader, net, criterion, optimizer, epoch)
        print(f"Epoch {epoch} Train, Loss Avg {loss_train}")
        train_log.write('{:.6f}\n'.format(loss_train))

        loss_val = validate(val_loader, net, criterion, epoch)
        val_log.write('{:.6f}\n'.format(loss_val))
        print(f"Epoch {epoch} Eval, Loss Avg {loss_val}")
        sched.step()

        save_dir = 'checkpoint_GRU_{}.pth.tar'.format(args.suffix)

        if (epoch+1)%args.save_epoch == 0:
            is_best = best_loss > loss_val
            best_loss = max(best_loss, loss_val)
            save_checkpoint({
                'epoch': epoch + 1,
                'loss_train': loss_train,
                'loss_val': loss_val,
                'state_dict': net.state_dict(),
            }, is_best, save_dir)

if __name__ == '__main__':
    main()
