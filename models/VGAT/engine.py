import os
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored

import torch.optim
import torch.nn.parallel

import torch.nn.functional as F
import torch.nn as nn


import warnings

# 忽略所有的警告
warnings.filterwarnings('ignore')



class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        # tensorboard
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            if not os.path.isdir(writer_dir):
                os.mkdir(writer_dir)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
            
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None
        self.best_risk = {}
        self.k1 = nn.KLDivLoss(reduction='batchmean')
        

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()

        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                self.epoch = checkpoint['epoch']
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            self.validate(val_loader, model, criterion)
            return True

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            self.train(train_loader, model, criterion, optimizer)
            # evaluate on validation set
            c_index,risk_list = self.validate(val_loader, model, criterion)
            # remember best c-index and save checkpoint
            is_best = c_index > self.best_score
            if is_best:
                self.best_score = c_index
                self.best_epoch = self.epoch
                self.best_risk = risk_list
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score})
            print(' *** best c-index={:.4f} at epoch {}'.format(self.best_score, self.best_epoch))
            if scheduler is not None:
                scheduler.step()
            print('>')
        return self.best_score, self.best_epoch, self.best_risk

    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        train_loss = 0.0
        train_surloss = 0.
        train_klloss = 0.
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Train Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI,omic,label, event_time, c) in enumerate(dataloader):

            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                omic = omic.cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()

            hazards, S, re_x,_ = model(x_path=data_WSI)
            
            
            #when train loss = surv_loss+sim_loss
            #similarity loss to resconstruct gene embedding
            
            surloss = criterion[0](hazards=hazards, S=S, Y=label, c=c)
            loss_kl = criterion[1](re_x[0],omic)#criterion[1] alwyas KL as sim_loss
            
            
            
            loss = surloss + loss_kl 

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time
            train_loss += loss.item()
            train_surloss += surloss.item()
            train_klloss += loss_kl.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # calculate loss and error for epoch
        train_loss /= len(dataloader)
        train_surloss /= len(dataloader)
        train_klloss /= len(dataloader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(train_loss, c_index))

        if self.writer:
            self.writer.add_scalar('train/loss', train_loss, self.epoch)
            self.writer.add_scalar('train/surloss', train_surloss, self.epoch)
            self.writer.add_scalar('train/klloss', train_klloss, self.epoch)
            self.writer.add_scalar('train/c_index', c_index, self.epoch)

    def validate(self, data_loader, model, criterion):
        model.eval()
        val_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        risk_list = {}
        dataloader = tqdm(data_loader, desc='Test Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI,omic,label, event_time, c) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                omic = omic.cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()

            with torch.no_grad():
                hazards, S,re_x,attn = model(x_path=data_WSI)  # return hazards, S, re_x,attn

            #when inference, we just use surv_loss
            loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)
            
            risk = -torch.sum(S, dim=1).cpu().numpy()
            risk_list[risk[0]] = [event_time[0], c.cpu().numpy()[0]]
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time
            val_loss += loss.item()

        val_loss /= len(dataloader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(val_loss, c_index))
        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, self.epoch)
            self.writer.add_scalar('val/c-index', c_index, self.epoch)
        
        print(attn.shape)
        
        return c_index,risk_list

    
    

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          'fold_' + str(self.fold),
                                          'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'], epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
