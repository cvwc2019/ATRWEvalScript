#!/usr/bin/env python3

import argparse
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from tensorboardX import SummaryWriter

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels

from models import C3D464
from models.TemporalAlignMoudle import TemporalAlignMoudleOTAM as TemporalAlignMoudle
from models.DynamicHead import DynamicHeadCond as DynamicHead
from models.DynamicHead import DynConv3DBlock,DynamicLinear
from models.STAttention import STAttention
from dataset.tsndataset import TSNDataSet,DataPrefetcher
# from dataset.uniformdataset import TSNDataSet,DataPrefetcher
from dataset import dataset_config
from ops.transforms import *
import losses

import os
import sys
import shutil
import ast



def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits

def pairwise_concat(support,query,sum_mask):
    '''
    Follow original RelationNet, we sum k-shot then concat
    '''
    with torch.no_grad():
        n = sum_mask.shape[0]
        m = query.shape[0]
        shape=support.shape
    support_sum= (sum_mask @ support.view(shape[0],-1)).view(n,*shape[-4:])
    pairs = torch.cat([support_sum.unsqueeze(1).expand(n, m, -1,-1,-1,-1),
                query.unsqueeze(0).expand(n, m, -1,-1,-1,-1)],-4)
    return pairs


def accuracy(logits, targets):
    with torch.no_grad():
        # classes=torch.unique(support_labels)
        # sum_mask= (classes.view(-1,1)==support_labels.view(1,-1)).float()
        # scores = sum_mask@logits
        # predictions = scores.argmax(dim=0)
        predictions = logits.argmax(dim=0)
        # predictions = support_labels[predictions]
        return (predictions == targets).sum().float() / targets.size(0)

class Convnet_pooling4(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = C3D464.Conv3DBase(output_size=z_dim,
                                         hidden=hid_dim,
                                         channels=x_dim,
                                         max_pool=True,
                                         temporal_pool=True,
                                         max_pool_factor=1.0)
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x

class Convnet(nn.Module):

    def __init__(self,st_attention=False, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = C3D464.Conv3DBase(output_size=z_dim,
                                         hidden=hid_dim,
                                         channels=x_dim,
                                         max_pool=[True,True,True,True],
                                         temporal_pool=[True,False,False,False],
                                         max_pool_factor=[1.0,1.0,1.0,1.0],)
        self.use_attention=st_attention
        if self.use_attention:
            self.attention=STAttention(alpha_s=1.0,alpha_t=0.5) #For HMDB!

    def forward(self, x):
        x = self.encoder(x)
        if self.use_attention:
            x=self.attention(x)
        return x

class Convnet2D(nn.Module):

    def __init__(self,st_attention=False, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = C3D464.ConvBase(output_size=z_dim,
                                         hidden=hid_dim,
                                         channels=x_dim,
                                         max_pool=True,
                                         # temporal_pool=[True,False,False,False],
                                         max_pool_factor=1.0,)
        self.use_attention=st_attention
        if self.use_attention:
            self.attention=STAttention(alpha_s=1.0,alpha_t=0.5) #For HMDB!

    def forward(self, x):
        with torch.no_grad():
            x=x[:,:,::2,:,:]
            n,c,t,h,w=x.shape
            x=x.transpose(1,2)#n,t,c,h,w
            x=x.reshape(n*t,c,h,w)
        x = self.encoder(x)
        _,c,h,w=x.shape
        x=x.view(n,t,c,h,w).transpose(1,2).contiguous()
        if self.use_attention:
            x=self.attention(x)
        return x


class RelationHead_pooling4(nn.Module):
    def __init__(self,cin=128):
        super(RelationHead_pooling4, self).__init__()
        self.net=nn.Sequential(
            C3D464.Conv3DBlock(cin,64,3),
            C3D464.Conv3DBlock(64,64,3,ceil_mode=True),
            nn.Flatten(),
            nn.Linear(256,8,bias=True),
            nn.ReLU(),
            nn.Linear(8,1,bias=True),
            nn.Sigmoid()
        )
    def forward(self,pair):
        return self.net(pair)

class RelationHead(nn.Module):
    def __init__(self,ways,cin=128,dynamic=False):
        super(RelationHead, self).__init__()
        self.dynamic=dynamic
        if not dynamic:
            conv=C3D464.Conv3DBlock
            self.net=nn.Sequential(
                conv(cin,64,3),
                conv(64,64,3,ceil_mode=True),
                nn.Flatten(),
                nn.Linear(768,8,bias=True),
                nn.ReLU(),
                nn.Linear(8,1,bias=True),
                nn.Sigmoid()
            )
        else:
            conv=DynConv3DBlock
            self.net=torch.nn.ModuleList([
                conv(cin,64,num_train_total_channels=64*ways),
                conv(64,64,num_train_total_channels=64*ways,ceil_mode=True),
                nn.Flatten(),
                DynamicLinear(768,8,num_train_total_channels=64*ways),
                nn.ReLU(),
                DynamicLinear(8,1,num_train_total_channels=64*ways),
                nn.Sigmoid()
            ])


    def forward(self,pair,support_mean=None):
        x=pair
        if not self.dynamic:
            x=self.net(x)
            return x
        else:
            for layer in self.net:
                if isinstance(layer,DynamicLinear) or isinstance(layer,DynConv3DBlock):
                    x=layer(support_mean,x)
                else:
                    x=layer(x)
            return x

def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None,epoch=None):
    global args
    if args.temporal_align:
        model,relation_head,temporal_align=model
    else:
        model,relation_head=model

    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    with torch.no_grad():
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    query = embeddings[query_indices]

    with torch.no_grad():
        support_labels=labels[support_indices].long()
        query_labels  =labels[query_indices].long()
        classes=torch.unique(support_labels)
        shape=support.shape
        sum_mask= (classes.view(-1,1)==support_labels.view(1,-1)).float()
    support_mean= ((sum_mask/shot) @ support.view(ways*shot,-1)).view(ways,*shape[-4:])
    #concat (s,q) pair-wise
    if args.temporal_align:
        pairs,support=temporal_align(support,query,sum_mask)
        support_mean= ((sum_mask) @ support.reshape(ways*shot,-1)).reshape(ways,*shape[-4:])
    else:
        pairs=pairwise_concat(support,query,sum_mask)
    #feed pairs to relation model
    shape=pairs.shape
    pairs=pairs.view(-1,*shape[2:])
    if not args.dynhead:
        logits=relation_head(pairs)#(N*K)*Q
    else:
        logits=relation_head(pairs,support_mean)#(N*K)*Q
    logits=logits.view(shape[:2],-1)

    #  convert label to one hot
    pair_labels = classes.view(-1,1)==query_labels.view(1,-1)
    #  change loss to mse
    loss=F.mse_loss(logits,pair_labels.float())
    # if epoch>=10:
    #     loss+=0.01*losses.align_pair_loss(pairs.view(*shape),pair_labels)
    acc = accuracy(logits, query_labels)
    return loss, acc


def save_checkpoint(state, is_best):
    if not os.path.exists('%s/%s' % (args.root_model, args.store_name)):
        os.mkdir('%s/%s' % (args.root_model, args.store_name))
    filename = '%s/%s/ckpt.pth' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        print('Saving best ckpt at epoch %d'%state['epoch'])
        shutil.copyfile(filename, filename.replace('.pth', '.best.pth'))

def train(args):
    logger = SummaryWriter(comment=args.comment)

    device = torch.device('cpu')
    if args.gpu and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(43)
        device = torch.device('cuda')
    if args.backbone2d:
        model=Convnet2D(st_attention=args.st_attention)
    else:
        model = Convnet(st_attention=args.st_attention)
    model.to(device)
    relation_head=RelationHead(ways=args.train_way,dynamic=args.dynhead)
    relation_head.to(device)
    if args.temporal_align:
        temporal_align=TemporalAlignMoudle(10,shot=args.shot)
        # d=torch.load('./ckpt/pretrain_ta/ckpt.pth','cpu')
        # temporal_align.load_state_dict(d['align_dict'])
        temporal_align.to(device)
        # del d

    # train_augmentation = get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)
    train_augmentation=Compose([GroupScale([128,128]),GroupRandomHorizontalFlip(is_flow=False)])
    normalize = IdentityTransform()

    num_class, args.train_list, args.val_list, args.root_path, prefix, anno_prefix = dataset_config.return_dataset(args.dataset,
                                                                                                     'RGB')
    args.test_list=args.train_list.replace('train','test')
    path_data=args.root_path
    num_segments=20

    train_dataset = TSNDataSet(path_data, args.train_list, num_segments=num_segments,
                   new_length=1,
                   modality='RGB',
                   image_tmpl=prefix,
                   transform=Compose([
                       train_augmentation,
                       StackBatch(roll=False),
                       To3DTorchFormatTensor(div=True)
                   ]),
                   dense_sample=False)
    train_dataset = l2l.data.MetaDataset(train_dataset,indices_to_labels=train_dataset.indices_to_labels)
    train_transforms = [
        NWays(train_dataset, args.train_way),
        KShots(train_dataset, args.train_query + args.shot),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)
    # train_loader = DataLoader(train_tasks,num_workers=1, pin_memory=True, shuffle=True)
    train_prefetcher=DataPrefetcher(train_tasks)

    valid_dataset = TSNDataSet(path_data, args.val_list, num_segments=num_segments,
                   new_length=1,
                   modality='RGB',
                   image_tmpl=prefix,
                   transform=Compose([
                       GroupScale([128,128]),
                       StackBatch(roll=False),
                       To3DTorchFormatTensor(div=True)
                   ]),
                   dense_sample=False)
    valid_dataset = l2l.data.MetaDataset(valid_dataset,indices_to_labels=valid_dataset.indices_to_labels)
    valid_transforms = [
        NWays(valid_dataset, args.test_way),
        KShots(valid_dataset, args.test_query + args.test_shot),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=100)
    valid_loader = DataLoader(valid_tasks,num_workers=1, pin_memory=True, shuffle=True)
    # valid_prefetcher=DataPrefetcher(valid_tasks)

    models=[model,relation_head]
    param_groups=[
        {'params':m.parameters()} for m in models
    ]
    if args.temporal_align:
        models.append(temporal_align)
        param_groups.append({'params':temporal_align.parameters(),'lr':1e-3})

    optimizer = torch.optim.Adam(param_groups,lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5)
    print('Start training')
    best_metric=0
    for epoch in range(1, args.max_epoch + 1):
        for m in models:
            m.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        batch=train_prefetcher.next()
        for i in range(100):
            # batch = next(iter(train_loader))

            loss, acc = fast_adapt(models,
                                   batch,
                                   args.train_way,
                                   args.shot,
                                   args.train_query,
                                   metric=pairwise_distances_logits,
                                   device=device,
                                   epoch=epoch)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc.item()

            batch=train_prefetcher.next()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1)%10==0:
                sys.stdout.write('\rWorking.... i=%d \033[K'%(i+1))

        lr_scheduler.step()

        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))
        logger.add_scalar('train_acc',n_acc/loss_ctr,global_step=epoch)
        ckpt={
            'epoch': epoch + 1,
            'arch': 'raw_relation',
            'state_dict': model.state_dict(),
            'head_dict': relation_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': -1,
        }
        if args.temporal_align:
            ckpt['align_dict']=temporal_align.state_dict()
        save_checkpoint(ckpt, False)

        for m in models:
            m.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        # batch=valid_prefetcher.next()
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
            # while batch is not None and batch[0] is not None:
                loss, acc = fast_adapt(models,
                                       batch,
                                       args.test_way,
                                       args.test_shot,
                                       args.test_query,
                                       metric=pairwise_distances_logits,
                                       device=device,
                                       epoch=epoch)

                loss_ctr += 1
                n_loss += loss.item()
                n_acc += acc.item()
                # batch=valid_prefetcher.next()

                if (i+1)%10==0:
                    sys.stdout.write('\rEvaling.... i=%d \033[K'%(i+1))

        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))
        logger.add_scalar('val_acc',n_acc/loss_ctr,global_step=epoch)
        metric_for_best=n_acc/loss_ctr
        is_best = metric_for_best > best_metric
        best_metric=max(metric_for_best,best_metric)
        ckpt={
            'epoch': epoch + 1,
            'arch': 'raw_relation',
            'state_dict': model.state_dict(),
            'head_dict': relation_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_metric,
        }
        if args.temporal_align:
            ckpt['align_dict']=temporal_align.state_dict()
        save_checkpoint(ckpt, is_best)

    torch.cuda.empty_cache()

def test(args):

    device = torch.device('cpu')
    if args.gpu and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(43)
        device = torch.device('cuda')

    if args.backbone2d:
        model=Convnet2D(st_attention=args.st_attention)
    else:
        model = Convnet(st_attention=args.st_attention)
    model.to(device)
    relation_head=RelationHead(ways=args.test_way,dynamic=args.dynhead)
    relation_head.to(device)
    if args.temporal_align:
        temporal_align=TemporalAlignMoudle(10,shot=args.test_shot)
        temporal_align.to(device)

    models=[model,relation_head]
    if args.temporal_align:
        models.append(temporal_align)

    num_class, args.train_list, args.val_list, args.root_path, prefix, anno_prefix = dataset_config.return_dataset(args.dataset,
                                                                                                     'RGB')
    args.test_list=args.train_list.replace('train','test')
    path_data=args.root_path
    num_segments=20

    test_dataset = TSNDataSet(path_data, args.test_list, num_segments=num_segments,
                   new_length=1,
                   modality='RGB',
                   image_tmpl=prefix,
                   transform=Compose([
                       GroupScale([128,128]),
                       StackBatch(roll=False),
                       To3DTorchFormatTensor(div=True)
                   ]),
                   dense_sample=False,
                   test_mode=True)

    test_dataset = l2l.data.MetaDataset(test_dataset,indices_to_labels=test_dataset.indices_to_labels)
    test_transforms = [
        NWays(test_dataset, args.test_way),
        KShots(test_dataset, args.test_query + args.test_shot),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=2000)
    test_loader = DataLoader(test_tasks,num_workers=1, pin_memory=True, shuffle=True)
    # test_prefetcher=DataPrefetcher(test_tasks)

    loss_ctr = 0
    n_acc = 0
    model.load_state_dict(torch.load('%s/%s/ckpt.pth' % (args.root_model, args.store_name))['state_dict'])
    relation_head.load_state_dict(torch.load('%s/%s/ckpt.pth' % (args.root_model, args.store_name))['head_dict'])
    if args.temporal_align:
        temporal_align.load_state_dict(torch.load('%s/%s/ckpt.pth' % (args.root_model, args.store_name))['align_dict'])

    for m in models:
        m.eval()

    for i, batch in enumerate(test_loader, 1):
        loss, acc = fast_adapt(models,
                               batch,
                               args.test_way,
                               args.test_shot,
                               args.test_query,
                               metric=pairwise_distances_logits,
                               device=device)
        loss_ctr += 1
        n_acc += acc.item()
        sys.stdout.write('\rbatch {}: {:.2f}({:.2f})  \033[K'.format(
            i, n_acc/loss_ctr * 100, acc * 100))
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=250)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--test-shot', type=int, default=1)
    parser.add_argument('--test-query', type=int, default=20)
    parser.add_argument('--train-query', type=int, default=5)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--train_list', type=str, default="")
    parser.add_argument('--val_list', type=str, default="")
    parser.add_argument('--test_list', type=str, default="")
    parser.add_argument('--root_path', type=str, default="")
    parser.add_argument('--root_model', type=str, default="")
    parser.add_argument('--store_name', type=str, default="")
    parser.add_argument('--istrain', type=ast.literal_eval, default=True)
    parser.add_argument('--temporal_align', type=ast.literal_eval, default=False)
    parser.add_argument('--dynhead', type=ast.literal_eval, default=False)
    parser.add_argument('--st_attention', type=ast.literal_eval, default=False)
    parser.add_argument('--backbone2d', type=ast.literal_eval, default=False)
    parser.add_argument('dataset', type=str)
    parser.add_argument('comment',type=str)
    global args
    args = parser.parse_args()
    print(args)
    if args.istrain:
        train(args)
        torch.cuda.empty_cache()
    with torch.no_grad():
        test(args)
