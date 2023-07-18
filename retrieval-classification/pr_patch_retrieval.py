#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
This code is used to retrieval image on CUB and COCO dataset
"""

import argparse
import builtins
import math
import os
import sys
import random
import shutil
import time
import warnings
from functools import partial

from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
import torch.nn.functional as F
import moco.builder
import moco.loader
import vits


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:

parser.add_argument('--pretrained_model', default='cae', type=str,)
parser.add_argument('--dataset_name', default='cub200', type=str,)
parser.add_argument('--model_scale', default='base', type=str,)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()

    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)


@torch.no_grad()
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))
    pretrain_model = args.pretrained_model
    model_scale = args.model_scale

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]  # we use imagenet mean and std
    
    if args.arch.startswith('vit'):
        if pretrain_model == 'vit':
            if model_scale == 'base':
                model = vits.vit_base_patch16_224(pretrained=True)
            elif model_scale == 'large':
                model = vits.vit_large_patch16_224(pretrained=True)
            else:
                raise NotImplementedError()
            
        elif pretrain_model == 'moco':
            assert model_scale == 'base'
            model = moco.builder.MoCo_ViT(
                partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
                args.moco_dim, args.moco_mlp_dim, args.moco_t)
            args.resume = 'pretrained_models/vit-b-300ep.pth.tar'

        elif pretrain_model == 'cae':
            if model_scale == 'base':
                model = vits.VisionTransformerCAE()
                checkpoint = torch.load('pretrained_models/cae_base_1600ep.pth', map_location="cpu")
            elif model_scale == 'large':
                model = vits.VisionTransformerCAE(embed_dim=1024, depth=24, num_heads=16)
                checkpoint = torch.load('pretrained_models/cae_large_model_run2.pth', map_location="cpu")
            elif model_scale == 'base_dvae':
                model = vits.VisionTransformerCAE()
                checkpoint = torch.load('pretrained_models/cae_dvae_base_1600ep.pth', map_location="cpu")
            elif model_scale == 'large_dvae':
                model = vits.VisionTransformerCAE(embed_dim=1024, depth=24, num_heads=16)
                checkpoint = torch.load('pretrained_models/cae_dvae_large_1600ep.pth', map_location="cpu")
            else:
                raise NotImplementedError()
            checkpoint = {k[8:]:v for k, v in checkpoint['model'].items() if k.startswith('encoder.')}
            model.load_state_dict(checkpoint)

        elif pretrain_model == 'clip':
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
            model = torch.jit.load('pretrained_models/CLIP-ViT-B-16.pt', map_location="cpu").eval()
            model = vits.build_clip_model(model.state_dict())

        elif pretrain_model == 'mae':
            if model_scale == 'base':
                model = vits.mae_vit_base_patch16()
                checkpoint = torch.load('../../patch_embed_search/pretrained_models/mae_pretrain_vit_base.pth', map_location='cpu')
            elif model_scale == 'large':
                model = vits.mae_vit_large_patch16()
                checkpoint = torch.load('pretrained_models/mae_pretrain_vit_large.pth', map_location='cpu')
            else:
                raise NotImplementedError()
            model.load_state_dict(checkpoint['model'])

        elif pretrain_model == 'dino':
            assert model_scale in ['base', 'base_teacher']
            model = vits.vit_base_dino()
            checkpoint = torch.load('pretrained_models/dino_vitbase16_pretrain_full_checkpoint.pth', map_location='cpu')
            if 'teacher' not in model_scale:
                checkpoint = {(k[16:] if k.startswith('module.backbone.') else k[7:]): v for k, v in checkpoint['student'].items() if k.startswith('module.')}
            else:
                checkpoint = {(k[9:] if k.startswith('backbone.') else k): v for k, v in checkpoint['teacher'].items()}
            model.load_state_dict(checkpoint)

        elif pretrain_model == 'ibot':
            assert model_scale in ['base']
            model = vits.vit_base_ibot()
            checkpoint = torch.load('pretrained_models/ibot_checkpoint.pth', map_location='cpu')
            checkpoint = {(k[9:] if k.startswith('backbone.') else k): v for k, v in checkpoint['teacher'].items()}
            model.load_state_dict(checkpoint)

        elif pretrain_model == 'beit':
            assert model_scale == 'base'
            model = vits.BEiT()
            model.init_weights('pretrained_models/beit_base_standard_epoch800_checkpoint-799.pth')
        else:
            raise NotImplementedError()
        
    else:
        model = moco.builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError("need specify the gpu id")
    
    # optionally resume from a checkpoint. related to moco v3
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            checkpoint['state_dict'] = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    model.eval()
    dataset_name = args.dataset_name
    include_other_parts = False


    if dataset_name == 'cub200':
        image_root = '/home/ssd9/qjy/patch_embed_search/CUB_200_2011/'
        dataset_cls = moco.loader.PatchLoaderCUB200
        keypoint_cls_list = [11, 12, 9, 14]  # start from 1
        keypoint_cls_set_name = '11120914'
    elif dataset_name == 'coco':
        image_root = '/home/ssd2/data/mscoco/'
        dataset_cls = moco.loader.PatchLoaderCOCO
        keypoint_cls_list = [1, 11, 16]  # start from 1
        keypoint_cls_set_name = '011116'
    else:
        raise NotImplementedError()

    print(pretrain_model)
    patch_root = f'./feature/patches_{dataset_name}_{keypoint_cls_set_name}/'
    embedding_root = f'./feature/patch_embed_search_embeddings_{dataset_name}_{keypoint_cls_set_name}/'

    cache_path = os.path.join(embedding_root, f'embeddings_{pretrain_model}.pth')

    if model_scale != 'base':
        cache_path = cache_path.replace('.pth', f'_{model_scale}.pth')
    if include_other_parts:
        cache_path = cache_path.replace('.pth', '_iop.pth')
        patch_root = patch_root[:-1] + '_iop/'

    # feature extraction
    if not os.path.exists(cache_path):
        val_dataset = dataset_cls(
            data_root=image_root, patch_root=patch_root, mean=mean, std=std,
            keypoint_cls_list=keypoint_cls_list, include_other_parts=include_other_parts)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)

        total_state_dict = {
            'embedding': [],
            'projection': [],
            'cls_token': [],
            'patch_names': [],
        }

        for i, (patches, patch_paths, target) in enumerate(val_loader):
            patches = patches.cuda(args.gpu, non_blocking=True)
            embedding, projection, cls_token = model.extract(patches)

            embedding = embedding.cpu()
            projection = projection.cpu()
            cls_token = cls_token.cpu()

            total_state_dict['embedding'].extend(list(embedding))
            total_state_dict['projection'].extend(list(projection))
            total_state_dict['cls_token'].extend(list(cls_token))
            total_state_dict['patch_names'].extend(patch_paths)
            if i % 20 == 0:
                print(i, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                sys.stdout.flush()
    
        """
        To save the extracted features to file in cache_path
        """
        # os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # if total_state_dict['projection'][0].shape[0] > 4096:
        #     print('projection output has too many channels:', total_state_dict['projection'][0].shape[0], ', projection output will not be saved')
        #     tmp = {k: v for k, v in total_state_dict.items() if k != 'projection'}
        #     torch.save(tmp, cache_path)
        # else:
        #     torch.save(total_state_dict, cache_path)

    else:
        print("=> loading cache '{}'".format(cache_path))
        total_state_dict = torch.load(cache_path)
        print("=> loaded cache '{}'".format(cache_path))

    if pretrain_model in ('moco', 'clip', 'dino', 'ibot'):
        calc_acc(total_state_dict,  'projection')
    calc_acc(total_state_dict,  'embedding')
    calc_acc(total_state_dict, 'cls_token')


def mean(l):
    return sum(l) / len(l)


def del_annid(s):
    if '_' in s:
        suffix = s.split('_')[-1]
        if suffix.isdigit():
            return s[:-len(suffix)-1]
    return s


def calc_acc(total_state_dict, target_feature='embedding'):
    patch_names = total_state_dict['patch_names']
    feature_list = total_state_dict[target_feature]
    feature_list = torch.stack(feature_list).float()

    print(feature_list.size(0))
    split_flag = True
    split_batch = 64000
    if feature_list.size(0) < 64000 or feature_list.size(1) < 10000: ###
        feature_list = feature_list.cuda()
        split_flag = False

    # print(patch_names[0])
    part_names = [del_annid(os.path.basename(x)[:-4]) for x in patch_names]
    part_mask_dict = split_part(part_names)
    # print(part_names[0])
    # print(part_mask_dict[part_names[0]])
    # print(part_mask_dict[part_names[0]].shape)

    # sorted_idx_list = []
    batch_size = 16
    if feature_list.size(1) > 6400:
        batch_size = 1

    if split_flag:
        all_similarity_list = []
        for splited_feature_list in feature_list.split(split_batch):
            splited_feature_list = splited_feature_list.cuda()
            similarity_list = []
            for i, x in enumerate(feature_list.split(batch_size)):
                x = x.cuda()
                similarity = torch.nn.functional.cosine_similarity(x[:, None], splited_feature_list[None], 2)
                similarity_list.extend(list(similarity.cpu()))
                if i % (20*256/batch_size) == 0:
                    print('sim & sort:', i, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                    sys.stdout.flush()
            all_similarity_list.append(torch.stack(similarity_list))
        similarity_list = torch.cat(all_similarity_list, dim=1)
    else:
        similarity_list = []
        for i, x in enumerate(feature_list.split(batch_size)):
            similarity = torch.nn.functional.cosine_similarity(x[:, None], feature_list[None], 2)
            similarity_list.extend(list(similarity.cpu()))
            if i % (20*256/batch_size) == 0:
                print('sim & sort:', i, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                sys.stdout.flush()

    ap_list = []
    ap_dict = {}
    for i, similarity in enumerate(similarity_list):
        part_name = part_names[i]
        ap = average_precision_score(part_mask_dict[part_name].int().numpy(), similarity.numpy())
        ap_list.append(ap)
        if part_name not in ap_dict:
            ap_dict[part_name] = []
        ap_dict[part_name].append(ap)
        if i % (20*256) == 0:
            print('ap:', i, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            sys.stdout.flush()
    len_dict = [(k, len(v)) for k, v in ap_dict.items()]
    ap_dict = [(k, mean(v)) for k, v in ap_dict.items()]
    len_dict.sort(key=lambda x: x[0])
    ap_dict.sort(key=lambda x: x[0])
    print('AP:', mean(ap_list), '    mAP:', mean([x[1] for x in ap_dict]))
    print(ap_dict)
    print(len_dict)


def split_part(part_names):
    part_mask_dict = {}
    for part_name in set(part_names):
        part_mask_dict[part_name] = torch.BoolTensor([x == part_name for x in part_names])
    return part_mask_dict


if __name__ == '__main__':
    main()
