#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This code is used to visualize the image retrieval results on imagenet1k.
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

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
import PIL
from PIL import Image, ImageDraw

import moco.builder
import moco.loader
import moco.optimizer

import vits


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
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
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

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
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')

parser.add_argument('--pretrained_model', default='cae', type=str,)
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

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


@torch.no_grad()
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))

    pretrain_model = args.pretrained_model
    model_scale = args.model_scale
    num_samples_per_cls = 50

    id_list = [
        # # # 
        # ('00039553', 32),  # cat beard 397520
        # ('00024863', 18),  # dog mouth 104290
        # ('00013660', 4),  # bird 934042
        ('00013660', 19),  # bird head
        # ('00020955', 4),  # head
        # ('00005774', 38),  # wheel
        ('00018569', 39),
        ('00001412', 16),
    ]
    # id_list = []

    all_patch_search = -1
    # all_patch_search = 0

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

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
            elif model_scale == 'rgb':
                model = vits.mae_vit_base_patch16()
                checkpoint = torch.load('pretrained_models/official_cotr_base_v1.2_300epoch_8self_w0.1_dp0.1_checkpoint-299.pth', map_location="cpu")
            elif model_scale == 'rgb_randommask_align0_300ep':
                model = vits.mae_vit_base_patch16()
                checkpoint = torch.load('pretrained_models/cae_vcxk_r8d8_base_300ep_alignw0.pth', map_location="cpu")
            elif model_scale == 'rgb_blockmask0.4_align0_300ep':
                model = vits.mae_vit_base_patch16()
                checkpoint = torch.load('pretrained_models/cae_vcxk_r8d8_base_300ep_alignw0_blockmask.pth', map_location="cpu")
            elif model_scale == 'rgb_blockmask0.5_align0.1_300ep':
                model = vits.mae_vit_base_patch16()
                checkpoint = torch.load('pretrained_models/cae_vcxk_r8d8_base_300ep_alignw0.1_blockmask0.5.pth', map_location="cpu")
            elif model_scale == 'mae_base_300ep':
                model = vits.mae_vit_base_patch16()
                checkpoint = torch.load('pretrained_models/mae_base_300ep.pth', map_location="cpu")
            elif model_scale == 'mae_base_300ep_blockmask':
                model = vits.mae_vit_base_patch16()
                checkpoint = torch.load('pretrained_models/mae_base_300ep_blockmask.pth', map_location="cpu")
            elif model_scale == '300ep':
                model = vits.VisionTransformerCAE()
                checkpoint = torch.load('pretrained_models/cae_base_300ep.pth', map_location="cpu")
            elif model_scale == '800ep':
                model = vits.VisionTransformerCAE()
                checkpoint = torch.load('pretrained_models/cae_base_800ep.pth', map_location="cpu")
            else:
                raise NotImplementedError()
            if 'rgb' in model_scale or 'mae' in model_scale:
                checkpoint = {k:v for k, v in checkpoint['model'].items() if not k.startswith('teacher') and not k.startswith('decoder') and k != 'mask_token' and not k.startswith('alignment') and not k.startswith('regressor')}
                model.load_state_dict(checkpoint)
            else:
                checkpoint = {k[8:]:v for k, v in checkpoint['model'].items() if k.startswith('encoder.')}
                model.load_state_dict(checkpoint)
            print('model loaded')
        elif pretrain_model == 'clip':
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
            model = torch.jit.load('pretrained_models/CLIP-ViT-B-16.pt', map_location="cpu").eval()
            model = vits.build_clip_model(model.state_dict())
        elif pretrain_model == 'mae':
            if model_scale == 'base':
                model = vits.mae_vit_base_patch16()
                checkpoint = torch.load('pretrained_models/mae_pretrain_vit_base.pth', map_location='cpu')
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
            # print(checkpoint['student'].keys())
            if 'teacher' not in model_scale:
                raise NotImplementedError()
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
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # print(model) # print model after SyncBatchNorm

    # optionally resume from a checkpoint
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

    image_root = '/home/ssd6/cxk/CAE/imagenet-samples_10K/'
    patch_root = '/home/ssd4/qjy/patch_embed_search/patches/'
    # embedding_root = '/home/ssd9/qjy/patch_embed_search_embeddings/'
    embedding_root = '/home/ssd7/qjy/patch_embed_search_embeddings/'

    zip_root = f'./patches_zip/{pretrain_model}'
    cache_path = os.path.join(embedding_root, f'embeddings_{pretrain_model}.pth')

    if num_samples_per_cls != 5:
        zip_root = zip_root + f'_{num_samples_per_cls}'
        cache_path = cache_path.replace('.pth', f'_{num_samples_per_cls}.pth')

    if model_scale != 'base':
        zip_root = zip_root + f'_{model_scale}'
        cache_path = cache_path.replace('.pth', f'_{model_scale}.pth')

    if all_patch_search >= 0:
        zip_root = os.path.join('/home/ssd4/qjy/patch_embed_search', zip_root + f'_all{all_patch_search}')
        id_list = [('{:08d}'.format(i), all_patch_search) for i in range(25000)]


    os.makedirs(zip_root, exist_ok=True)
    split_ratio = 4
    patch_num = ((split_ratio-1)*2 + 1)**2

    # feature extraction
    if not os.path.exists(cache_path):
        val_dataset = moco.loader.PatchLoader(
            image_root=image_root, patch_root=patch_root, split_ratio=split_ratio, num_samples_per_cls=num_samples_per_cls,
            image_root_all=args.data, mean=mean, std=std)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

        total_state_dict = {
            'embedding': [],
            'projection': [],
            'cls_token': [],
            'patch_names': [],
        }

        for i, (patches, img_name, patch_names) in enumerate(val_loader):
            assert patches.size(0) == 1, patches.size()
            img_name = img_name[0]
            patches = patches[0]
            image_level_cache_path = os.path.join(val_dataset.patch_root, img_name, os.path.basename(cache_path))
            if not os.path.exists(image_level_cache_path):
                patches = patches.cuda(args.gpu, non_blocking=True)
                embedding, projection, cls_token = model.extract(patches)

                embedding = embedding.cpu()
                projection = projection.cpu()
                cls_token = cls_token.cpu()

            else:
                state = torch.load(image_level_cache_path)
                embedding, projection, patch_names = state['embedding'], state['projection'], state['patch_names']

            total_state_dict['embedding'].extend(list(embedding))
            # total_state_dict['projection'].extend(list(projection))
            total_state_dict['cls_token'].extend(list(cls_token))
            assert len(patch_names) == len(patches) and len(patch_names[0]) == 1, (patch_names)
            total_state_dict['patch_names'].extend([os.path.join(patch_root, img_name, x[0]) for x in patch_names])
            if i % 20 == 0:
                print(i, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                sys.stdout.flush()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # torch.save(total_state_dict, cache_path)
    else:
        print("=> loading cache '{}'".format(cache_path))
        total_state_dict = torch.load(cache_path)
        print("=> loaded cache '{}'".format(cache_path))

    # embedding retrieval
    for id_tuple in id_list:
        for image_id, name in enumerate(total_state_dict['patch_names']):
            if id_tuple[0] in name:
                query_id = image_id + id_tuple[1]
                try:
                    retrieval([total_state_dict], query_id, zip_root, target_feature='embedding', patch_root=patch_root)
                    retrieval([total_state_dict], query_id, zip_root, target_feature='cls_token', patch_root=patch_root)
                except PIL.UnidentifiedImageError as e:
                    print(e)
                break


def retrieval(total_state_dicts, query_id, zip_root, target_feature='embedding', patch_root=None):
    imgs = [_retrieval(x, query_id, target_feature, patch_root=patch_root) for x in total_state_dicts]
    if len(imgs) > 1:
        img = Image.new("RGB", (imgs[0].width*len(imgs) + (len(imgs)-1)*20, imgs[0].height), "white")
        for i in range(len(imgs)):
            img.paste(imgs[i], ((imgs[0].width+20) * i, 0))
    else:
        img = imgs[0]

    target_path = os.path.join(zip_root, f'{query_id}_{target_feature}.jpg')
    img.save(target_path)
    print(target_path)
    print('#######################################')
    sys.stdout.flush()


def _retrieval(total_state_dict, query_id, target_feature='embedding', patch_root=None):
    patch_names = total_state_dict['patch_names']
    feature_list = total_state_dict[target_feature]
    assert len(feature_list) == len(patch_names), (len(feature_list), len(patch_names))
    query_name = os.path.dirname(patch_names[query_id])
    feature_list = torch.stack(feature_list).float()
    query = feature_list[query_id:query_id+1]

    similarity = torch.nn.functional.cosine_similarity(query, feature_list, 1)
    # similarity = -torch.nn.functional.pairwise_distance(query.expand(feature_list.size(0), -1), feature_list, 2)
    sims, neighbors = torch.sort(similarity, dim=0, descending=True)
    assert sims.dim() == 1, sims.size()

    w, h = 10, 10
    # w, h = 20, 20
    searched_ids = neighbors[:w*h+1].tolist()
    searched_patch_names = [patch_names[i] for i in searched_ids]

    img = Image.new("RGB", (56*w + 5*(w-1), 56*h + 5*(h-1)), "white")
    if patch_root is not None:
        query_name = os.path.join(patch_root, '..', query_name)
    img.paste(Image.open(os.path.join(query_name, 'origin.jpg')), (0, 0))
    drawer = ImageDraw.Draw(img)
    query_pos = [int(x)//4 for x in searched_patch_names[0].split('/')[-1].split('.')[0].split('_')]
    drawer.rectangle([(query_pos[2], query_pos[0]), (query_pos[3], query_pos[1])], outline="red", width=5)

    idx = 0
    searched_patch_names = searched_patch_names[1:]
    for i in range(h):
        for j in range(w):
            if i < 4 and j < 4: continue
            if patch_root is not None:
                searched_patch_names[idx] = os.path.join(patch_root, '..', searched_patch_names[idx])
            img.paste(Image.open(searched_patch_names[idx]).resize((56, 56)), (j*(56+5), i*(56+5)))
            idx += 1
    print([sims[1:pre].mean().item() for pre in [2, 5, 10, 50, 100]])
    return img

if __name__ == '__main__':
    main()
