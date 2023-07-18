""" DeiT - Data-efficient Image Transformers
DeiT model defs and weights from https://github.com/facebookresearch/deit, original copyright below
paper: `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
paper: `DeiT III: Revenge of the ViT` - https://arxiv.org/abs/2204.07118
Modifications copyright 2021, Ross Wightman
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from functools import partial

import torch
from torch import nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer, trunc_normal_

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES


@BACKBONES.register_module()
class DEiT(VisionTransformer):
    """ Vision Transformer w/ Distillation Token and Head
    Distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """
    def __init__(self, *args, **kwargs):

        weight_init = kwargs.pop('weight_init', '')

        ## parameter set fot segmentation
        self.out_indices = kwargs.pop('out_indices', '') 
        self.frozen_stages = kwargs.pop('frozen_stages', '')

        super().__init__(*args, **kwargs, weight_init='skip')
        assert self.global_pool in ('token',)
        self.num_prefix_tokens = 2
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed|dist_token',
            blocks=[
                (r'^blocks\.(\d+)', None),
                (r'^norm', (99999,))]  # final norm w/ last block
        )

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()

    @torch.jit.ignore
    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}  

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)


    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat((
            self.cls_token.expand(x.shape[0], -1, -1),
            self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_indices:
                features.append(x)    
        return tuple(features)


    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False
            self.pos_drop.eval()

            for block in self.blocks[:self.frozen_stages]:
                block.eval()
                for _, p in block.named_parameters():
                    p.requires_grad = False


    def forward(self, x):
        x = self.forward_features(x)
        return x

