import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .decode_head import BaseDecodeHead


class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels, conv_cfg, norm_cfg, act_cfg, out_channels=None, scale=1):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = ConvModule(self.in_channels, self.key_channels, 1, stride=1, padding=0, bias=True, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.f_query = self.f_key
        self.f_value = ConvModule(self.in_channels, self.value_channels, kernel_size=1, stride=1, padding=0, bias=True, act_cfg=None)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)


    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = resize(
                context,
                size=(h, w),
                mode='bilinear',
                align_corners=True)
                
        return context



class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, conv_cfg, norm_cfg, act_cfg, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                    key_channels,
                                                    value_channels,
                                                    conv_cfg, norm_cfg, act_cfg, 
                                                    out_channels,
                                                    scale)

class BaseOC_Context_Module(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """
    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, conv_cfg, norm_cfg, act_cfg, sizes=([1])):
        super(BaseOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels, key_channels, conv_cfg, norm_cfg, act_cfg, value_channels, size) for size in sizes])
        self.conv_bn_dropout = ConvModule(in_channels, out_channels, 1, \
        padding=0, conv_cfg=conv_cfg, norm_cfg= norm_cfg, act_cfg=act_cfg) 

    def _make_stage(self, in_channels, output_channels, key_channels, conv_cfg, norm_cfg, act_cfg, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    conv_cfg, norm_cfg, act_cfg,
                                    output_channels, 
                                    size)
        
    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output


@HEADS.register_module()
class ASPOCHead(BaseDecodeHead):
    def __init__(self, out_features=256, dilations=(12, 24, 36), **kwargs):
        super(ASPOCHead, self).__init__(**kwargs)
        self.in_channels 
        self.context = nn.Sequential(ConvModule(self.in_channels , out_features, kernel_size=3, padding=1, dilation=1, bias=True, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
                                   BaseOC_Context_Module(in_channels=out_features, out_channels=out_features, key_channels=out_features//2, value_channels=out_features, 
                                    dropout=0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, sizes=([2])))
                                    
        self.conv2 = ConvModule(self.in_channels, out_features, kernel_size=1, padding=0, dilation=1, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.conv3 = ConvModule(self.in_channels, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
                                
        self.conv4 = ConvModule(self.in_channels, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
                       
        self.conv5 = ConvModule(self.in_channels, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
                       
        self.conv_bn_dropout = nn.Sequential(
            ConvModule(out_features * 5, out_features * 2, kernel_size=1, padding=0, dilation=1, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            nn.Dropout2d(0.1)
            )

    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        _, _, h, w = x.size()

        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        output = self.conv_bn_dropout(out)
        output = self.cls_seg(output)

        return output