import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.abstract_arch import AbsArchitecture

class TransposeHead(nn.Module):
    """
    Head block for different tasks. Upsamples twice and applies a 1x1 convolution.
    :param int dim: Input dimension.
    :param int out_ch: Output channels.
    """

    def __init__(self, dim, out_ch):
        super().__init__()
        self.upsample1 = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2)
        self.last_conv = nn.Conv2d(dim // 4, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.last_conv(x)

        return x
class get_tasks():
    def __init__(self, task_str):
        self.task_str = task_str
    def get_task_keys(self, task_str):
        task_keys = self.task_str.split('_')
        replacement_dict = {'Seg': 'segmentation', 'Dep': 'depth', 'Nor': 'normal'}
        task_keys = [replacement_dict[key] for key in task_keys]
        return task_keys


class TIT(AbsArchitecture):
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(TIT, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        self.encoder = self.encoder_class()
        self.task = self.kwargs['task']
        self.head = TransposeHead(dim=96,out_ch=3)
        self.head2 = TransposeHead(dim=96,out_ch=1)
        self.head3 = TransposeHead(dim=96,out_ch=14)
        self.get_tasks = get_tasks(self.task)
    
    def forward(self, inputs, task_name=None):
        out = {}
        s_rep = self.encoder(inputs)
        task_keys = self.get_tasks.get_task_keys(self.task)
        if 'segmentation' in task_keys:
            out['segmentation'] = self.head3(self.decoders['all'](s_rep, 'segmentation'))
        
        if 'depth' in task_keys:
            out['depth'] = self.head2(self.decoders['all'](s_rep, 'depth'))
        
        if 'normal' in task_keys:
            out['normal'] = self.head(self.decoders['all'](s_rep, 'normal'))
        return out

