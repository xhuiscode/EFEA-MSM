import torch
import torch.nn as nn
import torch.nn.functional as F
#from __future__ import division

import math
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from .fusion import DirectAddFuse, ASKCFuse, ResGlobLocaforGlobLocaChaFuse


def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


class ResGlobLocaChaFuse(HybridBlock):
    def __init__(self, channels=64):
        super(ResGlobLocaChaFuse, self).__init__()

        with self.name_scope():

            self.local_att = nn.HybridSequential(prefix='local_att')
            self.local_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.local_att.add(nn.BatchNorm())

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sig = nn.Activation('sigmoid')

    def hybrid_forward(self, F, x, residual):

        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = F.broadcast_add(xl, xg)
        wei = self.sig(xlg)

        xo = 2 * F.broadcast_mul(x, wei) + 2 * F.broadcast_mul(residual, 1-wei)

        return xo