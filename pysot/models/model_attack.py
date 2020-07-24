# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck

torch.manual_seed(0)

class ModelAttacker(nn.Module):
    def __init__(self):
        super(ModelAttacker, self).__init__()

        self.adv = nn.Parameter(torch.rand([1, 3, 127, 127], requires_grad=True, dtype=torch.float))

    def perturb(self, img, epsilon):
        # x = (self.adv - self.adv.min()) / (self.adv.max() - self.adv.min())
        # pdb.set_trace()
        x = img + epsilon * (2 * self.adv - 1)
        x[x != x] = img[x != x]
        x[x > 255] = 255
        x[x < 0] = 0
        return x

    def template(self, z, tracker, epsilon=0):
        if epsilon != 0:
            z = self.perturb(z, epsilon)
            zf = tracker.backbone(z)
        else:
            zf = tracker.backbone(z)

        if cfg.ADJUST.ADJUST:
            zf = tracker.neck(zf)

        self.zf = zf

        return z

    def track(self, x, tracker, iter=0):

        if iter == 0:
            xf = tracker.backbone(x)
            if cfg.MASK.MASK:
                self.xf = xf[:-1]
                xf = xf[-1]
            if cfg.ADJUST.ADJUST:
                xf = tracker.neck(xf)

            self.xf = xf

        cls, loc = tracker.rpn_head(self.zf, self.xf)

        return {
                'cls': cls,
                'loc': loc
               }