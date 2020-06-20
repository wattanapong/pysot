import torch
import torch.nn as nn
import numpy as np
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss


class Steath(ModelBuilder):
    def __init__(self, dim):
        super(Steath, self).__init__()
        self.dx = nn.Parameter(torch.rand(dim, requires_grad=True, dtype=torch.float))

    def forward(self, data, epsilon):
        """ only used in training
                """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)

        search = search + self.dx

        xf = self.backbone(search)

        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, -label_loc, label_loc_weight)
        # import pdb
        # pdb.set_trace()
        outputs = {}
        outputs['search'] = search
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        return outputs


class WeightClipper(object):

    def __init__(self, limit=5):
        self.limit = limit

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-self.limit, self.limit)
