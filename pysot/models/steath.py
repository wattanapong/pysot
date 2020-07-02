import torch
import torch.nn as nn
import numpy as np
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
import torch.nn.functional as F

class Steath(ModelBuilder):
    def __init__(self, dim):
        super(Steath, self).__init__()
        # self.dx = nn.Parameter(torch.rand(dim, requires_grad=True, dtype=torch.float))
        # self.dx = nn.Parameter(torch.rand([1, 10, 25, 25], requires_grad=True, dtype=torch.float))
        self.cn = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1,  bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
                )

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def forward(self, data, epsilon):
        """ only used in training
                """
        template = data['template']
        search = data['search']
        label_cls = data['label_cls']
        label_loc = data['label_loc']
        label_loc_weight = data['label_loc_weight']

        # get feature
        zf = self.backbone(template)

        search = search + 0.1*self.cn(search)

        xf = self.backbone(search)

        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        import pdb
        pdb.set_trace()
        score = self._convert_score(cls)

        idx = np.argwhere(score < 0.5)
        lt = score[idx]
        idx05 = idx[np.argmax(lt)]
        lt05 = score[idx05]

        # get loss
        cls = self.log_softmax(cls)

        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {'search': search, 'cls_loss': cls_loss}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss
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
