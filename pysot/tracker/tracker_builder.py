# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
# from pysot.tracker.siamrpn_attack_tracker import SiamRPNAttackTracker
from pysot.tracker.siamrpn_attack_oneshot import SiamRPNAttackOneShot
from pysot.tracker.siamrpn_train_oneshot import SiamRPNTrainOneShot
from pysot.tracker.siamrpn_attack_oneshot_search import SiamRPNAttackSearch
from pysot.tracker.siamrpn_attack_2pass import SiamRPNAttack2Pass
from pysot.tracker.siammask_tracker import SiamMaskTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker

TRACKS = {
    'SiamRPNTracker': SiamRPNTracker,
    'SiamMaskTracker': SiamMaskTracker,
    'SiamRPNLTTracker': SiamRPNLTTracker,
    # 'SiamRPNAttackTracker': SiamRPNAttackTracker,
    'SiamRPNAttackOneShot': SiamRPNAttackOneShot,
    'SiamRPNAttack2Pass': SiamRPNAttack2Pass,
    'SiamRPNAttackSearch': SiamRPNAttackSearch,
    'SiamRPNTrainOneShot': SiamRPNTrainOneShot
}


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
