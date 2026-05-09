from tac_latent_il.common import (
    MidTrainTactileMixin,
    PolicyTactileEncoderBase,
)

from .DiffusionPolicyUniT import DiffusionPolicyUniT
from .TrainDiffusionPolicyUniT import TrainDiffusionPolicyUniT


class TactileEncoderUniT(PolicyTactileEncoderBase):
    PolicyClass = DiffusionPolicyUniT


class MidTrainUniT(MidTrainTactileMixin, TrainDiffusionPolicyUniT):
    TactileEncoderClass = TactileEncoderUniT
