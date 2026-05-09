from tac_latent_il.common import (
    MidTrainTactileMixin,
    PolicyTactileEncoderBase,
)

from .DiffusionPolicyT3 import DiffusionPolicyT3
from .TrainDiffusionPolicyT3 import TrainDiffusionPolicyT3


class TactileEncoderT3(PolicyTactileEncoderBase):
    PolicyClass = DiffusionPolicyT3


class MidTrainT3(MidTrainTactileMixin, TrainDiffusionPolicyT3):
    TactileEncoderClass = TactileEncoderT3
