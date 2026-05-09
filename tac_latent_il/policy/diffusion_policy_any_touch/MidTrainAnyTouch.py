from tac_latent_il.common import (
    MidTrainTactileMixin,
    PolicyTactileEncoderBase,
)

from .DiffusionPolicyAnyTouch import DiffusionPolicyAnyTouch
from .TrainDiffusionPolicyAnyTouch import TrainDiffusionPolicyAnyTouch


class TactileEncoderAnyTouch(PolicyTactileEncoderBase):
    PolicyClass = DiffusionPolicyAnyTouch


class MidTrainAnyTouch(MidTrainTactileMixin, TrainDiffusionPolicyAnyTouch):
    TactileEncoderClass = TactileEncoderAnyTouch
