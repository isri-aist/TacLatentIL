from tac_latent_il.common import (
    MidTrainTactileMixin,
    PolicyTactileEncoderBase,
)

from .DiffusionPolicySparsh import DiffusionPolicySparsh
from .TrainDiffusionPolicySparsh import TrainDiffusionPolicySparsh


class TactileEncoderSparsh(PolicyTactileEncoderBase):
    PolicyClass = DiffusionPolicySparsh


class MidTrainSparsh(MidTrainTactileMixin, TrainDiffusionPolicySparsh):
    TactileEncoderClass = TactileEncoderSparsh
