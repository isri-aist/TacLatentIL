from tac_latent_il.common import (
    MidTrainTactileMixin,
    PolicyTactileEncoderBase,
)
from tac_latent_il.policy.diffusion_policy_t3 import (
    DiffusionPolicyT3,
    TrainDiffusionPolicyT3,
)


class TactileEncoderT3(PolicyTactileEncoderBase):
    PolicyClass = DiffusionPolicyT3


class MidTrainT3(MidTrainTactileMixin, TrainDiffusionPolicyT3):
    TactileEncoderClass = TactileEncoderT3
