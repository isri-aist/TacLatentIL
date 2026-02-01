from tac_latent_il.common import (
    TactileEncoderBase,
    TactileMetricBase,
    TrainTactileMetricBase,
)
from tac_latent_il.policy.diffusion_policy_t3 import (
    DiffusionPolicyT3,
    TrainDiffusionPolicyT3,
)


class TactileEncoderT3(TactileEncoderBase):
    DiffusionPolicyClass = DiffusionPolicyT3


class TactileMetricT3(TactileMetricBase):
    TactileEncoderClass = TactileEncoderT3


class TrainTactileMetricT3(TrainTactileMetricBase, TrainDiffusionPolicyT3):
    TactileMetricClass = TactileMetricT3
