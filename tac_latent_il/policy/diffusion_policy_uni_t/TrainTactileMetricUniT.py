from tac_latent_il.common import (
    TactileEncoderBase,
    TactileMetricBase,
    TrainTactileMetricBase,
)
from tac_latent_il.policy.diffusion_policy_uni_t import (
    DiffusionPolicyUniT,
    TrainDiffusionPolicyUniT,
)


class TactileEncoderUniT(TactileEncoderBase):
    DiffusionPolicyClass = DiffusionPolicyUniT


class TactileMetricUniT(TactileMetricBase):
    TactileEncoderClass = TactileEncoderUniT


class TrainTactileMetricUniT(TrainTactileMetricBase, TrainDiffusionPolicyUniT):
    TactileMetricClass = TactileMetricUniT
