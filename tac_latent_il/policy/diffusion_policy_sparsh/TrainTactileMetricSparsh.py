from tac_latent_il.common import (
    TactileEncoderBase,
    TactileMetricBase,
    TrainTactileMetricBase,
)
from tac_latent_il.policy.diffusion_policy_sparsh import (
    DiffusionPolicySparsh,
    TrainDiffusionPolicySparsh,
)


class TactileEncoderSparsh(TactileEncoderBase):
    DiffusionPolicyClass = DiffusionPolicySparsh


class TactileMetricSparsh(TactileMetricBase):
    TactileEncoderClass = TactileEncoderSparsh


class TrainTactileMetricSparsh(TrainTactileMetricBase, TrainDiffusionPolicySparsh):
    TactileMetricClass = TactileMetricSparsh
