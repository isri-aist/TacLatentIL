from tac_latent_il.common import (
    MidTrainTactileMixin,
    PolicyTactileEncoderBase,
)
from tac_latent_il.policy.diffusion_policy_sparsh import (
    DiffusionPolicySparsh,
    TrainDiffusionPolicySparsh,
)


class TactileEncoderSparsh(PolicyTactileEncoderBase):
    PolicyClass = DiffusionPolicySparsh


class MidTrainSparsh(MidTrainTactileMixin, TrainDiffusionPolicySparsh):
    TactileEncoderClass = TactileEncoderSparsh
