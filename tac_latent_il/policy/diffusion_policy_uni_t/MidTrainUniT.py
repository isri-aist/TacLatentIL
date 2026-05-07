from tac_latent_il.common import (
    MidTrainTactileMixin,
    PolicyTactileEncoderBase,
)
from tac_latent_il.policy.diffusion_policy_uni_t import (
    DiffusionPolicyUniT,
    TrainDiffusionPolicyUniT,
)


class TactileEncoderUniT(PolicyTactileEncoderBase):
    PolicyClass = DiffusionPolicyUniT


class MidTrainUniT(MidTrainTactileMixin, TrainDiffusionPolicyUniT):
    TactileEncoderClass = TactileEncoderUniT
