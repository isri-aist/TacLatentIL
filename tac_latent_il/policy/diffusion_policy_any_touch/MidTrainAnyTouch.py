from tac_latent_il.common import (
    MidTrainTactileMixin,
    PolicyTactileEncoderBase,
)
from tac_latent_il.policy.diffusion_policy_any_touch import (
    DiffusionPolicyAnyTouch,
    TrainDiffusionPolicyAnyTouch,
)


class TactileEncoderAnyTouch(PolicyTactileEncoderBase):
    PolicyClass = DiffusionPolicyAnyTouch


class MidTrainAnyTouch(MidTrainTactileMixin, TrainDiffusionPolicyAnyTouch):
    TactileEncoderClass = TactileEncoderAnyTouch
