from tac_latent_il.common import (
    TactileEncoderBase,
    TactileMetricBase,
    TrainTactileMetricBase,
)
from tac_latent_il.policy.diffusion_policy_any_touch import (
    DiffusionPolicyAnyTouch,
    TrainDiffusionPolicyAnyTouch,
)


class TactileEncoderAnyTouch(TactileEncoderBase):
    DiffusionPolicyClass = DiffusionPolicyAnyTouch


class TactileMetricAnyTouch(TactileMetricBase):
    TactileEncoderClass = TactileEncoderAnyTouch


class TrainTactileMetricAnyTouch(TrainTactileMetricBase, TrainDiffusionPolicyAnyTouch):
    TactileMetricClass = TactileMetricAnyTouch
