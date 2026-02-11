from tac_latent_il.common import (
    EvalDiffusionPolicyTactileBase,
)

from .DiffusionPolicyUniT import DiffusionPolicyUniT
from .DiffusionPolicyUniTDataset import DiffusionPolicyUniTDataset


class EvalDiffusionPolicyUniT(EvalDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicyUniT
    DatasetClass = DiffusionPolicyUniTDataset

    def print_policy_info_for_tactile(self):
        pass
