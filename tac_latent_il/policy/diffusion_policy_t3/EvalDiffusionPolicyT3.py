from tac_latent_il.common import EvalDiffusionPolicyTactileBase

from .DiffusionPolicyT3 import DiffusionPolicyT3
from .DiffusionPolicyT3Dataset import DiffusionPolicyT3Dataset


class EvalDiffusionPolicyT3(EvalDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicyT3
    DatasetClass = DiffusionPolicyT3Dataset

    def print_policy_info_for_tactile(self):
        tactile_kwargs = self.model_meta_info["policy"]["args"]["tactile_kwargs"]
        print(f"  - t3 pooling type: {tactile_kwargs['pooling_type']}")
