from tac_latent_il.common import EvalDiffusionPolicyTactileBase

from .DiffusionPolicySparsh import DiffusionPolicySparsh
from .DiffusionPolicySparshDataset import DiffusionPolicySparshDataset


class EvalDiffusionPolicySparsh(EvalDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicySparsh
    DatasetClass = DiffusionPolicySparshDataset

    def print_policy_info_for_tactile(self):
        tactile_kwargs = self.model_meta_info["policy"]["args"]["tactile_kwargs"]
        print(f"  - sparsh encoder type: {tactile_kwargs['encoder_type']}")
