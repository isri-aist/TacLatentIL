from tac_latent_il.common import EvalDiffusionPolicyTactileBase

from .DiffusionPolicyAnyTouch import DiffusionPolicyAnyTouch
from .DiffusionPolicyAnyTouchDataset import DiffusionPolicyAnyTouchDataset


class EvalDiffusionPolicyAnyTouch(EvalDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicyAnyTouch
    DatasetClass = DiffusionPolicyAnyTouchDataset

    def print_policy_info_for_tactile(self):
        tactile_kwargs = self.model_meta_info["policy"]["args"]["tactile_kwargs"]
        print(
            f"  - tactile sensor name: {tactile_kwargs['sensor_name']}, universal token: {tactile_kwargs['use_universal_sensor_token']}\n"
            f"  - sensor token: {tactile_kwargs['use_sensor_token']}, same patchemb: {tactile_kwargs['use_same_patchemb']}, pooling type: {tactile_kwargs['pooling_method']}"
        )
