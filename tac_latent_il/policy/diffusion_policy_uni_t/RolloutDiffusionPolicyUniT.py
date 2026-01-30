import torch
from torchvision.transforms import v2

from tac_latent_il.common import RolloutDiffusionPolicyTactileBase

from .DiffusionPolicyUniT import DiffusionPolicyUniT


class RolloutDiffusionPolicyUniT(RolloutDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicyUniT

    def setup_variables(self):
        super().setup_variables()

        tactile_transform_list = []

        tactile_transform_list.append(v2.ToDtype(torch.float32, scale=True))

        self.tactile_transforms = v2.Compose(tactile_transform_list)

    def pre_convert_data_for_tactile(self, tactile):
        return tactile

    def print_policy_info_for_tactile(self):
        tactile_kwargs = self.model_meta_info["policy"]["args"]["tactile_kwargs"]
        print(f"  - fine-tune unit: {tactile_kwargs['enable_finetune']}")
