import torch
from torchvision.transforms import v2

from tac_latent_il.common import RolloutDiffusionPolicyTactileBase

from .DiffusionPolicyT3 import DiffusionPolicyT3


class RolloutDiffusionPolicyT3(RolloutDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicyT3

    def setup_variables(self):
        super().setup_variables()

        tactile_transform_list = []

        tactile_transform_list.append(v2.ToDtype(torch.float32, scale=True))

        # mean and std are copied from
        # https://github.com/alanzjl/t3/blob/341177f232df3b824a5246b0d1855cd9e4d2cf29/configs/datasets/eval_ds_cls_mini.yaml#L17-L19
        tactile_transform_list.append(
            v2.Normalize(
                mean=[0.02172, 0.04152, 0.04152], std=[0.10158, 0.14423, 0.12185]
            )
        )

        self.tactile_transforms = v2.Compose(tactile_transform_list)

    def print_policy_info_for_tactile(self):
        tactile_kwargs = self.model_meta_info["policy"]["args"]["tactile_kwargs"]
        print(
            f"  - t3 pooling type: {tactile_kwargs['pooling_type']}, fine-tune t3: {tactile_kwargs['enable_finetune']}"
        )
