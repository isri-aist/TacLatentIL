import torch
from torchvision.transforms import v2

from tac_latent_il.common import DiffusionPolicyTactileBaseDataset


class DiffusionPolicyT3Dataset(DiffusionPolicyTactileBaseDataset):
    """Dataset to train diffusion policy t3."""

    def setup_tactile_transforms(self):
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
