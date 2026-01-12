import torch
from torchvision.transforms import v2

from tac_latent_il.common import DiffusionPolicyTactileBaseDataset


class DiffusionPolicyUniTDataset(DiffusionPolicyTactileBaseDataset):
    """Dataset to train diffusion policy unit."""

    def setup_tactile_transforms(self):
        tactile_transform_list = []

        tactile_transform_list.append(v2.ToDtype(torch.float32, scale=True))

        self.tactile_transforms = v2.Compose(tactile_transform_list)

    def pre_convert_data_for_tactile(self, tactiles):
        return tactiles
