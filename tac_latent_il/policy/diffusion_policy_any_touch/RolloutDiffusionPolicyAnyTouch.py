import torch
from torchvision.transforms import v2

from tac_latent_il.common import RolloutDiffusionPolicyTactileBase

from .DiffusionPolicyAnyTouch import DiffusionPolicyAnyTouch


class RolloutDiffusionPolicyAnyTouch(RolloutDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicyAnyTouch

    def setup_variables(self):
        super().setup_variables()

        # Copied from:
        # https://github.com/GeWu-Lab/AnyTouch/blob/3ee6c09130baa9f47a14449b47fd151fbd17a163/dataloader/downstream_dataset.py#L110-L117
        self.tactile_transforms = v2.Compose(
            [
                # v2.RandomHorizontalFlip(p=0.5),
                # v2.RandomVerticalFlip(p=0.5),
                # v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def print_policy_info_for_tactile(self):
        tactile_kwargs = self.model_meta_info["policy"]["args"]["tactile_kwargs"]
        print(
            f"  - tactile sensor name: {tactile_kwargs['sensor_name']}, universal token: {tactile_kwargs['use_universal_sensor_token']}\n"
            f"  - sensor token: {tactile_kwargs['use_sensor_token']}, same patchemb: {tactile_kwargs['use_same_patchemb']}, pooling type: {tactile_kwargs['pooling_method']}"
        )
