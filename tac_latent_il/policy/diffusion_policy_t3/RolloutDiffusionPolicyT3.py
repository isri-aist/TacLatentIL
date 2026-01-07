import cv2
import numpy as np
import torch
from robo_manip_baselines.common import (
    DataKey,
    denormalize_data,
)
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

    def reset_variables(self):
        super().reset_variables()

        self.tactiles_buf = None

    def infer_policy(self):
        # Update observation buffer
        if len(self.state_keys) > 0:
            self.update_state_buf()
        self.update_images_buf()
        self.update_tactiles_buf()

        # Infer
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            input_data = {}
            if len(self.state_keys) > 0:
                input_data["state"] = self.get_state()
            for camera_name, image in zip(self.camera_names, self.get_images()):
                input_data[DataKey.get_rgb_image_key(camera_name)] = image
            for tactile_name, tactile in zip(self.tactile_names, self.get_tactiles()):
                input_data[tactile_name] = tactile
            action = self.policy.predict_action(input_data)["action"][0]
            self.policy_action_buf = list(
                action.cpu().detach().numpy().astype(np.float64)
            )

        # Store action
        self.policy_action = denormalize_data(
            self.policy_action_buf.pop(0), self.model_meta_info["action"]
        )
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def update_tactiles_buf(self):
        tactiles = []
        for tactile_name in self.tactile_names:
            tactile = self.info["rgb_images"][tactile_name]

            tactile = cv2.resize(tactile, (224, 224))

            tactile = np.moveaxis(tactile, -1, -3)
            tactile = torch.tensor(tactile, dtype=torch.uint8)
            tactile = self.tactile_transforms(tactile)

            tactiles.append(tactile)

        if self.tactiles_buf is None:
            self.tactiles_buf = [
                [tactile for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for tactile in tactiles
            ]
        else:
            for single_tactiles_buf, tactile in zip(self.tactiles_buf, tactiles):
                single_tactiles_buf.pop(0)
                single_tactiles_buf.append(tactile)

    def get_tactiles(self):
        return [
            torch.stack(single_tactiles_buf, dim=0)[torch.newaxis].to(self.device)
            for single_tactiles_buf in self.tactiles_buf
        ]

    def print_policy_info_for_tactile(self):
        tactile_kwargs = self.model_meta_info["policy"]["args"]["tactile_kwargs"]
        print(
            f"  - t3 pooling type: {tactile_kwargs['pooling_type']}, fine-tune t3: {tactile_kwargs['enable_finetune']}"
        )

    def override_policy_args_for_tactile(self, policy_args):
        tactile_kwargs = self.model_meta_info["policy"]["args"]["tactile_kwargs"]
        tactile_kwargs["enable_finetune"] = False
        tactile_kwargs["load_checkpoint"] = False
