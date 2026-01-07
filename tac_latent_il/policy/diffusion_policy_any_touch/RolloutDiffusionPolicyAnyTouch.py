import cv2
import numpy as np
import torch
from robo_manip_baselines.common import (
    DataKey,
    denormalize_data,
)
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
            f"  - tactile sensor name: {tactile_kwargs['sensor_name']}, fine-tune anytouch: {tactile_kwargs['enable_finetune']}"
            f"  - universal token: {tactile_kwargs['use_universal_sensor_token']}, sensor token: {tactile_kwargs['use_sensor_token']}, same patchemb: {tactile_kwargs['use_same_patchemb']}, pooling type: {tactile_kwargs['pooling_method']}"
        )

    def override_policy_args_for_tactile(self, policy_args):
        tactile_kwargs = self.model_meta_info["policy"]["args"]["tactile_kwargs"]
        tactile_kwargs["enable_finetune"] = False
        tactile_kwargs["load_checkpoint"] = False
