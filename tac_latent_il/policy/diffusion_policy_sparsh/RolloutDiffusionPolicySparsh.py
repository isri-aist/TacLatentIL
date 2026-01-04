import cv2
import numpy as np
import torch
from robo_manip_baselines.common import (
    DataKey,
    denormalize_data,
)
from tactile_ssl.data.digit.utils import compute_diff, enhance_image

from tac_latent_il.common import RolloutDiffusionPolicyTactileBase

from .DiffusionPolicySparsh import DiffusionPolicySparsh


class RolloutDiffusionPolicySparsh(RolloutDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicySparsh

    def reset_variables(self):
        super().reset_variables()

        self.init_tactile_dict = {}
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

            tactile = self.process_tactile_cv2(tactile)

            if tactile_name not in self.init_tactile_dict:
                self.init_tactile_dict[tactile_name] = tactile.copy()

            tactile = compute_diff(
                tactile, self.init_tactile_dict[tactile_name], offset=0.5
            )
            tactile = enhance_image(tactile, brightness=280, contrast=200)

            tactile = np.ascontiguousarray(np.moveaxis(tactile, -1, -3))
            tactile = torch.from_numpy(tactile).float().div(255.0)

            tactiles.append(tactile)

        if self.tactiles_buf is None:
            self.tactiles_buf = [
                [
                    tactile.clone()
                    for _ in range(self.model_meta_info["data"]["n_obs_steps"] + 1)
                ]
                for tactile in tactiles
            ]
        else:
            for single_tactiles_buf, tactile in zip(self.tactiles_buf, tactiles):
                single_tactiles_buf.pop(0)
                single_tactiles_buf.append(tactile)

    def get_tactiles(self):
        tactiles = []
        for single_tactiles_buf in self.tactiles_buf:
            current_tactile_buf = torch.stack(single_tactiles_buf[1:], dim=0)
            prev_tactile_buf = torch.stack(single_tactiles_buf[:-1], dim=0)
            combined_tactile_buf = torch.cat(
                [current_tactile_buf, prev_tactile_buf], dim=1
            )
            tactiles.append(combined_tactile_buf[torch.newaxis].to(self.device))

        return tactiles

    def process_tactile_cv2(self, img):
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 0)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    def print_policy_info_for_tactile(self):
        tactile_kwargs = self.model_meta_info["policy"]["args"]["tactile_kwargs"]
        print(
            f"  - sparsh type: {tactile_kwargs['encoder_type']}, fine-tune sparsh: {tactile_kwargs['enable_finetune']}"
        )

    def override_policy_args_for_tactile(self, policy_args):
        tactile_kwargs = self.model_meta_info["policy"]["args"]["tactile_kwargs"]
        tactile_kwargs["enable_finetune"] = False
        tactile_kwargs["load_checkpoint"] = False
