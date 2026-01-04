import os
import sys
from abc import ABC, abstractmethod

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../third_party/diffusion_policy")
)
from robo_manip_baselines.common import (
    RolloutBase,
    normalize_data,
)


class RolloutDiffusionPolicyTactileBase(RolloutBase, ABC):
    PolicyClass = None  # must be overridden

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.tactile_names = self.model_meta_info["tactile"]["tactile_names"]

    def setup_policy(self):
        # For backward compatibility
        if "backbone" not in self.model_meta_info["policy"]:
            self.model_meta_info["policy"]["backbone"] = "cnn"
        if "scheduler" not in self.model_meta_info["policy"]:
            self.model_meta_info["policy"]["scheduler"] = "ddpm"

        # Print policy information
        self.print_policy_info()
        print(
            f"  - use ema: {self.model_meta_info['policy']['use_ema']}, backbone: {self.model_meta_info['policy']['backbone']}, scheduler: {self.model_meta_info['policy']['scheduler']}\n"
            f"  - horizon: {self.model_meta_info['data']['horizon']}, obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}\n"
            f"  - image size: {self.model_meta_info['data']['image_size']}, image crop size: {self.model_meta_info['data']['image_crop_size']}\n"
            f"  - tactile names: {self.tactile_names}"
        )
        self.print_policy_info_for_tactile()

        # Override policy args for rollout
        self.override_policy_args_for_tactile(self.model_meta_info["policy"]["args"])

        # Construct scheduler
        if self.model_meta_info["policy"]["scheduler"] == "ddpm":
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

            noise_scheduler = DDPMScheduler(
                **self.model_meta_info["policy"]["noise_scheduler_args"]
            )
        elif self.model_meta_info["policy"]["scheduler"] == "ddim":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            noise_scheduler = DDIMScheduler(
                **self.model_meta_info["policy"]["noise_scheduler_args"]
            )
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid scheduler: {self.model_meta_info['policy']['scheduler']}"
            )

        # Construct policy
        if self.model_meta_info["policy"]["backbone"] == "cnn":
            pass
        elif self.model_meta_info["policy"]["backbone"] == "transformer":
            raise NotImplementedError(
                f"[{self.__class__.__name__}] The transformer backbone is not supported."
            )
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid backbone: {self.model_meta_info['policy']['backbone']}"
            )
        self.policy = self.PolicyClass(
            noise_scheduler=noise_scheduler,
            **self.model_meta_info["policy"]["args"],
        )

        # Load checkpoint
        self.load_ckpt()

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            len(self.camera_names) + len(self.tactile_names),
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )
        super().setup_plot(fig_ax)

    def reset_variables(self):
        super().reset_variables()

        self.state_buf = None
        self.images_buf = None
        self.policy_action_buf = None

    def update_state_buf(self):
        state = np.concatenate(
            [
                self.motion_manager.get_data(state_key, self.obs)
                for state_key in self.state_keys
            ]
        )
        state = normalize_data(state, self.model_meta_info["state"])
        state = torch.tensor(state, dtype=torch.float32)

        if self.state_buf is None:
            self.state_buf = [
                state for _ in range(self.model_meta_info["data"]["n_obs_steps"])
            ]
        else:
            self.state_buf.pop(0)
            self.state_buf.append(state)

    def get_state(self):
        return torch.stack(self.state_buf, dim=0)[torch.newaxis].to(self.device)

    def update_images_buf(self):
        images = []
        for camera_name in self.camera_names:
            image = self.info["rgb_images"][camera_name]

            image = cv2.resize(image, self.model_meta_info["data"]["image_size"])

            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image, dtype=torch.uint8)
            image = self.image_transforms(image)
            # Adjust to a range from -1 to 1 to match the original implementation
            image = image * 2.0 - 1.0

            images.append(image)

        if self.images_buf is None:
            self.images_buf = [
                [image for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for image in images
            ]
        else:
            for single_images_buf, image in zip(self.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)

    def get_images(self):
        return [
            torch.stack(single_images_buf, dim=0)[torch.newaxis].to(self.device)
            for single_images_buf in self.images_buf
        ]

    def draw_plot(self):
        # Clear plot
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Plot images
        self.plot_images(
            self.ax[0, 0 : len(self.camera_names) + len(self.tactile_names)]
        )

        # Plot action
        self.plot_action(self.ax[1, 0])

        # Finalize plot
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )

    def plot_images(self, axes):
        for camera_idx, camera_name in enumerate(
            self.camera_names + self.tactile_names
        ):
            axes[camera_idx].imshow(self.info["rgb_images"][camera_name])
            axes[camera_idx].set_title(camera_name, fontsize=20)

    @abstractmethod
    def print_policy_info_for_tactile(self):
        pass

    @abstractmethod
    def override_policy_args_for_tactile(self, policy_args):
        pass
