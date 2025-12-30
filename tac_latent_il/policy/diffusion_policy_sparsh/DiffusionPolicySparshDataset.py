import os
import sys

import cv2
import numpy as np
import torch
from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    DpStyleDatasetMixin,
    RmbData,
    get_skipped_data_seq,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/sparsh"))
from tactile_ssl.data.digit.utils import compute_diff, enhance_image


class DiffusionPolicySparshDataset(DatasetBase, DpStyleDatasetMixin):
    """Dataset to train diffusion policy sparsh."""

    def setup_variables(self):
        self.setup_dp_style_chunk()

    def __len__(self):
        return len(self.chunk_info_list)

    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            time_idxes = np.clip(
                np.arange(start_time_idx, start_time_idx + horizon), 0, episode_len - 1
            )

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load images
            images = np.stack(
                [
                    rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip][time_idxes]
                    for camera_name in self.model_meta_info["image"]["camera_names"]
                ],
                axis=0,
            )

            # Load tactiles
            prev_time_idxes = np.maximum(time_idxes - 1, 0)
            init_tactile_dict = {}
            tactiles_dict = {}
            for tactile_name in self.model_meta_info["tactile"]["tactile_names"]:
                tactile_data = rmb_data[DataKey.get_rgb_image_key(tactile_name)][::skip]
                init_tactile_dict[tactile_name] = tactile_data[0]
                tactiles_dict[tactile_name] = np.stack(
                    [tactile_data[time_idxes], tactile_data[prev_time_idxes]], axis=1
                )  # (T, 2, H, W, C)

        # Resize images
        K, T, H, W, C = images.shape
        image_size = self.model_meta_info["data"]["image_size"]
        images = np.array(
            [cv2.resize(img, image_size) for img in images.reshape(-1, H, W, C)]
        ).reshape(K, T, *image_size[::-1], C)

        # Process tactiles
        def _process_tactile_cv2(img):
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.flip(img, 0)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return img

        tactiles_tensor_dict = {}
        for tactile_name in self.model_meta_info["tactile"]["tactile_names"]:
            init_tactile_proc = _process_tactile_cv2(init_tactile_dict[tactile_name])
            tactiles = tactiles_dict[tactile_name]
            proc_list0 = []
            for proc_idx0 in range(tactiles.shape[0]):
                proc_list1 = []
                for proc_idx1 in range(tactiles.shape[1]):
                    tactile_proc = _process_tactile_cv2(tactiles[proc_idx0, proc_idx1])
                    tactile_proc = compute_diff(
                        tactile_proc, init_tactile_proc, offset=0.5
                    )
                    tactile_proc = enhance_image(
                        tactile_proc, brightness=280, contrast=200
                    )
                    tactile_proc = np.ascontiguousarray(
                        np.moveaxis(tactile_proc, -1, -3)
                    )  # (C, H, W)
                    tactile_proc = torch.from_numpy(tactile_proc).float().div(255.0)
                    proc_list1.append(tactile_proc)
                proc_list0.append(torch.cat(proc_list1, dim=0))  # (2 * C, H, W)
            tactiles_tensor_dict[tactile_name] = torch.stack(
                proc_list0, dim=0
            )  # (T, 2 * C, H, W)

        # Pre-convert data
        state, action, images = self.pre_convert_data(state, action, images)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )

        # Convert to data structure of policy input and output
        data = {"obs": {}, "action": action_tensor}
        if len(self.model_meta_info["state"]["keys"]) > 0:
            data["obs"]["state"] = state_tensor
        for camera_idx, camera_name in enumerate(
            self.model_meta_info["image"]["camera_names"]
        ):
            data["obs"][DataKey.get_rgb_image_key(camera_name)] = images_tensor[
                camera_idx
            ]
        for tactile_name in self.model_meta_info["tactile"]["tactile_names"]:
            data["obs"][tactile_name] = tactiles_tensor_dict[tactile_name]

        return data

    def augment_data(self, state, action, images):
        state, action, images = super().augment_data(state, action, images)

        # Adjust to a range from -1 to 1 to match the original implementation
        images = images * 2.0 - 1.0

        return state, action, images
