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
from torchvision.transforms import v2


class DiffusionPolicyAnyTouchDataset(DatasetBase, DpStyleDatasetMixin):
    """Dataset to train diffusion policy anytouch."""

    def setup_image_transforms(self):
        super().setup_image_transforms()

        # Copied from:
        # https://github.com/GeWu-Lab/AnyTouch/blob/3ee6c09130baa9f47a14449b47fd151fbd17a163/dataloader/downstream_dataset.py#L110-L117
        self.tactile_image_transforms = v2.Compose(
            [
                # v2.RandomHorizontalFlip(p=0.5),
                # v2.RandomVerticalFlip(p=0.5),
                # v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

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
            tactiles = np.stack(
                [
                    rmb_data[DataKey.get_rgb_image_key(tactile_name)][::skip][
                        time_idxes
                    ]
                    for tactile_name in self.model_meta_info["tactile"]["tactile_names"]
                ],
                axis=0,
            )

        # Resize images
        K, T, H, W, C = images.shape
        image_size = self.model_meta_info["data"]["image_size"]
        images = np.array(
            [cv2.resize(img, image_size) for img in images.reshape(-1, H, W, C)]
        ).reshape(K, T, *image_size[::-1], C)

        # Resize tactiles
        K, T, H, W, C = tactiles.shape
        tactile_size = (224, 224)
        tactiles = np.array(
            [cv2.resize(img, tactile_size) for img in tactiles.reshape(-1, H, W, C)]
        ).reshape(K, T, *tactile_size[::-1], C)

        # Pre-convert data
        state, action, images = self.pre_convert_data(state, action, images)
        tactiles = self.pre_convert_data_for_tactile(tactiles)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)
        tactiles_tensor = torch.tensor(tactiles, dtype=torch.uint8)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )
        tactiles_tensor = self.augment_data_for_tactile(tactiles_tensor)

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
        for tactile_idx, tactile_name in enumerate(
            self.model_meta_info["tactile"]["tactile_names"]
        ):
            data["obs"][tactile_name] = tactiles_tensor[tactile_idx]

        return data

    def augment_data(self, state, action, images):
        state, action, images = super().augment_data(state, action, images)

        # Adjust to a range from -1 to 1 to match the original implementation
        images = images * 2.0 - 1.0

        return state, action, images

    def pre_convert_data_for_tactile(self, tactiles):
        tactiles = np.moveaxis(tactiles, -1, -3)

        return tactiles

    def augment_data_for_tactile(self, tactiles):
        if tactiles.ndim < 3:
            raise ValueError(
                f"[{self.__class__.__name__}] Input must have at least 3 dimensions (C, H, W)"
            )
        elif tactiles.ndim == 3:
            tactiles = self.tactile_transforms(tactiles)
        else:
            orig_prefix_shape = tactiles.shape[:-3]
            orig_tactile_list = tactiles.reshape(-1, *tactiles.shape[-3:])
            transformed_tactile_list = [
                self.tactile_transforms(img) for img in orig_tactile_list
            ]
            tactiles = torch.stack(transformed_tactile_list)
            new_img_shape = tactiles.shape[-3:]
            tactiles = tactiles.view(*orig_prefix_shape, *new_img_shape)

        return tactiles
