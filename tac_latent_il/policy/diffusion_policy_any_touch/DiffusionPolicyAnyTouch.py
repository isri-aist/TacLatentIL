import copy
import os
import sys

import torch
from torch import nn
from transformers import CLIPConfig
from transformers.models.clip.modeling_clip import (
    CLIPVisionTransformer,
)

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../third_party/any_touch")
)
from model.linear_probe import TactileProbe

from tac_latent_il.common import DiffusionPolicyTactileBase


def load_model_from_multi_clip(ckpt, model, use_same_patchemb=False):
    new_ckpt = {}
    for key, item in ckpt.items():
        if (
            "touch_model" in key
            or "touch_projection" in key
            or "sensor_token" in key
            and "sensor_token_proj" not in key
        ):
            new_ckpt[key.replace("touch_mae_model.", "")] = copy.deepcopy(item)
        if use_same_patchemb:
            if "video_patch_embedding" in key:
                new_key = key.replace("touch_mae_model.", "")
                new_ckpt[
                    new_key.replace(
                        "video_patch_embedding",
                        "touch_model.embeddings.patch_embedding",
                    )
                ] = copy.deepcopy(item)

    for k, v in model.named_parameters():
        if k not in new_ckpt.keys():
            new_ckpt[k] = v

    model.load_state_dict(new_ckpt, strict=True)

    return model


class AnyTouchTactileEncoder(TactileProbe):
    def __init__(
        self,
        model_config: CLIPConfig,
        num_frames,
        add_time_attn,
        tube_size,
        use_sensor_token,
        use_same_patchemb,
        pooling,
        n_obs_steps,
    ):
        nn.Module.__init__(self)
        model_config.vision_config.num_frames = num_frames
        model_config.vision_config.tube_size = tube_size

        self.use_sensor_token = use_sensor_token
        self.use_same_patchemb = use_same_patchemb
        self.n_obs_steps = n_obs_steps

        if self.use_sensor_token:
            self.sensor_token = nn.Parameter(
                torch.zeros(10, 5, model_config.vision_config.hidden_size)
            )

        self.touch_model = CLIPVisionTransformer(model_config.vision_config)
        self.touch_projection = nn.Linear(
            model_config.vision_config.hidden_size,
            model_config.projection_dim,
            bias=False,
        )

        if self.use_same_patchemb:
            self.touch_model.embeddings.patch_embedding = nn.Conv3d(
                in_channels=model_config.vision_config.num_channels,
                out_channels=self.touch_model.embeddings.embed_dim,
                kernel_size=(
                    3,
                    self.touch_model.embeddings.patch_size,
                    self.touch_model.embeddings.patch_size,
                ),
                stride=(
                    3,
                    self.touch_model.embeddings.patch_size,
                    self.touch_model.embeddings.patch_size,
                ),
                bias=False,
            )

        self.pooling = pooling

        self.touch_model.forward = self.touch_forward
        self.touch_model.embeddings.forward = self.emb_forward

    def emb_forward(
        self, pixel_values: torch.FloatTensor, noise=None, sensor_type=None
    ) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.touch_model.embeddings.patch_embedding.weight.dtype
        patch_embeds = self.touch_model.embeddings.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        pos_emb = self.touch_model.embeddings.position_embedding(
            self.touch_model.embeddings.position_ids
        )

        embeddings = patch_embeds + pos_emb[:, 1:, :]

        class_embeds = self.touch_model.embeddings.class_embedding + pos_emb[:, 0, :]
        class_embeds = class_embeds.expand(batch_size, 1, -1)

        if self.use_sensor_token:
            sensor_emb = self.sensor_token[sensor_type.repeat(self.n_obs_steps)]
            embeddings = torch.cat([class_embeds, sensor_emb, embeddings], dim=1)
        else:
            embeddings = torch.cat([class_embeds, embeddings], dim=1)
        # embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

    def forward(self, x, sensor_type=None):
        if self.use_same_patchemb:
            x = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)

        # print(x.shape, sensor_type.shape)

        with torch.no_grad():
            x = self.touch_model(x, sensor_type=sensor_type)
            if self.pooling == "cls":
                out = self.touch_projection(x.pooler_output)
            else:
                out = self.touch_projection(x.last_hidden_state)

        if self.pooling == "global":
            if self.use_sensor_token:
                out = out[:, 6:, :].mean(dim=1)
            else:
                out = out[:, 1:, :].mean(dim=1)

        return out


class DiffusionPolicyAnyTouch(DiffusionPolicyTactileBase):
    def setup_tactile_encoder(
        self,
        anytouch_config: CLIPConfig,
        n_obs_steps,
        output_dim=64,
        use_sensor_token=False,
        use_same_patchemb=False,
        pooling_method="global",
    ):
        checkpoint_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "../../checkpoint/AnyTouch/anytouch.pth",
            )
        )
        self.anytouch_model = AnyTouchTactileEncoder(
            anytouch_config,
            1,
            None,
            1,
            use_sensor_token,
            use_same_patchemb,
            pooling_method,
            n_obs_steps,
        )
        embed_dim = anytouch_config.projection_dim
        self.linear = nn.Linear(embed_dim, output_dim)

        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
        self.anytouch_model = load_model_from_multi_clip(
            loaded_checkpoint, self.anytouch_model, use_same_patchemb
        )

        self.anytouch_model.to(self.device)
        self.anytouch_model.eval()

    def forward_tactile_encoder(self, tactile_image: dict):
        anytouch_output = self.anytouch_model(
            tactile_image["rgb_image"], sensor_type=tactile_image["sensor_type"]
        )
        tactile_feature = self.linear(anytouch_output)
        return tactile_feature
