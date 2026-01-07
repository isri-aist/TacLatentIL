from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import CLIPConfig, CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    CLIP_VISION_INPUTS_DOCSTRING,
    CLIPVisionTransformer,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)


class AnyTouchTactileEncoder(nn.Module):
    def __init__(
        self,
        model_config: CLIPConfig,
        num_frames,
        add_time_attn,
        tube_size,
        use_sensor_token,
        use_same_patchemb,
        pooling,
    ):
        super(AnyTouchTactileEncoder, self).__init__()
        model_config.vision_config.num_frames = num_frames
        model_config.vision_config.tube_size = tube_size

        self.use_sensor_token = use_sensor_token
        self.use_same_patchemb = use_same_patchemb

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

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig
    )
    def touch_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sensor_type=None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # a = self.sensor_token[sensor_type]
        # print(a.shape)
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.touch_model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.touch_model.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.touch_model.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.touch_model.embeddings(
            pixel_values, sensor_type=sensor_type
        )
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.touch_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.touch_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

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
            sensor_type = sensor_type.repeat(batch_size)
            sensor_emb = self.sensor_token[sensor_type]
            embeddings = torch.cat([class_embeds, sensor_emb, embeddings], dim=1)
        else:
            embeddings = torch.cat([class_embeds, embeddings], dim=1)
        # embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

    def forward(self, x, sensor_type=None):
        if self.use_same_patchemb:
            x = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)

        # print(x.shape, sensor_type.shape)

        x = self.touch_model(x, sensor_type=sensor_type)
        if self.pooling == "cls":
            out = self.touch_projection(x.pooler_output)
        elif self.pooling == "global":
            out = self.touch_projection(x.last_hidden_state)
            if self.use_sensor_token:
                out = out[:, 6:, :].mean(dim=1)
            else:
                out = out[:, 1:, :].mean(dim=1)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Unknown pooling: {self.pooling}"
            )

        return out
