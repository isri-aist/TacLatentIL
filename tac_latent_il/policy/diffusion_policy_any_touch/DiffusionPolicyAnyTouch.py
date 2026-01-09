import copy
import os

import torch
from transformers import AutoConfig

from tac_latent_il.common import DiffusionPolicyTactileBase

from .AnyTouchTactileEncoder import AnyTouchTactileEncoder


# Copied and modified from:
# https://github.com/GeWu-Lab/AnyTouch/blob/3ee6c09130baa9f47a14449b47fd151fbd17a163/main_probe.py#L66
def load_model_from_multi_clip(
    ckpt, model, use_same_patchemb=False, use_sensor_token=True
):
    new_ckpt = {}
    for key, item in ckpt.items():
        if (
            "touch_model" in key
            or "touch_projection" in key
            or ("sensor_token" in key and use_sensor_token)
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


class DiffusionPolicyAnyTouch(DiffusionPolicyTactileBase):
    def setup_tactile_encoder(
        self,
        output_dim=64,
        sensor_name=None,
        use_universal_sensor_token=False,
        use_sensor_token=False,
        use_same_patchemb=False,
        pooling_method="global",
        enable_finetune=False,
        load_checkpoint=True,
    ):
        anytouch_config = AutoConfig.from_pretrained(
            os.path.join(
                os.path.dirname(__file__),
                "../../../third_party/any_touch/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/config.json",
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
        )
        embed_dim = anytouch_config.projection_dim
        self.linear = torch.nn.Linear(embed_dim, output_dim)

        if use_universal_sensor_token:
            self.sensor_type = -1
        else:
            sensor_type_map = {
                "gelsight": 0,
                "digit": 1,
                "gelslim": 2,
                "gelsight_mini": 3,
                "duragel": 4,
            }
            self.sensor_type = sensor_type_map[sensor_name]

        if load_checkpoint:
            checkpoint_path = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../checkpoint/AnyTouch/checkpoint.pth",
                )
            )
            print(
                f"[{self.__class__.__name__}] Load AnyTouch pretrained checkpoint: {checkpoint_path}"
            )
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            self.anytouch_model = load_model_from_multi_clip(
                checkpoint_data["model"],
                self.anytouch_model,
                use_same_patchemb,
                use_sensor_token,
            )
        else:
            print(
                f"[{self.__class__.__name__}] Skip loading AnyTouch pretrained checkpoint."
            )

        if enable_finetune:
            print(f"[{self.__class__.__name__}] Fine-tune AnyTouch model.")
        else:
            print(f"[{self.__class__.__name__}] Freeze AnyTouch model.")
            for p in self.anytouch_model.parameters():
                p.requires_grad = False
            self.anytouch_model.eval()

    def forward_tactile_encoder(self, tactile_key, tactile_image):
        anytouch_output = self.anytouch_model(
            tactile_image, sensor_type=torch.tensor([self.sensor_type]).to(self.device)
        )
        tactile_feature = self.linear(anytouch_output)
        return tactile_feature
