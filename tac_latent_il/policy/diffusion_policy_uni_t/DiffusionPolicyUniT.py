import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/unit"))
from UniT.model.tactile.utils import ConvPoolingPolicyHead as ConvPoolingHead
from UniT.taming.models.vqgan import VQModel

from tac_latent_il.common import DiffusionPolicyTactileBase


class DiffusionPolicyUniT(DiffusionPolicyTactileBase):
    def setup_tactile_encoder(
        self,
        output_dim=64,
        enable_finetune=False,
        load_checkpoint=True,
    ):
        ddconfig = {
            "double_z": False,
            "z_channels": 3,
            "resolution": 128,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [16],
            "dropout": 0.0,
        }
        lossconfig = {
            "target": "UniT.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
            "params": {
                "disc_conditional": False,
                "disc_in_channels": 3,
                "disc_start": 10000,
                "disc_weight": 0.8,
                "codebook_weight": 1.0,
            },
        }
        config = {
            "embed_dim": 3,
            "n_embed": 1024,
            "ddconfig": ddconfig,
            "lossconfig": lossconfig,
        }
        self.unit_model = VQModel(**config)
        latent_shape = [3, 16, 20]
        temp_head = ConvPoolingHead(input_channels=latent_shape[0])
        with torch.no_grad():
            embed_dim = temp_head(
                torch.randn(1, latent_shape[0], latent_shape[1], latent_shape[2])
            ).shape[1]
        self.heads = nn.ModuleDict(
            {
                tactile_key: nn.Sequential(
                    ConvPoolingHead(input_channels=latent_shape[0]),
                    nn.Linear(embed_dim, output_dim),
                )
                for tactile_key in self.tactile_keys
            }
        )

        if load_checkpoint:
            checkpoint_path = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../checkpoint/UniT/checkpoint-epoch=342.ckpt",
                )
            )
            print(
                f"[{self.__class__.__name__}] Load UniT pretrain checkpoint: {checkpoint_path}"
            )
            self.unit_model.init_from_ckpt(checkpoint_path)
        else:
            print(
                f"[{self.__class__.__name__}] Skip loading UniT pretrained checkpoint."
            )

        if enable_finetune:
            print(f"[{self.__class__.__name__}] Fine-tune UniT model.")
        else:
            print(f"[{self.__class__.__name__}] Freeze UniT model.")
            for p in self.unit_model.parameters():
                p.requires_grad = False
            self.unit_model.eval()

    def forward_tactile_encoder(self, tactile_key, tactile_image):
        unit_input = self.unit_model.get_input({"image": tactile_image}, "image")
        tactile_feature = self.unit_model.to_latent(unit_input)
        tactile_feature = self.heads[tactile_key](tactile_feature)

        batch_size = tactile_image.shape[0]
        tactile_feature = tactile_feature.reshape(batch_size, -1)

        return tactile_feature
