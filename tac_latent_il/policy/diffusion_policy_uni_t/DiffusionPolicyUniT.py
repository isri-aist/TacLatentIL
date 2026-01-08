import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/unit"))
from UniT.taming.models.vqgan import VQModel

from tac_latent_il.common import DiffusionPolicyTactileBase


class DiffusionPolicyUniT(DiffusionPolicyTactileBase):
    def setup_tactile_encoder(
        self,
        output_dim=64,
        enable_finetune=False,
        load_checkpoint=True,
    ):
        embed_dim = 84
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
            "ddconfig": ddconfig,
            "lossconfig": lossconfig,
            "n_embed": 1024,
            "embed_dim": 3,
        }
        self.unit_model = VQModel(**config)
        self.linear = torch.nn.Linear(embed_dim, output_dim)

        if load_checkpoint:
            model_checkpoint_path = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../checkpoint/UniT/checkpoint-epoch=340.ckpt",
                )
            )
            print(
                f"[{self.__class__.__name__}] Load UniT encoder pretrain checkpoint: {model_checkpoint_path}"
            )
            self.unit_model.init_from_ckpt(model_checkpoint_path)
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

    def forward_tactile_encoder(self, tactile_image):
        tactile_image = tactile_image.permute(0, 2, 3, 1)

        img_batch = {"image": tactile_image}
        input = self.unit_model.get_input(img_batch, "image")
        z = self.unit_model.encoder(input)
        unit_output = z.mean(dim=3)
        unit_output = unit_output.view(tactile_image.shape[0], -1)
        tactile_feature = self.linear(unit_output)
        return tactile_feature
