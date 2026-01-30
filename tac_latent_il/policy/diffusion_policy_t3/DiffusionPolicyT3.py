import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/t3"))
from t3.models.encoder import ViTEncoder
from t3.models.trunk import TransformerTrunk

from tac_latent_il.common import DiffusionPolicyTactileBase


class DiffusionPolicyT3(DiffusionPolicyTactileBase):
    def setup_tactile_encoder(
        self,
        output_dim=64,
        pooling_type="cls",
        enable_finetune=False,
        load_checkpoint=True,
    ):
        embed_dim = 192
        t3_encoder = ViTEncoder(
            patch_size=16, embed_dim=embed_dim, depth=3, num_heads=3, mlp_ratio=4.0
        )
        t3_trunk = TransformerTrunk(
            embed_dim=embed_dim,
            depth=9,
            num_heads=3,
            mlp_ratio=4.0,
            pooling_type=pooling_type,
        )
        tactile_head = torch.nn.Linear(embed_dim, output_dim)

        if load_checkpoint:
            encoder_checkpoint_path = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../checkpoint/T3/mini.pth",
                )
            )
            print(
                f"[{self.__class__.__name__}] Load T3 encoder pretrain checkpoint: {encoder_checkpoint_path}"
            )
            t3_encoder.load(encoder_checkpoint_path)

            trunk_checkpoint_path = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../checkpoint/T3/trunk.pth",
                )
            )
            print(
                f"[{self.__class__.__name__}] Load T3 trunk pretrain checkpoint: {trunk_checkpoint_path}"
            )
            t3_trunk.load(trunk_checkpoint_path)
        else:
            print(f"[{self.__class__.__name__}] Skip loading T3 pretrained checkpoint.")

        if enable_finetune:
            print(f"[{self.__class__.__name__}] Fine-tune T3 model.")
        else:
            print(f"[{self.__class__.__name__}] Freeze T3 model.")
            for p in t3_encoder.parameters():
                p.requires_grad = False
            t3_encoder.eval()
            for p in t3_trunk.parameters():
                p.requires_grad = False
            t3_trunk.eval()

        self.tactile_model = torch.nn.Module()
        self.tactile_model.t3_encoder = t3_encoder
        self.tactile_model.t3_trunk = t3_trunk
        self.tactile_model.tactile_head = tactile_head

    def forward_tactile_encoder(self, tactile_key, tactile_image):
        t3_output = self.tactile_model.t3_trunk(
            self.tactile_model.t3_encoder(tactile_image)
        )
        tactile_feature = self.tactile_model.tactile_head(t3_output)
        return tactile_feature
