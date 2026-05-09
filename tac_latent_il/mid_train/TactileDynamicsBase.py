import argparse
import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../third_party/diffusion_policy")
)
from diffusion_policy.model.diffusion.ema_model import EMAModel

from tac_latent_il.common.mid_train_base import MidTrainModelBase


class TactileDynamicsBase(MidTrainModelBase):
    @classmethod
    def set_additional_args(cls, parser):
        parser.add_argument(
            "--use_teacher_ema",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="use EMA teacher encoder for targets",
        )

    @classmethod
    def get_additional_model_args(cls, args):
        return {
            "use_teacher_ema": args.use_teacher_ema,
        }

    def __init__(
        self,
        action_dim,
        tactile_encoder,
        use_teacher_ema=True,
    ):
        super().__init__()

        self.encoder = tactile_encoder
        self.tactile_keys = self.encoder.tactile_keys
        self.tactile_feature_dim = self.encoder.tactile_feature_dim
        self.use_teacher_ema = use_teacher_ema

        dynamics_hidden_dim = 256
        self.dynamics_head = nn.Sequential(
            nn.Linear(
                2 * self.tactile_feature_dim + action_dim,
                dynamics_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(dynamics_hidden_dim, self.tactile_feature_dim),
        )

        if self.use_teacher_ema:
            self.teacher_encoder = copy.deepcopy(self.encoder)
            self.teacher_ema = EMAModel(
                model=self.teacher_encoder,
                update_after_step=0,
                inv_gamma=1.0,
                power=0.75,
                min_value=0.0,
                max_value=0.9999,
                copy_buffers=True,
            )
            for p in self.teacher_encoder.parameters():
                p.requires_grad = False
            self.teacher_encoder.eval()
        else:
            self.teacher_encoder = None
            self.teacher_ema = None

    @torch.no_grad()
    def on_after_optimizer_step(self):
        if self.use_teacher_ema:
            self.teacher_ema.step(self.encoder)
