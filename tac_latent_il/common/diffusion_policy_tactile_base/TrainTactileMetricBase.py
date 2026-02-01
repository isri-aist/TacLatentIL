import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from tqdm import tqdm


class TactileEncoderBase(nn.Module):
    DiffusionPolicyClass = None  # must be overridden

    def __init__(
        self,
        tactile_keys,
        tactile_feature_dim=64,
        tactile_kwargs=None,
    ):
        super().__init__()
        if tactile_kwargs is None:
            tactile_kwargs = {}
        self.tactile_keys = tactile_keys

        self.policy = self.DiffusionPolicyClass.__new__(
            self.DiffusionPolicyClass
        )  # bypass __init__
        nn.Module.__init__(self.policy)
        self.policy.tactile_keys = self.tactile_keys
        self.policy.setup_tactile_encoder(
            output_dim=tactile_feature_dim, **tactile_kwargs
        )
        self.tactile_model = self.policy.tactile_model

    def forward_one(self, tactile_key, tactile_image):
        return self.policy.forward_tactile_encoder(tactile_key, tactile_image)


class TactileMetricBase(nn.Module):
    TactileEncoderClass = None  # must be overridden

    def __init__(
        self,
        tactile_keys,
        use_teacher_ema=True,
        tactile_feature_dim=64,
        tactile_kwargs=None,
    ):
        super().__init__()

        self.tactile_keys = tactile_keys
        self.use_teacher_ema = use_teacher_ema

        self.encoder = self.TactileEncoderClass(
            tactile_keys=self.tactile_keys,
            tactile_feature_dim=tactile_feature_dim,
            tactile_kwargs=tactile_kwargs,
        )

        dynamics_hidden_dim = 256
        self.dynamics_head = nn.Sequential(
            nn.Linear(2 * tactile_feature_dim, dynamics_hidden_dim),
            nn.ReLU(),
            nn.Linear(dynamics_hidden_dim, tactile_feature_dim),
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
    def step_teacher_ema(self):
        if self.use_teacher_ema:
            self.teacher_ema.step(self.encoder)

    def compute_infonce_loss(self, q, k, temperature=0.1):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = (q @ k.t()) / temperature  # (B, B)
        labels = torch.arange(q.shape[0], device=q.device)
        return F.cross_entropy(logits, labels), logits

    def forward_batch(self, batch, temperature=0.1):
        obs = batch["obs"]

        loss_sum = torch.zeros((), device=next(self.parameters()).device)
        logit_pos_sum = torch.zeros_like(loss_sum)

        for k in self.tactile_keys:
            tactile_seq = obs[k]  # (B, T, C, H, W)
            if tactile_seq.shape[1] < 3:
                raise ValueError(
                    f"[{self.__class__.__name__}] Horizon must be >= 3: tactile_seq.shape[1]"
                )

            x_prev = tactile_seq[:, 0]
            x_curr = tactile_seq[:, 1]
            x_next = tactile_seq[:, 2]

            z_prev = self.encoder.forward_one(k, x_prev)
            z_curr = self.encoder.forward_one(k, x_curr)
            z_hat_next = self.dynamics_head(torch.cat([z_prev, z_curr], dim=-1))

            with torch.no_grad():
                if self.use_teacher_ema:
                    z_next = self.teacher_encoder.forward_one(k, x_next)
                else:
                    z_next = self.encoder.forward_one(k, x_next)

            loss_k, logits = self.compute_infonce_loss(
                z_hat_next, z_next, temperature=temperature
            )
            logit_pos_sum += logits.diag().mean()
            loss_sum += loss_k

        loss = loss_sum / len(self.tactile_keys)
        logit_pos = logit_pos_sum / len(self.tactile_keys)
        return {"loss": loss, "loss_infonce": loss, "logit_pos": logit_pos}


class TrainTactileMetricBase:
    TactileMetricClass = None  # must be overridden

    def set_additional_args_for_tactile(self, parser):
        super().set_additional_args_for_tactile(parser)

        parser.set_defaults(state_keys=[])
        parser.set_defaults(camera_names=[])
        parser.set_defaults(train_ratio=0.8)
        parser.set_defaults(batch_size=32)
        parser.set_defaults(num_epochs=100)

        # add metric-specific args
        parser.add_argument("--tactile_feature_dim", type=int, default=64)

        parser.add_argument(
            "--use_teacher_ema",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="use EMA teacher encoder for targets",
        )

        parser.add_argument("--temperature", type=float, default=0.1)

    def setup_policy(self):
        # Construct only tactile metric model (no diffusion policy)
        self.policy = self.TactileMetricClass(
            tactile_keys=self.args.tactile_names,
            use_teacher_ema=self.args.use_teacher_ema,
            tactile_feature_dim=self.args.tactile_feature_dim,
            tactile_kwargs=self.get_tactile_kwargs(),
        )

        # Construct optimizer
        self.optimizer = torch.optim.AdamW(
            (p for p in self.policy.parameters() if p.requires_grad),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.95, 0.999),
            eps=1e-8,
        )

        num_warmup_steps = 500
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * self.args.num_epochs),
        )

        # Transfer to device
        self.policy.cuda()
        optimizer_to(self.optimizer, "cuda")

        # Print policy information
        self.print_policy_info()
        print(
            f"  - horizon: {self.args.horizon}, obs steps: {self.args.n_obs_steps}, action steps: {self.args.n_action_steps}\n"
            f"  - tactile names: {self.args.tactile_names}, tactile fine-tune: {self.args.enable_tactile_finetune}\n"
            f"  - teacher ema: {self.args.use_teacher_ema}, temperature: {self.args.temperature}"
        )

    def train_loop(self):
        metrics_path = Path(self.args.checkpoint_dir) / "tactile_metric.yaml"
        if metrics_path.exists():
            metrics_path.unlink()

        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                data = dict_apply(data, lambda x: x.cuda())
                loss_dict = self.policy.forward_batch(
                    data, temperature=self.args.temperature
                )
                loss = loss_dict["loss"]
                loss.backward()
                self.optimizer.step()
                self.policy.step_teacher_ema()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                batch_result_list.append(
                    self.detach_batch_result(
                        {**loss_dict, "lr": self.lr_scheduler.get_last_lr()[0]}
                    )
                )
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            self.policy.eval()
            with torch.inference_mode():
                batch_result_list = []
                for data in self.val_dataloader:
                    data = dict_apply(data, lambda x: x.cuda())
                    loss_dict = self.policy.forward_batch(
                        data, temperature=self.args.temperature
                    )
                    batch_result_list.append(self.detach_batch_result(loss_dict))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                self.update_best_ckpt(epoch_summary, policy=self.policy)

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>4}", policy=self.policy)

            # Save metrics
            metrics = {
                "epoch": int(epoch),
                "loss": float(epoch_summary["loss"]),
                "loss_infonce": float(epoch_summary["loss_infonce"]),
                "logit_pos": float(epoch_summary["logit_pos"]),
            }
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics_list = yaml.safe_load(f) or []
            else:
                metrics_list = []
            metrics_list.append(metrics)
            with open(metrics_path, "w") as f:
                yaml.safe_dump(metrics_list, f)

        # Save last checkpoint
        self.save_current_ckpt("last", policy=self.policy)

        # Save best checkpoint
        self.save_best_ckpt()
