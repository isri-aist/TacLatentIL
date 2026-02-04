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
from robo_manip_baselines.common import DataKey
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
        action_dim,
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
            nn.Linear(2 * tactile_feature_dim + action_dim, dynamics_hidden_dim),
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

        self.queue_size = 1024
        self.register_buffer(
            "neg_queue",
            torch.empty(self.queue_size, tactile_feature_dim),
            persistent=False,
        )
        self.register_buffer(
            "neg_queue_ptr", torch.zeros((), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "neg_queue_filled", torch.zeros((), dtype=torch.long), persistent=False
        )

    @torch.no_grad()
    def step_teacher_ema(self):
        if self.use_teacher_ema:
            self.teacher_ema.step(self.encoder)

    @torch.no_grad()
    def enqueue(self, k):
        k = k.detach()
        B, D = k.shape
        ptr = int(self.neg_queue_ptr.item())
        # write in a ring buffer
        if B >= self.queue_size:
            self.neg_queue.copy_(k[-self.queue_size :])
            ptr = 0
            filled = self.queue_size
        else:
            end = ptr + B
            if end <= self.queue_size:
                self.neg_queue[ptr:end] = k
            else:
                first = self.queue_size - ptr
                self.neg_queue[ptr:] = k[:first]
                self.neg_queue[: end - self.queue_size] = k[first:]
            ptr = end % self.queue_size
            filled = min(self.queue_size, int(self.neg_queue_filled.item()) + B)
        self.neg_queue_ptr.fill_(ptr)
        self.neg_queue_filled.fill_(filled)

    def compute_infonce_loss(self, q, k, temperature=0.1):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits_inbatch = (q @ k.t()) / temperature  # (B, B)

        # queue negatives
        logits_all = logits_inbatch
        qfilled = int(self.neg_queue_filled.item())
        if qfilled > 0:
            neg = self.neg_queue[:qfilled]
            neg = F.normalize(neg, dim=-1)
            logits_queue = (q @ neg.t()) / temperature  # (B, Q)
            logits_all = torch.cat([logits_inbatch, logits_queue], dim=1)  # (B, B+Q)

        labels = torch.arange(
            q.shape[0], device=q.device
        )  # positives are diagonal in first B columns
        loss = F.cross_entropy(logits_all, labels)
        return loss, logits_all

    def forward_batch(self, batch, temperature=0.1):
        obs = batch["obs"]
        action = batch["action"]  # (B, T, Da)

        loss_sum = torch.zeros((), device=next(self.parameters()).device)
        logit_pos_sum = torch.zeros_like(loss_sum)

        for k in self.tactile_keys:
            tactile_seq = obs[k]  # (B, T, C, H, W)
            if tactile_seq.shape[1] < 3:
                raise ValueError(
                    f"[{self.__class__.__name__}] Tactile horizon must be >= 3: {tactile_seq.shape[1]}"
                )
            if action.shape[1] < 2:
                raise ValueError(
                    f"[{self.__class__.__name__}] Action horizon must be >= 2: {action.shape[1]}"
                )

            x_prev = tactile_seq[:, 0]
            x_curr = tactile_seq[:, 1]
            x_next = tactile_seq[:, 2]
            a_curr = action[:, 1]

            z_prev = self.encoder.forward_one(k, x_prev)
            z_curr = self.encoder.forward_one(k, x_curr)

            # f([z_{t-1}, z_t, a_t]) -> z_{t+1}
            z_hat_next = self.dynamics_head(torch.cat([z_prev, z_curr, a_curr], dim=-1))

            with torch.no_grad():
                if self.use_teacher_ema:
                    z_next = self.teacher_encoder.forward_one(k, x_next)
                else:
                    z_next = self.encoder.forward_one(k, x_next)

            loss_k, logits = self.compute_infonce_loss(
                z_hat_next, z_next, temperature=temperature
            )
            with torch.no_grad():
                self.enqueue(z_next)

            logit_pos_sum += logits[:, : logits.shape[0]].diag().mean()
            loss_sum += loss_k

        loss = loss_sum / len(self.tactile_keys)
        logit_pos = logit_pos_sum / len(self.tactile_keys)
        return {"loss": loss, "loss_infonce": loss, "logit_pos": logit_pos}


class TrainTactileMetricBase:
    TactileMetricClass = None  # must be overridden

    def set_additional_args_for_tactile(self, parser):
        super().set_additional_args_for_tactile(parser)

        parser.set_defaults(state_keys=[])
        parser.set_defaults(
            action_keys=[
                DataKey.COMMAND_EEF_POSE_REL,
                DataKey.COMMAND_GRIPPER_JOINT_POS,
                DataKey.COMMAND_GRIPPER_JOINT_POS_REL,
            ]
        )
        parser.set_defaults(camera_names=[])
        parser.set_defaults(train_ratio=0.8)
        parser.set_defaults(batch_size=16)
        parser.set_defaults(num_epochs=100)
        parser.set_defaults(lr=1e-4)

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
        action_dim = len(self.model_meta_info["action"]["example"])
        self.policy = self.TactileMetricClass(
            action_dim=action_dim,
            tactile_keys=self.args.tactile_names,
            use_teacher_ema=self.args.use_teacher_ema,
            tactile_feature_dim=self.args.tactile_feature_dim,
            tactile_kwargs=self.get_tactile_kwargs(),
        )

        # Construct optimizer
        encoder_params = []
        head_params = []
        for name, p in self.policy.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("encoder."):
                encoder_params.append(p)
            else:
                head_params.append(p)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": 0.1 * self.args.lr},
                {"params": head_params, "lr": self.args.lr},
            ],
            weight_decay=self.args.weight_decay,
            betas=(0.95, 0.999),
            eps=1e-8,
        )

        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=50,
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
        self.print_policy_info_for_tactile()

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
