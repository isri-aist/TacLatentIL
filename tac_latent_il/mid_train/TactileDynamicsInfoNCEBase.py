import torch
import torch.nn.functional as F

from .TactileDynamicsBase import TactileDynamicsBase


class TactileDynamicsInfoNCEBase(TactileDynamicsBase):
    @classmethod
    def set_additional_args(cls, parser):
        super().set_additional_args(parser)
        parser.add_argument("--temperature", type=float, default=0.1)

    @classmethod
    def get_additional_model_args(cls, args):
        return {
            **super().get_additional_model_args(args),
            "temperature": args.temperature,
        }

    def __init__(
        self,
        action_dim,
        tactile_encoder,
        use_teacher_ema=True,
        use_action=True,
        temperature=0.1,
    ):
        super().__init__(
            action_dim=action_dim,
            tactile_encoder=tactile_encoder,
            use_teacher_ema=use_teacher_ema,
            use_action=use_action,
        )
        self.temperature = temperature

        self.queue_size = 1024
        self.register_buffer(
            "neg_queue",
            torch.empty(self.queue_size, self.tactile_feature_dim),
            persistent=False,
        )
        self.register_buffer(
            "neg_queue_ptr", torch.zeros((), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "neg_queue_filled", torch.zeros((), dtype=torch.long), persistent=False
        )

    def get_model_info(self):
        return f"{super().get_model_info()}\n  - temperature: {self.temperature}"

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

    def compute_infonce_loss(self, q, k):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits_inbatch = (q @ k.t()) / self.temperature  # (B, B)

        logits_all = logits_inbatch
        qfilled = int(self.neg_queue_filled.item())
        if qfilled > 0:
            neg = self.neg_queue[:qfilled]
            neg = F.normalize(neg, dim=-1)
            logits_queue = (q @ neg.t()) / self.temperature  # (B, Q)
            logits_all = torch.cat([logits_inbatch, logits_queue], dim=1)  # (B, B+Q)

        labels = torch.arange(q.shape[0], device=q.device)
        loss = F.cross_entropy(logits_all, labels)
        return loss, logits_all

    def forward_batch(self, batch):
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
            z_hat_next = self.dynamics_head(
                self.build_dynamics_input(z_prev, z_curr, a_curr)
            )

            with torch.no_grad():
                if self.use_teacher_ema:
                    z_next = self.teacher_encoder.forward_one(k, x_next)
                else:
                    z_next = self.encoder.forward_one(k, x_next)

            loss_k, logits = self.compute_infonce_loss(z_hat_next, z_next)
            with torch.no_grad():
                self.enqueue(z_next)

            logit_pos_sum += logits[:, : logits.shape[0]].diag().mean()
            loss_sum += loss_k

        loss = loss_sum / len(self.tactile_keys)
        logit_pos = logit_pos_sum / len(self.tactile_keys)
        return {"loss": loss, "loss_infonce": loss, "logit_pos": logit_pos}
