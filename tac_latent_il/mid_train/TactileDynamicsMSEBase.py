import torch
import torch.nn.functional as F

from .TactileDynamicsBase import TactileDynamicsBase


class TactileDynamicsMSEBase(TactileDynamicsBase):
    def forward_batch(self, batch):
        obs = batch["obs"]
        action = batch["action"]  # (B, T, Da)

        loss_sum = torch.zeros((), device=next(self.parameters()).device)

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

            loss_sum += F.mse_loss(z_hat_next, z_next)

        loss = loss_sum / len(self.tactile_keys)
        return {"loss": loss, "loss_mse": loss}
