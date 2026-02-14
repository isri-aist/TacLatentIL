import argparse
import os
import pickle
import sys
from abc import ABC, abstractmethod

import torch
import yaml
from diffusion_policy.common.pytorch_util import dict_apply
from robo_manip_baselines.common import find_rmb_files, remove_prefix
from torch.utils.data import DataLoader
from tqdm import tqdm


class EvalDiffusionPolicyTactileBase(ABC):
    PolicyClass = None  # must be overridden
    DatasetClass = None  # must be overridden

    @property
    def policy_name(self):
        return remove_prefix(self.__class__.__name__, "Eval")

    def __init__(self):
        self.setup_args()

        self.setup_model_meta_info()

        self.setup_dataset()

        self.setup_policy()

    def setup_args(self, parser=None, argv=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

        parser.add_argument(
            "--checkpoint", type=str, required=True, help="checkpoint file"
        )
        parser.add_argument(
            "--dataset_dir",
            type=str,
            required=True,
            help="dataset directory",
        )

        parser.add_argument(
            "--enable_rmb_cache",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Whether to enable data caching in RmbData. This uses RAM heavily, so it should be disabled on computers with small RAM.",
        )

        parser.add_argument("--batch_size", type=int, default=32, help="batch size")
        parser.add_argument(
            "--num_workers", type=int, default=4, help="number of workers in dataloader"
        )

        parser.add_argument(
            "--out",
            type=str,
            default=None,
            help="output yaml file",
        )

        if argv is None:
            argv = sys.argv
        self.args = parser.parse_args(argv[1:])

    def setup_model_meta_info(self):
        checkpoint_dir = os.path.split(self.args.checkpoint)[0]
        model_meta_info_path = os.path.join(checkpoint_dir, "model_meta_info.pkl")
        with open(model_meta_info_path, "rb") as f:
            self.model_meta_info = pickle.load(f)
        print(
            f"[{self.__class__.__name__}] Load model meta info: {model_meta_info_path}"
        )

    def setup_dataset(self):
        filenames = find_rmb_files(self.args.dataset_dir)
        dataset = self.DatasetClass(
            filenames, self.model_meta_info, self.args.enable_rmb_cache
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.args.num_workers,
            persistent_workers=(self.args.num_workers > 0),
            prefetch_factor=4 if self.args.num_workers > 0 else None,
        )

        print(
            f"[{self.__class__.__name__}] Load dataset from {self.args.dataset_dir}\n"
            f"  - size: {len(self.dataloader.dataset)}, files: {len(self.dataloader.dataset.filenames)}\n"
            f"  - episode len mean: {int(self.model_meta_info['data']['mean_episode_len'])}, min: {self.model_meta_info['data']['min_episode_len']}, max: {self.model_meta_info['data']['max_episode_len']}"
        )

    def setup_policy(self):
        # For backward compatibility
        if "backbone" not in self.model_meta_info["policy"]:
            self.model_meta_info["policy"]["backbone"] = "cnn"
        if "scheduler" not in self.model_meta_info["policy"]:
            self.model_meta_info["policy"]["scheduler"] = "ddpm"

        # Print policy information
        state_dim = len(self.model_meta_info["state"]["example"])
        action_dim = len(self.model_meta_info["action"]["example"])
        state_keys = self.model_meta_info["state"]["keys"]
        action_keys = self.model_meta_info["action"]["keys"]
        camera_names = self.model_meta_info["image"]["camera_names"]
        print(
            f"[{self.__class__.__name__}] Construct {self.policy_name} policy.\n"
            f"  - state dim: {state_dim}, action dim: {action_dim}, camera num: {len(camera_names)}\n"
            f"  - state keys: {state_keys}\n"
            f"  - action keys: {action_keys}\n"
            f"  - camera names: {camera_names}\n"
            f"  - skip: {self.model_meta_info['data']['skip']}"
        )
        print(
            f"  - use ema: {self.model_meta_info['policy']['use_ema']}, backbone: {self.model_meta_info['policy']['backbone']}, scheduler: {self.model_meta_info['policy']['scheduler']}\n"
            f"  - horizon: {self.model_meta_info['data']['horizon']}, obs steps: {self.model_meta_info['data']['n_obs_steps']}, action steps: {self.model_meta_info['data']['n_action_steps']}\n"
            f"  - image size: {self.model_meta_info['data']['image_size']}, image crop size: {self.model_meta_info['data']['image_crop_size']}\n"
            f"  - tactile names: {self.model_meta_info['tactile']['tactile_names']}, tactile fine-tune: {self.model_meta_info['policy']['args']['tactile_kwargs']['enable_finetune']}"
        )
        self.print_policy_info_for_tactile()

        # Override policy args for rollout
        self.override_policy_args_for_tactile()

        # Construct scheduler
        if self.model_meta_info["policy"]["scheduler"] == "ddpm":
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

            noise_scheduler = DDPMScheduler(
                **self.model_meta_info["policy"]["noise_scheduler_args"]
            )
        elif self.model_meta_info["policy"]["scheduler"] == "ddim":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            noise_scheduler = DDIMScheduler(
                **self.model_meta_info["policy"]["noise_scheduler_args"]
            )
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid scheduler: {self.model_meta_info['policy']['scheduler']}"
            )

        # Construct policy
        if self.model_meta_info["policy"]["backbone"] == "cnn":
            pass
        elif self.model_meta_info["policy"]["backbone"] == "transformer":
            raise NotImplementedError(
                f"[{self.__class__.__name__}] The transformer backbone is not supported."
            )
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid backbone: {self.model_meta_info['policy']['backbone']}"
            )
        self.policy = self.PolicyClass(
            noise_scheduler=noise_scheduler,
            **self.model_meta_info["policy"]["args"],
        )

        # Load checkpoint
        self.load_ckpt()

        self.install_tactile_feature_hook()

    def install_tactile_feature_hook(self):
        self._latest_tactile_feat = {}  # tactile_key -> Tensor

        orig_method = self.policy.forward_tactile_encoder

        def wrapped_method(tactile_key, tactile_image):
            feat = orig_method(tactile_key, tactile_image)
            self._latest_tactile_feat[tactile_key] = feat
            if feat.requires_grad:
                feat.retain_grad()
            return feat

        self.policy.forward_tactile_encoder = wrapped_method

    @abstractmethod
    def print_policy_info_for_tactile(self):
        pass

    def override_policy_args_for_tactile(self):
        tactile_kwargs = self.model_meta_info["policy"]["args"]["tactile_kwargs"]
        tactile_kwargs["enable_finetune"] = False
        tactile_kwargs["load_checkpoint"] = False

    def load_ckpt(self, device="cuda"):
        print(f"[{self.__class__.__name__}] Load {self.args.checkpoint}")
        self.device = torch.device(device)
        self.policy.load_state_dict(
            torch.load(
                self.args.checkpoint, map_location=self.device, weights_only=True
            )
        )
        self.policy.to(self.device)
        self.policy.eval()

    def run(self):
        self.reset_accumulators()

        for data in tqdm(self.dataloader, desc="Pass1 (no_grad)"):
            self._latest_tactile_feat.clear()
            data = dict_apply(data, lambda x: x.cuda())
            self.step_nograd(data)

        self.grad_znorm_sigma = self.compute_grad_znorm_sigma()

        for data in tqdm(self.dataloader, desc="Pass2 (grad)"):
            self._latest_tactile_feat.clear()
            data = dict_apply(data, lambda x: x.cuda())
            self.step_grad(data)

        out = self.finalize_accumulators()

        if self.args.out is not None:
            os.makedirs(os.path.dirname(self.args.out), exist_ok=True)
            with open(self.args.out, "w") as f:
                yaml.safe_dump(out, f, sort_keys=False)
            print(f"[{self.__class__.__name__}] Save {self.args.out}")

        return out

    def reset_accumulators(self):
        self.sum_act_err = 0.0
        self.count_act_err = 0

        self.sum_grad_sens = 0.0
        self.count_grad_sens = 0

        self.sum_grad_sens_znorm = 0.0
        self.count_grad_sens_znorm = 0

        self.latent_list = []
        self.action_norm_list = []

    def step_nograd(self, data):
        with torch.no_grad():
            out = self.policy.predict_action(data["obs"])
            act_pred = out["action"]  # [B, n_action_steps, Da]

            n_obs = self.model_meta_info["data"]["n_obs_steps"]
            n_act = act_pred.shape[1]
            act_gt = data["action"][:, n_obs - 1 : n_obs - 1 + n_act]

            act_err = torch.norm(act_pred - act_gt, dim=-1).mean(dim=1)  # [B]
            self.sum_act_err += float(act_err.sum().cpu())
            self.count_act_err += int(act_err.numel())

            z = self.extract_latent()  # [B, D]
            self.latent_list.append(z.detach().cpu())

            a = act_gt.reshape(act_gt.shape[0], -1)  # [B, T*Da]
            self.action_norm_list.append(a.norm(dim=1).cpu())  # [B]

    def extract_latent(self):
        feats = list(self._latest_tactile_feat.values())
        if len(feats) != len(self.model_meta_info["tactile"]["tactile_names"]):
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid number of tactile features extracted: {len(feats)}"
            )
        return torch.cat(feats, dim=-1)

    def compute_grad_znorm_sigma(self):
        if len(self.latent_list) == 0:
            raise ValueError(
                f"[{self.__class__.__name__}] No latent vectors collected."
            )
        latent = torch.cat(self.latent_list, dim=0)  # [N, D]
        sigma = latent.std(dim=0)
        sigma = sigma.clamp_min(1e-6)
        return sigma.cuda()

    def step_grad(self, data):
        self.policy.zero_grad(set_to_none=True)
        loss = self.policy.compute_loss(data)
        loss.backward()

        grad_sens, grad_sens_znorm = self.extract_grad_sensitivity()

        self.sum_grad_sens += float(grad_sens)
        self.count_grad_sens += 1

        self.sum_grad_sens_znorm += float(grad_sens_znorm)
        self.count_grad_sens_znorm += 1

    def extract_grad_sensitivity(self):
        feats = list(self._latest_tactile_feat.values())
        if len(feats) != len(self.model_meta_info["tactile"]["tactile_names"]):
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid number of tactile features extracted: {len(feats)}"
            )
        elif feats[0].grad is None:
            raise ValueError(
                f"[{self.__class__.__name__}] Gradient of tactile features disabled."
            )

        grad = torch.cat([f.grad for f in feats], dim=-1)  # [B, D]
        grad_sens = grad.norm(dim=-1).mean().item()

        grad_znorm = grad * self.grad_znorm_sigma
        grad_sens_znorm = grad_znorm.norm(dim=-1).mean().item()

        return grad_sens, grad_sens_znorm

    def finalize_accumulators(self):
        out = {}
        if self.count_act_err > 0:
            out["act_err"] = self.sum_act_err / self.count_act_err

        if self.count_grad_sens > 0:
            out["grad_sens"] = self.sum_grad_sens / self.count_grad_sens

        if self.count_grad_sens_znorm > 0:
            out["grad_sens_znorm"] = (
                self.sum_grad_sens_znorm / self.count_grad_sens_znorm
            )

        if len(self.latent_list) > 0:
            latent = torch.cat(self.latent_list, dim=0)  # [N, D]
            out.update(self.compute_latent_metrics(latent))

        return out

    def compute_latent_metrics(self, latent):
        out = {}

        out["lat_std"] = float(latent.std(dim=0).mean())

        X = latent - latent.mean(dim=0, keepdim=True)  # [N, D]
        s = torch.linalg.svdvals(X)  # [min(N, D)]
        total = s.sum()
        if float(total) > 0.0:
            p = (s / total).clamp_min(1e-12)
            entropy = -(p * p.log()).sum()
            out["lat_rank"] = float(entropy.exp())
        else:
            out["lat_rank"] = 0.0

        a = torch.cat(self.action_norm_list, dim=0)  # [N]
        qs = torch.quantile(a, torch.tensor([0.25, 0.5, 0.75], device=a.device))
        bins = torch.bucketize(a, qs)  # 0..3
        bin_stats = []
        for b in range(4):
            idx = (bins == b).nonzero(as_tuple=True)[0]
            if idx.numel() < 30:
                continue
            bin_stats.append(latent[idx].std(dim=0).mean())

        if len(bin_stats) > 0:
            overall_stats = latent.std(dim=0).mean().clamp_min(1e-12)
            out["act_mod"] = float(torch.stack(bin_stats).mean() / overall_stats)
        else:
            out["act_mod"] = 0.0

        return out
