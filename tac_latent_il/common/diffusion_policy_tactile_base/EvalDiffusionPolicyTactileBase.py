import argparse
import os
import pickle
import sys
from abc import ABC, abstractmethod

import numpy as np
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from robo_manip_baselines.common import RmbData, find_rmb_files, remove_prefix
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
            "--out",
            type=str,
            required=True,
            help="output npz file",
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

    def run(self):
        episode_files = find_rmb_files(self.args.dataset_dir)
        print(
            f"[{self.__class__.__name__}] Run episode-wise evaluation\n"
            f"  - #episodes: {len(episode_files)}"
        )

        all_data = []

        for ep_id, ep_file in enumerate(tqdm(episode_files, desc="Episodes")):
            ep_data = self.run_one_episode(ep_id, ep_file)
            all_data.append(ep_data)

        self.save_npz(all_data)

    def run_one_episode(self, episode_id, episode_file):
        with RmbData(episode_file) as rmb_data:
            task_cond = int(rmb_data.attrs.get("task_cond", 0))

        dataset = self.DatasetClass(
            [episode_file], self.model_meta_info, self.args.enable_rmb_cache
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        z_list = []
        grad_list = []
        act_gt_list = []
        act_pred_list = []

        for data in dataloader:
            data = dict_apply(data, lambda x: x.cuda())

            # forward action
            self.policy.zero_grad(set_to_none=True)
            self._latest_tactile_feat.clear()

            with torch.no_grad():
                out = self.policy.predict_action(data["obs"])
                act_pred = out["action"]  # [B, n_action_steps, Da]

            # forward loss
            self.policy.zero_grad(set_to_none=True)
            self._latest_tactile_feat.clear()

            loss = self.policy.compute_loss(data)
            loss.backward()

            # latent
            z = self.extract_latent().detach().cpu()
            z_list.append(z)

            # grad_sens
            grad = self.extract_grad_sensitivity().detach().cpu()
            grad_list.append(grad)

            # action gt / pred
            n_obs = self.model_meta_info["data"]["n_obs_steps"]
            n_act = act_pred.shape[1]
            act_gt = data["action"][:, n_obs - 1 : n_obs - 1 + n_act]

            act_gt_list.append(act_gt.detach().cpu())
            act_pred_list.append(act_pred.detach().cpu())

        z = torch.cat(z_list, dim=0).numpy()
        grad_sens = torch.cat(grad_list, dim=0).numpy()
        act_gt = torch.cat(act_gt_list, dim=0).numpy()
        act_pred = torch.cat(act_pred_list, dim=0).numpy()

        T = z.shape[0]
        t_norm = np.arange(T, dtype=np.float32) / max(T - 1, 1)

        return dict(
            episode_id=episode_id,
            task_cond=task_cond,
            t_norm=t_norm,
            z=z,
            grad_sens=grad_sens,
            action_gt=act_gt,
            action_pred=act_pred,
        )

    def extract_latent(self):
        feats = list(self._latest_tactile_feat.values())
        if len(feats) != len(self.model_meta_info["tactile"]["tactile_names"]):
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid number of tactile features extracted: {len(feats)}"
            )
        return torch.cat(feats, dim=-1)

    def extract_grad_sensitivity(self):
        feats = list(self._latest_tactile_feat.values())
        if feats[0].grad is None:
            raise ValueError(
                f"[{self.__class__.__name__}] Gradient of tactile features disabled."
            )
        grad = torch.cat([f.grad for f in feats], dim=-1)
        return grad.norm(dim=-1)

    def save_npz(self, episodes):
        os.makedirs(os.path.dirname(self.args.out), exist_ok=True)

        def _obj_list(key):
            return np.array([e[key] for e in episodes], dtype=object)

        np.savez(
            self.args.out,
            episode_id=np.array([e["episode_id"] for e in episodes], dtype=np.int64),
            task_cond=np.array([e["task_cond"] for e in episodes], dtype=np.int64),
            t_norm=_obj_list("t_norm"),
            z=_obj_list("z"),
            grad_sens=_obj_list("grad_sens"),
            action_gt=_obj_list("action_gt"),
            action_pred=_obj_list("action_pred"),
        )
        print(f"[{self.__class__.__name__}] Save {self.args.out}")
