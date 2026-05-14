import torch
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from robo_manip_baselines.common import DataKey
from tqdm import tqdm


class MidTrainTactileMixin:
    MidTrainModelClass = None  # must be overridden
    TactileEncoderClass = None  # must be overridden

    @property
    def policy_name(self):
        return self.__class__.__name__.removeprefix("MidTrain")

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
        parser.set_defaults(lr=1e-4)

        parser.add_argument("--tactile_feature_dim", type=int, default=64)
        self.MidTrainModelClass.set_additional_args(parser)

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["policy"]["mid_train_model"] = (
            self.MidTrainModelClass.__name__
        )
        self.model_meta_info["policy"]["mid_train_model_args"] = (
            self.MidTrainModelClass.get_additional_model_args(self.args)
        )
        self.model_meta_info["policy"]["tactile_encoder"] = (
            self.TactileEncoderClass.__name__
        )
        self.model_meta_info["policy"]["tactile_encoder_args"] = dict(
            tactile_keys=self.args.tactile_names,
            tactile_feature_dim=self.args.tactile_feature_dim,
            tactile_kwargs=self.get_tactile_kwargs(),
        )

    def setup_policy(self):
        # Construct only mid-training model (no diffusion policy)
        action_dim = len(self.model_meta_info["action"]["example"])

        self.model_meta_info["policy"]["args"] = dict(
            action_dim=action_dim,
            **self.model_meta_info["policy"]["mid_train_model_args"],
        )

        self.tactile_encoder = self.TactileEncoderClass(
            **self.model_meta_info["policy"]["tactile_encoder_args"],
        )

        self.policy = self.MidTrainModelClass(
            tactile_encoder=self.tactile_encoder,
            **self.model_meta_info["policy"]["args"],
        )

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

        self.policy.cuda()
        optimizer_to(self.optimizer, "cuda")

        self.print_policy_info()
        print(
            f"  - horizon: {self.args.horizon}, obs steps: {self.args.n_obs_steps}, action steps: {self.args.n_action_steps}\n"
            f"  - tactile names: {self.args.tactile_names}, tactile fine-tune: {self.args.enable_tactile_finetune}"
        )
        model_info = self.policy.get_model_info()
        if model_info:
            print(model_info)
        self.print_policy_info_for_tactile()

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                data = dict_apply(data, lambda x: x.cuda())
                loss_dict = self.policy.forward_batch(data)
                loss = loss_dict["loss"]
                loss.backward()
                self.optimizer.step()
                self.policy.on_after_optimizer_step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                batch_result_list.append(
                    self.detach_batch_result(
                        {**loss_dict, "lr": self.lr_scheduler.get_last_lr()[0]}
                    )
                )
            self.log_epoch_summary(batch_result_list, "train", epoch)

            self.policy.eval()
            with torch.inference_mode():
                batch_result_list = []
                for data in self.val_dataloader:
                    data = dict_apply(data, lambda x: x.cuda())
                    loss_dict = self.policy.forward_batch(data)
                    batch_result_list.append(self.detach_batch_result(loss_dict))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)
                self.update_best_ckpt(epoch_summary, policy=self.policy)

            # if epoch % max(self.args.num_epochs // 10, 1) == 0:
            #     self.save_current_ckpt(f"epoch{epoch:0>4}", policy=self.policy)

        self.save_current_ckpt("last", policy=self.policy)
        self.save_best_ckpt()

    def get_mid_trained_tactile_encoder_state_dict(self, policy=None):
        if policy is None:
            policy = self.policy
        return policy.encoder.get_tactile_encoder_state_dict()

    def update_best_ckpt(self, epoch_summary, policy=None):
        if policy is None:
            policy = self.policy

        if epoch_summary["loss"] < self.best_ckpt_info["loss"]:
            self.best_ckpt_info = {
                "epoch": epoch_summary["epoch"],
                "loss": epoch_summary["loss"],
                "state_dict": self.get_mid_trained_tactile_encoder_state_dict(policy),
            }

    def save_current_ckpt(self, ckpt_suffix, policy=None):
        ckpt_path = f"{self.args.checkpoint_dir}/tactile_encoder_{ckpt_suffix}.ckpt"
        torch.save(self.get_mid_trained_tactile_encoder_state_dict(policy), ckpt_path)

    def save_best_ckpt(self):
        ckpt_path = f"{self.args.checkpoint_dir}/tactile_encoder_best.ckpt"
        torch.save(self.best_ckpt_info["state_dict"], ckpt_path)
        print(
            f"[{self.__class__.__name__}] Best val loss is {self.best_ckpt_info['loss']:.3f} at epoch {self.best_ckpt_info['epoch']}"
        )
