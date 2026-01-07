from tac_latent_il.common import TrainDiffusionPolicyTactileBase

from .DiffusionPolicySparsh import DiffusionPolicySparsh
from .DiffusionPolicySparshDataset import DiffusionPolicySparshDataset


class TrainDiffusionPolicySparsh(TrainDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicySparsh
    DatasetClass = DiffusionPolicySparshDataset

    def set_additional_args_for_tactile(self, parser):
        parser.add_argument(
            "--sparsh_encoder_type",
            type=str,
            default="dino",
            choices=["mae", "dino", "dinov2", "ijepa"],
            help="sparsh encoder type",
        )

        parser.add_argument(
            "--enable_tactile_finetune",
            action="store_true",
            help="whether to fine-tune sparsh model",
        )

    def setup_model_meta_info_for_tactile(self):
        self.model_meta_info["policy"]["sparsh_encoder_type"] = (
            self.args.sparsh_encoder_type
        )

    def print_policy_info_for_tactile(self):
        print(
            f"  - sparsh encoder type: {self.args.sparsh_encoder_type}, fine-tune sparsh: {self.args.enable_tactile_finetune}"
        )

    def get_tactile_kwargs(self):
        return {
            "encoder_type": self.args.sparsh_encoder_type,
            "enable_finetune": self.args.enable_tactile_finetune,
        }
