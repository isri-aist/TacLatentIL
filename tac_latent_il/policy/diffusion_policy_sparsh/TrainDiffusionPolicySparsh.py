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

    def print_policy_info_for_tactile(self):
        print(f"  - sparsh encoder type: {self.args.sparsh_encoder_type}")

    def get_tactile_kwargs(self):
        return {
            "encoder_type": self.args.sparsh_encoder_type,
            "enable_finetune": self.args.enable_tactile_finetune,
        }
