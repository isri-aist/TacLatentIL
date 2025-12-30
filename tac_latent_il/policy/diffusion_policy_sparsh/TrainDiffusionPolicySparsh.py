import os

from tac_latent_il.common import TrainDiffusionPolicyTactileBase

from .DiffusionPolicySparsh import DiffusionPolicySparsh
from .DiffusionPolicySparshDataset import DiffusionPolicySparshDataset


class TrainDiffusionPolicySparsh(TrainDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicySparsh
    DatasetClass = DiffusionPolicySparshDataset

    def set_additional_args_for_tactile(self, parser):
        parser.add_argument(
            "--sparsh_checkpoint",
            type=str,
            default=None,
            help="sparsh pretrained checkpoint",
        )

        parser.add_argument(
            "--sparsh_encoder_type",
            type=str,
            default="dino",
            choices=["mae", "dino", "dinov2", "ijepa"],
            help="sparsh encoder type",
        )

    def setup_model_meta_info_for_tactile(self):
        self.model_meta_info["policy"]["sparsh_encoder_type"] = (
            self.args.sparsh_encoder_type
        )

    def print_policy_info_for_tactile(self):
        sparsh_checkpoint_str = (
            os.path.basename(self.args.sparsh_checkpoint)
            if isinstance(self.args.sparsh_checkpoint, str)
            else "None"
        )
        print(
            f"  - sparsh checkpoint: {sparsh_checkpoint_str}, sparsh type: {self.args.sparsh_encoder_type}"
        )

    def get_tactile_kwargs(self):
        return {
            "checkpoint": self.args.sparsh_checkpoint,
            "encoder_type": self.args.sparsh_encoder_type,
        }
