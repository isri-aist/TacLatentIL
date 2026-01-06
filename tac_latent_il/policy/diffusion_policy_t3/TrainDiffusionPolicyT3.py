from tac_latent_il.common import TrainDiffusionPolicyTactileBase

from .DiffusionPolicyT3 import DiffusionPolicyT3
from .DiffusionPolicyT3Dataset import DiffusionPolicyT3Dataset


class TrainDiffusionPolicyT3(TrainDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicyT3
    DatasetClass = DiffusionPolicyT3Dataset

    def set_additional_args_for_tactile(self, parser):
        parser.add_argument(
            "--t3_pooling_type",
            type=str,
            default="cls",
            choices=["global", "cls"],
            help="t3 pooling type",
        )

        parser.add_argument(
            "--enable_tactile_finetune",
            action="store_true",
            help="whether to fine-tune t3 model",
        )

    def setup_model_meta_info_for_tactile(self):
        self.model_meta_info["policy"]["t3_pooling_type"] = self.args.t3_pooling_type

    def print_policy_info_for_tactile(self):
        print(
            f"  - t3 type: {self.args.t3_pooling_type}, fine-tune t3: {self.args.enable_tactile_finetune}"
        )

    def get_tactile_kwargs(self):
        return {
            "pooling_type": self.args.t3_pooling_type,
            "enable_finetune": self.args.enable_tactile_finetune,
        }
