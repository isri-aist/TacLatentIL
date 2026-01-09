from tac_latent_il.common import TrainDiffusionPolicyTactileBase

from .DiffusionPolicyUniT import DiffusionPolicyUniT
from .DiffusionPolicyUniTDataset import DiffusionPolicyUniTDataset


class TrainDiffusionPolicyUniT(TrainDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicyUniT
    DatasetClass = DiffusionPolicyUniTDataset

    def set_additional_args_for_tactile(self, parser):
        parser.add_argument(
            "--enable_tactile_finetune",
            action="store_true",
            help="whether to fine-tune unit model",
        )

    def print_policy_info_for_tactile(self):
        print(f"  - fine-tune unit: {self.args.enable_tactile_finetune}")

    def get_tactile_kwargs(self):
        return {
            "enable_finetune": self.args.enable_tactile_finetune,
        }
