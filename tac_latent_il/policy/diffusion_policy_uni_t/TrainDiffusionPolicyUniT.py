from tac_latent_il.common import TrainDiffusionPolicyTactileBase

from .DiffusionPolicyUniT import DiffusionPolicyUniT
from .DiffusionPolicyUniTDataset import DiffusionPolicyUniTDataset


class TrainDiffusionPolicyUniT(TrainDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicyUniT
    DatasetClass = DiffusionPolicyUniTDataset

    def set_additional_args_for_tactile(self, parser):
        pass

    def print_policy_info_for_tactile(self):
        pass

    def get_tactile_kwargs(self):
        return {
            "enable_finetune": self.args.enable_tactile_finetune,
        }
