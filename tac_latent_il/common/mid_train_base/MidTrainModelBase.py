import torch.nn as nn


class MidTrainModelBase(nn.Module):
    @classmethod
    def set_additional_args(cls, parser):
        pass

    @classmethod
    def get_additional_model_args(cls, args):
        return {}

    def on_after_optimizer_step(self):
        pass

    def get_model_info(self):
        return ""
