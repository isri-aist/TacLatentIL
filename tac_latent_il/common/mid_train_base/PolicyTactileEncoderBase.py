import torch.nn as nn


class PolicyTactileEncoderBase(nn.Module):
    PolicyClass = None  # must be overridden

    def __init__(
        self,
        tactile_keys,
        tactile_feature_dim=64,
        tactile_kwargs=None,
    ):
        super().__init__()
        if tactile_kwargs is None:
            tactile_kwargs = {}
        self.tactile_keys = tactile_keys
        self.tactile_feature_dim = tactile_feature_dim

        self.policy = self.PolicyClass.__new__(self.PolicyClass)  # bypass __init__
        nn.Module.__init__(self.policy)
        self.policy.tactile_keys = self.tactile_keys
        self.policy.setup_tactile_encoder(
            output_dim=tactile_feature_dim, **tactile_kwargs
        )
        self.tactile_model = self.policy.tactile_model

    def forward_one(self, tactile_key, tactile_image):
        return self.policy.forward_tactile_encoder(tactile_key, tactile_image)

    def get_tactile_encoder_state_dict(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.tactile_model.state_dict().items()
        }
