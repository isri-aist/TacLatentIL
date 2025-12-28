import torch
from tactile_ssl.downstream_task.attentive_pooler import AttentivePooler
from tactile_ssl.model.vision_transformer import vit_base

from tac_latent_il.common import DiffusionPolicyTactileBase


class DiffusionPolicySparsh(DiffusionPolicyTactileBase):
    def setup_tactile_encoder(
        self, output_dim=64, encoder_type="dino", checkpoint=None
    ):
        self.sparsh_model = vit_base(
            in_chans=6, pos_embed_fn="sinusoidal", num_register_tokens=1
        )
        self.atten_pooler = AttentivePooler()
        embed_dim = self.atten_pooler.query_tokens.shape[-1]
        self.linear = torch.nn.Linear(embed_dim, output_dim)

        if checkpoint is not None:
            checkpoint_data = torch.load(checkpoint, weights_only=False)
        if "jepa" in encoder_type:
            encoder_key = "target_encoder"
        elif "dino" in encoder_type:
            encoder_key = "teacher_encoder.backbone"
        else:
            encoder_key = "encoder"
        target_keys = [
            key for key in checkpoint_data["model"].keys() if encoder_key in key
        ]
        if "backbone" in target_keys[0] and "backbone" not in encoder_key:
            encoder_key = encoder_key + ".backbone"
        new_keys = [key.replace(f"{encoder_key}.", "") for key in target_keys]
        new_state_dict = {
            new_key: checkpoint_data["model"][target_key]
            for new_key, target_key in zip(new_keys, target_keys)
        }
        self.sparsh_model.load_state_dict(new_state_dict, strict=False)

    def calc_tactile_feature(self, tactile_image):
        sparsh_output = self.sparsh_model(tactile_image)
        tactile_feature = self.linear(self.atten_pooler(sparsh_output).squeeze(1))
        return tactile_feature
