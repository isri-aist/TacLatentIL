from tac_latent_il.common import TrainDiffusionPolicyTactileBase

from .DiffusionPolicyAnyTouch import DiffusionPolicyAnyTouch
from .DiffusionPolicyAnyTouchDataset import DiffusionPolicyAnyTouchDataset


class TrainDiffusionPolicyAnyTouch(TrainDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicyAnyTouch
    DatasetClass = DiffusionPolicyAnyTouchDataset

    def set_additional_args_for_tactile(self, parser):
        parser.add_argument(
            "--tactile_sensor_name",
            type=str,
            default="gelsight_mini",
            help="sensor name of tactiles",
        )
        parser.add_argument(
            "--use_universal_sensor_token",
            action="store_true",
            help="whether to use universal sensor token instead of specific sensor token in anytouch model",
        )
        parser.add_argument(
            "--use_sensor_token",
            action="store_true",
            help="whether to use sensor token for tactile embedding in anytouch model",
        )
        parser.add_argument(
            "--use_same_patchemb",
            action="store_true",
            help="whether to use same patchemb in anytouch model",
        )
        parser.add_argument(
            "--tactile_pooling",
            type=str,
            default="cls",
            choices=["cls", "global"],
            help="pooling method in anytouch model",
        )

        parser.add_argument(
            "--enable_tactile_finetune",
            action="store_true",
            help="whether to fine-tune AnyTouch model",
        )

    def setup_model_meta_info_for_tactile(self):
        pass

    def print_policy_info_for_tactile(self):
        print(
            f"  - tactile sensor name: {self.args.tactile_sensor_name}, finetune anytouch: {self.args.enable_tactile_finetune}\n"
            f"  - universal token: {self.args.use_universal_sensor_token}, sensor token: {self.args.use_sensor_token}, same patchemb: {self.args.use_same_patchemb}, pooling type: {self.args.tactile_pooling}"
        )

    def get_tactile_kwargs(self):
        return {
            "sensor_name": self.args.tactile_sensor_name,
            "use_universal_sensor_token": self.args.use_universal_sensor_token,
            "use_sensor_token": self.args.use_sensor_token,
            "use_same_patchemb": self.args.use_same_patchemb,
            "pooling_method": self.args.tactile_pooling,
            "enable_finetune": self.args.enable_tactile_finetune,
        }
