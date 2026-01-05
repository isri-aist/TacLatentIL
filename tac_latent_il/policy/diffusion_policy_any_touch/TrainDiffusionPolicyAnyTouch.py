from transformers import CLIPConfig

from tac_latent_il.common import TrainDiffusionPolicyTactileBase

from .DiffusionPolicyAnyTouch import DiffusionPolicyAnyTouch
from .DiffusionPolicyAnyTouchDataset import DiffusionPolicyAnyTouchDataset


class TrainDiffusionPolicyAnyTouch(TrainDiffusionPolicyTactileBase):
    PolicyClass = DiffusionPolicyAnyTouch
    DatasetClass = DiffusionPolicyAnyTouchDataset

    def set_additional_args_for_tactile(self, parser):
        parser.add_argument(
            "--anytouch_config",
            type=str,
            default="./checkpoint/AnyTouch/config/AnyTouchConfig.json",
            help="model config file (JSON) of anytouch",
        )

        parser.add_argument(
            "--tactile_pooling",
            type=str,
            default="cls",
            choices=["cls", "global"],
            help="anytouch pooling method",
        )
        parser.add_argument(
            "--tactile_sensor_names",
            type=str,
            nargs="+",
            default=["gelsight_mini", "gelsight_mini"],
            help="sensor name of each tactile",
        )
        parser.add_argument(
            "--use_universal_sensor_token",
            action="store_true",
            help="Use universal sensor token instead of specific sensor token",
        )
        parser.add_argument(
            "--use_sensor_token",
            action="store_true",
            help="Use sensor token for tactile embedding",
        )
        parser.add_argument(
            "--use_same_patchemb",
            action="store_true",
            help="Use same patchemb in AnyTouch or not",
        )

    def setup_model_meta_info_for_tactile(self):
        self.model_meta_info["data"]["use_universal_sensor_token"] = (
            self.args.use_universal_sensor_token
        )
        self.model_meta_info["data"]["tactile_sensor_names"] = (
            self.args.tactile_sensor_names
        )
        self.model_meta_info["policy"]["anytouch_config"] = CLIPConfig.from_json_file(
            self.args.anytouch_config
        )

    def print_policy_info_for_tactile(self):
        print(
            f"  - pooling type: {self.args.tactile_pooling}, tactile sensor names: {self.args.tactile_sensor_names}"
        )
        print(
            f"  - same patchemb: {self.args.use_same_patchemb}, use sensor token: {self.args.use_sensor_token}, use universal sensor: {self.args.use_universal_sensor_token}"
        )

    def get_tactile_kwargs(self):
        return {
            "anytouch_config": self.model_meta_info["policy"]["anytouch_config"],
            "n_obs_steps": self.args.n_obs_steps,
            "use_sensor_token": self.args.use_sensor_token,
            "use_same_patchemb": self.args.use_same_patchemb,
            "pooling_method": self.args.tactile_pooling,
        }
