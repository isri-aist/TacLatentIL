import argparse
import importlib
import sys

from robo_manip_baselines.bin.Train import TrainMain
from robo_manip_baselines.common import remove_prefix

from tac_latent_il.mid_train import MID_TRAIN_METHODS


class MidTrainTLIL(TrainMain):
    policy_parent_module_str = "tac_latent_il.policy"
    policy_choices = [
        "DiffusionPolicySparsh",
        "DiffusionPolicyT3",
        "DiffusionPolicyAnyTouch",
        "DiffusionPolicyUniT",
    ]
    mid_train_method_choices = list(MID_TRAIN_METHODS.keys())

    def setup_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="This is a meta argument parser for mid-training switching between different policies and mid-training methods. The actual arguments are handled by another internal argument parser.",
            fromfile_prefix_chars="@",
            add_help=False,
        )
        parser.add_argument(
            "policy",
            type=str,
            nargs="?",
            default=None,
            choices=self.policy_choices,
            help="policy",
        )
        parser.add_argument(
            "mid_train_method",
            type=str,
            nargs="?",
            default=None,
            choices=self.mid_train_method_choices,
            help="mid-training method",
        )
        parser.add_argument(
            "-h",
            "--help",
            action="store_true",
            help="Show this help message and continue",
        )

        self.args, remaining_argv = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + remaining_argv
        if self.args.policy is None or self.args.mid_train_method is None:
            parser.print_help()
            sys.exit(1)
        elif self.args.help:
            parser.print_help()
            print("\n================================\n")
            sys.argv += ["--help"]

    def run(self):
        from robo_manip_baselines.common import camel_to_snake

        policy_module = importlib.import_module(
            f"{self.policy_parent_module_str}.{camel_to_snake(self.args.policy)}"
        )
        policy_name = "MidTrain" + remove_prefix(self.args.policy, "DiffusionPolicy")
        TrainPolicyClass = getattr(policy_module, policy_name)
        MidTrainModelClass = MID_TRAIN_METHODS[self.args.mid_train_method]

        TrainPolicyClass = type(
            f"{policy_name}{self.args.mid_train_method}",
            (TrainPolicyClass,),
            {"MidTrainModelClass": MidTrainModelClass},
        )

        train = TrainPolicyClass()
        train.run()
        train.close()


if __name__ == "__main__":
    main = MidTrainTLIL()
    main.run()
