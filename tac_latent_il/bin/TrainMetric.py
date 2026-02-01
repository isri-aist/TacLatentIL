import importlib

from robo_manip_baselines.bin.Train import TrainMain
from robo_manip_baselines.common import remove_prefix


class TrainMetricTLIL(TrainMain):
    policy_parent_module_str = "tac_latent_il.policy"
    policy_choices = [
        "DiffusionPolicySparsh",
        "DiffusionPolicyT3",
        "DiffusionPolicyAnyTouch",
        "DiffusionPolicyUniT",
    ]

    def run(self):
        from robo_manip_baselines.common import camel_to_snake

        policy_module = importlib.import_module(
            f"{self.policy_parent_module_str}.{camel_to_snake(self.args.policy)}"
        )
        policy_name = "TrainTactileMetric" + remove_prefix(
            self.args.policy, "DiffusionPolicy"
        )
        TrainPolicyClass = getattr(policy_module, policy_name)

        train = TrainPolicyClass()
        train.run()
        train.close()


if __name__ == "__main__":
    main = TrainMetricTLIL()
    main.run()
