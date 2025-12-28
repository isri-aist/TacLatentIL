from robo_manip_baselines.bin.Rollout import RolloutMain


class RolloutMainTLIL(RolloutMain):
    operation_parent_module_str = "robo_manip_baselines.envs.operation"
    policy_parent_module_str = "tac_latent_il.policy"
    policy_choices = [
        "DiffusionPolicySparsh",
    ]


if __name__ == "__main__":
    main = RolloutMainTLIL()
    main.run()
