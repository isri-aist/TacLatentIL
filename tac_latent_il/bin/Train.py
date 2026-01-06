from robo_manip_baselines.bin.Train import TrainMain


class TrainMainTLIL(TrainMain):
    policy_parent_module_str = "tac_latent_il.policy"
    policy_choices = [
        "DiffusionPolicySparsh",
        "DiffusionPolicyT3",
    ]


if __name__ == "__main__":
    main = TrainMainTLIL()
    main.run()
