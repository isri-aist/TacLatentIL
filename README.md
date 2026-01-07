# [TacLatentIL](https://github.com/isri-aist/TacLatentIL)

Tactile-based generative policies for imitation learning

## Install
Install RoboManipBaselines as described [here](https://github.com/isri-aist/RoboManipBaselines/blob/master/doc/install.md).

Install TacLatentIL:
```console
$ git clone git@github.com:isri-aist/TacLatentIL.git --recursive
$ cd TacLatentIL
$ pip install -e .
```

> [!NOTE]
> Clone both `RoboManipBaselines` and `TacLatentIL` under the same directory, e.g.:
> ```console
> $ ls <workspace_dir>
> RoboManipBaselines  TacLatentIL
> ```

### Install for each policy
Please refer to the README of each policy listed below.

## Policies
- **[DiffusionPolicySparsh](./tac_latent_il/policy/diffusion_policy_sparsh)**: Diffusion policy with Sparsh tactile encoder
- **[DiffusionPolicyT3](./tac_latent_il/policy/diffusion_policy_t3)**: Diffusion policy with T3 tactile encoder
