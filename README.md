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
- **[DiffusionPolicyAnyTouch](./tac_latent_il/policy/diffusion_policy_any_touch)**: Diffusion policy with AnyTouch tactile encoder
- **[DiffusionPolicyUniT](./tac_latent_il/policy/diffusion_policy_uni_t)**: Diffusion policy with UniT tactile encoder

## License

Original TacLatentIL code is licensed under the [BSD 2-Clause License](./LICENSE).
Adapted third-party code retains its original license; see
[THIRD_PARTY_NOTICES.txt](./THIRD_PARTY_NOTICES.txt) for attribution, usage details,
and license texts for adapted code and the principal integrated projects.

Third-party submodules retain their own licenses:

| Submodule | License |
| --- | --- |
| Diffusion Policy | [MIT](./third_party/diffusion_policy/LICENSE) |
| Sparsh | [CC BY-NC 4.0](./third_party/sparsh/LICENSE.md) (noncommercial) |
| T3 | [MIT](./third_party/t3/LICENSE) |
| AnyTouch | [MIT](./third_party/any_touch/LICENSE) |
| UniT | [MIT](./third_party/unit/LICENSE) |

The BSD license for TacLatentIL does not override these third-party terms,
including Sparsh's noncommercial restriction. Other dependencies, including
RoboManipBaselines, retain their own licenses. Pretrained models and datasets
are subject to their publishers' terms; check those terms separately from the
code licenses before use or redistribution.
