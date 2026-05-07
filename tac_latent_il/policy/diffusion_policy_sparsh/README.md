# Diffusion Policy with Sparsh Tactile Encoder

## Install
Install RoboManipBaselines:
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

$ git clone git@github.com:isri-aist/RoboManipBaselines.git --recursive
$ cd RoboManipBaselines
$ pip install -e .[diffusion-policy]

# Do not install the following third_party/diffusion_policy in RoboManipBaselines.
# $ cd third_party/diffusion_policy
# $ pip install -e .
```

Install TacLatentIL:
```console
$ git clone git@github.com:isri-aist/TacLatentIL.git --recursive
$ cd TacLatentIL
$ pip install -e .[sparsh]

# Go to the top directory of this repository
$ cd third_party/diffusion_policy
$ pip install -e .

# Go to the top directory of this repository
$ cd third_party/sparsh
$ pip install -e .
```

> [!NOTE]
> If you encounter the following error,
> ```console
> File ".../lib/python3.12/site-packages/xformers/ops/fmha/dispatch.py", line 83, in _run_priority_list
>     raise NotImplementedError(msg)
> NotImplementedError: No operator found for `memory_efficient_attention_forward` with inputs:
>      query       : shape=(32, 197, 12, 64) (torch.float32)
>      key         : shape=(32, 197, 12, 64) (torch.float32)
>      value       : shape=(32, 197, 12, 64) (torch.float32)
>      attn_bias   : <class 'NoneType'>
>      p           : 0.0
> `fa3F@0.0.0` is not supported because:
>     requires device with capability <= (9, 0) but your GPU has capability (12, 0) (too new)
>     dtype=torch.float32 (supported: {torch.float16, torch.bfloat16})
> `fa2F@2.5.7-pt` is not supported because:
>     dtype=torch.float32 (supported: {torch.float16, torch.bfloat16})
> `cutlassF-pt` is not supported because:
>     requires device with capability <= (9, 0) but your GPU has capability (12, 0) (too new)
> ```
> uninstall `xformers` by the following command.
> ```console
> $ pip uninstall -y xformers
> ```

## Download (required only for training)
Download the Sparsh pretrained models:
```console
# Go to the top directory of this repository
$ cd tac_latent_il/checkpoint/Sparsh
$ wget -O mae_vitbase.ckpt https://huggingface.co/facebook/sparsh-mae-base/resolve/main/mae_vitbase.ckpt?download=true
$ wget -O dino_vitbase.ckpt https://huggingface.co/facebook/sparsh-dino-base/resolve/main/dino_vitbase.ckpt?download=true
$ wget -O dinov2_vitbase.ckpt https://huggingface.co/facebook/sparsh-dinov2-base/resolve/main/dinov2_vitbase.ckpt?download=true
$ wget -O ijepa_vitbase.ckpt https://huggingface.co/facebook/sparsh-ijepa-base/resolve/main/ijepa_vitbase.ckpt?download=true
```

## Mid-training
Run tactile mid-training:
```console
# Go to the top directory of this repository
$ cd tac_latent_il
$ python ./bin/MidTrain.py DiffusionPolicySparsh TactileDynamicsInfoNCE --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/MidTrain/DiffusionPolicySparsh/<checkpoint_name>
```

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd tac_latent_il
$ python ./bin/Train.py DiffusionPolicySparsh --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/DiffusionPolicySparsh/<checkpoint_name>
```

> [!NOTE]
> If you encounter the following error,
> ```console
> ImportError: cannot import name 'cached_download' from 'huggingface_hub'
> ```
> downgrade `huggingface_hub` by the following command.
> ```console
> $ pip install huggingface_hub==0.21.4
> ```

## Policy rollout
Run a trained policy:
```console
# Go to the top directory of this repository
$ cd tac_latent_il
$ python ./bin/Rollout.py DiffusionPolicySparsh RealUR5eDemo --checkpoint ./checkpoint/DiffusionPolicySparsh/<checkpoint_name>/policy_last.ckpt
```
