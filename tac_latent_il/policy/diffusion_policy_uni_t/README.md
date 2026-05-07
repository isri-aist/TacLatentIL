# Diffusion Policy with UniT Tactile Encoder

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
$ pip install -e .[unit]

# Go to the top directory of this repository
$ cd third_party/diffusion_policy
$ pip install -e .

# Go to the top directory of this repository
$ cd third_party/unit
$ pip install -e .
```

## Download (required only for training)
Download the UniT pretrained model:
```console
# Go to the top directory of this repository
# Activate the venv environment for gdown
$ cd tac_latent_il/checkpoint/UniT
$ gdown https://drive.google.com/uc?id=12uVJZGyjcHazZ_bezORexZTzWtctOSDq
```

## Mid-training
Run tactile mid-training:
```console
# Go to the top directory of this repository
$ cd tac_latent_il
$ python ./bin/MidTrain.py DiffusionPolicyUniT TactileDynamicsInfoNCE --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/MidTrain/DiffusionPolicyUniT/<checkpoint_name>
```

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd tac_latent_il
$ python ./bin/Train.py DiffusionPolicyUniT --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/DiffusionPolicyUniT/<checkpoint_name>
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
$ python ./bin/Rollout.py DiffusionPolicyUniT RealUR5eDemo --checkpoint ./checkpoint/DiffusionPolicyUniT/<checkpoint_name>/policy_last.ckpt
```
