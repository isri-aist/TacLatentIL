# Diffusion Policy with T3 Tactile Encoder

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
$ pip install -e .

# Go to the top directory of this repository
$ cd third_party/diffusion_policy
$ pip install -e .

# Go to the top directory of this repository
$ cd third_party/t3
$ pip install -e .
```

## Download (required only for training)
Download the T3 pretrained models:
```console
# Go to the top directory of this repository
$ cd tac_latent_il/checkpoint/T3
$ wget -O mini.pth https://huggingface.co/datasets/alanz-mit/FoundationTactile/resolve/main/models/t3_tiny/encoders/mini.pth?download=true
$ wget -O trunk.pth https://huggingface.co/datasets/alanz-mit/FoundationTactile/resolve/main/models/t3_tiny/trunk.pth?download=true
```

## Mid-training
Run tactile mid-training:
```console
# Go to the top directory of this repository
$ cd tac_latent_il
$ python ./bin/MidTrain.py DiffusionPolicyT3 TactileDynamicsInfoNCE --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/MidTrain/DiffusionPolicyT3/<checkpoint_name>
```

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd tac_latent_il
$ python ./bin/Train.py DiffusionPolicyT3 --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/DiffusionPolicyT3/<checkpoint_name>
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
$ python ./bin/Rollout.py DiffusionPolicyT3 RealUR5eDemo --checkpoint ./checkpoint/DiffusionPolicyT3/<checkpoint_name>/policy_last.ckpt
```
