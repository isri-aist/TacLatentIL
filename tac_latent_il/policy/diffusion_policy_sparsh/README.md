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

## Download Sparsh pretrained models
Download the Sparsh pretrained models:
```console
# Go to the top directory of this repository
$ cd tac_latent_il/checkpoint/Sparsh
$ wget -O mae_vitbase.ckpt https://huggingface.co/facebook/sparsh-mae-base/resolve/main/mae_vitbase.ckpt?download=true
$ wget -O dino_vitbase.ckpt https://huggingface.co/facebook/sparsh-dino-base/resolve/main/dino_vitbase.ckpt?download=true
$ wget -O dinov2_vitbase.ckpt https://huggingface.co/facebook/sparsh-dinov2-base/resolve/main/dinov2_vitbase.ckpt?download=true
$ wget -O ijepa_vitbase.ckpt https://huggingface.co/facebook/sparsh-ijepa-base/resolve/main/ijepa_vitbase.ckpt?download=true
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
$ cd guided_attention
$ python ./bin/Rollout.py DiffusionPolicySparsh MujocoUR5eCable --checkpoint ./checkpoint/DiffusionPolicySparsh/<checkpoint_name>/policy_last.ckpt
```
