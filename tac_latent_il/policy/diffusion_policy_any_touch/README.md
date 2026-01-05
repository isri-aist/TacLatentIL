# Diffusion Policy with AnyTouch Tactile Encoder

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
$ pip install -e .[anytouch]

# Go to the top directory of this repository
$ cd third_party/diffusion_policy
$ pip install -e .
```

## Download
Download the AnyTouch pretrained models:
```console
# Download checkpoint file from https://drive.google.com/file/d/1L4jGUjIHNBMzOiD33Rv0jxWYKHBORD1R/view?usp=sharing
# Go to the top directory of this repository
$ cd tac_latent_il/checkpoint/AnyTouch
$ mv <download location dir>/checkpoint.pth anytouch.pth
```

Copy config file from AnyTouch repository:
```console
# Go to the top directory of this repository
$ cd third_party/any_touch
$ cp ./CLIP-ViT-L-14-DataComp.XL-s13B-b90K/config.json ../../tac_latent_il/checkpoint/AnyTouch/config/AnyTouchConfig.json
```

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd tac_latent_il
$ python ./bin/Train.py DiffusionPolicyAnyTouch --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/DiffusionPolicyAnyTouch/<checkpoint_name>
```

> [!NOTE]
> If you encounter the following error,
> ```console
> ImportError: cannot import name 'cached_download' from 'huggingface_hub'
> ```
> downgrade `huggingface_hub` by the following command.
> ```console
> $ pip install huggingface_hub==0.25.0
> ```

## Policy rollout
Run a trained policy:
```console
# Go to the top directory of this repository
$ cd tac_latent_il
$ python ./bin/Rollout.py DiffusionPolicyAnyTouch RealUR5eDemo --checkpoint ./checkpoint/DiffusionPolicySparsh/<checkpoint_name>/policy_last.ckpt
```
