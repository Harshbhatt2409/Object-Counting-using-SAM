# count-anything-yuhao


## Install
Install python dependencies. We use conda and python 3.10.4 and PyTorch 1.13.1
> conda env create -f env.yaml

## Dataset preparation
- For FSC-147:
    Images can be downloaded from here: https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing


## Test
Download the [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

- For FSC-147:
```
python test_FSC.py --data_path <FSC-147 dataset path> --model_path <path to ViT-H SAM model>
```


