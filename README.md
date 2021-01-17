# GrETTA
Gradient-Estimation Test-time Augmentation(Unofficial)

#### Prequisities

1. Install `efficientnet-pytorch`, `kornia`, and `tqdm` library
2. Download Imagenet-C dataset



### Usage

1. Generate corrupted validation set

   `python gretta/gen_val.py`

2. Train and validation

   `python imagenet.py`