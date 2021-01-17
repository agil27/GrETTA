# GrETTA
Gradient-Estimation Test-time Augmentation(Unofficial)

#### Prequisities

1. Install `efficientnet-pytorch`, `kornia`, and `tqdm` library
2. Download Imagenet-C dataset



### Usage

1. Generate corrupted validation set

   `python gretta/gen_val.py`

2. Train and validation

   - Train gradient oracle with lr set to 1e-8, batch size set to 256, using resnet50 as black box, using resnet18 as policy network backbone, for 50 epochs and evaluate on clean test set.
   
     `python imagenet.py -l 1e-8 -b 256 -m resnet50 -p resnet18 -e 50 -t oracle -v clean`
   
   - Train vanilla gradient estimation with num of samples set to 12, lr set to 1e-8, batch size set to 256, using augmix resnet50 as black box, using efficient net as policy network backbone, for 90 epochs and evaluate on corrupt test set.
   
     `python imagenet.py -l 1e-8 -b 256 -m augmix -p efficientnet -e 90 -t vanilla -v corrupt`
   
   - Train guided gradient estimation with the same hyperparameters as vanilla gradient estimation:
   
     `python imagenet.py -l 1e-8 -b 256 -m augmix -p efficientnet -e 90 -t vanilla -v corrupt`