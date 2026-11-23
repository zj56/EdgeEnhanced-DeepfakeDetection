#!/bin/bash


python main.py --batch_size=32 --method=xception_sobel_pass_npr  --lr=0.0002 --optim_type=SGD  --dataset=cnnsport --gpu=3
python main.py --batch_size=80 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=AdamW  --dataset=fdmas --gpu=3 --normalize=imagenet
python main.py --batch_size=80 --method=xception_sobel_pass_npr  --lr=0.02 --optim_type=AdamW  --dataset=assess --gpu=3 --normalize=imagenet
