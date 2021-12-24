#!/bin/sh

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 42 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.4 \
	--beta3 0.75 --ma_uncertainty --ma_beta 0.85 --device_num 0,1,2,3

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 42 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.5 \
	--beta3 0.75 --ma_uncertainty --ma_beta 0.85 --device_num 0,1,2,3

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 42 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.6 \
	--beta3 0.75 --ma_uncertainty --ma_beta 0.85 --device_num 0,1,2,3

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 42 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.7 \
	--beta3 0.75 --ma_uncertainty --ma_beta 0.85 --device_num 0,1,2,3

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 42 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.8 \
	--beta3 0.75 --ma_uncertainty --ma_beta 0.85 --device_num 0,1,2,3

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 42 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.9 \
	--beta3 0.75 --ma_uncertainty --ma_beta 0.85 --device_num 0,1,2,3

