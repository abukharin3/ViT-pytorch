#!/bin/sh

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 43 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.1 \
	--move_prune --device_num 0,1,2,3

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 43 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.2 \
	--move_prune --device_num 0,1,2,3

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 43 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.4 \
	--move_prune --device_num 0,1,2,3

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 42 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.6 \
	--move_prune --device_num 0,1,2,3

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 42 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.8 \
	--move_prune --device_num 0,1,2,3

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 42 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.9 \
	--move_prune --device_num 0,1,2,3

