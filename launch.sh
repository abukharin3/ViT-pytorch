#!/bin/sh

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 42 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.1 \
	--move_prune False

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 42 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.1 \
	--move_prune True

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 42 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.15 \
	--move_prune False

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz \
	--train_batch_size 512 --num_steps 20000 --seed 42 --initial_warmup 2000 --final_warmup 8000 --final_threshold 0.15 \
	--move_prune True
