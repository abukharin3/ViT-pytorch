#!/bin/sh

python3 train.py --name cifar100_500 --dataset cifar100 --model_type ViT-B_16 --prune_schedule cubic \
    --warmup_steps 3 --initial_threshold 1 --final_threshold 0.90 --initial_warmup 1 --final_warmup 3 \
    --beta3 0.85 --beta_meta 0.95 --deltaT 10 --train_batch_size 512 --num_steps 20000 --seed 9 \
    --device_num 0,1,2,3 --pretrained_dir ../model_cache/ViT-B_16.npz --output_root_folder ../DataLog/vit/adp_log --output_dir debug

python3 train.py --name cifar100_500 --device_num 0,1,2,3 --prune_schedule cubic --warmup_steps 3 --initial_threshold 1 --final_threshold 0.90 --initial_warmup 1 --final_warmup 3 --beta3 0.85 --beta_meta 0.95 --deltaT 10 --dataset cifar100 --model_type ViT-B_16 --train_batch_size 512 --num_steps 20000 --seed 9 --pretrained_dir ../model_cache/ViT-B_16.npz --output_root_folder ../DataLog/vit/adp_log --output_dir debug

python3 train.py --name cifar100_500 --device_num 4,5,6,7 --prune_schedule cubic --initial_threshold 1 --final_threshold 0.15 --warmup_steps 3000 --initial_warmup 1 --final_warmup 3 --beta3 0.85 --beta_meta 0.90 --deltaT 1  --dataset cifar100 --model_type ViT-B_16 --train_batch_size 512 --num_steps 20000 --eval_every 3000 --seed 9 --pretrained_dir ../model_cache/ViT-B_16.npz --output_root_folder ../DataLog/vit/adp_log --output_dir run
python3 train.py --name cifar100_500 --device_num 4,5,6,7 --prune_schedule cubic --initial_threshold 1 --final_threshold 0.15 --warmup_steps 3000 --initial_warmup 1 --final_warmup 3 --beta3 0.85 --beta_meta 0.90 --deltaT 5  --dataset cifar100 --model_type ViT-B_16 --train_batch_size 512 --num_steps 20000 --eval_every 3000 --seed 9 --pretrained_dir ../model_cache/ViT-B_16.npz --output_root_folder ../DataLog/vit/adp_log --output_dir run
python3 train.py --name cifar100_500 --device_num 0,1,6,7 --prune_schedule cubic --initial_threshold 1 --final_threshold 0.20 --warmup_steps 3000 --initial_warmup 1 --final_warmup 3 --beta3 0.85 --beta_meta 0.90 --deltaT 5  --dataset cifar100 --model_type ViT-B_16 --train_batch_size 512 --num_steps 20000 --eval_every 3000 --seed 9 --pretrained_dir ../model_cache/ViT-B_16.npz --output_root_folder ../DataLog/vit/adp_log --output_dir run
