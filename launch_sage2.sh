#!/bin/sh


python3 train.py --name cifar100_500 --device_num 0,1,2,3 --pruner_name Movement --prune_schedule cubic --initial_threshold 1 --final_threshold 0.3 --warmup_steps 1000 --initial_warmup 3 --final_warmup 10 --beta3 0.85 --beta_meta 1. --deltaT 5 --dataset ImageNet --model_type ViT-B_16 --train_batch_size 150 --num_steps 30000 --eval_every 3000 --seed 9 --pretrained_dir /mnt/data/vit_pruning/model_cache/ViT-B_16-finetune.npz --output_root_folder /mnt/data/vit_pruning/log/mp_log --output_dir run --optim_warmup_steps 1000 --img_size 384
python3 train.py --name cifar100_500 --device_num 0,1,2,3 --pruner_name Movement --prune_schedule cubic --initial_threshold 1 --final_threshold 0.2 --warmup_steps 1000 --initial_warmup 3 --final_warmup 10 --beta3 0.85 --beta_meta 1. --deltaT 5 --dataset ImageNet --model_type ViT-B_16 --train_batch_size 150 --num_steps 30000 --eval_every 3000 --seed 9 --pretrained_dir /mnt/data/vit_pruning/model_cache/ViT-B_16-finetune.npz --output_root_folder /mnt/data/vit_pruning/log/mp_log --output_dir run --optim_warmup_steps 1000 --img_size 384
python3 train.py --name cifar100_500 --device_num 0,1,2,3 --pruner_name Movement --prune_schedule cubic --initial_threshold 1 --final_threshold 0.15 --warmup_steps 1000 --initial_warmup 3 --final_warmup 10 --beta3 0.85 --beta_meta 1. --deltaT 5 --dataset ImageNet --model_type ViT-B_16 --train_batch_size 150 --num_steps 30000 --eval_every 3000 --seed 9 --pretrained_dir /mnt/data/vit_pruning/model_cache/ViT-B_16-finetune.npz --output_root_folder /mnt/data/vit_pruning/log/mp_log --output_dir run --optim_warmup_steps 1000 --img_size 384
python3 train.py --name cifar100_500 --device_num 0,1,2,3 --pruner_name Movement --prune_schedule cubic --initial_threshold 1 --final_threshold 0.1 --warmup_steps 1000 --initial_warmup 3 --final_warmup 10 --beta3 0.85 --beta_meta 1. --deltaT 5 --dataset ImageNet --model_type ViT-B_16 --train_batch_size 150 --num_steps 30000 --eval_every 3000 --seed 9 --pretrained_dir /mnt/data/vit_pruning/model_cache/ViT-B_16-finetune.npz --output_root_folder /mnt/data/vit_pruning/log/mp_log --output_dir run --optim_warmup_steps 1000 --img_size 384





# python3 train.py --name cifar100_500 --device_num 0,1,2,3 --prune_schedule cubic --pruner_name Movement --initial_threshold 1 --final_threshold 0.15 --warmup_steps 3000 --initial_warmup 1 --final_warmup 3 --beta3 0.85 --beta_meta 0.90 --deltaT 5  --dataset cifar100 --model_type ViT-B_16 --train_batch_size 512 --num_steps 20000 --eval_every 1000 --seed 9 --pretrained_dir model_cache/ViT-B_16.npz --output_root_folder DataLog/vit/adp_log --output_dir run
# python3 train.py --name cifar100_500 --device_num 0,1,2,3 --prune_schedule cubic --pruner_name Movement --initial_threshold 1 --final_threshold 0.2 --warmup_steps 3000 --initial_warmup 1 --final_warmup 3 --beta3 0.85 --beta_meta 0.90 --deltaT 5  --dataset cifar100 --model_type ViT-B_16 --train_batch_size 512 --num_steps 20000 --eval_every 1000 --seed 9 --pretrained_dir model_cache/ViT-B_16.npz --output_root_folder DataLog/vit/adp_log --output_dir run
# python3 train.py --name cifar100_500 --device_num 0,1,2,3 --prune_schedule cubic --pruner_name Movement --initial_threshold 1 --final_threshold 0.3 --warmup_steps 3000 --initial_warmup 1 --final_warmup 3 --beta3 0.85 --beta_meta 0.90 --deltaT 5  --dataset cifar100 --model_type ViT-B_16 --train_batch_size 512 --num_steps 20000 --eval_every 1000 --seed 9 --pretrained_dir model_cache/ViT-B_16.npz --output_root_folder DataLog/vit/adp_log --output_dir run
# python3 train.py --name cifar100_500 --device_num 0,1,2,3 --prune_schedule cubic --pruner_name Movement --initial_threshold 1 --final_threshold 0.1 --warmup_steps 3000 --initial_warmup 1 --final_warmup 3 --beta3 0.85 --beta_meta 0.90 --deltaT 5  --dataset cifar100 --model_type ViT-B_16 --train_batch_size 512 --num_steps 20000 --eval_every 1000 --seed 9 --pretrained_dir model_cache/ViT-B_16.npz --output_root_folder DataLog/vit/adp_log --output_dir run
# python3 train.py --name cifar100_500 --device_num 0,1,2,3 --prune_schedule cubic --pruner_name Magnitude --initial_threshold 1 --final_threshold 0.15 --warmup_steps 3000 --initial_warmup 1 --final_warmup 3 --beta3 0.85 --beta_meta 0.90 --deltaT 5  --dataset cifar100 --model_type ViT-B_16 --train_batch_size 512 --num_steps 20000 --eval_every 1000 --seed 9 --pretrained_dir model_cache/ViT-B_16.npz --output_root_folder DataLog/vit/adp_log --output_dir run
# python3 train.py --name cifar100_500 --device_num 0,1,2,3 --prune_schedule cubic --pruner_name Magnitude --initial_threshold 1 --final_threshold 0.2 --warmup_steps 3000 --initial_warmup 1 --final_warmup 3 --beta3 0.85 --beta_meta 0.90 --deltaT 5  --dataset cifar100 --model_type ViT-B_16 --train_batch_size 512 --num_steps 20000 --eval_every 1000 --seed 9 --pretrained_dir model_cache/ViT-B_16.npz --output_root_folder DataLog/vit/adp_log --output_dir run
# python3 train.py --name cifar100_500 --device_num 0,1,2,3 --prune_schedule cubic --pruner_name Magnitude --initial_threshold 1 --final_threshold 0.3 --warmup_steps 3000 --initial_warmup 1 --final_warmup 3 --beta3 0.85 --beta_meta 0.90 --deltaT 5  --dataset cifar100 --model_type ViT-B_16 --train_batch_size 512 --num_steps 20000 --eval_every 1000 --seed 9 --pretrained_dir model_cache/ViT-B_16.npz --output_root_folder DataLog/vit/adp_log --output_dir run
# python3 train.py --name cifar100_500 --device_num 0,1,2,3 --prune_schedule cubic --pruner_name Magnitude --initial_threshold 1 --final_threshold 0.1 --warmup_steps 3000 --initial_warmup 1 --final_warmup 3 --beta3 0.85 --beta_meta 0.90 --deltaT 5  --dataset cifar100 --model_type ViT-B_16 --train_batch_size 512 --num_steps 20000 --eval_every 1000 --seed 9 --pretrained_dir model_cache/ViT-B_16.npz --output_root_folder DataLog/vit/adp_log --output_dir run
