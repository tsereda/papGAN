#!/bin/bash

#SBATCH --job-name=papGAN
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G # Adjust memory as needed
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2 # Request 1 GPU

#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err



source ~/.bashrc

conda activate papGAN

nvidia-smi

python -m visdom.server > visdom.log 2>&1 &
# Wait a moment to ensure Visdom has time to start
sleep 5

python gpucheck.py
echo "Starting training..."
python pytorch-CycleGAN-and-pix2pix/train.py \
    --dataroot cyclegan_dataset_256_split/ \
    --name healthy2unhealthy_cyclegan \
    --model cycle_gan \
    --batch_size 6 \
    --n_epochs 40 --n_epochs_decay 20 \
    --display_freq 100 --print_freq 100 \
    --lambda_B 7.5 \
    --lambda_A 7.5