https://drive.google.com/file/d/1q5czzLDE_lQLJlko5Zpb10lUbPwVqkHW/view

!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

!pip install -r pytorch-CycleGAN-and-pix2pix/requirements.txt

!python pytorch-CycleGAN-and-pix2pix/train.py \
    --dataroot /content/drive/MyDrive/data/cyclegan_dataset_matched_256_with_splits/ \
    --name healthy2unhealthy_cyclegan \
    --model cycle_gan \
    --batch_size 12 \
    --n_epochs 40 --n_epochs_decay 20 \
    --display_freq 100 --print_freq 100 \
    --lambda_B 7.5 \
    --lambda_A 7.5