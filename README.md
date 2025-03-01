

Do not run preprocessing on login node!

bash ```
srun --pty -p himem  bash
```
unzip isbi2025-ps3c-train-dataset.zip

run each preprocessing script (there is no 2_)

After preprocessing

bash```
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
pip install -r pytorch-CycleGAN-and-pix2pix/requirements.txt

mkdir logs
sbatch train.sh 
```

on local machine 
ssh -L 8097:localhost:8097 user@lawrence.usd.edu

in the login node, using the above shell
ssh -L 8097:localhost:8097 gpu001

browse to http://localhost:8097/