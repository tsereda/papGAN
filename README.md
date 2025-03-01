https://drive.google.com/file/d/1q5czzLDE_lQLJlko5Zpb10lUbPwVqkHW/view

Do not run preprocessing on login node!

bash ```
srun --pty -p himem  bash
```

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