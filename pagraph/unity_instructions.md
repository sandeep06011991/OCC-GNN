Notes to run pagraph:

This is a general outline. Do check the documentation, welcome to come up with better ideas.

Target environment
PyTorch (v >= 1.3)
DGL (v == 0.4.1)
dgl-cu102 == 0.4.3

Login into the unity cluster and set up ssh keys
https://unity.rc.umass.edu/panel/account.php
ssh <NET_ID>_umass_edu@unity.rc.umass.edu
https://unity.rc.umass.edu/docs/index.html
This will get you to the head node.
Unity cluster uses slurm which is a cluster job scheduler.
To test and build we need interactive jobs using srun
srun -p gpu-preempt --gres=gpu:4 --mem=64G --pty bash
Gives you a node with 4 gpus
Load modules needed
module load cuda/10.2
module load miniconda


Create a conda environment to locally manipulate similar to virtual env
mkdir packages
conda config --add  pkgs_dirs packages
[Adding a package directory was a problem I had, conda is a standardized library. Do try to debug for other issues]
conda create -p myenv
conda activate myenv/
conda install pip==20.0.2

pip install numpy==1.18.5 (install numpy before dgl as dgl might choose a higher version of numpy causing broken dependencies)

Install torch:
Try removing the flag –no-cache-dir
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install --no-cache-dir torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102
pip install dgl-cu102==0.4.3

python3 server/pa_server.py --dataset ogbn-arxiv &
