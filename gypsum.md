## Contains GYPSUM command reference. 

1. Run a bash script to get a single node for development with GPU
srun --pty -p m40-short --mem=24000 --gres=gpu:1 bash

2. module load cuda11

3. torch 1.8.1,
   cuda111
   pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html


