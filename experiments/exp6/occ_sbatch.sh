#!/bin/bash
#SBATCH --job-name=test_run_occ    # Job name
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=spolisetty@cs.umass.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=200G                     # Job memory request
#SBATCH --time=8:00:00               # Time limit hrs:min:sec
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu-preempt
#SBATCH --output=/home/spolisetty_umass_edu/OCC-GNN/pagraph/slurm-job.log   # Standard output and error log
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
pwd; hostname; date

module load conda
module load cuda/11.3


source /home/spolisetty_umass_edu/.bashrc 

conda deactivate
conda activate /work/spolisetty_umass_edu/occ-env/

echo "Conda setup done!"

export PYTHONPATH=/home/spolisetty_umass_edu/OCC-GNN/cslicer

export DGLBACKEND=pytorch
export HOME=/home/spolisetty_umass_edu/OCC-GNN/

echo "Start experiment"


cd $HOME; python3 experiments/exp6/occ.py > out.log

echo "All Done "


date

