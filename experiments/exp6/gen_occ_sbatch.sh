#!/bin/bash
exists_in_list () {
    LIST=$1
    DELIMITER=","
    VALUE=$2
    LIST_WHITESPACES=`echo $LIST | tr "$DELIMITER" " "`
    for x in $LIST_WHITESPACES; do
        if [ "$x" = "$VALUE" ]; then
	   echo 0
	   return
        fi
    done
    echo 1
}
GRAPH=$1
BATCH=$3
MODEL=$4
CACHE=$2
GRAPHLIST="ogbn-arxiv,ogbn-products,reorder-papers100M,amazon"
BATCHLIST="1024,4096,16384"
MODELLIST="gcn,gat"
CACHELIST="0,.10,.25,.5,1"
if [ "$(exists_in_list $GRAPHLIST $GRAPH)" = "1" ] ; then
  echo "graph Not found"
  return 1
fi
if [ "$(exists_in_list $BATCHLIST $BATCH)" = "1" ] ; then
  echo "batch Not found"
  return 1
fi
if [ "$(exists_in_list $MODELLIST $MODEL)" = "1" ] ; then
  echo "model Not found"
  return 1
fi
if [ "$(exists_in_list $CACHELIST $CACHE)" = "1" ] ; then
  echo "cache Not found" $CACHELIST $CACHE
  return 1
fi

SLURMJOBNAME=gbmc_occ_$GRAPH\_$BATCH\_$MODEL\_$CACHE
echo $SLURMJOBNAME
DIR=/home/spolisetty_umass_edu/OCC-GNN/experiments/exp6/slurm-log-occ
mkdir -p $DIR
JOBFILE=$DIR/$SLURMJOBNAME\_sbatch.sh
rm -rf $JOBFILE
echo "#!/bin/bash" >> $JOBFILE
echo "#SBATCH --job-name=$SLURMJOBNAME   # Job name" >> $JOBFILE
echo "#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL)" >> $JOBFILE
echo "#SBATCH --mail-user=spolisetty@cs.umass.edu     # Where to send mail" >> $JOBFILE
echo "#SBATCH --ntasks=1                    # Run on a single CPU" >> $JOBFILE
echo "#SBATCH --mem=200G                     # Job memory request" >> $JOBFILE
echo "#SBATCH --time=8:00:00               # Time limit hrs:min:sec" >> $JOBFILE
echo "#SBATCH --gres=gpu:4" >> $JOBFILE
echo "#SBATCH --partition=gypsum-2080ti" >> $JOBFILE
echo "#SBATCH --gpus-per-node=4" >> $JOBFILE
echo "#SBATCH --output=$DIR/slurm-job-$SLURMJOBNAME.log   # Standard output and error log" >> $JOBFILE
echo "#SBATCH -e $DIR/slurm-$SLURMJOBNAME.err" >> $JOBFILE
echo "#SBATCH -o $DIR/slurm-$SLURMJOBNAME.out" >> $JOBFILE
echo "pwd; hostname; date" >> $JOBFILE

echo "module load conda" >> $JOBFILE
echo "module load cuda/11.3">> $JOBFILE


echo "source /home/spolisetty_umass_edu/.bashrc " >> $JOBFILE

echo "conda deactivate" >> $JOBFILE
echo "conda activate /work/spolisetty_umass_edu/occ-env/" >> $JOBFILE

echo "echo conda setup done!" >> $JOBFILE

echo "export PYTHONPATH=/home/spolisetty_umass_edu/OCC-GNN/pagraph" >> $JOBFILE

echo "export DGLBACKEND=pytorch" >> $JOBFILE
echo "export HOME=/home/spolisetty_umass_edu/OCC-GNN/" >> $JOBFILE

echo "echo start experiment $CACHE $MODEL $BATCH $GRAPH" >> $JOBFILE


echo "cd \$HOME; python3 experiments/exp6/occ.py --graph $GRAPH --cache-per $CACHE --model $MODEL --batch-size $BATCH" >> $JOBFILE

echo "echo All Done " >> $JOBFILE

echo "written to $JOBFILE"
sbatch $JOBFILE
