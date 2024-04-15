#!/bin/sh            


### select queue, choose the gpuv100 queue when training "correctly"
#BSUB -q gpuv100
### BSUB -q hpcintrogpu

### name of job, output file and err
#BSUB -J train_model
#BSUB -o HPC_logs/train_model_%J.out
#BSUB -e HPC_logs/train_model_%J.err

### request the number of GPUs
#BSUB -gpu "num=1:mode=exclusive_process"
### request 32GB of GPU memory
#BSUB -R "select[gpu32gb]"
### request the number of CPU cores (at least 4x the number of GPUs)
#BSUB -n 4 
### we want to have this on a single node
#BSUB -R "span[hosts=1]"
### we need to request CPU memory, too (note: this is per CPU core)
#BSUB -R "rusage[mem=32GB]"


##BSUB -u s204090@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 


### wall time limit - the maximum time the job will run. Current 5 min
#BSUB -W 02:30
# end of BSUB options          


# load the correct  scipy module and python

module load python3/3.10.13
module load cuda/11.8

# activate the virtual environment
source PII_env/bin/activate

python model_train.py
            