#!/bin/sh
#SBATCH --job-name=scale-sim
#SBATCH -N 1
#SBATCH -n 40
#SBATCH --gres=gpu:1
##SBATCH -exclude=node[178-238]
#SBATCH --output gpu_opt_output.out
#SBATCH --error gpu_opt_error.err
#SBATCH -p gpu-v100-32gb
#SBATCH --mem=150G
##SBATCH --cpus-per-task 26

# v100-16gb-hiprio,v100-32gb-hiprio,AI_Center,gpu-v100-16gb,gpu-v100-32gb
# set to use first visible GPU in the machine
#export CUDA VISIBLE DEVICES=0
#export CUDA_VISIBLE_DEVICES=0,1

# the environment variable PYTHONUNBUFFERED to set unbuffered I/O for the whole batch script
export PYTHONUNBUFFERED=TRUE

# Load modules
module load python3/anaconda/2023.9
source activate /work/jmalekar/MNSIM-2.0/gpu_test/env
module load cuda/12.1
cd /work/jmalekar/MNSIM-2.0/gpu_test


# Function to print system information
print_system_info() {
    echo "------------------------------------"
    echo "Configuration Information:"
    echo "------------------------------------"
    echo "$(nvidia-smi)"
    echo "Date: $(date)"
    echo "Time: $(date +%T)"
    echo "OS: $(uname -s)"
    echo "Kernel: $(uname -r)"
    echo "Memory: $(free -m | grep Mem | awk '{print $2}') MB"
    echo "GPU Model: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
    echo "GPU Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
    echo "CUDA Version: $(nvcc --version | awk '/release/ {print $6, $7, $8}')"
    echo "Python Version: $(python --version)"
    echo "NumPy Version: $(python -c 'import numpy as np; print(np.__version__)')"
    echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
    # Project Information
    echo "Project Directory: $(git rev-parse --show-toplevel)"
    #echo "Git Commit: $(git rev-parse HEAD)"
    # Resources
    echo "PWD: $(pwd)"
    echo "GPUs: $CUDA_VISIBLE_DEVICES"
    echo "CPUS: $SLURM_CPUS_ON_NODE"
    echo "HOST: $HOSTNAME"
    echo "Python: $(which python)"
    echo "CONDA: $CONDA_PREFIX"
    echo "Partition: $SLURM_JOB_PARTITION"
    # Job Information (if applicable)
    echo "Job ID: $SLURM_JOB_ID"
    # User Info
    echo "User: $(whoami)"
}

# Print system information
print_system_info

# Function to run the pipeline with different stages
# astar batch size should be 1000
run_pipeline() {
    local CMD="python test_gpu_opt.py"

    echo "Running command:"
    while IFS= read -r line; do
        echo "$line"
    done <<< "$(echo "$CMD" | sed 's/ --/\n--/g')"
    echo ""

    echo "------------------------------------"
    echo "------------------------------------"

    # Capture start time
    START_TIME=$(date +%s%3N)

    # Run the pipeline script
    $CMD

    # Capture end time
    END_TIME=$(date +%s%3N)

    # Calculate execution time in milliseconds
    ELAPSED_TIME=$((END_TIME - START_TIME))

    # Convert milliseconds to days, hours, minutes, seconds, and milliseconds
    DAYS=$((ELAPSED_TIME / 86400000))
    ELAPSED_TIME=$((ELAPSED_TIME % 86400000))

    HOURS=$((ELAPSED_TIME / 3600000))
    ELAPSED_TIME=$((ELAPSED_TIME % 3600000))

    MINUTES=$((ELAPSED_TIME / 60000))
    ELAPSED_TIME=$((ELAPSED_TIME % 60000))

    SECONDS=$((ELAPSED_TIME / 1000))
    MILLISECONDS=$((ELAPSED_TIME % 1000))

    # Display elapsed time in the desired format
    echo "Elapsed Time for $1 stage (D:H:M:S:MS): $DAYS:$HOURS:$MINUTES:$SECONDS:$MILLISECONDS"

    echo "------------------------------------"
    echo "------------------------------------"
    echo ""
}


run_pipeline