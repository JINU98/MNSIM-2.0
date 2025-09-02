#!/bin/sh
#SBATCH --job-name=test
#SBATCH --output=job%j.%N.out
#SBATCH --error=job%j.%N.err
#SBATCH -N 1  # Request 2 nodes
#SBATCH -n 16  # Request 96 cores (48 cores per node)
##SBATCH -p defq
##SBATCH --mem=100G
## Load your modules and run code here

hostname
date

module load python3/anaconda/2023.9
cd /work/jmalekar/MNSIM-2.0
source activate /work/jmalekar/MNSIM-2.0/mnsim

python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 2 -d 2048 -dff 16384 > micro_2b/gemma2.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 2 -d 2048 -dff 8192 > micro_2b/llama32_1B.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 2 -d 3072 -dff 8192 > micro_2b/llama32_3B.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 2 -d 4096 -dff 14336 > micro_2b/llama3_8B.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 2 -d 3072 -dff 8192 > micro_2b/phi3.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 2 -d 1536 -dff 4096 > micro_2b/bitnet_07.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 2 -d 4096 -dff 14336 > micro_2b/bitnet_8B.txt



python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 4 -d 2048 -dff 16384 > micro_4b/gemma2.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 4 -d 2048 -dff 8192 > micro_4b/llama32_1B.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 4 -d 3072 -dff 8192 > micro_4b/llama32_3B.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 4 -d 4096 -dff 14336 > micro_4b/llama3_8B.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 4 -d 3072 -dff 8192 > micro_4b/phi3.txt


python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 6 -d 2048 -dff 16384 > micro_6b/gemma2.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 6 -d 2048 -dff 8192 > micro_6b/llama32_1B.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 6 -d 3072 -dff 8192 > micro_6b/llama32_3B.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 6 -d 4096 -dff 14336 > micro_6b/llama3_8B.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 6 -d 3072 -dff 8192 > micro_6b/phi3.txt



python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 8 -d 2048 -dff 16384 > micro_8b/gemma2.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 8 -d 2048 -dff 8192 > micro_8b/llama32_1B.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 8 -d 3072 -dff 8192 > micro_8b/llama32_3B.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 8 -d 4096 -dff 14336 > micro_8b/llama3_8B.txt
python main.py --disable_accuracy_simulation -NN transformers -Weights None -b 8 -d 3072 -dff 8192 > micro_8b/phi3.txt







