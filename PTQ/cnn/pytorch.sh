
cd <insert your full project directory here>
# module load pytorch/1
module load pytorch
#module load anaconda3
#conda activate darts

python test_modified.py --auxiliary --do_quant 1 --model_path cifar10_model.pt  --param_bits 8 --fwd_bits 8 --n_sample 10
