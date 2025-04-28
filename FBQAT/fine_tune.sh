cd <insert your darts directory>
module load pytorch
echo "Search cifar"

nvidia-smi
python search_cifar.py --epochs 50 --step-epoch 15 --lr .1 --lra .01 --cd .00335 -j 3 --batch_size 128