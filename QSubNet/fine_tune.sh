cd <insert project directory here>
module load pytorch
echo "Search cifar"

nvidia-smi
python main_cifar.py --epochs 50 --step-epoch 15 --lr .1 -j 3 --batch-size 128 --ac arch_model_best.pth.tar
