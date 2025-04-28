## 𝚫QDARTS
## Usage for CIFAR-10
### Search on CIFAR-10 (How to search for the best architecture & mixed precision network)

To search with [2,4] as possible bit precisions,
```
python train_search.py --data <data_dir>
```
To search with [2,4,8] as possible bit precisions,
```
python train_search.py --data <data_dir> --allow8bit

```

### Evaluation on CIFAR-10 (How to Fine tune the mixed precision & architecture you searched for above)
The genotype of the architecture identified during the search phase is to be added to genotypes.py and the variable name corresponding to be it is to be provided for the --arch.
The values are gamma parameters are stored in the checkpoint folder as a result of the search process. The path to this checkpoint is to be provided as input to the train script
```
python train.py --data <data_dir> --arch <architecture_to_train> --weights_path <path_to_gamma_checkpoint>/gamma_checkpoint.t7 [--allow8bit]
```

## Usage for Imagenet
### Pre-requisites for experiments with Imagenet
The imagenet experiments use ffcv data loaders to reduce data bottle which requires the before steps
Clone https://github.com/libffcv/ffcv & https://github.com/libffcv/ffcv-imagenet

Run the below to create the conda environment with the necessary dependencies
```
conda create -y -n qdarts_ffcv_env python=3.10.8 cupy pkg-config libjpeg-turbo opencv pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge
conda activate qdarts_ffcv_env
pip install ffcv
```

### Search on ImageNet (How to search for the best architecture & mixed precision network)
```
export NCCL_P2P_DISABLE=1

python3 -m torch.distributed.launch --nproc_per_node=<num_gpus> --nnodes=<num_nodes> --node_rank=<node_rank> --master_addr=<ip_address> --master_port=<port>  train_search_imagenet.py --dist --workers 2 --num_nodes <num_nodes> --num_gpus_per_node <num_gpus> --n_node <node_rank> --save <checkpoint_folder_path> --complexity-decay <control_parameter_to_pick; Example, 1e-5> [--allow8bit]
```
Similar to the above, the genotype of the architecture identified during the search phase is to be added to genotypes.py and the variable name corresponding to be it is to be provided for the --arch. The values are gamma parameters are stored in the checkpoint folder as a result of the search process. The path to this checkpoint is to be provided as input to the train script. This folder needs to be created before running the above script.

### Evaluation on ImageNet (How to Fine tune the mixed precision & architecture you searched for above)
```
python3 -m torch.distributed.launch --nproc_per_node=<num_gpus> --nnodes=<num_nodes> --node_rank=<node_rank> --master_addr=<ip_address> --master_port=<port> train_imagenet.py --dist --weights_path .<path_to_gamma_checkpoint>/gamma_checkpoint.t7 --arch <architecture_to_train> --workers 2 --num_nodes <num_nodes> --num_gpus_per_node <num_gpus> --n_node <node_rank> --save <log_path> --auxiliary [--allow8bit]
```
The path_to_gamma_checkpoint above will be the same as the checkpoint folder used in the Search phase.
