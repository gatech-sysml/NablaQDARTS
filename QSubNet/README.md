## Qsubnet

#### Search on CIFAR-10 (How to search for a mixed precision network)
```
python search_cifar.py --epochs 50 --step-epoch 15 --lr .1 --lra .01 --cd .00335 -j 3 --batch-size <batch_size> --data <data_dir>
```

#### Evaluation on CIFAR-10 (How to Fine tune the mixed precision network you searched for above)
```
python main_cifar.py --epochs 50 --step-epoch 15 --lr .1 -j 3 --batch-size <batch_size> --ac <architecture_to_train, Example DARTS> --data <data_dir>
```

#### Search on ImageNet (How to search for a mixed precision network)
```
python3 -m torch.distributed.launch --nproc_per_node=<num_gpus> --nnodes=<num_nodes> --node_rank=<node_rank> --master_addr=<ip_address> --master_port=<port> search.py --epochs 50 --step-epoch 15 --lr .1 --lra .01 -j 3 --batch-size 1024 --distributed --workers 2 --num_nodes <num_nodes> --num_gpus_per_node <num_gpus> --n_node <node_rank>
```
##### Evaluation on ImageNet (How to Fine tune the mixed precision network you searched for above)
```
python3 -m torch.distributed.launch --nproc_per_node=<num_gpus> --nnodes=<num_nodes> --node_rank=<node_rank> --master_addr=<ip_address> --master_port=<port> main.py --epochs 500 --step-epoch 15 --lr .1 -j 3 --batch-size 2048 --distributed --ac <architecture_to_train, Example PCDARTS> --workers 2 --num_nodes <num_nodes> --num_gpus_per_node <num_gpus> --n_node <node_rank> --fixed_bit -1
```
