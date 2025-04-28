## Fixed-Bit QAT

### To quantize aware train the darts model
For CIFAR-10,
```
1) edit fixed_bit2.sh or fixed_bit4.sh and replace "cd <insert your home directory>" with the directory path the FBQAT folder
2) qsub fixed_bit2.sh or qsub fixed_bit4.sh
```
For Imagenet,
```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> --nnodes=<num_nodes> --node_rank=<node_rank> --master_addr=<ip_address> --master_port=<port> main.py --epochs 500 --step-epoch 15 --lr .1 -j 3 --batch-size <batch_size> --ac <architecture_to_quantize, Example PCDARTS or any other architecture specified in cnn/genotypes.py> --fixed_bit <2|4> --distributed --workers 2 --num_nodes <num_nodes> --num_gpus_per_node <num_gpus> --n_node <node_rank>
```
