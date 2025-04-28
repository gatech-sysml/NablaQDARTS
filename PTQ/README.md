# PC-DARTS_with_Post_Training_Quantization

## Running post-training quantization experiments:

### Step 1: clone this repo
`git clone <clone_url>`

### Step 2: cd into the cnn directory
`cd cnn`

### Step 3: run the command (baseline):
For CIFAR-10,
```
python test_modified.py --auxiliary --do_quant 0 --model_path <path_to_trained_model>
```
For Imagenet,
```
python test_modified_imagenet.py --auxiliary --do_quant 0 --model_path <path_to_trained_model>
```
here, do_quant parameter defines whether we want to perform quantization or not. by default its 0 (means no quantization)

### Step 4: run the command (with quant):
For CIFAR-10,
```
python test_modified.py --auxiliary --do_quant 1 --model_path <path_to_trained_model_to_be_quantized> --param_bits <2|4|8> --fwd_bits <2|4|8> --n_sample 10
```
For Imagenet,
```
python test_modified_imagenet.py --auxiliary --do_quant 1 --model_path <path_to_trained_model_to_be_quantized> --param_bits <2|4|8> --fwd_bits <2|4|8> --n_sample 10
```

### Quantization parameters:
These can be found in the **test_modified.py** script:\
**quant_method:**, quantization function: possible choice: linear|minmax|log|tanh\
**param_bits:** bit-width for parameters\
**bn_bits:** bit-width for running mean and std in batchnorm layer (should be higher)\
**fwd_bits:** bit-width for layer output (activation)\
**n_sample:** number of samples to calculate the scaling factor

