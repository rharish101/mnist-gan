# MNIST GAN

This is a repository for training a conditional MNIST BiGAN.
This requires Python 3.6+.

## Instructions

All scripts use argparse to parse commandline arguments.
For viewing the list of all positional and optional arguments for any script, type:
```
./script.py --help
```

### Setup
1. Install all required Python libraries:
```
pip install -r requirements.txt
```
2. Download the MNIST dataset using the provided script (requires cURL >= 7.19.0):
```
./download_mnist.sh [/path/where/dataset/should/be/saved/]
```

### Training
Run `train.py`:
```
./train.py
```
The trained model is saved in TensorFlow's ckpt format (to the directory given by the `--save-dir` argument).
The training logs are by default stored inside an ISO 8601 timestamp named subdirectory, which is stored in a parent directory (as given by the `--log-dir` argument).

### Generation
Run `generate.py`:
```
./generate.py
```
The generated images are saved in the directory given by the `--output-dir` argument.
