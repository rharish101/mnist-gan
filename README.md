# MNIST GAN

This is a repository for training a conditional MNIST BiGAN.
This requires Python 3.6+.

## Instructions

All scripts use argparse to parse commandline arguments.
For viewing the list of all positional and optional arguments for any script, type:
```sh
./script.py --help
```

### Setup
1. Install all required Python libraries:
    ```sh
    pip install -r requirements.txt
    ```

2. Download the MNIST dataset using the provided script (requires cURL >= 7.19.0):
    ```sh
    ./download_mnist.sh [/path/where/dataset/should/be/saved/]
    ```

#### For Contributing
1. Install extra dependencies for development (with Python 3.6+):
    ```sh
    pip install -r requirements-dev.txt
    ```

2. Install pre-commit hooks:
    ```sh
    pre-commit install
    ```

**NOTE**: You need to be inside the virtual environment where you installed the above dependencies every time you commit.

### Training
Run `train.py`:
```sh
./train.py
```
The trained model is saved in TensorFlow's ckpt format (to the directory given by the `--save-dir` argument).
The training logs are by default stored inside an ISO 8601 timestamp named subdirectory, which is stored in a parent directory (as given by the `--log-dir` argument).

### Generation
Run `generate.py`:
```sh
./generate.py
```
The generated images are saved in the directory given by the `--output-dir` argument.
