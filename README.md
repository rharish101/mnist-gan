# MNIST GAN

This is a repository for training a [conditional GAN](https://arxiv.org/abs/1411.1784) for the [MNIST dataset](yann.lecun.com/exdb/mnist/).
The GAN is optimized using the [Wasserstein loss](https://arxiv.org/abs/1701.07875) and the [Wasserstein gradient penalty](https://arxiv.org/abs/1704.00028).
A [DCGAN-like](https://arxiv.org/abs/1511.06434) architecture is used along with [spectral normalization](https://arxiv.org/abs/1802.05957) for the critic.

This implementation requires Python 3.6+.
This supports multi-GPU training on a single machine using TensorFlow's [`tf.distribute.MirroredStrategy`](https://www.tensorflow.org/tutorials/distribute/custom_training#create_a_strategy_to_distribute_the_variables_and_the_graph).

## Instructions

All Python scripts use argparse to parse commandline arguments.
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

    By default, this dataset is saved to the directory `datasets/MNIST`.

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
* Classifier: Run `classifier.py`:
    ```sh
    ./classifier.py
    ```

* GAN: Run `train.py` after training a classifier:
    ```sh
    ./train.py
    ```

The weights of trained models are saved in TensorFlow's ckpt format to the directory given by the `--save-dir` argument.
By default, this directory is `checkpoints` for both the classifier and the GAN.

Training logs are by default stored inside an ISO 8601 timestamp named subdirectory, which is stored in a parent directory (as given by the `--log-dir` argument).
By default, this directory is `logs/classifier` for classifier, and `logs/gan` for the GAN.

Copies of the CLI arguments are saved as a YAML file in both the model checkpoint directory and the timestamped log directory.
For the classifier, it is named `config-cls.yaml`, and for the GAN, it is named `config-gan.yaml`.

### Generation
Run `generate.py`:
```sh
./generate.py
```
The generated images are saved in the directory given by the `--output-dir` argument.
By default, this directory is `outputs`.
