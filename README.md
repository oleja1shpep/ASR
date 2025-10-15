# Automatic Speech Recognition (ASR) with PyTorch


## Installation

Follow these steps to install the project:

1. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).


   ```bash
   # create env
   conda create -n project_env python=3.10

   # activate env
   conda activate project_env
   ```

2. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## Checkpoints

1. Model checkpoint:

   ```bash
   gdown https://drive.google.com/uc?id=1LVKpi6Nu7Rk8FcNs88paNXbCqe52kl7e -O model_best.pth
   ```

2. LM download

   ```python
   from torchaudio.models.decoder._ctc_decoder import download_pretrained_files


   download_pretrained_files("librispeech-3-gram")
   ```

3. Logs:

   a. Overfit logs download

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

To run inference on custom dataset

```bash
python3 inference.py -cn=inference_custom HYDRA_CONFIG_ARGUMENTS
```

More information about inference in demo.ipynb

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
