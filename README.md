# BERT-gang

Using Knowledge Distillation to learn from BERT-like models.

## Setup

You need `Ubuntu18.04+` and `python3.8+` to run all the code.

```sh
# clone the project
git clone https://github.com/ProFatXuanAll/BERT-gang

# change to the project directory
cd BERT-gang

# create data folder
mkdir data data/fine_tune

# create virtual environment
python3.8 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel

# install dependencies
pip install -r requirements.txt
```

## Profiling

- Use `htop` to monitor CPU and memory usage.
  - You need to install `htop` first by `apt-get install htop`.
- Use `nvidia-smi` to monitor GPU and memory usage.
  - You need to install `cuda` driver first.
  - Required `cuda10+`.
- Use `tensorboard --logdir='./data/fine_tune_experiment/log'` to monitor loss, learning rate and accuracy.
