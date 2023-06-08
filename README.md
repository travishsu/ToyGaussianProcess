# Toy example for Gaussian Process

## Description

This is a toy example for Gaussian Process. The script `test.py` generates a dataset from a sine function with some noise. Then, it fits a Gaussian Process to the dataset and plots the result.

## Installation

```bash
# Recommended to use a virtual environment
# python -m venv venv
# source venv/bin/activate
# pip install --upgrade pip
pip install -r requirements.txt
```

## Run the script

```bash
python test.py \
    --n_samples 10 \
    --noise 0.1 \
    --xmin 0 \
    --xmax 3.14 \
    --n_iter 5000 \
    --lr 1e-2 
```
