# Differential Privacy in Deep Learning Models

This repository contains code and results for experiments on applying differential privacy techniques to deep learning models.

## Repository Structure

- `code/`: Python scripts for training models
  - `sample.py`: Sample CNN model
  - `lstm.py`: LSTM model
  - `bert.py`: BERT model
- `data/`: Dataset files
  - `MNIST/`: MNIST image data
- `results/`: Output logs and metrics
  - `run_results_*.pt`: Accuracy and privacy metrics
- `slurm/`: SLURM job files and output logs
- `README.md`: This file

## Experiments

The experiments train image classifiers on MNIST and text classifiers on IMDB using different model architectures:

- Sample CNN
- LSTM
- BERT

Differential privacy is applied using the Opacus library in PyTorch.

Key metrics reported:

- Test accuracy
- Privacy budget (epsilon)
- Training time overhead

## Usage

To run an experiment:

```bash
sbatch code/sample.py

This will train a model and log results to `results/run_results_mnist_<params>.pt`.

## Citation

```latex
@misc{differential-privacy-deep-learning,
  author = {John Doe},
  title = {Differential Privacy in Deep Learning Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/user/differential-privacy-deep-learning}}
}
