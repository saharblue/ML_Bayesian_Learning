# ML_Bayesian_Learning

This repository contains an implementation of a Bayesian Learning model. The goal of this model is to classify a new instance according to its feature values by maximizing the posterior probability, provided with various class options. The repository includes a Python file with the model implementation, a data directory containing several CSV files as examples of datasets for training and testing purposes, and a Jupyter notebook file for running tests and visual demonstrations.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)

## Requirements

This project is implemented in Python and requires the following Python libraries:

- pandas
- numpy
- scikit-learn

Additionally, you'll need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html).

## Installation

To get started, clone this repository to your machine:

```
git clone https://github.com/saharblue/ML_Bayesian_Learning.git
```

## Usage

1. **Load your dataset:** This implementation uses CSV files in the 'data' directory as examples of datasets. You can replace these files with your own CSV data. Make sure that your CSV files are in the correct format and placed in the 'data' directory.

2. **Train the model:** Run the Python file `hw3.py` to train the model on your dataset.

```python
python hw3.py
```

3. **Test and visualize:** Open the provided Jupyter notebook (`hw3.ipynb`) to run tests and visualize the model performance. You can view the notebook by using the following command:

```bash
jupyter notebook hw3.ipynb
```

