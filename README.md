# Manual Linear and Logistic Regression (NumPy Implementation)

## Overview

This repository contains a **clean-room implementation of Linear Regression and Logistic Regression using only NumPy**.  
The project was designed to demonstrate a detailed understanding of the mathematical foundations, optimization behavior, and validation methodology behind classical machine learning algorithms.

Instead of relying on machine learning libraries for model training logic, the algorithms are implemented directly using **vectorized matrix operations and gradient descent**. This exposes the internal mechanics of parameter estimation, loss minimization, and convergence behavior.

The project includes:

- Linear Regression implemented from first principles
- Logistic Regression implemented from first principles
- Batch Gradient Descent optimization
- Feature standardization
- L2 regularization
- Convergence diagnostics
- Benchmark comparison against Scikit-Learn
- Experiments on real datasets

The objective is to validate both the **correctness and stability** of the implementations through structured experimentation.

---

## Repository Structure


manual-linear-logistic-regression/
│
├── manual_ml_model.py
├── run_full_project_validation.py
├── plot_training_convergence.py
├── plot_training_convergence_report.py
│
├── experiments/
│ ├── benchmark_sklearn_linear.py
│ ├── benchmark_sklearn_logistic.py
│ ├── real_dataset_linear_experiment.py
│ ├── real_dataset_logistic_experiment_l2.py
│ ├── test_manual_linear.py
│ ├── test_manual_logistic.py
│ ├── test_linear_raw_coefficients.py
│ ├── test_logistic_raw_coefficients.py
│ └── test_logistic_large_sample.py
│
├── figures/
│ └── training_convergence_plot.png
│
├── report/
│ └── clean_room_ml_report.pdf
│
├── requirements.txt
├── README.md
└── .gitignore



---

## Model Implementation

The core model implementation is located in:

manual_ml_model.py


The main class:

ManualMLModel



supports two modes:
mode="linear"
mode="logistic"



### Key features

- Fully vectorized NumPy implementation
- Gradient descent optimization
- Feature scaling (standardization)
- L2 regularization
- Loss tracking for convergence analysis
- Defensive input validation

---

## Validation Pipeline

The full validation workflow is executed using:


run_full_project_validation.py


This script performs the following steps:

1. Synthetic dataset validation  
2. Benchmark comparison against Scikit-Learn models  
3. Real dataset regression experiment  
4. Real dataset classification experiment  
5. Convergence visualization

Running the script reproduces the complete set of experimental results.

---

## Convergence Visualization

During training, the loss value is recorded at each iteration of gradient descent.  
This information is used to generate convergence plots showing how the optimization progresses.

Example output:
<img width="799" height="455" alt="image" src="https://github.com/user-attachments/assets/b5b24f12-5e8e-4fc3-8dd1-09631729d835" />


These plots illustrate the characteristic phases of gradient descent:

1. rapid initial loss reduction  
2. stabilization as gradients decrease  
3. convergence plateau near the optimum

---

## Datasets Used

### California Housing Dataset

- Task: Regression
- Metric: R² score
- Source: Scikit-Learn dataset repository

This dataset predicts median housing prices based on demographic and geographic variables.

---

### Breast Cancer Wisconsin Dataset

- Task: Binary classification
- Metric: Accuracy
- Source: Scikit-Learn dataset repository

The dataset contains measurements derived from breast cancer biopsy images used to distinguish malignant and benign tumors.

---

## Example Results

### Synthetic Benchmark

| Metric | Manual Implementation | Scikit-Learn |
|------|------|------|
| Linear Regression R² | 0.9937 | 0.9937 |
| Logistic Regression Accuracy | 0.8435 | 0.8435 |

---

### Real Dataset Results

| Dataset | Model | Manual | Scikit-Learn |
|------|------|------|------|
| California Housing | Linear Regression | R² = 0.5756 | R² = 0.5758 |
| Breast Cancer Wisconsin | Logistic Regression | Accuracy = 0.9824 | Accuracy = 0.9824 |

These results demonstrate that the manual implementations produce predictions nearly identical to the reference implementations.

---

## Installation

Install dependencies using:

These plots illustrate the characteristic phases of gradient descent:

1. rapid initial loss reduction  
2. stabilization as gradients decrease  
3. convergence plateau near the optimum

---

## Datasets Used

### California Housing Dataset

- Task: Regression
- Metric: R² score
- Source: Scikit-Learn dataset repository

This dataset predicts median housing prices based on demographic and geographic variables.

---

### Breast Cancer Wisconsin Dataset

- Task: Binary classification
- Metric: Accuracy
- Source: Scikit-Learn dataset repository

The dataset contains measurements derived from breast cancer biopsy images used to distinguish malignant and benign tumors.

---

## Example Results

### Synthetic Benchmark

| Metric | Manual Implementation | Scikit-Learn |
|------|------|------|
| Linear Regression R² | 0.9937 | 0.9937 |
| Logistic Regression Accuracy | 0.8435 | 0.8435 |

---

### Real Dataset Results

| Dataset | Model | Manual | Scikit-Learn |
|------|------|------|------|
| California Housing | Linear Regression | R² = 0.5756 | R² = 0.5758 |
| Breast Cancer Wisconsin | Logistic Regression | Accuracy = 0.9824 | Accuracy = 0.9824 |

These results demonstrate that the manual implementations produce predictions nearly identical to the reference implementations.

---

## Installation

Install dependencies using:


pip install -r requirements.txt



Dependencies include:

- numpy
- matplotlib
- scikit-learn


---

## Running Experiments

Run the complete validation pipeline:


python run_full_project_validation.py



The script will:

- train the manual models
- compare results against Scikit-Learn
- evaluate real datasets
- generate convergence plots

---

## Report

A detailed report describing the mathematical derivations, implementation design, and experimental findings is available in:


report/clean_room_ml_report.pdf



Report title:

**Clean-Room Implementation of Linear and Logistic Regression Using NumPy: Mathematical Derivation, Optimization Behavior, and Empirical Validation**

---

## Concepts Demonstrated

This project demonstrates understanding of:

- supervised learning algorithms
- gradient-based optimization
- convex loss functions
- numerical stability in machine learning
- model validation methodology
- comparison with reference implementations

---

## Author


Meherab Hossain Shafin


Daffodil International University


## License

This repository is provided for educational and research purposes.





