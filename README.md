# Hickathon: Water shortage prediction

A machine learning and deep learning pipeline designed to predict groundwater levels and water shortages using piezometric, meteorological, and hydrological data. **This project was selected as a finalist and reached the final pitch.** 🏆

## 🛠 Environment Setup

This project uses `pyenv` for Python version management and `poetry` for dependency tracking to ensure full reproducibility.

```bash
# 1. Set the local Python version
$ pyenv local 3.11.2

# 2. Configure Poetry to create the virtual environment inside the project folder
$ poetry config virtualenvs.in-project true

# 3. Install all dependencies from the poetry.lock file
$ poetry install

## 🧠 Machine Learning Methodology

To solve the water shortage prediction challenge, we implemented a robust pipeline focusing on time-series forecasting and feature density.

### 1. Data Preprocessing & Engineering
* **Handling Missing Values:** Used iterative imputation and specialized "fill-nan" logic to maintain temporal consistency in hydrological data.
* **Categorical Encoding:** Applied **One-Hot Encoding** for static features and **Embeddings** for complex categorical variables to capture higher-dimensional relationships.
* **Dimensionality Reduction:** Utilized PCA/Autoencoders to condense high-dimensional meteorological inputs while retaining 95% of variance.

### 2. Model Architectures
We experimented with and implemented two primary architectures:
* **LSTM (Long Short-Term Memory):** A Recurrent Neural Network (RNN) approach designed to capture long-term temporal dependencies in piezometric levels.
* **Deep Neural Network (DNN):** A multi-layer perceptron (MLP) optimized for static and non-sequential feature integration.

### 3. Training Strategy
* **Loss Function:** Optimized using Mean Squared Error (MSE) to penalize large deviations in water level predictions.
* **Optimization:** Utilized Adam optimizer with a dynamic learning rate scheduler to prevent overfitting during long training epochs.
