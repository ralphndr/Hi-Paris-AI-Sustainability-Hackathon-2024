# Hickathon: Water shortage prediction

A machine learning and deep learning pipeline designed to predict groundwater levels and water shortages using piezometric, meteorological, and hydrological data. 

## 🛠 Environment Setup

This project uses `pyenv` for Python version management and `poetry` for dependency tracking to ensure full reproducibility.

```bash
# 1. Set the local Python version
$ pyenv local 3.11.2

# 2. Configure Poetry to create the virtual environment inside the project folder
$ poetry config virtualenvs.in-project true

# 3. Install all dependencies from the poetry.lock file
$ poetry install