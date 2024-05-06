# nuclei-segmentation

Development of machine learning model for instance segmentation of nuclei cells for Kaggle Data Science Bowl 2018 challenge. 

Link to challenge: https://www.kaggle.com/c/data-science-bowl-2018

Link to download dataset: https://drive.google.com/file/d/16gp8kPxMFrSDiLjgw2eaZmtrDccOjKKR/view?usp=sharing

Link to download model: https://drive.google.com/file/d/1UbvdHFv5pvSWy_V-yNSDmrNQmF2bbLSB/view?usp=sharing

## Data set

This application use the dataset from Kaggle

## Data structure

```text
nuclei-segmentation
    data
        data-science-bowl-2018.zip (should place there)
    doc
        report.pdf
    models
        model_name
            model.h5
            history.csv
    notebook
        nuclei.ipynb (Google Colab)
    src
    submissions
    poetry.lock
    pyproject.toml
```

## Installation

Make sure first that poetry is installed:

```bash
pip install poetry
```

Then, install all necessary packages:

```bash
poetry install
```

## Usage

Once this is done, you can run the application:

```bash
poetry run python src/main.py
```

## Models

This application uses deep learning and convolutionnal neural networks (CNN) with tensorflow/keras library. You can either train a model inside the application using `Train a new model` part but a GPU is recommended to function properly. Another option is to use Google Colab to train a model and then insert in the `models` folder following this structure:

```markdown
models
    ├── model_name
    │   ├── model.h5
    │   └── history.csv
```

Google colab notebook is `nuclei.ipynb` available in `notebook` folder. Model and history should normally be saved inside your Google Drive.

## Screenshots of application

![Training of a model with DSB dataset](screenshots/training.png)
![Prediction of nuclei with trained model with some metrics](screenshots/inference.png)
![Performance of a mode](screenshots/performance.png)
