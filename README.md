# Skin Cancer Detection

This repository contains code and resources for a skin cancer detection project using deep learning. The goal is to classify skin lesions as benign or malignant by analyzing medical images.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

Skin cancer is one of the most prevalent types of cancer worldwide. Early detection improves treatment outcomes and survival rates. This project uses deep learning models to classify skin lesions, helping with early identification of potentially cancerous growths.

## Dataset

This project uses the **HAM10000** ("Human Against Machine with 10000 training images") dataset:
- Source: [Kaggle Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Description: The dataset consists of 10,000 dermatoscopic images of skin lesions, each labeled as one of several categories (e.g., melanoma, nevus, benign keratosis).

**Dataset Usage:**  
Download the HAM10000 dataset from Kaggle and place it in the `data/` directory of this repository.

## Installation

### Prerequisites

- Python 3.x
- pip

### Install Dependencies

Clone the repository and install required packages:
```bash
git clone https://github.com/Vamsi1807/Skin_Cancer_Detection.git
cd Skin_Cancer_Detection
pip install -r requirements.txt
```

## Usage

1. Place the HAM10000 dataset in the `data/` directory.
2. Run the preprocessing script:
    ```bash
    python preprocess.py
    ```
3. Train the model:
    ```bash
    python train.py
    ```
4. Evaluate the trained model:
    ```bash
    python evaluate.py
    ```
5. Predict on new images:
    ```bash
    python predict.py --image path_to_image.jpg
    ```

## Results

After training, performance metrics (accuracy, precision, recall, F1-score) are displayed. Model files are saved in the `models/` directory and sample outputs in `results/`.
