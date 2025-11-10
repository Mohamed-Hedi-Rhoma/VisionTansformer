# Plant Disease Classification using Hybrid CNN-Vision Transformer

## Overview
This project implements a hybrid CNN-Vision Transformer model for plant disease classification using data scraped from iNaturalist website.

## Data Collection
The training data is scraped from the iNaturalist website using the provided scraping tool.

**Steps:**
- Use the `download_data.py` file located in the `dataset/` directory
- You can modify the list of diseases you want to work with based on image availability on iNaturalist
- The script will automatically download and organize the plant disease images

## Model Architecture
The model combines CNN feature extraction with Vision Transformer attention mechanisms:

- **CNN Backbone**: Extracts local features from plant images  
- **Vision Transformer**: Processes image patches using multi-head self-attention mechanism
- **Classification Head**: Final layer that predicts the specific disease type


## Installation
First, install pixi package manager:
```bash
curl -fsSL https://pixi.sh/install.sh | bash 
# Install project dependencies
pixi install
```


## Training
To train the model with your configuration:
```bash
python train.py
```
The training script will:

- Preprocess the scraped data
- Train the hybrid CNN-ViT model
- Save checkpoints and training results

## Usage Summary

- Run python dataset/get_data.py to scrape plant disease data from iNaturalist
- Edit disease list in the scraping script based on available images
- Start training with python train.py