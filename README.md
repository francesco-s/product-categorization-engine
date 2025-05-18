# Smart Product Categorization

A machine learning solution for automatically categorizing products based on their names and brands, delivered as a REST API. This system implements fine-tuning of transformer models (BERT/DistilBERT/RoBERTa) to accurately predict product categories.

## Overview

This project implements an end-to-end ML pipeline for product categorization in e-commerce. The solution fine-tunes transformer-based models on product data to achieve high accuracy while maintaining reasonable inference times.

## Features

- **Transformer-Based Fine-Tuning**: Fine-tunes state-of-the-art language models on product data
- **Flexible Model Selection**: Support for BERT, DistilBERT, and RoBERTa models
- **REST API**: Simple, scalable interface for making predictions
- **Containerized Deployment**: Easy deployment with Docker and Docker Compose
- **Monitoring**: Built-in monitoring with Prometheus and Grafana
- **Optimized Training**: GPU support, batch processing, and early stopping

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK resources
python scripts/download_nltk_data.py

# 4. Download the dataset
python scripts/download_data.py

# 5. Train the model with DistilBERT (faster)
python scripts/train.py --model_type distilbert --batch_size 16 --num_epochs 3

# 6. Start the API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Project Structure

```
product-categorization/
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies
├── setup.py                     # Package installation
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Services definition
├── docs/                        # Extended documentation
│   ├── design.md                # Design decisions, architecture
│   ├── data_analysis.md         # Data analysis findings
│   └── evaluation.md            # Model evaluation results
├── notebooks/                   # Jupyter notebooks for exploration
├── data/                        # Data directory
│   ├── raw/                     # Raw data
│   ├── processed/               # Processed data
│   └── models/                  # Saved models
├── src/                         # Source code
│   ├── config.py                # Configuration
│   ├── data/                    # Data processing modules
│   ├── models/                  # Model definitions
│   ├── training/                # Training logic
│   ├── utils/                   # Utility functions
│   └── api/                     # API module
├── scripts/                     # Utility scripts
│   ├── download_data.py         # Data download script
│   ├── download_nltk_data.py    # NLTK resources download
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── start.sh                 # API startup script
├── configs/                     # Configuration files
│   ├── training.yaml            # Training configuration
│   └── evaluation.yaml          # Evaluation configuration
├── monitoring/                  # Monitoring configuration
└── tests/                       # Unit and integration tests
```

## Architecture

This solution follows a modular architecture:

1. **Data Processing Module**: Handles data loading, cleaning, preprocessing, and feature engineering
2. **Model Module**: Implements transformer-based classifier fine-tuning using PyTorch
3. **Training Module**: Manages the training and evaluation process
4. **API Module**: Provides a REST interface for model inference

## Design Patterns

Several design patterns have been implemented:

- **Factory Pattern**: For creating different model variants
- **Strategy Pattern**: For different text processing strategies
- **Repository Pattern**: For data access abstraction
- **Singleton Pattern**: For model loading in the API

## Detailed Setup

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)
- 8GB+
- NVIDIA GPU (optional, for faster training)

### Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK resources:
   ```bash
   python scripts/download_nltk_data.py
   ```

4. Create necessary directories:
   ```bash
   mkdir -p data/raw data/processed data/models/checkpoints results logs
   ```

### GPU Acceleration (Optional)

For faster training, you can use GPU acceleration if you have an NVIDIA GPU:

1. Install CUDA and cuDNN following [NVIDIA's instructions](https://developer.nvidia.com/cuda-downloads)

2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Verify GPU availability:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

## Model Information

This project implements fine-tuning of transformer models for text classification. Available models include:

| Model | Type | Parameters | Training Speed | Inference Speed | Relative Accuracy |
|-------|------|------------|----------------|-----------------|-------------------|
| BERT | bert-base-uncased | 110M | Slower | Slower | Baseline (100%) |
| RoBERTa | roberta-base | 125M | Slower | Slower | 100-102% |
| DistilBERT | distilbert-base-uncased | 66M | 60% faster | 60% faster | ~97% |

### Fine-tuning vs. Feature Extraction

This project uses **fine-tuning**, not just feature extraction. This means:
- The pre-trained transformer weights are updated during training
- The model adapts to the specific language patterns of product descriptions
- Better performance on the categorization task at the cost of more computation

## Training Process

### Downloading the Data

```bash
# Download the Amazon products dataset
python scripts/download_data.py
```

### Training Options

```bash
# Train with default settings (BERT)
python scripts/train.py

# Use DistilBERT for faster training
python scripts/train.py --model_type distilbert

# Customize batch size and epochs
python scripts/train.py --model_type distilbert --batch_size 16 --num_epochs 3

# See all options
python scripts/train.py --help
```

### Training with Limited Resources

If you have limited RAM or GPU memory:

```bash
# Use smallest model with reduced batch size
python scripts/train.py --model_type distilbert --batch_size 8 --num_epochs 2
```

### Training on Google Colab

For free GPU acceleration:

1. Upload your project to Google Drive or GitHub
2. Create a Colab notebook and mount your Drive or clone from GitHub
3. Install dependencies and run the training script with GPU runtime

## Evaluation

Evaluate model performance:

```bash
python scripts/evaluate_model.py
```

This generates:
- Performance metrics (accuracy, F1 score, etc.)
- Confusion matrix visualization
- Analysis of common misclassifications

## API Usage

### Start the API Server

```bash
# Using uvicorn directly
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Or using Docker
docker-compose up -d
```

### Making Predictions

#### Using curl:

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"name": "Samsung Galaxy S23 Ultra, 512GB, Phantom Black", "brand": "Samsung"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict-batch" \
     -H "Content-Type: application/json" \
     -d '[
           {"name": "Samsung Galaxy S23 Ultra, 512GB", "brand": "Samsung"},
           {"name": "Nike Air Force 1 07 Sneaker", "brand": "Nike"}
         ]'
```

#### Using Python:

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Product data
product = {
    "name": "Apple iPhone 14 Pro Max, 256GB, Deep Purple",
    "brand": "Apple"
}

# Make prediction
response = requests.post(url, json=product)
result = response.json()

print(f"Predicted category: {result['category']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Deployment

The solution is containerized for easy deployment:

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Troubleshooting

### Missing NLTK Resources

If you encounter errors about missing NLTK resources:

```bash
python scripts/download_nltk_data.py
```

### Dataset Column Issues

The system automatically handles the Amazon Products dataset's column structure by:
- Extracting brand from product name if 'brand' column is missing
- Creating categories from main_category and sub_category

### Memory Issues During Training

If you encounter memory issues:

```bash
# Use a smaller model
python scripts/train.py --model_type distilbert

# Reduce batch size
python scripts/train.py --batch_size 8

# Edit train.py to use a subset of data
df = df.sample(n=10000, random_state=42)  # Use only 10k samples
```

### CUDA Issues

If you encounter CUDA errors:

1. Verify CUDA is installed: `nvcc --version`
2. Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Install compatible PyTorch version: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Future Improvements

1. **Model Optimization**: Quantization for faster inference
2. **Active Learning**: Feedback loop for continuous model improvement
3. **Enhanced Features**: Incorporate product image data
4. **A/B Testing Framework**: For comparing model versions
