# Product Categorization Engine

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/francesco-s/product-categorization-engine)

A machine learning solution for automatically categorizing products based on their names and brands, delivered as a REST API. This system implements fine-tuning of transformer models (BERT/DistilBERT/RoBERTa) to achieve high accuracy while maintaining reasonable inference times.

## Overview

This project implements an end-to-end ML pipeline for product categorization in e-commerce. The solution fine-tunes transformer-based models on product data to achieve high accuracy while maintaining reasonable inference times.

![High-Level Architecture](images/High-LevelComponentDiagram.png)

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

## System Architecture

### Training and Evaluation Workflow

![Training and Evaluation Workflow](images/TrainingandEvaluationWorkflow.png)

The training pipeline consists of several key stages:
1. **Data Ingestion**: Raw product data is loaded and validated
2. **Preprocessing**: Text cleaning, tokenization, and feature engineering
3. **Model Training**: Fine-tuning transformer models with cross-validation
4. **Evaluation**: Comprehensive performance assessment with multiple metrics
5. **Model Persistence**: Saving trained models with metadata for inference

### Inference Pipeline

![Inference Workflow](images/InferenceWorkflow.png)

The inference system provides real-time predictions through:
1. **API Gateway**: RESTful endpoints for single and batch predictions
2. **Model Loading**: Efficient model loading with caching mechanisms
3. **Text Processing**: Same preprocessing pipeline as training
4. **Prediction**: Fast inference with confidence scoring
5. **Response Formatting**: Structured JSON responses with category mappings

### Production Deployment

![Production Deployment Stack](images/ProductionDeploymentStack.png)

The production environment includes:
- **Load Balancer**: Traffic distribution across multiple API instances
- **Container Orchestration**: Docker containers with auto-scaling
- **Model Registry**: Versioned model storage and management
- **Monitoring Stack**: Prometheus metrics and Grafana dashboards
- **Logging**: Centralized logging with ELK stack integration

## Detailed Setup

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)
- 4GB+ RAM (8GB+ recommended for training)
- NVIDIA GPU (optional, for faster training)

### Installation

1. Move to the project root:
   ```bash
   cd smart-product-categorization-engine
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required NLTK resources:
   ```bash
   python scripts/download_nltk_data.py
   ```

5. Create necessary directories:
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

## Alternative Approaches to Product Classification

Though this project centers on transformer fine-tuning, there are various alternative strategies to consider:

### Traditional Machine Learning

#### TF-IDF + SVM / Random Forest  
Converts text into sparse vectors based on term frequencies:

```python
# Example: "Samsung Galaxy S23 Ultra"
# → [0.42, 0.00, 0.31, 0.78, ...]  # sparse vector representation
```

* Pros: Fast, interpretable
* Cons: No understanding of word semantics (e.g., "smartphone" ≠ "mobile")

#### Naive Bayes with N-grams

Builds feature sets from contiguous word sequences:

```python
# Example: "Apple iPhone 15 Pro Max"
# → ["Apple", "iPhone", "15", "Pro", "Max", "Apple_iPhone", "iPhone_15"]
```

* Pros: Simple and effective
* Cons: No semantic awareness, fails on unseen n-grams

#### XGBoost / LightGBM on Engineered Features

Requires manual feature crafting:

```python
# Example features:
# brand = Apple, title_length = 28, has_storage = True, contains_number = True
```

* Pros: Strong performance when domain knowledge is encoded
* Cons: High maintenance, limited language understanding

### Deep Learning Approaches

#### CNN-Based Text Classifier

Learns local n-gram patterns using convolutional filters:

```python
# Learns patterns like: "Pro Max", "Gaming Laptop", "Ultra HD"
```

* Pros: Lightweight and fast
* Cons: Poor global context understanding

#### LSTM / GRU (RNNs)

Sequential models that process tokens one at a time:

```python
# "Samsung Galaxy S23 Ultra" → LSTM processes tokens in order
```

* Pros: Good at modeling order and context
* Cons: Slower training, less efficient than transformers

#### Static Word Embeddings + Classifier

Uses pre-trained vectors like Word2Vec or GloVe:

```python
# "Samsung Galaxy S23" → average of word embeddings → fed into a classifier
```

* Pros: Fast and compact
* Cons: No context-awareness (e.g., "Apple" as fruit vs brand)

### Hybrid and Advanced Techniques

#### Rule-Based + ML Combination

Uses expert knowledge for known patterns, ML for the rest:

```python
if brand == "Apple" and "iPhone" in title:
    category = "Electronics > Mobile Phones"
else:
    category = ML_model.predict(title)
```

* Pros: Highly interpretable and controlled
* Cons: Hard to scale and maintain

#### Multi-Modal Classification

Combines product image and text features:

```python
# image_embedding = CNN(image)
# text_embedding = BERT(description)
# prediction = classifier(concat(image_embedding, text_embedding))
```

* Pros: Improves performance where text alone is ambiguous
* Cons: Requires access to and preprocessing of images

#### Graph-Based Methods

Leverages relationships across product metadata:

```python
# Example: Graph with nodes → [Product] — [Brand] — [Category]
# Node embeddings capture similarity across product families
```

* Pros: Effective for hierarchical taxonomies and relational data
* Cons: Requires clean graph structure and advanced implementation

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

### Standalone Evaluation (Without Training)

You can evaluate a trained model without retraining using the standalone evaluation script:

```bash
# Basic usage with default paths
python scripts/evaluate_model.py

# With custom paths
python scripts/evaluate_model.py --model_path data/models/best_model.pt --test_file data/processed/test.csv

# With specific batch size
python scripts/evaluate_model.py --batch_size 16
```

This script:
- Loads a pre-trained model directly from a checkpoint
- Evaluates it on test data
- Generates comprehensive reports and visualizations

### After Training

If you've just trained a model, the evaluation metrics are also generated at the end of training:

```bash
python scripts/train.py
```

### Evaluation Outputs

Both evaluation methods generate:
- Performance metrics (accuracy, F1 score, precision, recall)
- Confusion matrix visualization
- Analysis of common misclassifications
- Detailed JSON report in the results directory

## API Usage

### Start the API Server

```bash
# Using uvicorn directly
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Or using Docker
docker-compose up -d
```

### Category Mapping

For the API to return meaningful category names instead of generic labels like "Category 0", there are two approaches:

1. **Use the model's embedded category mapping** (default):
   - During training, category mappings are saved in the model checkpoint
   - The API automatically loads these mappings when available

2. **Create an external category mapping file** (recommended):
   - Useful if the model's mapping is missing or incomplete
   - Creates a standalone mapping file that can be edited or extended
   ```bash
   # Generate category mapping from processed data
   python scripts/create_category_mapping.py
   
   # The mapping will be saved to data/processed/category_mapping.json
   ```
   
   This external mapping will be automatically loaded by the API if the model's internal mapping is missing.

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

## Project Structure

```
smart-product-categorization-engine/
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies
├── setup.py                     # Package installation
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Services definition
├── images/                      # Architecture diagrams
│   ├── High-LevelComponentDiagram.png
│   ├── InferenceWorkflow.png
│   ├── ProductionDeploymentStack.png
│   └── TrainingandEvaluationWorkflow.png
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

The architecture diagrams above illustrate how these components interact throughout the ML lifecycle, from data ingestion to production deployment.

## Design Patterns

Several design patterns have been implemented:

- **Factory Pattern**: For creating different model variants
- **Strategy Pattern**: For different text processing strategies
- **Repository Pattern**: For data access abstraction
- **Singleton Pattern**: For model loading in the API

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

### Evaluation Issues

If you encounter errors during evaluation:

1. **Tuple index error**: Use the standalone evaluation script instead:
   ```bash
   python scripts/evaluate_model.py
   ```

2. **Missing columns in test data**: The standalone script automatically handles missing columns:
   ```python
   # It will create combined_text if missing
   if 'combined_text' not in test_df.columns:
       test_df['combined_text'] = test_df['name'] + ' ' + test_df['brand']
   ```

3. **Model loading errors**: Check that the model path is correct and the model was properly saved:
   ```bash
   python scripts/evaluate_model.py --model_path path/to/your/model.pt
   ```

### API Category Label Issues

If the API returns generic labels like "Category 0" instead of meaningful names:

1. **Create an external category mapping file**:
   ```bash
   python scripts/create_category_mapping.py
   ```

2. **Check logs for mapping information**:
   ```bash
   # Run the API with more verbose logging
   uvicorn src.api.app_fixed:app --log-level debug
   ```

3. **See detailed documentation**: 
   Review the [Category Mapping Documentation](docs/category-mapping.md) for more details

## Future Improvements

1. **Model Optimization**: Quantization for faster inference
2. **Active Learning**: Feedback loop for continuous model improvement
3. **Enhanced Features**: Incorporate product image data
4. **A/B Testing Framework**: For comparing model versions